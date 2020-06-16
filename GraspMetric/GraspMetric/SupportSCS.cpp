#include "Support.h"
#ifdef SCS_SUPPORT
#include "Utils.h"
#include "SparseUtils.h"
#include "ScsInterface.h"
#include <Eigen/Eigenvalues>

USE_PRJ_NAMESPACE

//SupportQ1SCS
SupportQ1SCS::SupportQ1SCS(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt):Support(mesh,metric,metricSqrt) {}
scalarD SupportQ1SCS::supportPoint(const Vec6d& d,const IDSET& ids,bool directed)
{
  _errorFlag=0;
  if(_scs==NULL || directed) {
    clearModel();
    _scs.reset(new ScsInterface);
    sizeType off=0;
    Mat6Xd W=Mat6Xd::Zero(6,(sizeType)ids.size()*3+6);
    W.block<6,6>(0,(sizeType)ids.size()*3).setIdentity();
    for(IDSET::const_iterator beg=ids.begin(),end=ids.end(); beg!=end; beg++,off++) {
      //frictional cone
      Mat3d m;
      m.row(0)=_mesh->inNormal(*beg).transpose()*_mesh->theta();
      m.row(1)=_mesh->tangent1(*beg).transpose();
      m.row(2)=_mesh->tangent2(*beg).transpose();
      _scs->constraintSecondOrder(-m,off*3,Vec3d::Zero(),ScsInterface::SECOND_ORDER);
      //compute w-\sum(GI*F)
      W.block<6,3>(0,off*3)=-_mesh->getG(*beg);
    }
    //||f||^2
    fNormConstraint(ids);
    //0=w-\sum(GI*F)
    _scs->constraintLinear(W,0,Vec6d::Zero(),ScsInterface::EQUALITY);
    //directed
    if(directed) {
      Vec6d dN=d.normalized();
      Mat6d projT=(Mat6d::Identity()-dN*dN.transpose())*_metricSqrt;
      _scs->constraintLinear(projT,(sizeType)ids.size()*3,Vec6d::Zero(),ScsInterface::EQUALITY);
    }
  }
  //solve
  Cold c=concat(Cold::Zero((sizeType)ids.size()*3),-_metricSqrt*d);
  if(_scs->solve(c,_fx)<0) {
    writeError(d,ids,directed);
    _errorFlag=1;
    return ScalarUtil<scalarD>::scalar_max();
  }
  _fOut=_fx.segment(0,(sizeType)ids.size()*3);
  _wOut=_fx.segment<6>((sizeType)ids.size()*3);
  return -c.dot(_fx);
}
void SupportQ1SCS::fNormConstraint(const IDSET& ids)
{
  //\sum f_N<=1
  sizeType off=0;
  ScsInterface::Vec coef;
  coef.setZero(ids.size()*3);
  for(IDSET::const_iterator beg=ids.begin(),end=ids.end(); beg!=end; beg++,off++)
    coef.segment<3>(off*3)=_mesh->inNormal(*beg);
  _scs->constraintLinear(coef,0,1,ScsInterface::INEQUALITY);
}
void SupportQ1SCS::clearModel()
{
  _scs=NULL;
}
//SupportQInfSCS
SupportQInfSCS::SupportQInfSCS(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt):SupportQ1SCS(mesh,metric,metricSqrt) {}
void SupportQInfSCS::fNormConstraint(const IDSET& ids)
{
  //f_N<=1
  sizeType off=0;
  for(IDSET::const_iterator beg=ids.begin(),end=ids.end(); beg!=end; beg++,off++)
    _scs->constraintLinear(_mesh->inNormal(*beg),off*3,1,ScsInterface::INEQUALITY);
}
//SupportQSMSCS
SupportQSMSCS::SupportQSMSCS(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt,
                             const IDSET& sigmaIds,const sizeType& progressive,const scalarD& scale)
  :SupportQ1SCS(mesh,metric,metricSqrt),_sigmaIds(sigmaIds),_progressive(progressive),_scale(scale),_scaleF(1000) {}
scalarD SupportQSMSCS::supportPoint(const Vec6d& d,const IDSET& ids,bool directed)
{
  _initFlag=false;
  _scaleFCurr=_scaleF;
  _sigmaIdsActiveSet.clear();
  if((sizeType)ids.size()>_progressive || (sizeType)_sigmaIds.size()==_mesh->nrSigma()) {
    //not using progressive
    return SupportQ1SCS::supportPoint(d,ids,directed);
  } else {
    //using progressive
    _initFlag=true;
    sizeType maxId=-1;
    std::vector<scalarD> H;
    scalarD metric=ScalarUtil<scalarD>::scalar_max(),metric2;
    _sigmaIdsActiveSet=_sigmaIds;
    //progressive loop
    while(true) {
      if(maxId==-1) {
        clearModel();
        metric2=SupportQ1SCS::supportPoint(d,ids,directed);
        _initFlag=false;
        //debugFNorm(ids);
        if(_errorFlag==1)
          return ScalarUtil<scalarD>::scalar_max();
        //fetch force
        _fOut=_fx.segment(0,(sizeType)ids.size()*3);
      } else {
        //just add one more constraint and solve again
        Matd coef(_mesh->sigmaFromF(maxId,ids)/_scale);
        Mat3Xd coefs=Mat3Xd::Zero(3,coef.cols()*3);
        for(sizeType i=0; i<coef.cols(); i++)
          coefs.block<3,3>(0,i*3)=Eigen::Map<Mat3d>(&(coef.coeffRef(0,i)));
        _scs->constraintLinearMatrixInequality( coefs,0,Mat3d::Identity(),ScsInterface::SEMI_DEFINITE);
        _scs->constraintLinearMatrixInequality(-coefs,0,Mat3d::Identity(),ScsInterface::SEMI_DEFINITE);
        //solve
        Cold c=concat(Cold::Zero((sizeType)ids.size()*3),-_metricSqrt*d);
        if(_scs->solve(c,_fx)<0) {
          writeError(d,ids,directed);
          _errorFlag=1;
          return ScalarUtil<scalarD>::scalar_max();
        }
        //fetch force
        _fOut=_fx.segment(0,(sizeType)ids.size()*3);
        //fetch
        _wOut=_fx.segment<6>((sizeType)ids.size()*3);
        //fetch objective
        metric2=-c.dot(_fx);
      }
      if(metric2>=metric || (sizeType)_sigmaIdsActiveSet.size()==_mesh->nrSigma())
        break;
      metric=metric2;
      //add constraint with max violation
      maxId=-1;
      _mesh->getH(H,_fOut,ids);
      for(sizeType i=0; i<(sizeType)H.size(); i++)
        if(_sigmaIdsActiveSet.find(i)==_sigmaIdsActiveSet.end())
          if(maxId==-1 || H[i]>H[maxId])
            maxId=i;
      //if the max violation is smaller than _scale, we are done
      if(H[maxId]<_scale) {
        //scaleF reached
        if(_fOut.norm()>=_scaleFCurr*0.99f) {
          Cold val=Cold::Zero(_fOut.size()+1);
          val[0]=_scaleFCurr;
          _scaleFCurr*=2;
          _scs->add(_initScaleF,val);
          continue;
        } else break;
      }
      _sigmaIdsActiveSet.insert(maxId);
    }
    return metric;
  }
}
void SupportQSMSCS::fNormConstraint(const IDSET& ids)
{
  if(_initFlag) {
    sizeType off=0;
    Objective<scalarD>::SMat A;
    Objective<scalarD>::STrips ATrips;
    for(IDSET::const_iterator beg=ids.begin(),end=ids.end(); beg!=end; beg++,off+=3)
      addBlockId(ATrips,off+1,off,3,-1);
    A.resize(off+1,off);
    A.setFromTriplets(ATrips.begin(),ATrips.end());
    _initScaleF=_scs->constraintSecondOrder(A,0,Cold::Unit(A.rows(),0)*_scaleFCurr,ScsInterface::SECOND_ORDER);
  }
  //-I<=\sigma_{max}(\stress)/_scale<=I
  Mat3d Id=Mat3d::Identity();
  const IDSET& sigmaIds=_sigmaIdsActiveSet.empty()?_sigmaIds:_sigmaIdsActiveSet;
  for(IDSET::const_iterator beg=sigmaIds.begin(),end=sigmaIds.end(); beg!=end; beg++) {
    Matd coef(_mesh->sigmaFromF(*beg,ids)/_scale);
    Mat3Xd coefs=Mat3Xd::Zero(3,coef.cols()*3);
    for(sizeType i=0; i<coef.cols(); i++)
      coefs.block<3,3>(0,i*3)=Eigen::Map<Mat3d>(&(coef.coeffRef(0,i)));
    _scs->constraintLinearMatrixInequality( coefs,0,Id,ScsInterface::SEMI_DEFINITE);
    _scs->constraintLinearMatrixInequality(-coefs,0,Id,ScsInterface::SEMI_DEFINITE);
  }
}
void SupportQSMSCS::debugFNorm(const IDSET& ids)
{
  INFO("debugFNorm:")
  const IDSET& sigmaIds=_sigmaIdsActiveSet.empty()?_sigmaIds:_sigmaIdsActiveSet;
  for(IDSET::const_iterator beg=sigmaIds.begin(),end=sigmaIds.end(); beg!=end; beg++) {
    Vec9d sigma=Matd(_mesh->sigmaFromF(*beg,ids)/_scale)*_fOut;
    Eigen::Map<const Mat3d> sigma3x3(sigma.data());
    std::cout << Eigen::SelfAdjointEigenSolver<Mat3d>(Mat3d::Identity()-sigma3x3).eigenvalues().minCoeff() << " ";
    std::cout << Eigen::SelfAdjointEigenSolver<Mat3d>(Mat3d::Identity()+sigma3x3).eigenvalues().minCoeff() << " ";
  }
  std::cout << std::endl;
}
#endif
