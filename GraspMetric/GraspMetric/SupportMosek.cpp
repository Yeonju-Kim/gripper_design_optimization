#include "Support.h"
#include "Utils.h"
#include <Eigen/Eigenvalues>
#ifdef MOSEK_SUPPORT

USE_PRJ_NAMESPACE

//SupportQ1Mosek
SupportQ1Mosek::SupportQ1Mosek(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt):Support(mesh,metric,metricSqrt) {}
scalarD SupportQ1Mosek::supportPoint(const Vec6d& d,const IDSET& ids,bool directed)
{
  _errorFlag=0;
  if(_model.get()==NULL || directed) {
    clearModel();
    _model=new ModelM;
    _w=_model->variable("w",6);
    monty::rc_ptr<ExpressionM> GF;
    for(IDSET::const_iterator beg=ids.begin(),end=ids.end(); beg!=end; beg++) {
      monty::rc_ptr<VariableM> f=_model->variable("f"+std::to_string(*beg),3);
      _fss.push_back(f);
      //frictional cone
      Mat3d m;
      m.row(0)=_mesh->inNormal(*beg).transpose()*_mesh->theta();
      m.row(1)=_mesh->tangent1(*beg).transpose();
      m.row(2)=_mesh->tangent2(*beg).transpose();
      _model->constraint("f"+std::to_string(*beg)+"Cone",ExprM::mul(MosekInterface::toMosek(Matd(m)),f),DomainM::inQCone());
      //compute w-\sum(GI*F)
      monty::rc_ptr<ExpressionM> GFI=ExprM::mul(MosekInterface::toMosek(Matd(_mesh->getG(*beg))),f);
      if(GF.get()!=NULL)
        GF=ExprM::add(GF,GFI);
      else GF=GFI;
    }
    //||f||^2
    for(sizeType i=1; i<(sizeType)_fss.size(); i++)
      if(i==1)
        _fStacked=ExprM::vstack(_fss[0],_fss[i]);
      else _fStacked=ExprM::vstack(_fStacked,_fss[i]);
    fNormConstraint(ids);
    //0=w-\sum(GI*F)
    _model->constraint("wSubF",ExprM::sub(_w,GF),DomainM::equalsTo(MosekInterface::toMosek(Cold(Cold::Zero(6)))));
    //directed
    if(directed) {
      Cold dN=d.normalized();
      monty::rc_ptr<MatrixM> projT=MosekInterface::toMosek(Matd((Mat6d::Identity()-dN*dN.transpose())*_metricSqrt));
      _model->constraint("directed",ExprM::mul(projT,_w),DomainM::equalsTo(MosekInterface::toMosek(Cold(Cold::Zero(6)))));
    }
  }
  //objective
  monty::rc_ptr<ExpressionM> expr=ExprM::dot(MosekInterface::toMosek(Cold(_metricSqrt*d)),_w);
  _model->objective(ObjectiveSenseM::Maximize,expr);
  //solve
  std::string str;
  if(!MosekInterface::trySolve(*_model,str)) {
    writeError(d,ids,directed);
    _errorFlag=1;
    return ScalarUtil<scalarD>::scalar_max();
  }
  //fetch force
  _fOut=Cold::Zero((sizeType)_fss.size()*3);
  for(sizeType i=0; i<(sizeType)_fss.size(); i++) {
    std::shared_ptr<monty::ndarray<double,1>> vals=_fss[i]->level();
    _fOut.segment<3>(i*3)=Eigen::Map<Eigen::Matrix<double,3,1>>(vals->raw()).cast<scalarD>();
  }
  //fetch
  std::shared_ptr<monty::ndarray<double,1>> vals=_w->level();
  _wOut=_metricSqrt*Eigen::Map<Eigen::Matrix<double,6,1>>(vals->raw()).cast<scalarD>();
  //fetch objective
  return _model->primalObjValue();
}
void SupportQ1Mosek::fNormConstraint(const IDSET& ids)
{
  //\sum f_N<=1
  sizeType off=0;
  monty::rc_ptr<ExpressionM> fN;
  for(IDSET::const_iterator beg=ids.begin(),end=ids.end(); beg!=end; beg++,off++)
    if(fN.get()==NULL)
      fN=ExprM::dot(MosekInterface::toMosek(Cold(_mesh->inNormal(*beg))),_fss[off]);
    else fN=ExprM::add(fN,ExprM::dot(MosekInterface::toMosek(Cold(_mesh->inNormal(*beg))),_fss[off]));
  _model->constraint("FN",fN,DomainM::lessThan(1));
}
void SupportQ1Mosek::clearModel()
{
  if(_model.get()!=NULL) {
    _model->dispose();
    _model=NULL;
    _fss.clear();
    _fStacked=NULL;
    _w=NULL;
  }
}
//SupportQInfMosek
SupportQInfMosek::SupportQInfMosek(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt):SupportQ1Mosek(mesh,metric,metricSqrt) {}
void SupportQInfMosek::fNormConstraint(const IDSET& ids)
{
  //f_N<=1
  sizeType off=0;
  for(IDSET::const_iterator beg=ids.begin(),end=ids.end(); beg!=end; beg++,off++) {
    monty::rc_ptr<ExpressionM> fN=ExprM::dot(MosekInterface::toMosek(Cold(_mesh->inNormal(*beg))),_fss[off]);
    _model->constraint("FN"+std::to_string(off),fN,DomainM::lessThan(1));
  }
}
//SupportQSMMosek
SupportQSMMosek::SupportQSMMosek(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt,
                                 const IDSET& sigmaIds,const sizeType& progressive,const scalarD& scale)
  :SupportQ1Mosek(mesh,metric,metricSqrt),_sigmaIds(sigmaIds),_progressive(progressive),_scale(scale),_scaleF(1E3f) {}
scalarD SupportQSMMosek::supportPoint(const Vec6d& d,const IDSET& ids,bool directed)
{
  _initFlag=false;
  _scaleFCurr=_scaleF;
  _sigmaIdsActiveSet.clear();
  if((sizeType)ids.size()>_progressive || (sizeType)_sigmaIds.size()==_mesh->nrSigma()) {
    //not using progressive
    return SupportQ1Mosek::supportPoint(d,ids,directed);
  } else {
    //using progressive
    _initFlag=true;
    sizeType maxId=-1;
    std::vector<scalarD> H;
    scalarD metric=ScalarUtil<scalarD>::scalar_max(),metric2;
    _sigmaIdsActiveSet=_sigmaIds;
    monty::rc_ptr<MatrixM> I=MosekInterface::toMosek(Matd(Mat3d::Identity()));
    //progressive loop
    while(true) {
      if(_initFlag) {
        clearModel();
        metric2=SupportQ1Mosek::supportPoint(d,ids,directed);
        _initFlag=false;
        //debugFNorm(ids);
        if(_errorFlag==1)
          return ScalarUtil<scalarD>::scalar_max();
        //fetch force
        _fOut=Cold::Zero((sizeType)_fss.size()*3);
        for(sizeType i=0; i<(sizeType)_fss.size(); i++) {
          std::shared_ptr<monty::ndarray<double,1>> vals=_fss[i]->level();
          _fOut.segment<3>(i*3)=Eigen::Map<Eigen::Matrix<double,3,1>>(vals->raw()).cast<scalarD>();
        }
      } else {
        //just add one more constraint and solve again
        monty::rc_ptr<MatrixM> sigmaFromF=MosekInterface::toMosek(Matd(_mesh->sigmaFromF(maxId,ids)/_scale));
        monty::rc_ptr<ExpressionM> sigma=ExprM::mul(sigmaFromF,_fStacked);
        monty::rc_ptr<ExpressionM> sigmaM=ExprM::reshape(sigma,3,3);
        _model->constraint(ExprM::sub(I,sigmaM),DomainM::inPSDCone(3));
        _model->constraint(ExprM::add(I,sigmaM),DomainM::inPSDCone(3));
        //solve
        std::string str;
        if(!MosekInterface::trySolve(*_model,str)) {
          writeError(d,ids,directed);
          _errorFlag=1;
          return ScalarUtil<scalarD>::scalar_max();
        }
        //fetch force
        _fOut=Cold::Zero((sizeType)_fss.size()*3);
        for(sizeType i=0; i<(sizeType)_fss.size(); i++) {
          std::shared_ptr<monty::ndarray<double,1>> vals=_fss[i]->level();
          _fOut.segment<3>(i*3)=Eigen::Map<Eigen::Matrix<double,3,1>>(vals->raw()).cast<scalarD>();
        }
        //fetch
        std::shared_ptr<monty::ndarray<double,1>> vals=_w->level();
        _wOut=_metricSqrt*Eigen::Map<Eigen::Matrix<double,6,1>>(vals->raw()).cast<scalarD>();
        //fetch objective
        metric2=_model->primalObjValue();
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
#ifdef USE_MOSEK_8
          _initScaleF->add(MosekInterface::toMosekE(val));
#else
          _initScaleF->update(MosekInterface::toMosekE(val));
#endif
          continue;
        } else break;
      }
      _sigmaIdsActiveSet.insert(maxId);
    }
    return metric;
  }
}
void SupportQSMMosek::fNormConstraint(const IDSET& ids)
{
  if(_initFlag)
    _initScaleF=_model->constraint("sumF",ExprM::vstack(_scaleFCurr,_fStacked),DomainM::inQCone());
  //-I<=\sigma_{max}(\stress)/_scale<=I
  monty::rc_ptr<MatrixM> I=MosekInterface::toMosek(Matd(Mat3d::Identity()));
  const IDSET& sigmaIds=_sigmaIdsActiveSet.empty()?_sigmaIds:_sigmaIdsActiveSet;
  for(IDSET::const_iterator beg=sigmaIds.begin(),end=sigmaIds.end(); beg!=end; beg++) {
    monty::rc_ptr<MatrixM> sigmaFromF=MosekInterface::toMosek(Matd(_mesh->sigmaFromF(*beg,ids)/_scale));
    monty::rc_ptr<ExpressionM> sigma=ExprM::mul(sigmaFromF,_fStacked);
    monty::rc_ptr<ExpressionM> sigmaM=ExprM::reshape(sigma,3,3);
    _model->constraint(ExprM::sub(I,sigmaM),DomainM::inPSDCone(3));
    _model->constraint(ExprM::add(I,sigmaM),DomainM::inPSDCone(3));
  }
}
void SupportQSMMosek::debugFNorm(const IDSET& ids)
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
