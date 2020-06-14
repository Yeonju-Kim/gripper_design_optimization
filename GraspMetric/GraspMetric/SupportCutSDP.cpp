#include "Support.h"
#include "Utils.h"
#include <Eigen/Eigen>
#ifdef MOSEK_SUPPORT

USE_PRJ_NAMESPACE

SupportQSMCutMosek::SupportQSMCutMosek(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt,
                                       const IDSET& sigmaIds,const sizeType& progressive,const scalarD& scale,scalarD eps)
  :SupportQSMMosek(mesh,metric,metricSqrt,sigmaIds,progressive,scale),_eps(eps) {}
scalarD SupportQSMCutMosek::supportPoint(const Vec6d& d,const IDSET& ids,bool directed)
{
  _errorFlag=0;
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
  //0=w-\sum(GI*F)
  _model->constraint("wSubF",ExprM::sub(_w,GF),DomainM::equalsTo(MosekInterface::toMosek(Cold(Cold::Zero(6)))));
  //directed
  if(directed) {
    Cold dN=d.normalized();
    monty::rc_ptr<MatrixM> projT=MosekInterface::toMosek(Matd((Mat6d::Identity()-dN*dN.transpose())*_metricSqrt));
    _model->constraint("directed",ExprM::mul(projT,_w),DomainM::equalsTo(MosekInterface::toMosek(Cold(Cold::Zero(6)))));
  }
  //objective
  Cold c=_metricSqrt*d;
  monty::rc_ptr<ExpressionM> expr=ExprM::dot(MosekInterface::toMosek(c),_w);
  _model->objective(ObjectiveSenseM::Maximize,expr);
  //solve
  bool more=true;
  std::string str;
  //initial cut plane makes sure that problem is bounded
  generateInitialCutPlane();
  _fOut=Cold::Zero((sizeType)_fss.size()*3);
  do {
    if(!MosekInterface::trySolve(*_model,str)) {
      //std::cout << "Error Message: " << str << std::endl;
      writeError(d,ids,directed);
      _errorFlag=1;
      return ScalarUtil<scalarD>::scalar_max();
      //exit(EXIT_FAILURE);
    }
    //fetch force
    for(sizeType i=0; i<(sizeType)_fss.size(); i++) {
      std::shared_ptr<monty::ndarray<double,1>> vals=_fss[i]->level();
      _fOut.segment<3>(i*3)=Eigen::Map<Eigen::Matrix<double,3,1>>(vals->raw()).cast<scalarD>();
    }
    //fetch
    std::shared_ptr<monty::ndarray<double,1>> vals=_w->level();
    _wOut=_metricSqrt*Eigen::Map<Eigen::Matrix<double,6,1>>(vals->raw()).cast<scalarD>();
    more=generateNewCutPlane(ids);
    //update cut plane
    if(!more && updateInitialCutPlane())
      more=true;
  } while(more);
  //fetch objective
  return _model->primalObjValue();
}
bool SupportQSMCutMosek::generateNewCutPlane(const IDSET& ids)
{
  //add constraint with max violation
  sizeType maxId=-1;
  std::vector<scalarD> H;
  _mesh->getH(H,_fOut,ids);
  for(sizeType i=0; i<(sizeType)H.size(); i++)
    if(_sigmaIdsActiveSet.find(i)==_sigmaIdsActiveSet.end())
      if(maxId==-1 || H[i]>H[maxId])
        maxId=i;
  if(H[maxId]<_scale*(1+_eps))
    return false;
  //create constraint
  Vec3d d;
  sizeType index;
  Mat9Xd sigmaCoef=_mesh->sigmaFromF(maxId,ids)/_scale;
  Vec9d sigma=sigmaCoef*_fOut;
  Eigen::SelfAdjointEigenSolver<Mat3d> eig(Eigen::Map<const Mat3d>(sigma.data()));
  //std::cout << eig.eigenvalues() << std::endl;
  eig.eigenvalues().cwiseAbs().maxCoeff(&index);
  d=eig.eigenvectors().col(index).normalized();
  //build mosek constraint
  Cold fCoef=_fOut;
  for(sizeType i=0; i<_fOut.size(); i++) {
    Eigen::Map<const Mat3d> sigmaCol(sigmaCoef.data()+i*9);
    fCoef[i]=d.dot(sigmaCol*d);
  }
  monty::rc_ptr<ExpressionM> LHS=ExprM::dot(_fStacked,MosekInterface::toMosek(fCoef));
  if(eig.eigenvalues()[index]>0)
    _model->constraint(ExprM::sub(1,LHS),DomainM::greaterThan(0));
  else _model->constraint(ExprM::add(1,LHS),DomainM::greaterThan(0));
  return true;
}
void SupportQSMCutMosek::generateInitialCutPlane()
{
  //\sum||f||^2<=_initCutScale*_initCutScale
  _initCutScale=_scale;
  _initCutPlane=_model->constraint("sumF",ExprM::vstack(_initCutScale,_fStacked),DomainM::inQCone());
}
bool SupportQSMCutMosek::updateInitialCutPlane()
{
  if(_fOut.norm()>=_scale*(1-_eps)) {
    Cold val=Cold::Zero(_fOut.size()+1);
    val[0]=_initCutScale;
    _initCutScale*=2;
#ifdef USE_MOSEK_8
    _initCutPlane->add(MosekInterface::toMosekE(val));
#else
    _initCutPlane->update(MosekInterface::toMosekE(val));
#endif
    return true;
  }
  return false;
}

#endif
