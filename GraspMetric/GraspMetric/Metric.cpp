#include "Utils.h"
#include "Metric.h"
#include "Support.h"
#include "ConvexHull.h"
#include "DebugGradient.h"
#include <Eigen/Eigenvalues>

USE_PRJ_NAMESPACE

//Q1Metric
Q1Metric::Q1Metric()
{
  _support=Support::createQ1(_mesh,_metric,_metricSqrt);
}
Q1Metric::Q1Metric(const Q1Metric& other):_mesh(other._mesh),_metric(other._metric),_metricSqrt(other._metricSqrt),_thres(other._thres)
{
  _support=Support::createQ1(_mesh,_metric,_metricSqrt);
}
Q1Metric::Q1Metric(std::shared_ptr<GraspMesh> mesh,const Mat6d& metric,scalarD thres):_mesh(mesh),_metric(metric),_thres(thres)
{
  Eigen::SelfAdjointEigenSolver<Mat6d> eig(metric,Eigen::ComputeEigenvectors);
  _metricSqrt=eig.eigenvectors()*eig.eigenvalues().cwiseSqrt().asDiagonal()*eig.eigenvectors().transpose();
  _support=Support::createQ1(_mesh,_metric,_metricSqrt);
}
Q1Metric::~Q1Metric()
{
  if(_support)
    _support->clearModel();
}
std::string Q1Metric::type() const
{
  return typeid(Q1Metric).name();
}
bool Q1Metric::read(std::istream& is,IOData* dat)
{
  registerType<GraspMesh>(dat);
  readBinaryData(_mesh,is,dat);
  readBinaryData(_metric,is);
  readBinaryData(_metricSqrt,is);
  readBinaryData(_thres,is);
  _support=Support::createQ1(_mesh,_metric,_metricSqrt);
  return is.good();
}
bool Q1Metric::write(std::ostream& os,IOData* dat) const
{
  registerType<GraspMesh>(dat);
  writeBinaryData(_mesh,os,dat);
  writeBinaryData(_metric,os);
  writeBinaryData(_metricSqrt,os);
  writeBinaryData(_thres,os);
  return os.good();
}
std::shared_ptr<SerializableBase> Q1Metric::copy() const
{
  std::shared_ptr<Q1Metric> metric(new Q1Metric);
  metric->_mesh=_mesh;
  metric->_metric=_metric;
  metric->_metricSqrt=_metricSqrt;
  metric->_thres=_thres;
  return metric;
}
//main function
void Q1Metric::printIterationLog() const
{
  INFO("Metric Computation")
  INFOV("%10s %10s %10s","Iter","Q","Dist")
  for(sizeType i=0; i<(sizeType)_iterations.size(); i++) {
    INFOV("%10d %10f %10f",i+1,_iterations[i].first,_iterations[i].second)
  }
  INFO("Metric Computation")
}
template <int DIM>
scalarD Q1Metric::computeMetric(scalarD* LB,scalarD* UB,const IDSET& ids,bool directed,const Eigen::Matrix<scalarD,DIM,6>* basis)
{
  if(_support)
    _support->clearModel();
  clearSolution();
  _iterations.clear();
  //initial construction, i.e., force closure check
  if(!hasForceClosure<DIM>(UB,ids,true,basis))
    return 0;
  if(_support->errorFlag()>0) {
    _wssSols.clear();
    _fssSols.assign(1,Cold::Zero((sizeType)ids.size()*3));
    return ScalarUtil<scalarD>::scalar_max();
  }
  //incremental convex hull computation
  QHullConvexHull<DIM> hull;
  //CGALConvexHull<DIM> hull;
  typename CGALConvexHull<DIM>::PTS pss;
  for(sizeType i=0; i<DIM*2; i++)
    pss.push_back(ConvexHull<DIM>::mul(basis,_wssSols[i]));
  hull.insertInit(pss);
  //iterative refinement
  scalarD Q=ScalarUtil<scalarD>::scalar_max();
  Eigen::Matrix<scalarD,DIM,1> pt;
  while(true) {
    //update Q
    typename ConvexHull<DIM>::PT blockingPN;
    Q=hull.distToOrigin(blockingPN);
    if(Q<=0)
      return 0;
    if(LB && Q>*LB) //early stop condition
      break;
    //update convex hull
    blockingPN.normalize();
    scalarD dist=_support->supportPoint(ConvexHull<DIM>::mulT(basis,blockingPN),ids,directed);
    _wssSols.push_back(_support->w());
    _fssSols.push_back(_support->f());
    if(_support->errorFlag()>0) {
      _wssSols.clear();
      _fssSols.assign(1,Cold::Zero((sizeType)ids.size()*3));
      return ScalarUtil<scalarD>::scalar_max();
    }
    if(dist-Q<_thres*Q) //stopping criterion
      break;
    pt=ConvexHull<DIM>::mul(basis,_wssSols.back());
    if(UB && pt.norm()<*UB) //early stop condition
      break;
    hull.insert(pt);
    //log the computation
    _iterations.push_back(std::make_pair(Q,dist));
  }
  return Q;
}
scalarD Q1Metric::computeMetric(scalarD* LB,scalarD* UB,const IDSET& ids,bool directed,const Vec3d& twoPoint)
{
  if(_mesh->contacts().size()<2 || ids.size()<2)
    return 0;
  if(!twoPoint.isZero()) {
    Eigen::Matrix<scalarD,5,6> basis;
    basis.row(0)=Vec6d::Unit(0);
    basis.row(1)=Vec6d::Unit(1);
    basis.row(2)=Vec6d::Unit(2);

    sizeType minId;
    twoPoint.cwiseAbs().minCoeff(&minId);
    Vec3d t1=twoPoint.cross(Vec3d::Unit(minId)).normalized();
    Vec3d t2=twoPoint.cross(t1).normalized();
    basis.row(3)=concat<Cold,Cold>(Vec3d::Zero(),t1);
    basis.row(4)=concat<Cold,Cold>(Vec3d::Zero(),t2);
    return computeMetric<5>(LB,UB,ids,directed,&basis);
  } else return computeMetric<6>(LB,UB,ids,directed,NULL);
}
template <int DIM>
bool Q1Metric::hasForceClosure(scalarD* UB,const IDSET& ids,bool directed,const Eigen::Matrix<scalarD,DIM,6>* basis)
{
  Eigen::Matrix<scalarD,DIM,1> pt;
  Eigen::Matrix<scalarD,DIM*2,1> dist;
  dist.setZero();
  for(sizeType d=0; d<DIM; d++) {
    Vec6d D=Vec6d::Unit(d);
    if(basis)
      D=basis->row(d);
    //minus
    dist[d*2+0]=_support->supportPoint( D,ids,directed);
    _wssSols.push_back(_support->w());
    _fssSols.push_back(_support->f());
    if(_support->errorFlag()>0)
      return true;
    pt=ConvexHull<DIM>::mul(basis,_wssSols.back());
    if(UB && pt.norm()<*UB)
      break;
    //plus
    dist[d*2+1]=_support->supportPoint(-D,ids,directed);
    _wssSols.push_back(_support->w());
    _fssSols.push_back(_support->f());
    if(_support->errorFlag()>0)
      return true;
    pt=ConvexHull<DIM>::mul(basis,_wssSols.back());
    if(UB && pt.norm()<*UB)
      break;
  }
  if(directed && _support)
    _support->clearModel();
  return dist.minCoeff()>validThres();
}
bool Q1Metric::hasForceClosure(scalarD* UB,const IDSET& ids,bool directed,const Vec3d& twoPoint)
{
  if(_mesh->contacts().size()<2)
    return 0;
  if(!twoPoint.isZero()) {
    Eigen::Matrix<scalarD,5,6> basis;
    basis.row(0)=Vec6d::Unit(0);
    basis.row(1)=Vec6d::Unit(1);
    basis.row(2)=Vec6d::Unit(2);

    sizeType minId;
    twoPoint.cwiseAbs().minCoeff(&minId);
    Vec3d t1=twoPoint.cross(Vec3d::Unit(minId)).normalized();
    Vec3d t2=twoPoint.cross(t1).normalized();
    basis.row(3)=concat<Cold,Cold>(Vec3d::Zero(),t1);
    basis.row(4)=concat<Cold,Cold>(Vec3d::Zero(),t2);
    return hasForceClosure<5>(UB,ids,directed,&basis);
  } else return hasForceClosure<6>(UB,ids,directed,NULL);
}
//getter
Cold Q1Metric::getForce() const
{
  ASSERT_MSG(_fssSols.size()>0,"Cannot compute forces when there are no force solutions!")
  Cold ret=Cold::Zero(_fssSols[0].size()/3);
  for(sizeType i=0; i<(sizeType)_fssSols.size(); i++)
    for(sizeType j=0; j<ret.size(); j++)
      ret[j]=_fssSols[i].segment<3>(j*3).norm();
  return ret;
}
Cold Q1Metric::getBlockingForce() const
{
  ASSERT(!_fssSols.empty())
  return _fssSols.back();
}
scalarD Q1Metric::validThres() const
{
  return 1E-2f;
}
const scalarD& Q1Metric::thres() const
{
  return _thres;
}
scalarD& Q1Metric::thres()
{
  return _thres;
}
//IO
void Q1Metric::clearSolution()
{
  _fssSols.clear();
  _wssSols.clear();
}
sizeType Q1Metric::nrSolution() const
{
  return (sizeType)_fssSols.size();
}
const Cold& Q1Metric::getSolution(sizeType sid) const
{
  return _fssSols[sid];
}
Vec6d Q1Metric::computeGSolution(sizeType sid,const IDSET& ids,bool metric) const
{
  sizeType offF=0;
  Vec6d ret=Vec6d::Zero();
  const Cold& f=getSolution(sid);
  for(IDSET::const_iterator beg=ids.begin(),end=ids.end(); beg!=end; beg++) {
    ret+=_mesh->getG(*beg)*f.segment<3>(offF);
    offF+=3;
  }
  return metric?_metricSqrt*ret:ret;
}
void Q1Metric::writeForceVTK(const std::string& path,const IDSET& ids,sizeType sid,scalar coef) const
{
  ASSERT((sizeType)ids.size()==_fssSols[0].size()/3)
  ASSERT(sid>=0 && sid<(sizeType)_fssSols.size())
  Pts vss;
  sizeType offF=0;
  Cold fss=_fssSols[sid];
  for(IDSET::const_iterator beg=ids.begin(),end=ids.end(); beg!=end; beg++) {
    vss.push_back(_mesh->contacts()[*beg]);
    vss.push_back(_mesh->contacts()[*beg]+fss.segment<3>(offF));
    offF+=3;
  }
  VTKWriter<scalarD> os("force",path,true);
  os.appendPoints(vss.begin(),vss.end());
  os.appendCells(VTKWriter<scalarD>::IteratorIndex<Vec3i>(0,2,0),
                 VTKWriter<scalarD>::IteratorIndex<Vec3i>(ids.size(),2,0),
                 VTKWriter<scalarD>::LINE);
}
//QInfMetric
QInfMetric::QInfMetric()
{
  _support=Support::createQInf(_mesh,_metric,_metricSqrt);
}
QInfMetric::QInfMetric(const QInfMetric& other):Q1Metric(other)
{
  _support=Support::createQInf(_mesh,_metric,_metricSqrt);
}
QInfMetric::QInfMetric(std::shared_ptr<GraspMesh> mesh,const Mat6d& metric,scalarD thres):Q1Metric(mesh,metric,thres)
{
  _support=Support::createQInf(_mesh,_metric,_metricSqrt);
}
std::string QInfMetric::type() const
{
  return typeid(QInfMetric).name();
}
std::shared_ptr<SerializableBase> QInfMetric::copy() const
{
  std::shared_ptr<QInfMetric> metric(new QInfMetric);
  metric->_mesh=_mesh;
  metric->_metric=_metric;
  metric->_metricSqrt=_metricSqrt;
  metric->_thres=_thres;
  return metric;
}
//QSMMetric
struct LSS
{
  bool operator()(std::pair<sizeType,scalar> a,std::pair<sizeType,scalar> b) const {
    return a.second>b.second;
  }
};
QSMMetric::QSMMetric():Q1Metric()
{
  _support=Support::createQSM(_mesh,_metric,_metricSqrt,_sigmaIds,_progressive,_scale);
}
QSMMetric::QSMMetric(const QSMMetric& other):Q1Metric(other)
{
  _sigmaIds=other._sigmaIds;
  _progressive=false;
  _scale=1000;
  _support=Support::createQSM(_mesh,_metric,_metricSqrt,_sigmaIds,_progressive,_scale);
}
QSMMetric::QSMMetric(std::shared_ptr<GraspMesh> mesh,const Mat6d& metric,scalarD thres):Q1Metric(mesh,metric,thres)
{
  for(sizeType i=0; i<mesh->nrSigma(); i++)
    _sigmaIds.insert(i);
  _progressive=false;
  _scale=1000;
  _support=Support::createQSM(_mesh,_metric,_metricSqrt,_sigmaIds,_progressive,_scale);
}
std::string QSMMetric::type() const
{
  return typeid(QSMMetric).name();
}
bool QSMMetric::read(std::istream& is,IOData* dat)
{
  Q1Metric::read(is,dat);
  std::vector<sizeType> sigmaIds;
  readBinaryData(sigmaIds,is);
  _sigmaIds=IDSET(sigmaIds.begin(),sigmaIds.end());
  readBinaryData(_progressive,is);
  readBinaryData(_scale,is);
  _support=Support::createQSM(_mesh,_metric,_metricSqrt,_sigmaIds,_progressive,_scale);
  return is.good();
}
bool QSMMetric::write(std::ostream& os,IOData* dat) const
{
  Q1Metric::write(os,dat);
  std::vector<sizeType> sigmaIds(_sigmaIds.begin(),_sigmaIds.end());
  writeBinaryData(sigmaIds,os);
  writeBinaryData(_progressive,os);
  writeBinaryData(_scale,os);
  return os.good();
}
std::shared_ptr<SerializableBase> QSMMetric::copy() const
{
  std::shared_ptr<QSMMetric> metric(new QSMMetric);
  metric->_mesh=_mesh;
  metric->_metric=_metric;
  metric->_metricSqrt=_metricSqrt;
  metric->_thres=_thres;
  metric->_sigmaIds=_sigmaIds;
  metric->_scale=_scale;
  return metric;
}
//solve an optimization to make F valid
Cold QSMMetric::makeValidF(const Cold& F)
{
  Cold fOut=Cold::Zero(F.size());
#ifdef MOSEK_SUPPORT
  DECL_MOSEK_TYPES
  monty::rc_ptr<ModelM> model=new ModelM;
  std::vector<monty::rc_ptr<VariableM>> fss;
  monty::rc_ptr<ExpressionM> fStacked;
  monty::rc_ptr<VariableM> t=model->variable("t");
  for(sizeType i=0; i<(sizeType)_mesh->contacts().size(); i++) {
    monty::rc_ptr<VariableM> f=model->variable("f"+std::to_string(i),3);
    fss.push_back(f);
    //frictional cone
    Mat3d m;
    m.row(0)=_mesh->inNormal(i).transpose()*_mesh->theta();
    m.row(1)=_mesh->tangent1(i).transpose();
    m.row(2)=_mesh->tangent2(i).transpose();
    model->constraint("f"+std::to_string(i)+"Cone",ExprM::mul(MosekInterface::toMosek(Matd(m)),f),DomainM::inQCone());
    //||f||^2
    std::shared_ptr<monty::ndarray<double,1>> fStar=MosekInterface::toMosek(Cold(F.segment<3>(i*3)));
    monty::rc_ptr<ExpressionM> df=ExprM::sub(f,fStar);
    if(fStacked.get()==NULL)
      fStacked=df;
    else fStacked=ExprM::vstack(fStacked,df);
  }
  //objective
  model->constraint(ExprM::vstack(t,fStacked),DomainM::inQCone());
  model->objective("obj",ObjectiveSenseM::Minimize,t);
  MosekInterface::ensureSolve(*model);
  //fetch
  for(sizeType i=0; i<(sizeType)fss.size(); i++) {
    std::shared_ptr<monty::ndarray<double,1>> vals=fss[i]->level();
    fOut.segment<3>(i*3)=Eigen::Map<Eigen::Matrix<double,3,1>>(vals->raw()).cast<scalarD>();
  }
#else
  ASSERT_MSG(false,"makeValidF only supports Mosek!")
#endif
  return fOut;
}
//main function
void QSMMetric::setScale(scalarD scale)
{
  _scale=scale;
}
void QSMMetric::setProgressive(sizeType progressive)
{
  _progressive=progressive;
  _support=Support::createQSM(_mesh,_metric,_metricSqrt,_sigmaIds,_progressive,_scale);
}
void QSMMetric::simplifyConstraintPoisson(sizeType nrConstraint,bool add)
{
  GraspMesh::IDVEC ids;
  Pts contacts;
  Pts inNormals;
  _mesh->subSampleTriangleLevel(ids,contacts,inNormals,nrConstraint);
  if(!add)
    _sigmaIds.clear();
  _sigmaIds.insert(ids.begin(),ids.end());
}
void QSMMetric::simplifyConstraintRandom(sizeType nrSample,sizeType nrConstraint,bool add)
{
  std::vector<std::pair<sizeType,scalar>> H;
  for(sizeType s=0; s<_mesh->nrSigma(); s++)
    H.push_back(std::make_pair(s,0));
  for(sizeType s=0; s<nrSample; s++) {
    if((s%10)==0) {
      INFOV("Generating sample: %d/%d!",s,nrSample)
    }
    Cold F=makeValidF(Cold::Random(_mesh->contacts().size()*3));
    //compute sigma
    std::vector<scalarD> HH=_mesh->getH(F,_mesh->allIds());
    for(sizeType h=0; h<_mesh->nrSigma(); h++)
      H[h].second+=HH[h];
//#define DEBUG_GETH
#ifdef DEBUG_GETH
    scalarD err=0;
    std::vector<scalarD> HHRef=_mesh->getHRef(F,_mesh->allIds());
    for(sizeType h=0; h<_mesh->nrSigma(); h++)
      err+=std::abs(HHRef[h]-HH[h]);
    INFOV("Err: %f",err)
#endif
  }
  INFOV("Generating sample: %d/%d!",nrSample,nrSample)
  //find biggest indices
  if(!add)
    _sigmaIds.clear();
  std::sort(H.begin(),H.end(),LSS());
  for(sizeType s=0; s<std::min<sizeType>((sizeType)H.size(),nrConstraint); s++)
    _sigmaIds.insert(H[s].first);
  //writeVTK
  std::vector<scalarD> cellColor(_mesh->mesh().getI().size(),0);
  for(IDSET::const_iterator beg=_sigmaIds.begin(),end=_sigmaIds.end(); beg!=end; beg++)
    cellColor[*beg]=1;
  _mesh->mesh().writeVTK("sigmaIds.vtk",true,false,false,NULL,&cellColor);
}
//debug
void QSMMetric::debugSolver(sizeType nr,sizeType nrPass)
{
#ifdef MOSEK_SUPPORT
  DEFINE_NUMERIC_DELTA
  std::shared_ptr<Support> mosek,mosekCut,scs;
  nr=std::min(nr,(sizeType)_mesh->contacts().size());
  for(sizeType tid=0; tid<3; tid++) {
    std::string name="Mosek/SCS-";
    if(tid==0) {
      mosek.reset(new SupportQ1Mosek(_mesh,_metric,_metricSqrt));
      scs.reset(new SupportQ1SCS(_mesh,_metric,_metricSqrt));
      name+="Q1";
    } else if(tid==1) {
      mosek.reset(new SupportQInfMosek(_mesh,_metric,_metricSqrt));
      scs.reset(new SupportQInfSCS(_mesh,_metric,_metricSqrt));
      name+="QInf";
    } else {
      mosek.reset(new SupportQSMMosek(_mesh,_metric,_metricSqrt,_sigmaIds,_progressive,_scale));
      //mosekCut.reset(new SupportQSMCutMosek(_mesh,_metric,_metricSqrt,_sigmaIds,_progressive,_scale));
      scs.reset(new SupportQSMSCS(_mesh,_metric,_metricSqrt,_sigmaIds,_progressive,_scale));
      name+="QSM";
    }
    for(sizeType i=0; i<nrPass; i++) {
      Vec6d d=Vec6d::Random();
      scalarD retMosek=mosek->supportPoint(d,_mesh->randomIds(nr),true);
      //scalarD retMosekCut=mosekCut?mosekCut->supportPoint(d,_mesh->randomIds(nr),true):0;
      scalarD retSCS=scs->supportPoint(d,_mesh->randomIds(nr),true);
      DEBUG_GRADIENT(name,retMosek,retMosek-retSCS)
      //if(mosekCut) {
      //  DEBUG_GRADIENT(name+"Cut",retMosek,retMosek-retMosekCut)
      //}
    }
  }
#endif
  exit(EXIT_SUCCESS);
}
//QMSVMetric
QMSVMetric::QMSVMetric() {}
QMSVMetric::QMSVMetric(const QMSVMetric& other):_mesh(other._mesh) {}
QMSVMetric::QMSVMetric(std::shared_ptr<GraspMesh> mesh):_mesh(mesh) {}
std::string QMSVMetric::type() const
{
  return typeid(QMSVMetric).name();
}
bool QMSVMetric::read(std::istream& is,IOData* dat)
{
  readBinaryData(_mesh,is);
  return is.good();
}
bool QMSVMetric::write(std::ostream& os,IOData* dat) const
{
  writeBinaryData(_mesh,os);
  return os.good();
}
std::shared_ptr<SerializableBase> QMSVMetric::copy() const
{
  return std::shared_ptr<SerializableBase>(new QMSVMetric(_mesh));
}
scalarD QMSVMetric::computeMetric(const IDSET& ids) const
{
  Mat6Xd GI;
  Mat6d GGT=Mat6d::Zero();
  for(IDSET::const_iterator beg=ids.begin(),end=ids.end(); beg!=end; beg++) {
    GI=_mesh->getG(*beg);
    GGT+=GI*GI.transpose();
  }
  return Eigen::SelfAdjointEigenSolver<Mat6d>(GGT).eigenvalues().minCoeff();
}
//QVEWMetric
QVEWMetric::QVEWMetric() {}
QVEWMetric::QVEWMetric(const QVEWMetric& other):QMSVMetric(other) {}
QVEWMetric::QVEWMetric(std::shared_ptr<GraspMesh> mesh):QMSVMetric(mesh) {}
std::string QVEWMetric::type() const
{
  return typeid(QVEWMetric).name();
}
std::shared_ptr<SerializableBase> QVEWMetric::copy() const
{
  return std::shared_ptr<SerializableBase>(new QVEWMetric(_mesh));
}
scalarD QVEWMetric::computeMetric(const IDSET& ids) const
{
  Mat6Xd GI;
  Mat6d GGT=Mat6d::Zero();
  for(IDSET::const_iterator beg=ids.begin(),end=ids.end(); beg!=end; beg++) {
    GI=_mesh->getG(*beg);
    GGT+=GI*GI.transpose();
  }
  return std::sqrt(GGT.determinant());
}
//QG11Metric
QG11Metric::QG11Metric() {}
QG11Metric::QG11Metric(const QG11Metric& other):QMSVMetric(other) {}
QG11Metric::QG11Metric(std::shared_ptr<GraspMesh> mesh):QMSVMetric(mesh) {}
std::string QG11Metric::type() const
{
  return typeid(QG11Metric).name();
}
std::shared_ptr<SerializableBase> QG11Metric::copy() const
{
  return std::shared_ptr<SerializableBase>(new QG11Metric(_mesh));
}
scalarD QG11Metric::computeMetric(const IDSET& ids) const
{
  Mat6Xd GI;
  Mat6d GGT=Mat6d::Zero();
  for(IDSET::const_iterator beg=ids.begin(),end=ids.end(); beg!=end; beg++) {
    GI=_mesh->getG(*beg);
    GGT+=GI*GI.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Mat6d> sol(GGT);
  return sol.eigenvalues().minCoeff()/std::max<scalarD>(sol.eigenvalues().maxCoeff(),1E-9f);
}
