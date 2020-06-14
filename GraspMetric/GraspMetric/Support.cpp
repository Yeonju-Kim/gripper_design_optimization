#include "Support.h"
#include "Utils.h"

USE_PRJ_NAMESPACE

#ifdef MOSEK_SUPPORT
//#define USE_MOSEK_CUTSDP
#define USE_MOSEK
#endif
#define USE_SCS
//Support
Support::Support(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt):_mesh(mesh),_metric(metric),_metricSqrt(metricSqrt) {}
void Support::clearModel() {}
sizeType Support::errorFlag() const
{
  return _errorFlag;
}
const Cold& Support::f() const
{
  return _fOut;
}
const Cold& Support::w() const
{
  return _wOut;
}
//utility
std::shared_ptr<Support> Support::createQ1
(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt)
{
#ifdef MOSEK_SUPPORT
  if(_useMosek)
    return std::shared_ptr<Support>(new SupportQ1Mosek(mesh,metric,metricSqrt));
#endif
  if(_useSCS)
    return std::shared_ptr<Support>(new SupportQ1SCS(mesh,metric,metricSqrt));
#ifdef USE_MOSEK
  //INFO("Using Mosek!")
  return std::shared_ptr<Support>(new SupportQ1Mosek(mesh,metric,metricSqrt));
#endif
#ifdef USE_SCS
  //INFO("Using SCS!")
  return std::shared_ptr<Support>(new SupportQ1SCS(mesh,metric,metricSqrt));
#endif
  ASSERT_MSG(false,"Cannot find support configuration!")
  return std::shared_ptr<Support>();
}
std::shared_ptr<Support> Support::createQInf
(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt)
{
#ifdef MOSEK_SUPPORT
  if(_useMosek)
    return std::shared_ptr<Support>(new SupportQInfMosek(mesh,metric,metricSqrt));
#endif
  if(_useSCS)
    return std::shared_ptr<Support>(new SupportQInfSCS(mesh,metric,metricSqrt));
#ifdef USE_MOSEK
  //INFO("Using Mosek!")
  return std::shared_ptr<Support>(new SupportQInfMosek(mesh,metric,metricSqrt));
#endif
#ifdef USE_SCS
  //INFO("Using SCS!")
  return std::shared_ptr<Support>(new SupportQInfSCS(mesh,metric,metricSqrt));
#endif
  ASSERT_MSG(false,"Cannot find support configuration!")
  return std::shared_ptr<Support>();
}
std::shared_ptr<Support> Support::createQSM
(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt,
 const IDSET& sigmaIds,const sizeType& progressive,const scalarD& scale)
{
#ifdef MOSEK_SUPPORT
  if(_useMosekCutSDP)
    return std::shared_ptr<Support>(new SupportQSMCutMosek(mesh,metric,metricSqrt,sigmaIds,progressive,scale));
  if(_useMosek)
    return std::shared_ptr<Support>(new SupportQSMMosek(mesh,metric,metricSqrt,sigmaIds,progressive,scale));
#endif
  if(_useSCS)
    return std::shared_ptr<Support>(new SupportQSMSCS(mesh,metric,metricSqrt,sigmaIds,progressive,scale));
#ifdef USE_MOSEK_CUTSDP
  //INFO("Using Mosek CutSDP!")
  return std::shared_ptr<Support>(new SupportQSMCutMosek(mesh,metric,metricSqrt,sigmaIds,progressive,scale));
#endif
#ifdef USE_MOSEK
  //INFO("Using Mosek!")
  return std::shared_ptr<Support>(new SupportQSMMosek(mesh,metric,metricSqrt,sigmaIds,progressive,scale));
#endif
#ifdef USE_SCS
  //INFO("Using SCS!")
  return std::shared_ptr<Support>(new SupportQSMSCS(mesh,metric,metricSqrt,sigmaIds,progressive,scale));
#endif
  ASSERT_MSG(false,"Cannot find support configuration!")
  return std::shared_ptr<Support>();
}
//helper
void Support::writeError(const Vec6d& d,const IDSET& ids,bool directed) const
{
  INFO("Mosek failed, writing error!")
  std::vector<sizeType> idsVec(ids.begin(),ids.end());
  std::ofstream os("error.dat",std::ios::binary);
  writeBinaryData(d,os);
  writeBinaryData(idsVec,os);
  writeBinaryData(directed,os);
}
void Support::readError(Vec6d& d,IDSET& ids,bool& directed) const
{
  std::vector<sizeType> idsVec;
  std::ifstream is("error.dat",std::ios::binary);
  readBinaryData(d,is);
  readBinaryData(idsVec,is);
  readBinaryData(directed,is);
  ids.clear();
  ids.insert(idsVec.begin(),idsVec.end());
}
void Support::checkAndTestError()
{
  if(!exists("error.dat"))
    return;
  Vec6d d;
  IDSET ids;
  bool directed;
  readError(d,ids,directed);
  supportPoint(d,ids,directed);
  INFO("Tested error!")
}
//setting
bool Support::_useMosekCutSDP=false;
bool Support::_useMosek=false;
bool Support::_useSCS=false;