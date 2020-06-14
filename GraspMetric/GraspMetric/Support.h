#ifndef SUPPORT_H
#define SUPPORT_H

#include "Metric.h"
#include "MosekInterface.h"

PRJ_BEGIN

class ScsInterface;
class Support
{
public:
  typedef GraspMesh::IDSET IDSET;
  Support(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt);
  virtual scalarD supportPoint(const Vec6d& d,const IDSET& ids,bool directed)=0;
  virtual void clearModel();
  sizeType errorFlag() const;
  const Cold& f() const;
  const Cold& w() const;
  //utility
  static std::shared_ptr<Support> createQ1
  (std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt);
  static std::shared_ptr<Support> createQInf
  (std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt);
  static std::shared_ptr<Support> createQSM
  (std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt,
   const IDSET& sigmaIds,const sizeType& progressive,const scalarD& scale);
  //setting
  static bool _useMosekCutSDP;
  static bool _useMosek;
  static bool _useSCS;
protected:
  void writeError(const Vec6d& d,const IDSET& ids,bool directed) const;
  void readError(Vec6d& d,IDSET& ids,bool& directed) const;
  void checkAndTestError();
  //member
  Cold _fOut,_wOut;
  sizeType _errorFlag;
  std::shared_ptr<GraspMesh> _mesh;
  const Mat6d& _metric;
  const Mat6d& _metricSqrt;
};
//Mosek
#ifdef MOSEK_SUPPORT
class SupportQ1Mosek : public Support
{
public:
  DECL_MOSEK_TYPES
  SupportQ1Mosek(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt);
  virtual scalarD supportPoint(const Vec6d& d,const IDSET& ids,bool directed) override;
  virtual void fNormConstraint(const IDSET& ids);
  virtual void clearModel() override;
protected:
  monty::rc_ptr<ModelM> _model;
  std::vector<monty::rc_ptr<VariableM>> _fss;
  monty::rc_ptr<ExpressionM> _fStacked;
  monty::rc_ptr<VariableM> _w;
};
class SupportQInfMosek : public SupportQ1Mosek
{
public:
  SupportQInfMosek(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt);
  virtual void fNormConstraint(const IDSET& ids);
};
class SupportQSMMosek : public SupportQ1Mosek
{
public:
  SupportQSMMosek(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt,
                  const IDSET& sigmaIds,const sizeType& progressive,const scalarD& scale);
  virtual scalarD supportPoint(const Vec6d& d,const IDSET& ids,bool directed) override;
  virtual void fNormConstraint(const IDSET& ids);
  virtual void debugFNorm(const IDSET& ids);
protected:
  const IDSET& _sigmaIds;
  const sizeType& _progressive;
  const scalarD& _scale;
  //avoid unboundedness
  monty::rc_ptr<ConstraintM> _initScaleF;
  scalarD _scaleF,_scaleFCurr;
  bool _initFlag;
  //tmp
  IDSET _sigmaIdsActiveSet;
};
class SupportQSMCutMosek : public SupportQSMMosek
{
public:
  SupportQSMCutMosek(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt,
                     const IDSET& sigmaIds,const sizeType& progressive,const scalarD& scale,scalarD eps=1E-5f);
  virtual scalarD supportPoint(const Vec6d& d,const IDSET& ids,bool directed) override;
  bool generateNewCutPlane(const IDSET& ids);
  void generateInitialCutPlane();
  bool updateInitialCutPlane();
private:
  monty::rc_ptr<ConstraintM> _initCutPlane;
  scalarD _initCutScale;
  const scalarD _eps;
};
#endif
//SCS
class SupportQ1SCS : public Support
{
public:
  SupportQ1SCS(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt);
  virtual scalarD supportPoint(const Vec6d& d,const IDSET& ids,bool directed) override;
  virtual void fNormConstraint(const IDSET& ids);
  virtual void clearModel() override;
protected:
  std::shared_ptr<ScsInterface> _scs;
  Cold _fx;
};
class SupportQInfSCS : public SupportQ1SCS
{
public:
  SupportQInfSCS(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt);
  virtual void fNormConstraint(const IDSET& ids) override;
};
class SupportQSMSCS : public SupportQ1SCS
{
public:
  SupportQSMSCS(std::shared_ptr<GraspMesh> mesh,const Matd& metric,const Mat6d& metricSqrt,
                const IDSET& sigmaIds,const sizeType& progressive,const scalarD& scale);
  virtual scalarD supportPoint(const Vec6d& d,const IDSET& ids,bool directed) override;
  virtual void fNormConstraint(const IDSET& ids) override;
  virtual void debugFNorm(const IDSET& ids);
protected:
  const IDSET& _sigmaIds;
  const sizeType& _progressive;
  const scalarD& _scale;
  //avoid unboundedness
  sizeType _initScaleF;
  scalarD _scaleF,_scaleFCurr;
  bool _initFlag;
  //tmp
  IDSET _sigmaIdsActiveSet;
};

PRJ_END

#endif
