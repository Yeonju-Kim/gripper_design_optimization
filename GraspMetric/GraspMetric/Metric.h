#ifndef METRIC_H
#define METRIC_H

#include "GraspMesh.h"

PRJ_BEGIN

class Support;
//Q1,QInf,QSM
class Q1Metric : public SerializableBase
{
public:
  typedef GraspMesh::Pts Pts;
  typedef GraspMesh::Vss Vss;
  typedef GraspMesh::IDSET IDSET;
  Q1Metric();
  Q1Metric(const Q1Metric& other);
  Q1Metric(std::shared_ptr<GraspMesh> mesh,const Mat6d& metric,scalarD thres=1E-2f);
  ~Q1Metric();
  std::string type() const override;
  bool read(std::istream& is,IOData* dat) override;
  bool write(std::ostream& os,IOData* dat) const override;
  std::shared_ptr<SerializableBase> copy() const override;
  //main function
  void printIterationLog() const;
  template <int DIM>
  scalarD computeMetric(scalarD* LB,scalarD* UB,const IDSET& ids,bool directed=false,const Eigen::Matrix<scalarD,DIM,6>* basis=NULL);
  scalarD computeMetric(scalarD* LB,scalarD* UB,const IDSET& ids,bool directed=false,const Vec3d& twoPoint=Vec3d::Zero());
  template <int DIM>
  bool hasForceClosure(scalarD* UB,const IDSET& ids,bool directed=false,const Eigen::Matrix<scalarD,DIM,6>* basis=NULL);
  bool hasForceClosure(scalarD* UB,const IDSET& ids,bool directed=false,const Vec3d& twoPoint=Vec3d::Zero());
  //getter
  Cold getForce() const;
  Cold getBlockingForce() const;
  scalarD validThres() const;
  const scalarD& thres() const;
  scalarD& thres();
  //IO
  void clearSolution();
  sizeType nrSolution() const;
  const Cold& getSolution(sizeType sid) const;
  Vec6d computeGSolution(sizeType sid,const IDSET& ids,bool metric=true) const;
  void writeForceVTK(const std::string& path,const IDSET& ids,sizeType sid,scalar coef=1) const;
protected:
  //log
  std::vector<std::pair<scalarD,scalarD>> _iterations;
  std::shared_ptr<Support> _support;
  //param
  std::shared_ptr<GraspMesh> _mesh;
  Vss _fssSols,_wssSols;
  Mat6d _metric,_metricSqrt;
  scalarD _thres;
};
class QInfMetric : public Q1Metric
{
public:
  QInfMetric();
  QInfMetric(const QInfMetric& other);
  QInfMetric(std::shared_ptr<GraspMesh> mesh,const Mat6d& metric,scalarD thres=1E-2f);
  std::string type() const override;
  std::shared_ptr<SerializableBase> copy() const override;
};
class QSMMetric : public Q1Metric
{
public:
  QSMMetric();
  QSMMetric(const QSMMetric& other);
  QSMMetric(std::shared_ptr<GraspMesh> mesh,const Mat6d& metric,scalarD thres=1E-2f);
  std::string type() const override;
  bool read(std::istream& is,IOData* dat) override;
  bool write(std::ostream& os,IOData* dat) const override;
  std::shared_ptr<SerializableBase> copy() const override;
  //solve an optimization to make F valid
  Cold makeValidF(const Cold& F);
  //main function
  void setScale(scalarD scale);
  void setProgressive(sizeType progressive=10000);
  void simplifyConstraintPoisson(sizeType nrConstraint,bool add);
  void simplifyConstraintRandom(sizeType nrSample,sizeType nrConstraint,bool add);
  //debug
  virtual void debugSolver(sizeType nr=10,sizeType nrPass=100);
protected:
  IDSET _sigmaIds;
  sizeType _progressive;
  scalarD _scale;
  //tmp
  IDSET _sigmaIdsActiveSet;
};
//QMSV,QVEW,QG11
class QMSVMetric : public SerializableBase
{
public:
  typedef GraspMesh::IDSET IDSET;
  QMSVMetric();
  QMSVMetric(const QMSVMetric& other);
  QMSVMetric(std::shared_ptr<GraspMesh> mesh);
  std::string type() const override;
  bool read(std::istream& is,IOData* dat) override;
  bool write(std::ostream& os,IOData* dat) const override;
  std::shared_ptr<SerializableBase> copy() const override;
  virtual scalarD computeMetric(const IDSET& ids) const;
protected:
  //param
  std::shared_ptr<GraspMesh> _mesh;
};
class QVEWMetric : public QMSVMetric
{
public:
  QVEWMetric();
  QVEWMetric(const QVEWMetric& other);
  QVEWMetric(std::shared_ptr<GraspMesh> mesh);
  std::string type() const override;
  std::shared_ptr<SerializableBase> copy() const override;
  virtual scalarD computeMetric(const IDSET& ids) const override;
};
class QG11Metric : public QMSVMetric
{
public:
  QG11Metric();
  QG11Metric(const QG11Metric& other);
  QG11Metric(std::shared_ptr<GraspMesh> mesh);
  std::string type() const override;
  std::shared_ptr<SerializableBase> copy() const override;
  virtual scalarD computeMetric(const IDSET& ids) const override;
};

PRJ_END

#endif
