#include "pyGraspMetric.h"

PRJ_BEGIN

scalarD computeQ1(scalarD theta,const GraspMesh::Pts& pss,const GraspMesh::Pts& nss,const Mat6d& metric,bool callback) {
  std::shared_ptr<GraspMesh> mesh(new GraspMesh(theta,pss,nss));
  Q1Metric Q1(mesh,metric);
  std::set<sizeType> ids;
  for(sizeType i=0; i<(sizeType)pss.size(); i++)
    ids.insert(i);
  scalarD ret=Q1.computeMetric(NULL,NULL,ids);
  if(callback)
    Q1.printIterationLog();
  return ret;
}
scalarD computeQInf(scalarD theta,const GraspMesh::Pts& pss,const GraspMesh::Pts& nss,const Mat6d& metric,bool callback) {
  std::shared_ptr<GraspMesh> mesh(new GraspMesh(theta,pss,nss));
  QInfMetric QInf(mesh,metric);
  std::set<sizeType> ids;
  for(sizeType i=0; i<(sizeType)pss.size(); i++)
    ids.insert(i);
  scalarD ret=QInf.computeMetric(NULL,NULL,ids);
  if(callback)
    QInf.printIterationLog();
  return ret;
}
scalarD computeQMSV(scalarD theta,const GraspMesh::Pts& pss,const GraspMesh::Pts& nss) {
  std::shared_ptr<GraspMesh> mesh(new GraspMesh(theta,pss,nss));
  QMSVMetric QMSV(mesh);
  std::set<sizeType> ids;
  for(sizeType i=0; i<(sizeType)pss.size(); i++)
    ids.insert(i);
  return QMSV.computeMetric(ids);
}
scalarD computeQVEW(scalarD theta,const GraspMesh::Pts& pss,const GraspMesh::Pts& nss) {
  std::shared_ptr<GraspMesh> mesh(new GraspMesh(theta,pss,nss));
  QVEWMetric QVEW(mesh);
  std::set<sizeType> ids;
  for(sizeType i=0; i<(sizeType)pss.size(); i++)
    ids.insert(i);
  return QVEW.computeMetric(ids);
}
scalarD computeQG11(scalarD theta,const GraspMesh::Pts& pss,const GraspMesh::Pts& nss) {
  std::shared_ptr<GraspMesh> mesh(new GraspMesh(theta,pss,nss));
  QG11Metric QG11(mesh);
  std::set<sizeType> ids;
  for(sizeType i=0; i<(sizeType)pss.size(); i++)
    ids.insert(i);
  return QG11.computeMetric(ids);
}

PRJ_END
