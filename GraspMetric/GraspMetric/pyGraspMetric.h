#ifndef PY_GRASP_METRIC_H
#define PY_GRASP_METRIC_H

#include <GraspMetric/GraspMesh.h>
#include <GraspMetric/Metric.h>

PRJ_BEGIN

extern scalarD computeQ1(scalarD theta,const GraspMesh::Pts& pss,const GraspMesh::Pts& nss,const Mat6d& metric,bool callback);
extern scalarD computeQInf(scalarD theta,const GraspMesh::Pts& pss,const GraspMesh::Pts& nss,const Mat6d& metric,bool callback);
extern scalarD computeQMSV(scalarD theta,const GraspMesh::Pts& pss,const GraspMesh::Pts& nss);
extern scalarD computeQVEW(scalarD theta,const GraspMesh::Pts& pss,const GraspMesh::Pts& nss);
extern scalarD computeQG11(scalarD theta,const GraspMesh::Pts& pss,const GraspMesh::Pts& nss);

PRJ_END

#endif
