#include <GraspMetric/pyGraspMetric.h>

USE_PRJ_NAMESPACE

int main(int,char**)
{
  scalarD fri=0.7f;

  GraspMesh::Pts pss,nss;
  pss.push_back(Vec3d( 1,0,0));
  pss.push_back(Vec3d(-1,0,0));
  pss.push_back(Vec3d(0, 1,0));
  pss.push_back(Vec3d(0,-1,0));

  nss.push_back(Vec3d(-1,0,0));
  nss.push_back(Vec3d( 1,0,0));
  nss.push_back(Vec3d(0,-1,0));
  nss.push_back(Vec3d(0, 1,0));

  Mat6d metric=Mat6d::Identity();
  INFOV("Q1=%f QInf=%f QMSV=%f QVEW=%f QG11=%f",
        computeQ1(fri,pss,nss,metric,true),
        computeQInf(fri,pss,nss,metric,true),
        computeQMSV(fri,pss,nss),
        computeQVEW(fri,pss,nss),
        computeQG11(fri,pss,nss))
  return 0;
}
