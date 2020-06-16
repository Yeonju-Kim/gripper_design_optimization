#include <GraspMetric/Support.h>
#include <Eigen/Eigen>

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

  Mat6d metric=Mat6d::Random();
  metric=(metric*metric.transpose()).eval();
  Eigen::SelfAdjointEigenSolver<Mat6d> eig(metric,Eigen::ComputeEigenvectors);
  Mat6d metricSqrt=eig.eigenvectors()*eig.eigenvalues().cwiseSqrt().asDiagonal()*eig.eigenvectors().transpose();
  std::shared_ptr<GraspMesh> mesh(new GraspMesh(fri,pss,nss));
  std::set<sizeType> ids;
  for(sizeType i=0; i<(sizeType)pss.size(); i++)
    ids.insert(i);

  std::shared_ptr<Support> supportMosek,supportSCS,supportAnalytic;

  Support::_useMosek=true;
  supportMosek=Support::createQ1(mesh,metric,metricSqrt);
  Support::_useMosek=false;
  Support::_useSCS=true;
  supportSCS=Support::createQ1(mesh,metric,metricSqrt);
  Support::_useSCS=false;
  supportAnalytic=Support::createQ1(mesh,metric,metricSqrt);
  for(sizeType it=0; it<10; it++) {
    Vec6d d=Vec6d::Random();
    scalarD A=supportMosek->supportPoint(d,ids,false);
    scalarD B=supportSCS->supportPoint(d,ids,false);
    scalarD C=supportAnalytic->supportPoint(d,ids,false);
    INFOV("Q1: Mosek=%f SCS=%f Analytic=%f!",A,B,C)
  }

  Support::_useMosek=true;
  supportMosek=Support::createQInf(mesh,metric,metricSqrt);
  Support::_useMosek=false;
  Support::_useSCS=true;
  supportSCS=Support::createQInf(mesh,metric,metricSqrt);
  Support::_useSCS=false;
  supportAnalytic=Support::createQInf(mesh,metric,metricSqrt);
  for(sizeType it=0; it<10; it++) {
    Vec6d d=Vec6d::Random();
    scalarD A=supportMosek->supportPoint(d,ids,false);
    scalarD B=supportSCS->supportPoint(d,ids,false);
    scalarD C=supportAnalytic->supportPoint(d,ids,false);
    INFOV("QInf: Mosek=%f SCS=%f Analytic=%f!",A,B,C)
  }
  return 0;
}
