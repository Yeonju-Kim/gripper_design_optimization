#include "SparseUtils.h"
#include "EigenInterface.h"
#include "CholmodInterface.h"
#include "UmfpackInterface.h"

PRJ_BEGIN

//#define CHECK_SCALE
//maxAbs
scalarD absMax(const Objective<scalarD>::SMat& h)
{
  scalarD ret=0;
  for(sizeType k=0; k<h.outerSize(); ++k)
    for(Objective<scalarD>::SMat::InnerIterator it(h,k); it; ++it)
      ret=std::max(ret,std::abs(it.value()));
  return ret;
}
scalarD absMaxRel(const Objective<scalarD>::SMat& h,const Objective<scalarD>::SMat& hRef,bool detail)
{
  sizeType row=-1,col=-1;
  scalarD ret=0,num=0,denom=0;
  //check against h
  for(sizeType k=0; k<h.outerSize(); ++k)
    for(Objective<scalarD>::SMat::InnerIterator it(h,k); it; ++it) {
      scalarD err=abs(it.value()-hRef.coeff(it.row(),it.col()));
      scalarD ref=abs(hRef.coeff(it.row(),it.col()));
      scalarD rel=err/std::max<scalarD>(ref,1E-6f);
      if(err<1E-6f)
        continue;
      if(rel>ret) {
        num=err;
        denom=ref;
        row=it.row();
        col=it.col();
      }
      ret=std::max(ret,rel);
    }
  //check against hRef
  for(sizeType k=0; k<hRef.outerSize(); ++k)
    for(Objective<scalarD>::SMat::InnerIterator it(hRef,k); it; ++it) {
      scalarD err=abs(h.coeff(it.row(),it.col())-it.value());
      scalarD ref=abs(it.value());
      scalarD rel=err/std::max<scalarD>(ref,1E-6f);
      if(err<1E-6f)
        continue;
      if(rel>ret) {
        num=err;
        denom=ref;
        row=it.row();
        col=it.col();
      }
      ret=std::max(ret,rel);
    }
  if(detail) {
    INFOV("(%d,%d) Num: %f Denom: %f",row,col,num,denom)
  }
  return ret;
}
//build KKT matrix
Matd buildKKT(const Matd& h,const Matd& a,scalarD shift)
{
  Matd kkt;
  kkt.setZero(h.rows()+a.rows(),h.rows()+a.rows());
  kkt.block(0,0,h.rows(),h.rows())=h;
  kkt.block(0,0,h.rows(),h.rows()).diagonal().array()+=shift;
  kkt.block(h.rows(),0,a.rows(),a.cols())=a;
  kkt.block(0,h.rows(),a.cols(),a.rows())=a.transpose();
  return kkt;
}
Objective<scalarD>::SMat buildKKT(const Objective<scalarD>::SMat& h,const Objective<scalarD>::SMat& a,scalarD shift)
{
  typedef Objective<scalarD>::SMat SMat;
  SMat kkt;
  Objective<scalarD>::STrips trips;
  kkt.resize(h.rows()+a.rows(),h.rows()+a.rows());
  for(sizeType k=0; k<h.outerSize(); ++k)
    for(SMat::InnerIterator it(h,k); it; ++it)
      trips.push_back(Eigen::Triplet<scalarD,sizeType>(it.row(),it.col(),it.value()));
  if(shift!=0)
    for(sizeType k=0; k<h.rows(); ++k)
      trips.push_back(Eigen::Triplet<scalarD,sizeType>(k,k,shift));
  for(sizeType k=0; k<a.outerSize(); ++k)
    for(SMat::InnerIterator it(a,k); it; ++it) {
      trips.push_back(Eigen::Triplet<scalarD,sizeType>(h.rows()+it.row(),it.col(),it.value()));
      trips.push_back(Eigen::Triplet<scalarD,sizeType>(it.col(),h.rows()+it.row(),it.value()));
    }
  kkt.setFromTriplets(trips.begin(),trips.end());
  return kkt;
}
//kronecker-product
Matd kronecker(const Matd& h,sizeType n)
{
  Objective<scalarD>::STrips trips;
  for(sizeType r=0; r<h.rows(); r++)
    for(sizeType c=0; c<h.cols(); c++)
      for(sizeType d=0; d<n; d++)
        trips.push_back(Objective<scalarD>::STrip(r*n+d,c*n+d,h(r,c)));

  Objective<scalarD>::SMat ret;
  ret.resize(h.rows()*n,h.cols()*n);
  ret.setFromTriplets(trips.begin(),trips.end());
  return ret.toDense();
}
Objective<scalarD>::SMat kronecker(const Objective<scalarD>::SMat& h,sizeType n)
{
  Objective<scalarD>::STrips trips;
  for(sizeType k=0; k<h.outerSize(); ++k)
    for(Objective<scalarD>::SMat::InnerIterator it(h,k); it; ++it)
      for(sizeType d=0; d<n; d++)
        trips.push_back(Objective<scalarD>::STrip(it.row()*n+d,it.col()*n+d,it.value()));

  Objective<scalarD>::SMat ret;
  ret.resize(h.rows()*n,h.cols()*n);
  ret.setFromTriplets(trips.begin(),trips.end());
  return ret;
}
//concat-diag
Objective<scalarD>::SMat concatDiag(const Objective<scalarD>::SMat& a,const Objective<scalarD>::SMat& b)
{
  Objective<scalarD>::STrips trips;
  for(sizeType k=0; k<a.outerSize(); ++k)
    for(Objective<scalarD>::SMat::InnerIterator it(a,k); it; ++it)
      trips.push_back(Objective<scalarD>::STrip(it.row(),it.col(),it.value()));
  for(sizeType k=0; k<b.outerSize(); ++k)
    for(Objective<scalarD>::SMat::InnerIterator it(b,k); it; ++it)
      trips.push_back(Objective<scalarD>::STrip(a.rows()+it.row(),a.cols()+it.col(),it.value()));

  Objective<scalarD>::SMat ret;
  ret.resize(a.rows()+b.rows(),a.cols()+b.cols());
  ret.setFromTriplets(trips.begin(),trips.end());
  return ret;
}
//sparse dense adaptation
//LinearSolverTraits<Matd>
std::shared_ptr<LinearSolverInterface> LinearSolverTraits<Matd>::getCholSolver()
{
  return std::shared_ptr<LinearSolverInterface>(new EigenDenseCholInterface);
}
std::shared_ptr<LinearSolverInterface> LinearSolverTraits<Matd>::getLUSolver()
{
  return std::shared_ptr<LinearSolverInterface>(new EigenDenseLUInterface);
}
void LinearSolverTraits<Matd>::rescale(const Cold& c,const Matd& a,Cold& scale,Cold& cOut,Matd& aOut)
{
  //rescale constraints for better conditioning
  scale.setZero(a.rows());
  for(sizeType k=0; k<scale.size(); k++) {
    scalarD rowSqrNorm=a.row(k).squaredNorm();
    scale[k]=1/std::sqrt(rowSqrNorm);
  }
  cOut=scale.asDiagonal()*c;
  aOut=scale.asDiagonal()*a;
#ifdef CHECK_SCALE
  //check
  Cold scaleCheck=Cold::Zero(a.rows());
  for(sizeType k=0; k<aOut.rows(); ++k)
    scaleCheck[k]=aOut.row(k).squaredNorm();
  std::cout << scaleCheck.transpose() << std::endl;
#endif
}
//LinearSolverTraits<Objective<scalarD>::SMat>
std::shared_ptr<LinearSolverInterface> LinearSolverTraits<Objective<scalarD>::SMat>::getCholSolver()
{
#ifdef CHOLMOD_SUPPORT
#ifdef QUADMATH_SUPPORT
  return std::shared_ptr<LinearSolverInterface>(new EigenCholInterface);
#else
  return std::shared_ptr<LinearSolverInterface>(new CholmodInterface);
#endif
#endif

  return std::shared_ptr<LinearSolverInterface>(new EigenCholInterface);
}
std::shared_ptr<LinearSolverInterface> LinearSolverTraits<Objective<scalarD>::SMat>::getLUSolver()
{
#ifdef SuperLU_SUPPORT
#ifdef QUADMATH_SUPPORT
  return std::shared_ptr<LinearSolverInterface>(new EigenLUInterface);
#else
  return std::shared_ptr<LinearSolverInterface>(new EigenSuperLUInterface);
#endif
#endif

#ifdef UMFPACK_SUPPORT
#ifdef QUADMATH_SUPPORT
  return std::shared_ptr<LinearSolverInterface>(new EigenLUInterface);
#else
  return std::shared_ptr<LinearSolverInterface>(new UmfpackInterface);
#endif
#endif

  return std::shared_ptr<LinearSolverInterface>(new EigenLUInterface);
}
void LinearSolverTraits<Objective<scalarD>::SMat>::rescale(const Cold& c,const SMat& a,Cold& scale,Cold& cOut,SMat& aOut)
{
  //rescale constraints for better conditioning
  scale.setZero(a.rows());
  for(sizeType k=0; k<a.outerSize(); ++k)
    for(SMat::InnerIterator it(a,k); it; ++it)
      scale[it.row()]+=it.value()*it.value();
  for(sizeType k=0; k<scale.size(); k++)
    scale[k]=1/std::sqrt(scale[k]);
  cOut=scale.asDiagonal()*c;
  aOut=scale.asDiagonal()*a;
#ifdef CHECK_SCALE
  //check
  Cold scaleCheck=Cold::Zero(a.rows());
  for(sizeType k=0; k<aOut.outerSize(); ++k)
    for(SMat::InnerIterator it(aOut,k); it; ++it)
      scaleCheck[it.row()]+=it.value()*it.value();
  std::cout << scaleCheck.transpose() << std::endl;
#endif
}

PRJ_END
