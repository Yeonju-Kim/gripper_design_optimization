#include "EigenInterface.h"
#include "SparseUtils.h"

USE_PRJ_NAMESPACE

//CPU
bool LinearSolverInterface::recompute(const Matd& smat,scalarD shift,bool sameA)
{
  ASSERT_MSG(false,"This is not a dense matrix solver!")
  return false;
}
bool LinearSolverInterface::recompute(const SMat& smat,scalarD shift,bool sameA)
{
  ASSERT_MSG(false,"This is not a sparse matrix solver!")
  return false;
}
bool LinearSolverInterface::recomputeKKT(const Matd& h,const Matd& a,scalarD shift,bool sameA)
{
  return recompute(buildKKT(h,a,shift),0,sameA);
}
bool LinearSolverInterface::recomputeKKT(const SMat& h,const SMat& a,scalarD shift,bool sameA)
{
  return recompute(buildKKT(h,a,shift),0,sameA);
}
bool LinearSolverInterface::recomputeAAT(const Matd& a,scalarD shift,bool sameA)
{
  return recompute(Matd(a*a.transpose()),shift,sameA);
}
bool LinearSolverInterface::recomputeAAT(const SMat& a,scalarD shift,bool sameA)
{
  return recompute(SMat(a*a.transpose()),shift,sameA);
}
LinearSolverInterface::SMat LinearSolverInterface::analyzeSymmetricNullspace(const SMat& a)
{
  STrips trips;
  for(sizeType k=0; k<a.outerSize(); ++k) {
    bool isZero=true;
    for(SMat::InnerIterator it(a,k); it; ++it)
      if(it.value()!=0)
        isZero=false;
    if(isZero)
      trips.push_back(STrip(k,k,1));
  }
  SMat diag;
  diag.resize(a.rows(),a.cols());
  diag.setFromTriplets(trips.begin(),trips.end());
  return a+diag;
}
Matd LinearSolverInterface::analyzeSymmetricNullspace(const Matd& a)
{
  Cold d=Cold::Zero(a.rows());
  for(sizeType i=0; i<a.rows(); i++)
    if(a.row(i).isZero())
      d[i]=1;
  return a+Matd(d.asDiagonal());
}
//EigenCholInterface
std::shared_ptr<LinearSolverInterface> EigenCholInterface::copy() const
{
  return std::shared_ptr<LinearSolverInterface>(new EigenCholInterface);
}
bool EigenCholInterface::recompute(const SMat& smat,scalarD shift,bool sameA)
{

  _sol.setMode(Eigen::SimplicialCholeskyLLT);
  _sol.setShift(shift);
  _sol.compute(analyzeSymmetricNullspace(smat));
  if(_sol.info() == Eigen::Success)
    return true;
  return false;
}
const Matd& EigenCholInterface::solve(const Matd& b)
{
  if(_sol.info() == Eigen::Success)
    return _ret=_sol.solve(b);
  ASSERT(false)
  return _ret;
}
//EigenLUInterface
std::shared_ptr<LinearSolverInterface> EigenLUInterface::copy() const
{
  return std::shared_ptr<LinearSolverInterface>(new EigenLUInterface);
}
bool EigenLUInterface::recompute(const SMat& smat,scalarD shift,bool sameA)
{
  ASSERT(shift == 0)
  _sol.compute(smat);
  if(_sol.info() == Eigen::Success)
    return true;
  return false;
}
const Matd& EigenLUInterface::solve(const Matd& b)
{
  if(_sol.info() == Eigen::Success)
    return _ret=_sol.solve(b);
  ASSERT(false)
  return _ret;
}
//EigenSuperLUInterface
#ifdef SuperLU_SUPPORT
std::shared_ptr<LinearSolverInterface> EigenSuperLUInterface::copy() const
{
  return std::shared_ptr<LinearSolverInterface>(new EigenSuperLUInterface);
}
bool EigenSuperLUInterface::recompute(const SMat& smat,scalarD shift,bool sameA)
{
  ASSERT(shift == 0)
  _sol.analyzePattern(smat);
  _sol.factorize(smat);
  if(_sol.info() == Eigen::Success)
    return true;
  return false;
}
const Matd& EigenSuperLUInterface::solve(const Matd& b)
{
  if(_sol.info() == Eigen::Success)
    return _ret=_sol.solve(b);
  ASSERT(false)
  return _ret;
}
#endif
//EigenDenseCholInterface
std::shared_ptr<LinearSolverInterface> EigenDenseCholInterface::copy() const
{
  return std::shared_ptr<LinearSolverInterface>(new EigenDenseCholInterface);
}
bool EigenDenseCholInterface::recompute(const Matd& smat,scalarD shift,bool sameA)
{
  _sol.compute(analyzeSymmetricNullspace(smat)+Matd::Identity(smat.rows(),smat.cols())*shift);
  if(_sol.info() == Eigen::Success)
    return true;
  return false;
}
const Matd& EigenDenseCholInterface::solve(const Matd& b)
{
  if(_sol.info() == Eigen::Success)
    return _ret=_sol.solve(b);
  ASSERT(false)
  return _ret;
}
//EigenDenseLUInterface
std::shared_ptr<LinearSolverInterface> EigenDenseLUInterface::copy() const
{
  return std::shared_ptr<LinearSolverInterface>(new EigenDenseLUInterface);
}
bool EigenDenseLUInterface::recompute(const Matd& smat,scalarD shift,bool sameA)
{
  _sol.compute(smat+Matd::Identity(smat.rows(),smat.cols())*shift);
  return true;
}
const Matd& EigenDenseLUInterface::solve(const Matd& b)
{
  return _ret=_sol.solve(b);
}
