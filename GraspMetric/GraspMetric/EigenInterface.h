#ifndef EIGEN_INTERFACE_H
#define EIGEN_INTERFACE_H

#include <memory>
#include "Objective.h"
#include <Eigen/Cholesky>
#include <Eigen/LU>
#ifdef SuperLU_SUPPORT
#include <Eigen/SuperLUSupport>
#endif

PRJ_BEGIN

struct CudaTridiagMatrix;
template <typename T> struct CudaArray;
template <typename T> struct CudaMatrix;
class LinearSolverInterface
{
public:
  typedef Objective<scalarD>::SMat SMat;
  typedef Objective<scalarD>::STrips STrips;
  typedef Objective<scalarD>::STrip STrip;
  virtual ~LinearSolverInterface() {}
  //CPU
  virtual std::shared_ptr<LinearSolverInterface> copy() const=0;
  virtual bool recompute(const Matd& smat,scalarD shift,bool sameA);
  virtual bool recompute(const SMat& smat,scalarD shift,bool sameA);
  virtual bool recomputeKKT(const Matd& h,const Matd& a,scalarD shift,bool sameA);
  virtual bool recomputeKKT(const SMat& h,const SMat& a,scalarD shift,bool sameA);
  virtual bool recomputeAAT(const Matd& a,scalarD shift,bool sameA);
  virtual bool recomputeAAT(const SMat& a,scalarD shift,bool sameA);
  virtual const Matd& solve(const Matd& b)=0;
protected:
  SMat analyzeSymmetricNullspace(const SMat& a);
  Matd analyzeSymmetricNullspace(const Matd& a);
};
class EigenCholInterface : public LinearSolverInterface
{
public:
  using LinearSolverInterface::recompute;
  using LinearSolverInterface::recomputeKKT;
  using LinearSolverInterface::solve;
  virtual std::shared_ptr<LinearSolverInterface> copy() const override;
  virtual bool recompute(const SMat& smat,scalarD shift,bool sameA) override;
  virtual const Matd& solve(const Matd& b) override;
protected:
  Eigen::SimplicialCholesky<SMat> _sol;
  //transient data
  Matd _ret;
};
class EigenLUInterface : public LinearSolverInterface
{
public:
  using LinearSolverInterface::recompute;
  using LinearSolverInterface::recomputeKKT;
  using LinearSolverInterface::solve;
  virtual std::shared_ptr<LinearSolverInterface> copy() const override;
  virtual bool recompute(const SMat& smat,scalarD shift,bool sameA) override;
  virtual const Matd& solve(const Matd& b) override;
protected:
  Eigen::SparseQR<SMat,Eigen::COLAMDOrdering<sizeType> > _sol;
  //transient data
  Matd _ret;
};
#ifdef SuperLU_SUPPORT
class EigenSuperLUInterface : public LinearSolverInterface
{
public:
  using LinearSolverInterface::recompute;
  using LinearSolverInterface::recomputeKKT;
  using LinearSolverInterface::solve;
  virtual std::shared_ptr<LinearSolverInterface> copy() const override;
  virtual bool recompute(const SMat& smat,scalarD shift,bool sameA) override;
  virtual const Matd& solve(const Matd& b) override;
protected:
  Eigen::SuperLU<SMat> _sol;
  //transient data
  Matd _ret;
};
#endif
class EigenDenseCholInterface : public LinearSolverInterface
{
public:
  using LinearSolverInterface::recompute;
  using LinearSolverInterface::recomputeKKT;
  using LinearSolverInterface::solve;
  virtual std::shared_ptr<LinearSolverInterface> copy() const override;
  virtual bool recompute(const Matd& smat,scalarD shift,bool sameA) override;
  virtual const Matd& solve(const Matd& b) override;
protected:
  Eigen::LDLT<Matd> _sol;
  //transient data
  Matd _ret;
};
class EigenDenseLUInterface : public LinearSolverInterface
{
public:
  using LinearSolverInterface::recompute;
  using LinearSolverInterface::recomputeKKT;
  using LinearSolverInterface::solve;
  virtual std::shared_ptr<LinearSolverInterface> copy() const override;
  virtual bool recompute(const Matd& smat,scalarD shift,bool sameA) override;
  virtual const Matd& solve(const Matd& b) override;
protected:
  Eigen::FullPivLU<Matd> _sol;
  //transient data
  Matd _ret;
};

PRJ_END

#endif
