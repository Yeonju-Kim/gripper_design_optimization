#ifndef UMFPACK_INTERFACE_H
#define UMFPACK_INTERFACE_H
#ifdef UMFPACK_SUPPORT

#include "EigenInterface.h"
#include "umfpack.h"

PRJ_BEGIN

class UmfpackInterface : public LinearSolverInterface
{
public:
  UmfpackInterface();
  virtual ~UmfpackInterface();
  virtual std::shared_ptr<LinearSolverInterface> copy() const override;
  bool recompute(const SMat& smat,scalarD shift,bool sameA) override;
  const Matd& solve(const Matd& b) override;
  void setSameStruct(bool same);
  void setUseEigen(bool useEigen);
  void debug();
private:
  void copyData(const SMat& smat,scalarD shift);
  bool tryFactorize(const SMat& smat);
  void clear();
  //data
  Eigen::SparseLU<SMat> _eigenSol;
  void *_numeric;
  void *_symbolic;
  int _status;
  Matd _retCast;
  Eigen::Matrix<double,-1,-1> _ret,_b;
  Eigen::SparseMatrix<double,0,long> _A;
  bool _built,_same,_useEigen;
};

PRJ_END

#endif
#endif
