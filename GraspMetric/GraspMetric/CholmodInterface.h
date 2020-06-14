#ifndef CHOLMOD_INTERFACE_H
#define CHOLMOD_INTERFACE_H
#ifdef CHOLMOD_SUPPORT

#include "EigenInterface.h"
#include "cholmod.h"

PRJ_BEGIN

class CholmodInterface : public LinearSolverInterface
{
public:
  typedef Eigen::SparseMatrix<double,0,sizeType> SMatD;
  CholmodInterface();
  virtual ~CholmodInterface();
  virtual std::shared_ptr<LinearSolverInterface> copy() const override;
  bool recompute(const SMat& smat,scalarD shift,bool sameA) override;
  bool recomputeAAT(const SMat& smat,scalarD shift,bool sameA) override;
  const Matd& solve(const Matd& b) override;
  void setSupernodal(bool super);
  void setSameStruct(bool same);
  void setUseEigen(bool useEigen);
private:
  void copyData(const SMatD& smat);
  bool recomputeInner(const SMat& smat,scalarD shift,bool sameA,bool sym);
  bool tryFactorize(const SMatD& smat,scalarD shift,bool sym);
  void clear();
  Eigen::SimplicialCholesky<SMat> _eigenSol;
  //cholmod data
  cholmod_factor* _L;
  cholmod_sparse* _A;
  cholmod_common _c;
  //transient data
  Matd _ret;
  bool _built;
  bool _same;
  bool _useEigen;
  bool _supernodal;
};

PRJ_END

#endif
#endif
