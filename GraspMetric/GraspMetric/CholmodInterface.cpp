#include "CholmodInterface.h"
#include <CommonFile/Timing.h>

USE_PRJ_NAMESPACE

#ifdef CHOLMOD_SUPPORT
CholmodInterface::CholmodInterface():_built(false),_same(false),_useEigen(true),_supernodal(false) {}
CholmodInterface::~CholmodInterface()
{
  clear();
}
std::shared_ptr<LinearSolverInterface> CholmodInterface::copy() const
{
  return std::shared_ptr<LinearSolverInterface>(new CholmodInterface);
}
bool CholmodInterface::recompute(const SMat& smat,scalarD shift,bool sameA)
{
  return recomputeInner(smat,shift,sameA,true);
}
bool CholmodInterface::recomputeAAT(const SMat& smat,scalarD shift,bool sameA)
{
  return recompute(analyzeSymmetricNullspace(SMat(smat*smat.transpose())),shift,sameA);
  //return recomputeInner(smat,shift,sameA,false);
}
const Matd& CholmodInterface::solve(const Matd& b)
{
  TBEG("Solve-Cholmod");
  if(_c.status != CHOLMOD_OK) {
    if(_useEigen)
      _ret=_eigenSol.solve(b);
    else {
      ASSERT_MSG(false,"Cannot solve with cholmod_common.status != CHOLMOD_OK.")
    }
  } else {
    ASSERT(b.rows() == (sizeType)_A->nrow)
    cholmod_dense* B=cholmod_zeros(_A->nrow,b.cols(),_A->xtype,&_c);
    Eigen::Map<Eigen::Matrix<double,-1,-1> >((double*)B->x,B->nrow,B->ncol)=b.cast<double>();

    cholmod_dense* X=cholmod_solve(CHOLMOD_A,_L,B,&_c);
    _ret=Eigen::Map<Eigen::Matrix<double,-1,-1> >((double*)X->x,B->nrow,B->ncol).cast<scalarD>();
    ASSERT_MSGV(_c.status == CHOLMOD_OK,"Cholmod solve failed, status=%d!",_c.status)

    cholmod_free_dense(&B,&_c);
    cholmod_free_dense(&X,&_c);
  }
  TEND();
  return _ret;
}
void CholmodInterface::setSupernodal(bool super)
{
  _supernodal=super;
  clear();
}
void CholmodInterface::setSameStruct(bool same)
{
  _same=same;
}
void CholmodInterface::setUseEigen(bool useEigen)
{
  _useEigen=useEigen;
}
void CholmodInterface::copyData(const SMatD& smat)
{
  memcpy(_A->x,smat.valuePtr(),sizeof(double)*smat.nonZeros());
}
bool CholmodInterface::recomputeInner(const SMat& smat,scalarD shift,bool sameA,bool sym)
{
  bool succ;
  SMatD smatC=smat.cast<double>();
  smatC.makeCompressed();
  TBEG("Factorize-Cholmod");
  if(_same && _built) {
    ASSERT_MSG(_A->nzmax == (size_t)smatC.nonZeros(),"We must have same number of nonzeros in sameMode of CholmodWrapper!")
    if(!sameA)
      copyData(smatC);
    succ=tryFactorize(smatC,shift,sym);
  } else {
    clear();
    cholmod_start(&_c);
    //_c.useGPU=1;
    _c.supernodal=_supernodal ? CHOLMOD_SUPERNODAL : CHOLMOD_SIMPLICIAL;
    _A=cholmod_allocate_sparse((size_t)smatC.rows(),(size_t)smatC.cols(),(size_t)smatC.nonZeros(),1,1,sym?-1:0,CHOLMOD_REAL,&_c);
    ASSERT_MSG(_c.status == CHOLMOD_OK,"Cholmod create sparse failed!")
    int* P=(int*)_A->p;
    int* I=(int*)_A->i;
    ASSERT(_A->itype == CHOLMOD_INT)
    OMP_PARALLEL_FOR_
    for(sizeType i=0; i<=smatC.cols(); i++)
      P[i]=smatC.outerIndexPtr()[i];
    OMP_PARALLEL_FOR_
    for(sizeType i=0; i<smatC.nonZeros(); i++)
      I[i]=smatC.innerIndexPtr()[i];
    copyData(smatC);

    _L=cholmod_analyze(_A,&_c);
    ASSERT_MSG(_c.status == CHOLMOD_OK,"Cholmod analyze failed!")
    succ=tryFactorize(smatC,shift,sym);
    _built=true;
  }
  TEND();
  return succ;
}
bool CholmodInterface::tryFactorize(const SMatD& smat,scalarD shift,bool sym)
{
  double beta[2]= {(double)shift,0};
  cholmod_factorize_p(_A,beta,NULL,0,_L,&_c);
  if(_c.status != CHOLMOD_OK) {
    WARNINGV("Factorize failed status=%d, fallback to eigen!",_c.status)
    if(_useEigen) {
      _eigenSol.setShift(shift);
      if(!sym)
        _eigenSol.compute((smat*smat.transpose()).cast<scalarD>());
      else _eigenSol.compute(smat.cast<scalarD>());
      return _eigenSol.info() == Eigen::Success;
    } else return false;
  } else {
    return true;
  }
}
void CholmodInterface::CholmodInterface::clear()
{
  if(_built) {
    if(_L)
      cholmod_free_factor(&_L,&_c);
    cholmod_free_sparse(&_A,&_c);
    cholmod_finish(&_c);
    _built=false;
  }
}
#endif
