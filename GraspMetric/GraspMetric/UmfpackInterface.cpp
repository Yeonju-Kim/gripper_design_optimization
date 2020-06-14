#include "UmfpackInterface.h"
#include <CommonFile/Timing.h>

USE_PRJ_NAMESPACE

#ifdef UMFPACK_SUPPORT
#include "umfpack.h"
UmfpackInterface::UmfpackInterface():_built(false),_same(false),_useEigen(true) {}
UmfpackInterface::~UmfpackInterface()
{
  clear();
}
std::shared_ptr<LinearSolverInterface> UmfpackInterface::copy() const
{
  return std::shared_ptr<LinearSolverInterface>(new UmfpackInterface());
}
bool UmfpackInterface::recompute(const SMat& smat,scalarD shift,bool sameA)
{
  bool succ;
  SMat smatC=analyzeSymmetricNullspace(smat);
  smatC.makeCompressed();
  TBEG("Factorize");
  if(_same && _built) {
    smatC.makeCompressed();
    ASSERT_MSG(_A.nonZeros() == (int)smatC.nonZeros(),"We must have same number of nonzeros in sameMode of UmfpackWrapper!")
    copyData(smatC,shift);
    succ=tryFactorize(smatC);
  } else {
    clear();
    _A=smatC.cast<double>();
    _A.makeCompressed();
    copyData(smatC,shift);
    _status=umfpack_dl_symbolic(_A.rows(),_A.cols(),_A.outerIndexPtr(),_A.innerIndexPtr(),_A.valuePtr(),&_symbolic,NULL,NULL);
    ASSERT_MSGV(_status == UMFPACK_OK,"Analyze failed status=%d!",_status)
    succ=tryFactorize(smatC);
    _built=true;
  }
  TEND();
  return succ;
}
const Matd& UmfpackInterface::solve(const Matd& b)
{
  TBEG("Solve");
  if(_status != UMFPACK_OK) {
    if(_useEigen)
      _ret=_eigenSol.solve(b).cast<double>();
    else {
      ASSERT_MSG(false,"Cannot solve with _status != UMFPACK_OK.")
    }
  } else {
    _ret.resize(b.rows(),b.cols());
    _b=b.cast<double>();
    for(sizeType j=0; j<b.cols(); j++) {
      int status=umfpack_dl_solve(UMFPACK_A,_A.outerIndexPtr(),_A.innerIndexPtr(),_A.valuePtr(),
                                  _ret.data()+_ret.rows()*j,_b.data()+_b.rows()*j,_numeric,NULL,NULL);
      ASSERT_MSG(status == UMFPACK_OK,"Umfpack solve failed!")
    }
  }
  TEND();
  _retCast=_ret.cast<scalarD>();
  return _retCast;
}
void UmfpackInterface::setSameStruct(bool same)
{
  _same=same;
}
void UmfpackInterface::setUseEigen(bool useEigen)
{
  _useEigen=useEigen;
}
void UmfpackInterface::debug()
{
#define N 10
  Matd A(N,N);
  A.setRandom();
  A=(A.transpose()*A).eval();

  SMat AS;
  AS.resize(N,N);
  for(sizeType r=0; r<N; r++)
    for(sizeType c=0; c<N; c++)
      AS.coeffRef(r,c)=A(r,c);

  scalarD shift=0.1f;
  Cold B=Cold::Random(N),X,X2;
  recompute(AS,shift,false);
  X=solve(B);
  X2=(A+Matd::Identity(N,N)*shift).llt().solve(B);
  INFOV("X: %f Err: %f",X.norm(),(X-X2).norm())

  shift=0.2f;
  recompute(AS,shift,false);
  X=solve(B);
  X2=(A+Matd::Identity(N,N)*shift).llt().solve(B);
  INFOV("X: %f Err: %f",X.norm(),(X-X2).norm())
#undef N
}
void UmfpackInterface::copyData(const SMat& smat,scalarD shift)
{
  if(shift != 0)
    for(sizeType i=0; i<_A.rows(); i++)
      _A.coeffRef(i,i)+=shift;
  _A.makeCompressed();
}
bool UmfpackInterface::tryFactorize(const SMat& smat)
{
  //factorize
  _status=umfpack_dl_numeric(_A.outerIndexPtr(),_A.innerIndexPtr(),_A.valuePtr(),_symbolic,&_numeric,NULL,NULL);
  if(_status != UMFPACK_OK) {
    WARNINGV("Factorize failed status=%d, fallback to eigen!",_status)
    if(_useEigen) {
      _eigenSol.compute(smat);
      return _eigenSol.info() == Eigen::Success;
    } else return false;
  } else {
    return true;
  }
}
void UmfpackInterface::clear()
{
  if(_built) {
    if(_symbolic) {
      umfpack_dl_free_symbolic(&_symbolic);
      _symbolic=NULL;
    }
    if(_numeric) {
      umfpack_dl_free_numeric(&_numeric);
      _numeric=NULL;
    }
    _built=false;
  }
}
#endif
