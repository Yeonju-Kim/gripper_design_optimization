#ifndef SPARSE_UTILS_H
#define SPARSE_UTILS_H

#include "Objective.h"
#include <CommonFile/IO.h>

PRJ_BEGIN

//sparse matrix building
//dense
template <typename MAT,typename Derived>
EIGEN_DEVICE_FUNC static void addBlock(MAT& H,sizeType r,sizeType c,const Eigen::MatrixBase<Derived>& coef)
{
  H.block(r,c,coef.rows(),coef.cols())+=coef;
}
template <typename MAT>
EIGEN_DEVICE_FUNC static void addBlock(MAT& H,sizeType r,sizeType c,scalarD coef)
{
  H(r,c)+=coef;
}
template <typename MAT,typename T>
EIGEN_DEVICE_FUNC static void addBlockId(MAT& H,sizeType r,sizeType c,sizeType nr,T coefId)
{
  H.block(r,c,nr,nr).diagonal().array()+=coefId;
}
//sparse
template <typename Derived>
EIGEN_DEVICE_FUNC static void addBlock(ParallelVector<Eigen::Triplet<scalarD,sizeType> >& H,sizeType r,sizeType c,const Eigen::MatrixBase<Derived>& coef)
{
#ifndef __CUDACC__
  sizeType nrR=coef.rows();
  sizeType nrC=coef.cols();
  for(sizeType i=0; i<nrR; i++)
    for(sizeType j=0; j<nrC; j++)
      H.push_back(Eigen::Triplet<scalarD,sizeType>(r+i,c+j,coef(i,j)));
#else
  FUNCTION_NOT_IMPLEMENTED
#endif
}
EIGEN_DEVICE_FUNC static void addBlock(ParallelVector<Eigen::Triplet<scalarD,sizeType> >& H,sizeType r,sizeType c,scalarD coef)
{
#ifndef __CUDACC__
  H.push_back(Eigen::Triplet<scalarD,sizeType>(r,c,coef));
#else
  FUNCTION_NOT_IMPLEMENTED
#endif
}
template <typename Derived>
EIGEN_DEVICE_FUNC static void addBlockI(ParallelVector<Eigen::Triplet<scalarD,sizeType> >& H,sizeType r,sizeType c,const Eigen::MatrixBase<Derived>& coefI)
{
#ifndef __CUDACC__
  sizeType nr=coefI.size();
  for(sizeType i=0; i<nr; i++)
    H.push_back(Eigen::Triplet<scalarD,sizeType>(r+i,c+i,coefI[i]));
#else
  FUNCTION_NOT_IMPLEMENTED
#endif
}
template <typename T>
EIGEN_DEVICE_FUNC static void addBlockId(ParallelVector<Eigen::Triplet<scalarD,sizeType> >& H,sizeType r,sizeType c,sizeType nr,T coefId)
{
#ifndef __CUDACC__
  for(sizeType i=0; i<nr; i++)
    H.push_back(Eigen::Triplet<scalarD,sizeType>(r+i,c+i,coefId));
#else
  FUNCTION_NOT_IMPLEMENTED
#endif
}
template <typename S,int O,typename I>
EIGEN_DEVICE_FUNC static void addBlock(ParallelVector<Eigen::Triplet<scalarD,sizeType> >& H,sizeType r,sizeType c,const Eigen::SparseMatrix<S,O,I>& coef)
{
#ifndef __CUDACC__
  for(sizeType k=0; k<coef.outerSize(); ++k)
    for(typename Eigen::SparseMatrix<S,O,I>::InnerIterator it(coef,k); it; ++it)
      H.push_back(Eigen::Triplet<scalarD,sizeType>(r+it.row(),c+it.col(),it.value()));
#else
  FUNCTION_NOT_IMPLEMENTED
#endif
}
//sparseVec
template <typename Derived>
EIGEN_DEVICE_FUNC static void addBlock(Eigen::SparseVector<scalarD,0,sizeType>& H,sizeType r,const Eigen::MatrixBase<Derived>& coef)
{
  for(sizeType i=0; i<coef.rows(); i++)
    H.coeffRef(r+i)+=coef[i];
}
//maxAbs
scalarD absMax(const Objective<scalarD>::SMat& h);
scalarD absMaxRel(const Objective<scalarD>::SMat& h,const Objective<scalarD>::SMat& hRef,bool detail=false);
//build KKT matrix
Matd buildKKT(const Matd& h,const Matd& a,scalarD shift);
Objective<scalarD>::SMat buildKKT(const Objective<scalarD>::SMat& h,const Objective<scalarD>::SMat& a,scalarD shift);
//kronecker-product
Matd kronecker(const Matd& h,sizeType n);
Objective<scalarD>::SMat kronecker(const Objective<scalarD>::SMat& h,sizeType n);
//concat-diag
Objective<scalarD>::SMat concatDiag(const Objective<scalarD>::SMat& a,const Objective<scalarD>::SMat& b);
//sparseIO
template <typename T,typename MT>
Eigen::SparseMatrix<T,0,sizeType> toSparse(const MT& m)
{
  Eigen::SparseMatrix<T,0,sizeType> ret;
  ParallelVector<Eigen::Triplet<T,sizeType> > trips;
  for(sizeType r=0; r<m.rows(); r++)
    for(sizeType c=0; c<m.cols(); c++)
      trips.push_back(Eigen::Triplet<T,sizeType>(r,c,m(r,c)));
  ret.resize(m.rows(),m.cols());
  ret.setFromTriplets(trips.begin(),trips.end());
  return ret;
}
template <typename T>
Eigen::SparseMatrix<T,0,sizeType> concat(const Eigen::SparseMatrix<T,0,sizeType>& A,const Eigen::SparseMatrix<T,0,sizeType>& B)
{
  ASSERT(A.cols()==B.cols())
  Eigen::SparseMatrix<T,0,sizeType> M(A.rows()+B.rows(),A.cols());
  M.reserve(A.nonZeros()+B.nonZeros());
  for(sizeType c=0; c<A.cols(); ++c) {
    M.startVec(c);
    for(typename Eigen::SparseMatrix<T,0,sizeType>::InnerIterator itA(A,c); itA; ++itA)
      M.insertBack(itA.row(),c)=itA.value();
    for(typename Eigen::SparseMatrix<T,0,sizeType>::InnerIterator itB(B,c); itB; ++itB)
      M.insertBack(itB.row()+A.rows(),c)=itB.value();
  }
  M.finalize();
  return M;
}
template <typename T,int O,typename I>
bool readBinaryData(Eigen::SparseMatrix<T,O,I>& m,std::istream& is,IOData* dat=NULL)
{
  sizeType rows,cols;
  std::vector<I> r;
  std::vector<I> c;
  std::vector<T> v;
  readBinaryData(rows,is);
  readBinaryData(cols,is);
  readBinaryData(r,is);
  readBinaryData(c,is);
  readBinaryData(v,is);
  m.resize(rows,cols);
  m.reserve(v.size());
  for(sizeType ci=0; ci<m.cols(); ++ci) {
    m.startVec(ci);
    for(sizeType off=c[ci]; off<c[ci+1]; off++)
      m.insertBack(r[off],ci)=v[off];
  }
  m.finalize();
  return is.good();
}
template <typename T,int O,typename I>
bool writeBinaryData(const Eigen::SparseMatrix<T,O,I>& m,std::ostream& os,IOData* dat=NULL)
{
  std::vector<I> r(m.nonZeros());
  std::vector<I> c(m.cols()+1);
  std::vector<T> v(m.nonZeros());
  for(sizeType k=0,offr=0; k<m.outerSize(); ++k)
    for(typename Eigen::SparseMatrix<T,O,I>::InnerIterator it(m,k); it; ++it) {
      v[offr]=it.value();
      r[offr++]=it.row();
      c[k+1]++;
    }
  for(sizeType k=0; k<m.outerSize(); ++k)
    c[k+1]+=c[k];
  writeBinaryData((sizeType)m.rows(),os);
  writeBinaryData((sizeType)m.cols(),os);
  writeBinaryData(r,os);
  writeBinaryData(c,os);
  writeBinaryData(v,os);
  return os.good();
}
//sparse dense adaptation
class LinearSolverInterface;
template <typename T>
struct LinearSolverTraits;
template <>
struct LinearSolverTraits<Matd> {
  static std::shared_ptr<LinearSolverInterface> getCholSolver();
  static std::shared_ptr<LinearSolverInterface> getLUSolver();
  static void rescale(const Cold& c,const Matd& a,Cold& scale,Cold& cOut,Matd& aOut);
};
template <>
struct LinearSolverTraits<Objective<scalarD>::SMat> {
  typedef Objective<scalarD>::SMat SMat;
  static std::shared_ptr<LinearSolverInterface> getCholSolver();
  static std::shared_ptr<LinearSolverInterface> getLUSolver();
  static void rescale(const Cold& c,const SMat& a,Cold& scale,Cold& cOut,SMat& aOut);
};

PRJ_END

#endif
