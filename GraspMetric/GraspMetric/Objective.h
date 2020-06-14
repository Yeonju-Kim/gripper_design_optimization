#ifndef OBJECTIVE_H
#define OBJECTIVE_H

#include <CommonFile/MathBasic.h>
#include "ParallelVector.h"
#include <Eigen/Sparse>

PRJ_BEGIN

template < typename T>
struct Objective {
public:
  typedef typename Eigen::SparseMatrix<T,0,sizeType> SMat;
  typedef typename Eigen::Matrix<T,-1,-1> DMat;
  typedef typename Eigen::Matrix<T,-1,1> Vec;
  typedef Eigen::Triplet<T,sizeType> STrip;
  typedef ParallelVector<STrip> STrips;
  //for L-BFGS Optimization
  virtual int operator()(const Vec& x,T& FX,Vec& DFDX,const T& step,bool wantGradient) {
    ASSERT_MSG(false,"Not Implemented: Objective Function Without Hessian!")
    return -1;
  }
  //for Line Search Modification
  virtual void beginLineSearch() {}
  virtual void endLineSearch(const Vec& x,T& FX,Vec& DFDX) {}
  virtual void profileLineSearch(const sizeType& k,const Vec& x,const Vec& d,const T& step) {
    return;
  }
  //for LM Optimization, for Cons in FilterSQP
  virtual int operator()(const Vec& x,Vec& fvec,DMat* fjac,bool modifiableOrSameX) {
    int ret=operator()(x,fvec,fjac?&_tmpFjacs:NULL,modifiableOrSameX);
    if(fjac)
      *fjac=_tmpFjacs.toDense();
    return ret;
  }
  virtual int operator()(const Vec& x,Vec& fvec,SMat* fjac,bool modifiableOrSameX) {
    if(fjac)  //to remember number of entries
      _tmp.clear();
    int ret=operator()(x,fvec,fjac?&_tmp:NULL,modifiableOrSameX);
    if(fjac) {
      fjac->resize(values(),inputs());
      if(!_tmp.getVector().empty())
        fjac->setFromTriplets(_tmp.begin(),_tmp.end());
    }
    return ret;
  }
  virtual int operator()(const Vec& x,Vec& fvec,STrips* fjac,bool modifiableOrSameX) {
    ASSERT_MSG(false,"Not Implemented: Sparse Least Square Function!")
    return -1;
  }
  //for SQP Optimization
  virtual T operator()(const Vec& x,Vec* fgrad,DMat* fhess) {
    T ret=operator()(x,fgrad,fhess?&_tmpFjacs:NULL);
    if(fhess)
      *fhess=_tmpFjacs.toDense();
    return ret;
  }
  virtual T operator()(const Vec& x,Vec* fgrad,SMat* fhess) {
    if(fhess)  //to remember number of entries
      _tmp.clear();
    T ret=operator()(x,fgrad,fhess?&_tmp:NULL);
    if(fhess) {
      fhess->resize(inputs(),inputs());
      fhess->setFromTriplets(_tmp.begin(),_tmp.end());
    }
    return ret;
  }
  virtual T operator()(const Vec& x,Vec* fgrad,STrips* fhess) {
    if(fgrad)
      fgrad->setZero(inputs());
    if(fhess)
      fhess->clear();
    return 0;
  }
  virtual bool onNewIter(const Vec& x) {
    return true;
  }
  virtual void setLambda(const Vec& lambda) {}
  virtual void setUseFirstOrder(bool firstOrder) {}
  //for Stochastic Optimization
  virtual T operator()(const Vec& x) {
    ASSERT_MSG(false, "Not Implemented: Function Value Evaluation!")
    return 0;
  }
  virtual bool feasible(const Vec& x) {
    ASSERT_MSG(false, "Not Implemented: Feasibility Check!")
    return false;
  }
  //Dimension Info
  virtual int inputs() const {
    return 0;
  }
  virtual int values() const {
    return 0;
  }
protected:
  SMat _tmpFjacs;
  STrips _tmp;
};

PRJ_END

#endif
