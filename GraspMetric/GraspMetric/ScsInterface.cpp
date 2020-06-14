#include "ScsInterface.h"
#include "util.h"
#include "amatrix.h"

USE_PRJ_NAMESPACE

ScsInterface::ScsInterface()
{
  _k=NULL;
  _d=NULL;
  _sol=NULL;
  clearCones();
}
//b-A*x \in K
void ScsInterface::add(sizeType cid,const Vec& b)
{
  _coneIndices.find(cid)->second->_b+=b;
}
sizeType ScsInterface::constraintLinear(const Matd& coef,sizeType off,const Vec& b,CONE_TYPE type)
{
  SMat A;
  STrips trips;
  A.resize(coef.rows(),off+coef.cols());
  for(sizeType r=0; r<coef.rows(); r++)
    for(sizeType c=0; c<coef.cols(); c++)
      trips.push_back(STrip(r,off+c,coef(r,c)));
  A.setFromTriplets(trips.begin(),trips.end());
  ASSERT(type==EQUALITY || type==INEQUALITY)
  return constraint(A,b,type);
}
sizeType ScsInterface::constraintLinear(const Vec& coef,sizeType off,scalarD b,CONE_TYPE type)
{
  SMat A;
  STrips trips;
  A.resize(1,off+coef.size());
  for(sizeType i=0; i<coef.size(); i++)
    trips.push_back(STrip(0,off+i,coef[i]));
  A.setFromTriplets(trips.begin(),trips.end());
  ASSERT(type==EQUALITY || type==INEQUALITY)
  return constraint(A,Vec::Constant(1,b),type);
}
sizeType ScsInterface::constraintSecondOrder(const Matd& coef,sizeType off,const Vec& b,CONE_TYPE type)
{
  SMat A;
  STrips trips;
  A.resize(coef.rows(),off+coef.cols());
  for(sizeType r=0; r<coef.rows(); r++)
    for(sizeType c=0; c<coef.cols(); c++)
      trips.push_back(STrip(r,c+off,coef(r,c)));
  A.setFromTriplets(trips.begin(),trips.end());
  ASSERT(type==SECOND_ORDER)
  return constraint(A,b,type);
}
sizeType ScsInterface::constraintSecondOrder(const SMat& coef,sizeType off,const Vec& b,CONE_TYPE type)
{
  SMat A;
  STrips trips;
  A.resize(coef.rows(),off+coef.cols());
  for(sizeType k=0; k<coef.outerSize(); ++k)
    for(SMat::InnerIterator it(coef,k); it; ++it)
      trips.push_back(STrip(it.row(),it.col()+off,it.value()));
  A.setFromTriplets(trips.begin(),trips.end());
  ASSERT(type==SECOND_ORDER)
  return constraint(A,b,type);
}
sizeType ScsInterface::constraintLinearMatrixInequality(const Matd& coef,sizeType off,const Matd& c0,CONE_TYPE type)
{
  ASSERT(c0.rows()==c0.cols() && coef.rows()==c0.rows())
  sizeType nrVar=coef.cols()/c0.cols();

  SMat A;
  STrips trips;
  Vec b=toLower(c0);
  A.resize(b.size(),off+nrVar);
  for(sizeType i=0; i<nrVar; i++) {
    Vec tmp=-toLower(coef.block(0,c0.cols()*i,c0.rows(),c0.cols()));
    for(sizeType r=0; r<tmp.size(); r++)
      trips.push_back(STrip(r,off+i,tmp[r]));
  }
  A.setFromTriplets(trips.begin(),trips.end());
  ASSERT(type==SEMI_DEFINITE)
  return constraint(A,b,type);
}
sizeType ScsInterface::constraint(const SMat& A,const Vec& b,CONE_TYPE type)
{
  std::shared_ptr<CONE> c(new CONE);
  c->_A=A;
  c->_b=b;
  c->_cid=_cid++;
  _cones[type].push_back(c);
  _coneIndices[c->_cid]=c;
  return c->_cid;
}
sizeType ScsInterface::solve(const Vec& c,Vec& x,bool report)
{
  SMat A;
  Vec b;
  //initialize
  _k=(ScsCone*)scs_calloc(1,sizeof(ScsCone));
  _d=(ScsData*)scs_calloc(1,sizeof(ScsData));
  _d->stgs=(ScsSettings*)scs_calloc(1,sizeof(ScsSettings));
  _sol=(ScsSolution*)scs_calloc(1,sizeof(ScsSolution));
  _info= {0};
  SCS(set_default_settings)(_d);
  _d->stgs->max_iters=1E8;
  _d->stgs->eps=1E-8f;
  //set cones
  getCones(A,b,c.size());
  _d->n=c.size();
  _d->m=b.size();
  _d->A=(ScsMatrix*)scs_calloc(1,sizeof(ScsMatrix));
  _d->b=(scs_float*)scs_calloc(_d->m,sizeof(scs_float));
  _d->A->p=(scs_int*)scs_calloc(_d->n+1,sizeof(scs_int));
  _d->A->i=(scs_int*)scs_calloc(A.nonZeros(),sizeof(scs_int));
  _d->A->x=(scs_float*)scs_calloc(A.nonZeros(),sizeof(scs_float));
  _d->A->n=_d->n;
  _d->A->m=_d->m;
  A.makeCompressed();
  Eigen::Map<Eigen::Matrix<scs_int,-1,1>>(_d->A->p,_d->n+1)=Eigen::Map<Eigen::Matrix<sizeType,-1,1>>(A.outerIndexPtr(),_d->n+1).template cast<scs_int>();
  Eigen::Map<Eigen::Matrix<scs_int,-1,1>>(_d->A->i,A.nonZeros())=Eigen::Map<Eigen::Matrix<sizeType,-1,1>>(A.innerIndexPtr(),A.nonZeros()).template cast<scs_int>();
  Eigen::Map<Eigen::Matrix<scs_float,-1,1>>(_d->A->x,A.nonZeros())=Eigen::Map<Eigen::Matrix<scalarD,-1,1>>(A.valuePtr(),A.nonZeros()).template cast<scs_float>();
  Eigen::Map<Eigen::Matrix<scs_float,-1,1>>(_d->b,_d->m)=b.template cast<scs_float>();
  //set objective function coefficient
  _d->c=(scs_float*)scs_calloc(_d->n,sizeof(scs_float));
  Eigen::Map<Eigen::Matrix<scs_float,-1,1>>(_d->c,_d->n)=c.template cast<scs_float>();
  //solve
  _d->stgs->verbose=report;
  scs(_d,_k,_sol,&_info);
  //fetch result
  x=Eigen::Map<Eigen::Matrix<scs_float,-1,1>>(_sol->x,_d->n).template cast<scalarD>();
  //finalize
  SCS(free_data)(_d,_k);
  SCS(free_sol)(_sol);
  //fetch status
  return _info.status_val;
}
void ScsInterface::clearCones()
{
  _cones.clear();
  _coneIndices.clear();
  _cid=0;
}
//helper
void ScsInterface::getCones(SMat& A,Vec& b,sizeType n)
{
  sizeType m=0,off=0,lastOff=-1;
  for(CONES::const_iterator beg=_cones.begin(),end=_cones.end(); beg!=end; beg++)
    for(sizeType i=0; i<(sizeType)beg->second.size(); i++)
      m+=beg->second[i]->_A.rows();
  //setup ScsCone
  _k->f=0;
  _k->l=0;
  _k->q=NULL;
  _k->qsize=0;
  _k->s=NULL;
  _k->ssize=0;
  _k->ep=0;
  _k->ed=0;
  _k->p=NULL;
  _k->psize=0;
  //A,b
  STrips trips;
  A.resize(m,n);
  b.resize(m);
  for(CONES::const_iterator beg=_cones.begin(),end=_cones.end(); beg!=end; beg++) {
    lastOff=off;
    sizeType nrCone=(sizeType)beg->second.size();
    for(sizeType i=0; i<nrCone; i++) {
      sizeType mI=beg->second[i]->_A.rows();
      const SMat& AI=beg->second[i]->_A;
      for(sizeType k=0; k<AI.outerSize(); ++k)
        for(SMat::InnerIterator it(AI,k); it; ++it)
          trips.push_back(STrip(it.row()+off,it.col(),it.value()));
      b.segment(off,mI)=beg->second[i]->_b;
      off+=mI;
    }
    if(beg->first==EQUALITY)
      _k->f=off-lastOff;
    else if(beg->first==INEQUALITY)
      _k->l=off-lastOff;
    else if(beg->first==SECOND_ORDER) {
      _k->q=(scs_int*)scs_calloc(nrCone,sizeof(scs_int));
      for(sizeType i=0; i<nrCone; i++)
        _k->q[i]=beg->second[i]->_A.rows();
      _k->qsize=nrCone;
    } else if(beg->first==SEMI_DEFINITE) {
      _k->s=(scs_int*)scs_calloc(nrCone,sizeof(scs_int));
      for(sizeType i=0; i<nrCone; i++)
        _k->s[i]=SCRows(beg->second[i]->_A.rows());
      _k->ssize=nrCone;
    }
  }
  A.setFromTriplets(trips.begin(),trips.end());
}
ScsInterface::Vec ScsInterface::toLower(Matd m)
{
  static scalarD sqrt2=std::sqrt(2.0);
  for(sizeType r=0; r<m.rows(); r++)
    for(sizeType c=0; c<m.cols(); c++)
      if(r>c)
        m(r,c)*=sqrt2;
  Vec ret=Vec::Zero(m.rows()*(m.rows()+1)/2);
  for(sizeType i=0,off=0; i<m.cols(); i++) {
    ret.segment(off,m.rows()-i)=m.col(i).segment(i,m.rows()-i);
    off+=m.rows()-i;
  }
  return ret;
}
sizeType ScsInterface::SCRows(sizeType entries) const
{
  return std::floor(std::sqrt(entries*2));
}
