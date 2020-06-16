#include "ConvexHull.h"

USE_PRJ_NAMESPACE

PRJ_BEGIN
//ConvexHull<DIM>
template <int DIM>
void ConvexHull<DIM>::insertInit(const PTS& pss)
{
  for(sizeType i=0; i<(sizeType)pss.size(); i++)
    insert(pss[i]);
}
template <int DIM>
typename ConvexHull<DIM>::PT ConvexHull<DIM>::mul(const Eigen::Matrix<scalarD,DIM,6>* basis,const Vec6d& p)
{
  if(basis)
    return *basis*p;
  else return p.template segment<DIM>(0);
}
template <int DIM>
Vec6d ConvexHull<DIM>::mulT(const Eigen::Matrix<scalarD,DIM,6>* basis,const PT& p)
{
  if(basis)
    return basis->transpose()*p;
  else {
    Vec6d ret=Vec6d::Zero();
    ret.template segment<DIM>(0)=p;
    return ret;
  }
}
template class ConvexHull<5>;
template class ConvexHull<6>;
PRJ_END

#ifdef CGAL_SUPPORT
#include <CGAL/Convex_hull_d.h>
#include <CGAL/Cartesian_d.h>
#include <CGAL/Gmpq.h>
PRJ_BEGIN
//CGALConvexHull<DIM>
template <int DIM>
CGALConvexHull<DIM>::CGALConvexHull()
{
  typedef CGAL::Gmpq NumberType;
  typedef CGAL::Convex_hull_d<CGAL::Cartesian_d<NumberType>> CGAL_QHULL;
  _hull=new CGAL_QHULL(DIM);
}
template <int DIM>
CGALConvexHull<DIM>::~CGALConvexHull<DIM>()
{
  typedef CGAL::Gmpq NumberType;
  typedef CGAL::Convex_hull_d<CGAL::Cartesian_d<NumberType>> CGAL_QHULL;
  delete reinterpret_cast<CGAL_QHULL*>(_hull);
}
template <int DIM>
void CGALConvexHull<DIM>::insert(const PT& p)
{
  typedef CGAL::Gmpq NumberType;
  typedef CGAL::Convex_hull_d<CGAL::Cartesian_d<NumberType>> CGAL_QHULL;
  CGAL_QHULL* hull=reinterpret_cast<CGAL_QHULL*>(_hull);

  Eigen::Matrix<NumberType,DIM,1> pNum=p.template cast<double>().template cast<NumberType>();
  hull->insert(CGAL::Cartesian_d<NumberType>::Point_d(DIM,pNum.data(),pNum.data()+DIM));
}
template <int DIM>
scalarD CGALConvexHull<DIM>::distToOrigin(PT& blockingPN)
{
  typedef CGAL::Gmpq NumberType;
  typedef CGAL::Convex_hull_d<CGAL::Cartesian_d<NumberType>> CGAL_QHULL;
  CGAL_QHULL* hull=reinterpret_cast<CGAL_QHULL*>(_hull);

  PT n;
  scalarD dist;
  scalarD Q=ScalarUtil<scalarD>::scalar_max();
  for(CGAL::Convex_hull_d<CGAL::Cartesian_d<NumberType>>::Facet_iterator beg=hull->facets_begin(),end=hull->facets_end(); beg!=end; beg++)
  {
    CGAL::Cartesian_d<NumberType>::Hyperplane_d p=hull->hyperplane_supporting(beg);
    if(p.dimension()!=DIM)
      continue;
    for(sizeType d=0; d<DIM; d++)
      n[d]=CGAL::to_double(p[d]);
    dist=-CGAL::to_double(p[DIM])/n.norm();
    if(dist<Q) {
      Q=dist;
      blockingPN=n;
    }
  }
  return Q;
}
template class CGALConvexHull<5>;
template class CGALConvexHull<6>;
PRJ_END
#endif

#ifdef QHULL_SUPPORT
extern "C"
{
#include <libqhull/qhull_a.h>
}
PRJ_BEGIN
//QHullConvexHull<DIM>
#define NR_PT_INIT 100
template <int DIM>
QHullConvexHull<DIM>::QHullConvexHull():_memPt(NULL)
{
  reinitQHull(0,0,NULL);
}
template <int DIM>
QHullConvexHull<DIM>::~QHullConvexHull<DIM>()
{
  freeQHull();
}
template <int DIM>
void QHullConvexHull<DIM>::insert(const PT& p)
{
  if(_nrPtUsed==_nrPtAll) {
    void* memPt=malloc((_nrPtUsed+1)*DIM*sizeof(coordT));
    if(_nrPtUsed>0)
      memcpy(memPt,_memPt,_nrPtUsed*DIM*sizeof(coordT));
    coordT* memPtC=(coordT*)memPt;
    for(sizeType i=0; i<DIM; i++)
      memPtC[DIM*_nrPtUsed+i]=p[i];
    reinitQHull(std::max<sizeType>(_nrPtAll*2,NR_PT_INIT),_nrPtUsed+1,memPt);
    free(memPt);
  } else {
    coordT *memPtC=((coordT*)_memPt)+_nrPtUsed*DIM;
    facetT *facet;
    boolT isoutside;
    realT bestdist;
    for(sizeType i=0; i<DIM; i++)
      memPtC[i]=p[i];
    facet=qh_findbestfacet(memPtC,!qh_ALL,&bestdist,&isoutside);
    if(isoutside)
      qh_addpoint(memPtC,facet,False);  /* user requested an early exit with 'TVn' or 'TCn' */
    _nrPtUsed++;
  }
}
template <int DIM>
void QHullConvexHull<DIM>::insertInit(const PTS& pss)
{
  _nrPtUsed=(sizeType)pss.size();
  void* memPt=malloc(_nrPtUsed*DIM*sizeof(coordT));
  coordT* memPtC=(coordT*)memPt;
  for(sizeType p=0,off=0; p<_nrPtUsed; p++)
    for(sizeType i=0; i<DIM; i++)
      memPtC[off++]=pss[p][i];
  reinitQHull(std::max<sizeType>(_nrPtAll*2,NR_PT_INIT),_nrPtUsed,memPt);
  free(memPt);
}
template <int DIM>
scalarD QHullConvexHull<DIM>::distToOrigin(PT& blockingPN)
{
  PT n;
  facetT* facet;
  scalarD dist;
  scalarD Q=ScalarUtil<scalarD>::scalar_max();
  //qh_triangulate();
  FORALLfacets {
    for(sizeType d=0; d<DIM; d++)
      n[d]=facet->normal[d];
    dist=-facet->offset/n.norm();
    if(dist<Q) {
      Q=dist;
      blockingPN=n;
    }
  }
  return Q;
}
template <int DIM>
void QHullConvexHull<DIM>::reinitQHull(sizeType nrPtAll,sizeType nrPtUsed,void* memPt)
{
  freeQHull();
  ASSERT_MSGV(nrPtUsed==0 || nrPtUsed>=DIM+1,"Either add 0 points or add more than %d points",DIM+1)
  if(nrPtUsed==0) {
    _nrPtAll=_nrPtUsed=0;
    return;
  } else {
    char options[2000];
    memset(options,'\0',2000);
    QHULL_LIB_CHECK
    qh_init_A(stdin,stdout,stderr,0,NULL);
    int exitcode=setjmp(qh errexit);
    ASSERT(!exitcode)
    qh NOerrexit=False;
    qh_initflags(options);

    ASSERT(nrPtAll>0)
    _nrPtAll=nrPtAll; //initially we have 500 points
    _nrPtUsed=nrPtUsed;
    _memPt=malloc(_nrPtAll*DIM*sizeof(coordT));
    if(nrPtUsed>0)
      memcpy(_memPt,memPt,nrPtUsed*DIM*sizeof(coordT));
    qh_init_B((coordT*)_memPt,_nrPtUsed,DIM,true);
    qh_qhull();
    qh_triangulate();
  }
}
template <int DIM>
void QHullConvexHull<DIM>::freeQHull()
{
  if(_memPt!=NULL) {
    qh NOerrexit=False;
    int curlong,totlong;
#ifdef qh_NOmem
    qh_freeqhull(qh_ALL);
#else
    qh_freeqhull(!qh_ALL);
    qh_memfreeshort(&curlong,&totlong);
    if(curlong || totlong)
      fprintf(stderr,"qhull warning (QHullConvexHull, run 1): did not free %d bytes of long memory (%d pieces)\n",totlong,curlong);
#endif
    //free(_memPt);
    _memPt=NULL;
  }
}
template class QHullConvexHull<5>;
template class QHullConvexHull<6>;
PRJ_END
#endif
