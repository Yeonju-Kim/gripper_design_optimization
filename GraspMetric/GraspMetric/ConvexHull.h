#ifndef CONVEX_HULL_H
#define CONVEX_HULL_H

#include "CommonFile/MathBasic.h"

PRJ_BEGIN

template <int DIM>
struct ConvexHull
{
public:
  typedef Eigen::Matrix<scalarD,DIM,1> PT;
  typedef std::vector<PT,Eigen::aligned_allocator<PT>> PTS;
  virtual ~ConvexHull() {}
  virtual void insert(const PT& p)=0;
  virtual void insertInit(const PTS& pss);
  virtual scalarD distToOrigin(PT& blockingPN)=0;
  static PT mul(const Eigen::Matrix<scalarD,DIM,6>* basis,const Vec6d& p);
  static Vec6d mulT(const Eigen::Matrix<scalarD,DIM,6>* basis,const PT& p);
};
#ifdef CGAL_SUPPORT
template <int DIM>
struct CGALConvexHull : public ConvexHull<DIM>
{
public:
  using typename ConvexHull<DIM>::PT;
  using typename ConvexHull<DIM>::PTS;
  CGALConvexHull();
  virtual ~CGALConvexHull();
  virtual void insert(const PT& p) override;
  virtual scalarD distToOrigin(PT& blockingPN) override;
private:
  void* _hull;
};
#endif
#ifdef QHULL_SUPPORT
template <int DIM>
struct QHullConvexHull : public ConvexHull<DIM>
{
public:
  using typename ConvexHull<DIM>::PT;
  using typename ConvexHull<DIM>::PTS;
  QHullConvexHull();
  virtual ~QHullConvexHull();
  virtual void insert(const PT& p) override;
  virtual void insertInit(const PTS& pss) override;
  virtual scalarD distToOrigin(PT& blockingPN) override;
private:
  void reinitQHull(sizeType nrPtAll,sizeType nrPtUsed,void* memPt);
  void freeQHull();
  sizeType _nrPtAll,_nrPtUsed;
  void* _memPt;
};
#endif

PRJ_END

#endif
