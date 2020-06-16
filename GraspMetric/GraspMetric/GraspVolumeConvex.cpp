#ifdef CGAL_SUPPORT
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/convex_hull_3.h>
#include <iostream>
#include <fstream>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Polyhedron_3<K> Polyhedron_3;
typedef K::Point_3 Point_3;
#endif

#include "GraspVolumeMesh.h"

USE_PRJ_NAMESPACE

//convexification
ObjMeshD GraspVolumeMesh::makeConvex(const ObjMeshD& in)
{
#ifdef CGAL_SUPPORT
  std::vector<Point_3> points;
  for(sizeType i=0; i<(sizeType)in.getV().size(); i++) {
    Point_3 pt(in.getV(i)[0],in.getV(i)[1],in.getV(i)[2]);
    points.push_back(pt);
  }
  Polyhedron_3 poly;
  CGAL::convex_hull_3(points.begin(),points.end(),poly);
  {
    std::ofstream os("tmp.off");
    os << poly;
  }
  ObjMeshD out;
  std::ifstream is("tmp.off");
  readOFF(is,out);
  return out;
#else
  FUNCTION_NOT_IMPLEMENTED
  return ObjMeshD();
#endif
}
