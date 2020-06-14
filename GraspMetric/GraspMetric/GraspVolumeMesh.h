#ifndef GRASP_VOLUME_MESH
#define GRASP_VOLUME_MESH

#include "CommonFile/ObjMesh.h"

PRJ_BEGIN

class StaticGeom;
class CollisionHandler;
class GraspVolumeMesh : public SerializableBase
{
public:
  //conversion
  static void meshToABQ(std::istream& is,std::ostream& os,bool CGAL);
  static void meshToABQ(const std::string& is,const std::string& os,bool CGAL);
  static void objToOFF(const ObjMeshD& mesh,std::ostream& os);
  static void objToOFF(const ObjMeshD& mesh,const std::string& os);
  static void OFFToMesh(std::istream& is,std::ostream& os,scalar sizeF);
  static void OFFToMesh(const std::string& is,const std::string& os,scalar sizeF);
  static void makeBoxes(ObjMeshD& mesh,std::vector<Vec3i,Eigen::aligned_allocator<Vec3i>>& css,scalarD coef);
  static void readOFF(std::istream& is,ObjMeshD& mesh);
  static ObjMeshD makeConvex(const ObjMeshD& in);
};

PRJ_END

#endif
