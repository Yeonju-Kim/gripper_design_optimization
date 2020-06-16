#ifdef CGAL_SUPPORT
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/Polyhedral_mesh_domain_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/refine_mesh_3.h>
#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/polygon_mesh_processing.h>
#include <CGAL/IO/print_wavefront.h>
// Domain
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Polyhedron_3<K> Polyhedron;
typedef CGAL::Polyhedral_mesh_domain_3<Polyhedron,K> Mesh_domain;
#ifdef CGAL_CONCURRENT_MESH_3
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif
// Triangulation
typedef CGAL::Mesh_triangulation_3<Mesh_domain,K,Concurrency_tag>::type Tr;
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;
// Criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;
// To avoid verbose function and named parameters call
using namespace CGAL::parameters;
#endif
#include "GraspVolumeMesh.h"
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include "Utils.h"

USE_PRJ_NAMESPACE

//conversion
void GraspVolumeMesh::meshToABQ(std::istream& is,std::ostream& os,bool CGAL)
{
  Vec4i tets;
  Vec3d pos;
  sizeType nr,one;
  std::string line;

  while(CGAL) {
    getline(is,line);
    if(beginsWith(line,"Vertices"))
      break;
    if(!is.good() || is.eof())
      return;
  }
  getline(is,line);
  std::istringstream(line) >> nr;
  os << "*NODE" << std::endl;
  for(sizeType i=0; i<nr; i++) {
    getline(is,line);
    std::istringstream iss(line);
    iss >> pos[0] >> pos[1] >> pos[2];
    os << (i+1) << "," << pos[0] << "," << pos[1] << "," << pos[2] << std::endl;
  }

  os << "*ELEMENT, type=C3D4, ELSET=PART1" << std::endl;
  while(CGAL) {
    getline(is,line);
    if(beginsWith(line,"Tetrahedra"))
      break;
    if(!is.good() || is.eof())
      return;
  }
  getline(is,line);
  std::istringstream(line) >> nr;
  for(sizeType i=0; i<nr; i++) {
    getline(is,line);
    std::istringstream iss(line);
    if(CGAL)
      iss >> tets[0] >> tets[1] >> tets[2] >> tets[3];
    else iss >> one >> tets[0] >> tets[1] >> tets[2] >> tets[3];
    os << (i+1) << "," <<
       tets[0] << "," << tets[1] << "," <<
       tets[2] << "," << tets[3] << std::endl;
  }

  os << "*ELSET,ELSET=EALL,GENERATE" << std::endl;
  os << "1," << nr << std::endl;
}
void GraspVolumeMesh::meshToABQ(const std::string& is,const std::string& os,bool CGAL)
{
  std::ifstream inf(is);
  std::ofstream outf(os);
  meshToABQ(inf,outf,CGAL);
}
void GraspVolumeMesh::objToOFF(const ObjMeshD& mesh,std::ostream& os)
{
  os << "OFF" << std::endl;
  os << mesh.getV().size() << " " << mesh.getI().size() << " 0" << std::endl;
  for(sizeType i=0; i<(sizeType)mesh.getV().size(); i++) {
    const Vec3d& V=mesh.getV()[i];
    os << V[0] << " " << V[1] << " " << V[2] << std::endl;
  }
  for(sizeType i=0; i<(sizeType)mesh.getI().size(); i++) {
    const Vec3i& I=mesh.getI()[i];
    os << "3 " << I[0] << " " << I[1] << " " << I[2] << std::endl;
  }
}
void GraspVolumeMesh::objToOFF(const ObjMeshD& mesh,const std::string& os)
{
  std::ofstream outf(os);
  objToOFF(mesh,outf);
}
void GraspVolumeMesh::OFFToMesh(std::istream& is,std::ostream& os,scalar sizeF)
{
#ifdef CGAL_SUPPORT
  Polyhedron polyhedron;
  is >> polyhedron;
  Mesh_domain domain(polyhedron);
  //generate
  Mesh_criteria criteria(facet_angle=30.0f,
                         facet_size=sizeF,
                         facet_distance=0.025f,
                         cell_radius_edge_ratio=2.0f,
                         cell_size=sizeF);
  C3t3 c3t3=CGAL::make_mesh_3<C3t3>(domain,criteria,no_perturb(),no_exude());
  CGAL::refine_mesh_3(c3t3,domain,criteria,lloyd());
  c3t3.output_to_medit(os);
#else
  FUNCTION_NOT_IMPLEMENTED
#endif
}
void GraspVolumeMesh::OFFToMesh(const std::string& is,const std::string& os,scalar sizeF)
{
  std::ifstream inf(is);
  std::ofstream outf(os);
  OFFToMesh(inf,outf,sizeF);
}
void GraspVolumeMesh::makeBoxes(ObjMeshD& mesh,std::vector<Vec3i,Eigen::aligned_allocator<Vec3i>>& css,scalarD coef)
{
  std::unordered_set<Vec3i,Hash> cssSet;
  std::unordered_map<Vec3i,sizeType,Hash> vssMap;
  cssSet.insert(css.begin(),css.end());

  //vertices
  for(std::unordered_set<Vec3i,Hash>::const_iterator
      beg=cssSet.begin(),end=cssSet.end(); beg!=end; beg++)
  {
    const Vec3i& id=*beg;
    for(sizeType x=0; x<=1; x++)
      for(sizeType y=0; y<=1; y++)
        for(sizeType z=0; z<=1; z++)
          if(vssMap.find(id+Vec3i(x,y,z))==vssMap.end()) {
            sizeType off=(sizeType)vssMap.size();
            vssMap[id+Vec3i(x,y,z)]=off;
          }
  }
  mesh.getV().resize(vssMap.size());
  for(std::unordered_map<Vec3i,sizeType,Hash>::const_iterator
      beg=vssMap.begin(),end=vssMap.end(); beg!=end; beg++)
    mesh.getV()[beg->second]=(beg->first.cast<scalarD>()-Vec3d::Constant(0.5f))*coef;

  //indices
  mesh.getI().clear();
  for(std::unordered_set<Vec3i,Hash>::const_iterator
      beg=cssSet.begin(),end=cssSet.end(); beg!=end; beg++)
  {
    const Vec3i& id=*beg;
    for(sizeType dim=0; dim<3; dim++)
      for(sizeType dir=-1; dir<=1; dir+=2) {
        if(cssSet.find(id+Vec3i::Unit(dim)*dir)!=cssSet.end())
          continue;
        Vec3i base=id+Vec3i::Unit(dim)*(dir+1)/2;
        mesh.getI().push_back(Vec3i(vssMap[base],
                                    vssMap[base+Vec3i::Unit((dim+1)%3)],
                                    vssMap[base+Vec3i::Unit((dim+1)%3)+Vec3i::Unit((dim+2)%3)]));
        mesh.getI().push_back(Vec3i(vssMap[base],
                                    vssMap[base+Vec3i::Unit((dim+1)%3)+Vec3i::Unit((dim+2)%3)],
                                    vssMap[base+Vec3i::Unit((dim+2)%3)]));
      }
  }
  mesh.makeUnique();
  mesh.smooth();
  mesh.makeUniform();
  mesh.smooth();
}
void GraspVolumeMesh::readOFF(std::istream& is,ObjMeshD& mesh)
{
#ifdef CGAL_SUPPORT
  Polyhedron poly;
  is >> poly;
  {
    std::ofstream os("tmp.obj");
    print_polyhedron_wavefront(os,poly);
  }
  std::ifstream isObj("tmp.obj");
  mesh.read(isObj,false,false);
#else
  FUNCTION_NOT_IMPLEMENTED
#endif
}
