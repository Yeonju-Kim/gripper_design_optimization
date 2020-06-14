#ifndef GRASP_MESH_H
#define GRASP_MESH_H

#include "CommonFile/ParallelPoissonDiskSampling.h"
#include "Objective.h"
#include "CommonFile/ObjMesh.h"
#include "EigenInterface.h"
//#include <boost/unordered_set.hpp>

PRJ_BEGIN

class StaticGeom;
class DeformableBody;
class GraspMesh : public SerializableBase
{
public:
  typedef std::vector<sizeType> IDVEC;
  //typedef boost::unordered_set<sizeType> IDSET;
  typedef std::set<sizeType> IDSET;
  typedef Objective<scalarD>::SMat SMat;
  typedef Objective<scalarD>::STrip STrip;
  typedef Objective<scalarD>::STrips STrips;
  typedef std::vector<Cold,Eigen::aligned_allocator<Cold>> Vss;
  typedef std::vector<Vec3d,Eigen::aligned_allocator<Vec3d>> Pts;
  typedef std::vector<Vec6d,Eigen::aligned_allocator<Vec6d>> Dirs;
  typedef std::vector<Matd,Eigen::aligned_allocator<Matd>> Mats;
  GraspMesh();
  GraspMesh(scalarD theta,const Pts& contacts,const Pts& inNormals);
  GraspMesh(const ObjMeshD& mesh,scalarD theta,const Dirs* dss,sizeType subsampleTo);
  GraspMesh(std::shared_ptr<StaticGeom> mesh,scalarD theta,const Dirs* dss,sizeType subsampleTo);
  GraspMesh(const std::string& path,scalarD theta,const Dirs* dss,sizeType subsampleTo);
  std::string type() const override;
  bool read(std::istream& is,IOData* dat) override;
  bool write(std::ostream& os,IOData* dat) const override;
  std::shared_ptr<SerializableBase> copy() const override;
  //processing
  void readContactPoints(const std::string& path);
  void subSampleTriangleLevel(IDVEC& ids,Pts& contacts,Pts& inNormals,sizeType subsampleTo) const;
  void subSamplePixelLevel(IDVEC& ids,Pts& contacts,Pts& inNormals,sizeType subsampleTo) const;
  void generateSigmaCoefFEM(scalar sizeF=1,scalar poisson=0.45f,bool writeVTK=true,const std::string& dir=".");
  const DeformableBody& body() const;
  SMat buildFEMKMat() const;
  //debug
  void debugRandomSigma(const std::string& path,const IDSET& ids) const;
  void debugMatrices(const Mat3Xd& coefAll,const Mat3Xd& coefTorqueAll) const;
  void debugNullSpace() const;
  //getter
  IDSET allIds() const;
  IDSET allIdsDss() const;
  IDSET randomIds(sizeType nr) const;
  void getH(std::vector<scalarD>& H,const Cold& F,const IDSET& ids) const;
  std::vector<scalarD> getHRef(const Cold& F,const IDSET& ids) const;
  std::vector<scalarD> getH(const Cold& F,const IDSET& ids) const;
  Cold getDisplacement(const Cold& F,const IDSET& ids) const;
  sizeType collCellId(sizeType id) const;
  const Vec3d& inNormal(sizeType id) const;
  const Vec3d tangent1(sizeType id) const;
  const Vec3d tangent2(sizeType id) const;
  Eigen::Block<const Matd,6,3> getG(sizeType id) const;
  scalar avgEdgeLength() const;
  const ObjMeshD& mesh() const;
  const Pts& contacts() const;
  scalarD theta() const;
  sizeType nrSigma() const;
  Matd sigmaFromF(sizeType id,const IDSET& ids) const;
  //io
  Cold getNormalF(const IDSET& ids) const;
  void writeNormalHVTK(const std::string& path,const IDSET& ids) const;
  void writeNormalDisplacementVTK(const std::string& path,const IDSET& ids,scalar coef=1.0f) const;
  void writeDisplacementVTK(const std::string& path,const Cold& F,const IDSET& ids,scalar coef=1.0f) const;
  void writeTestPtsVTK(const std::string& path,scalar coef=0.25f,sizeType slice=6) const;
  void writeContactPtsVTK(const std::string& path,const IDSET* ids=NULL,scalar coef=0.25f,sizeType slice=6) const;
  void writeFrictionConesPtsVTK(const std::string& path,const IDSET* ids=NULL,scalar coef=10.0f,sizeType slice=32) const;
protected:
  void subSampleInner(ParallelPoissonDiskSampling& sampler,ParticleSetN pSet,sizeType subsampleTo) const;
  void detectContactIds(const ParticleSetN& pSet,IDVEC& ids,Pts& contacts,Pts& inNormals) const;
  Vec3d reset(std::shared_ptr<StaticGeom> mesh,scalarD theta);
  Vec3d reset(const ObjMeshD& mesh,scalarD theta,bool keepOriginalTopology=false);
  void generateContactsTriangleLevel(sizeType subsampleTo);
  void generateContactsTriangleLevel(const Dirs& dss);
  void generateContactsPixelLevel(sizeType subsampleTo);
  void generateContactsPixelLevel(const Dirs& dss);
  void assembleContacts();
  //data
  scalarD _theta;
  ObjMeshD _mesh;
  Pts _testPts;
  IDVEC _contactIds;
  Pts _contacts,_inNormals;
  Matd _wFromF,_gFromW;
  Mats _sigmaFromF;
  Dirs _dss;
  //FEM
  std::shared_ptr<LinearSolverInterface> _sol;
  std::shared_ptr<DeformableBody> _body;
  SMat _bodyEnergyCoefF;
  //collCellId, not written to file for backward consistency
  std::vector<sizeType> _collCellId;
};

PRJ_END

#endif
