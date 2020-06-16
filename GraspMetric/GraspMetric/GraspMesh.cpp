#include "Utils.h"
#include "GraspMesh.h"
#include "SparseUtils.h"
#include "RotationUtil.h"
#include "RigidBodyMass.h"
#include "DeformableBody.h"
#include "DebugGradient.h"
#include "GraspVolumeMesh.h"
#include "CommonFile/IO.h"
#include "CommonFile/Timing.h"
#include "CommonFile/MakeMesh.h"
#include "CommonFile/CollisionDetection.h"
#include "CommonFile/geom/StaticGeom.h"
#include <Eigen/Eigen>

USE_PRJ_NAMESPACE

#define SOLVER_REP_INTERVAL 1
void bodyForceCoef(const Vec3d& av,const Vec3d& bv,const Vec3d& cv,const Vec3d& dv,scalarD V,Mat12d& VI)
{
  //input
  //scalarD V;
  //Vec3 av;
  //Vec3 bv;
  //Vec3 cv;
  //Vec3 dv;

  //temp
  scalarD tt1;
  scalarD tt2;
  scalarD tt3;
  scalarD tt4;
  scalarD tt5;
  scalarD tt6;
  scalarD tt7;
  scalarD tt8;
  scalarD tt9;
  scalarD tt10;
  scalarD tt11;
  scalarD tt12;
  scalarD tt13;

  tt1=V/4.0;
  tt2=((2*av[0]+bv[0]+cv[0]+dv[0])*V)/20.0;
  tt3=((2*av[1]+bv[1]+cv[1]+dv[1])*V)/20.0;
  tt4=((2*av[2]+bv[2]+cv[2]+dv[2])*V)/20.0;
  tt5=((av[0]+2*bv[0]+cv[0]+dv[0])*V)/20.0;
  tt6=((av[1]+2*bv[1]+cv[1]+dv[1])*V)/20.0;
  tt7=((av[2]+2*bv[2]+cv[2]+dv[2])*V)/20.0;
  tt8=((av[0]+bv[0]+2*cv[0]+dv[0])*V)/20.0;
  tt9=((av[1]+bv[1]+2*cv[1]+dv[1])*V)/20.0;
  tt10=((av[2]+bv[2]+2*cv[2]+dv[2])*V)/20.0;
  tt11=((av[0]+bv[0]+cv[0]+2*dv[0])*V)/20.0;
  tt12=((av[1]+bv[1]+cv[1]+2*dv[1])*V)/20.0;
  tt13=((av[2]+bv[2]+cv[2]+2*dv[2])*V)/20.0;
  VI(0,0)=tt1;
  VI(0,1)=0;
  VI(0,2)=0;
  VI(0,3)=tt2;
  VI(0,4)=0;
  VI(0,5)=0;
  VI(0,6)=tt3;
  VI(0,7)=0;
  VI(0,8)=0;
  VI(0,9)=tt4;
  VI(0,10)=0;
  VI(0,11)=0;
  VI(1,0)=0;
  VI(1,1)=tt1;
  VI(1,2)=0;
  VI(1,3)=0;
  VI(1,4)=tt2;
  VI(1,5)=0;
  VI(1,6)=0;
  VI(1,7)=tt3;
  VI(1,8)=0;
  VI(1,9)=0;
  VI(1,10)=tt4;
  VI(1,11)=0;
  VI(2,0)=0;
  VI(2,1)=0;
  VI(2,2)=tt1;
  VI(2,3)=0;
  VI(2,4)=0;
  VI(2,5)=tt2;
  VI(2,6)=0;
  VI(2,7)=0;
  VI(2,8)=tt3;
  VI(2,9)=0;
  VI(2,10)=0;
  VI(2,11)=tt4;
  VI(3,0)=tt1;
  VI(3,1)=0;
  VI(3,2)=0;
  VI(3,3)=tt5;
  VI(3,4)=0;
  VI(3,5)=0;
  VI(3,6)=tt6;
  VI(3,7)=0;
  VI(3,8)=0;
  VI(3,9)=tt7;
  VI(3,10)=0;
  VI(3,11)=0;
  VI(4,0)=0;
  VI(4,1)=tt1;
  VI(4,2)=0;
  VI(4,3)=0;
  VI(4,4)=tt5;
  VI(4,5)=0;
  VI(4,6)=0;
  VI(4,7)=tt6;
  VI(4,8)=0;
  VI(4,9)=0;
  VI(4,10)=tt7;
  VI(4,11)=0;
  VI(5,0)=0;
  VI(5,1)=0;
  VI(5,2)=tt1;
  VI(5,3)=0;
  VI(5,4)=0;
  VI(5,5)=tt5;
  VI(5,6)=0;
  VI(5,7)=0;
  VI(5,8)=tt6;
  VI(5,9)=0;
  VI(5,10)=0;
  VI(5,11)=tt7;
  VI(6,0)=tt1;
  VI(6,1)=0;
  VI(6,2)=0;
  VI(6,3)=tt8;
  VI(6,4)=0;
  VI(6,5)=0;
  VI(6,6)=tt9;
  VI(6,7)=0;
  VI(6,8)=0;
  VI(6,9)=tt10;
  VI(6,10)=0;
  VI(6,11)=0;
  VI(7,0)=0;
  VI(7,1)=tt1;
  VI(7,2)=0;
  VI(7,3)=0;
  VI(7,4)=tt8;
  VI(7,5)=0;
  VI(7,6)=0;
  VI(7,7)=tt9;
  VI(7,8)=0;
  VI(7,9)=0;
  VI(7,10)=tt10;
  VI(7,11)=0;
  VI(8,0)=0;
  VI(8,1)=0;
  VI(8,2)=tt1;
  VI(8,3)=0;
  VI(8,4)=0;
  VI(8,5)=tt8;
  VI(8,6)=0;
  VI(8,7)=0;
  VI(8,8)=tt9;
  VI(8,9)=0;
  VI(8,10)=0;
  VI(8,11)=tt10;
  VI(9,0)=tt1;
  VI(9,1)=0;
  VI(9,2)=0;
  VI(9,3)=tt11;
  VI(9,4)=0;
  VI(9,5)=0;
  VI(9,6)=tt12;
  VI(9,7)=0;
  VI(9,8)=0;
  VI(9,9)=tt13;
  VI(9,10)=0;
  VI(9,11)=0;
  VI(10,0)=0;
  VI(10,1)=tt1;
  VI(10,2)=0;
  VI(10,3)=0;
  VI(10,4)=tt11;
  VI(10,5)=0;
  VI(10,6)=0;
  VI(10,7)=tt12;
  VI(10,8)=0;
  VI(10,9)=0;
  VI(10,10)=tt13;
  VI(10,11)=0;
  VI(11,0)=0;
  VI(11,1)=0;
  VI(11,2)=tt1;
  VI(11,3)=0;
  VI(11,4)=0;
  VI(11,5)=tt11;
  VI(11,6)=0;
  VI(11,7)=0;
  VI(11,8)=tt12;
  VI(11,9)=0;
  VI(11,10)=0;
  VI(11,11)=tt13;
}
void bodyForceTorqueCoef(const Vec3d& av,const Vec3d& bv,const Vec3d& cv,const Vec3d& dv,scalarD V,Mat3Xd& VI)
{
  //input
  //scalarD V;
  //Vec3 av;
  //Vec3 bv;
  //Vec3 cv;
  //Vec3 dv;

  //temp
  scalarD tt1;
  scalarD tt2;
  scalarD tt3;
  scalarD tt4;
  scalarD tt5;
  scalarD tt6;
  scalarD tt7;
  scalarD tt8;
  scalarD tt9;
  scalarD tt10;
  scalarD tt11;
  scalarD tt12;
  scalarD tt13;
  scalarD tt14;
  scalarD tt15;
  scalarD tt16;
  scalarD tt17;
  scalarD tt18;
  scalarD tt19;

  tt1=dv[2]+cv[2]+bv[2]+av[2];
  tt2=dv[1]+cv[1]+bv[1]+av[1];
  tt3=dv[0]+cv[0]+bv[0]+2*av[0];
  tt4=dv[0]+cv[0]+2*bv[0]+av[0];
  tt5=dv[0]+2*cv[0]+bv[0]+av[0];
  tt6=2*dv[0]+cv[0]+bv[0]+av[0];
  tt7=tt6*dv[2]+tt5*cv[2]+tt4*bv[2]+tt3*av[2];
  tt8=-(tt7*V)/20.0;
  tt9=tt6*dv[1]+tt5*cv[1]+tt4*bv[1]+tt3*av[1];
  tt10=(tt9*V)/20.0;
  tt11=(av[1]+bv[1]+cv[1]+2*dv[1])*dv[2]+(av[1]+bv[1]+2*cv[1]+dv[1])*cv[2]+(av[1]+2*bv[1]+cv[1]+dv[1])*bv[2]+(2*av[1]+bv[1]+cv[1]+dv[1])*av[2];
  tt12=-(tt11*V)/20.0;
  tt13=pow(dv[1],2)+(av[1]+bv[1]+cv[1])*dv[1]+pow(cv[1],2)+(av[1]+bv[1])*cv[1]+pow(bv[1],2)+av[1]*bv[1]+pow(av[1],2);
  tt14=pow(dv[2],2)+(av[2]+bv[2]+cv[2])*dv[2]+pow(cv[2],2)+(av[2]+bv[2])*cv[2]+pow(bv[2],2)+av[2]*bv[2]+pow(av[2],2);
  tt15=(tt11*V)/20.0;
  tt16=dv[0]+cv[0]+bv[0]+av[0];
  tt17=(tt7*V)/20.0;
  tt18=pow(dv[0],2)+(av[0]+bv[0]+cv[0])*dv[0]+pow(cv[0],2)+(av[0]+bv[0])*cv[0]+pow(bv[0],2)+av[0]*bv[0]+pow(av[0],2);
  tt19=-(tt9*V)/20.0;
  VI(0,0)=0;
  VI(0,1)=-(tt1*V)/4.0;
  VI(0,2)=(tt2*V)/4.0;
  VI(0,3)=0;
  VI(0,4)=tt8;
  VI(0,5)=tt10;
  VI(0,6)=0;
  VI(0,7)=tt12;
  VI(0,8)=(tt13*V)/10.0;
  VI(0,9)=0;
  VI(0,10)=-(tt14*V)/10.0;
  VI(0,11)=tt15;
  VI(1,0)=(tt1*V)/4.0;
  VI(1,1)=0;
  VI(1,2)=-(tt16*V)/4.0;
  VI(1,3)=tt17;
  VI(1,4)=0;
  VI(1,5)=-(tt18*V)/10.0;
  VI(1,6)=tt15;
  VI(1,7)=0;
  VI(1,8)=tt19;
  VI(1,9)=(tt14*V)/10.0;
  VI(1,10)=0;
  VI(1,11)=tt8;
  VI(2,0)=-(tt2*V)/4.0;
  VI(2,1)=(tt16*V)/4.0;
  VI(2,2)=0;
  VI(2,3)=tt19;
  VI(2,4)=(tt18*V)/10.0;
  VI(2,5)=0;
  VI(2,6)=-(tt13*V)/10.0;
  VI(2,7)=tt10;
  VI(2,8)=0;
  VI(2,9)=tt12;
  VI(2,10)=tt17;
  VI(2,11)=0;
}
//GraspMesh
GraspMesh::GraspMesh() {}
GraspMesh::GraspMesh(scalarD theta,const Pts& contacts,const Pts& inNormals):_contacts(contacts),_inNormals(inNormals)
{
  _theta=theta;
  _dss.clear();
  _contactIds.clear();
  assembleContacts();
}
GraspMesh::GraspMesh(const ObjMeshD& mesh,scalarD theta,const Dirs* dss,sizeType subsampleTo)
{
  reset(mesh,theta);
  if(dss)
    generateContactsPixelLevel(*dss);
  else if(subsampleTo>=0)
    generateContactsPixelLevel(subsampleTo);
  else  {
    ASSERT_MSG(false,"Define you sample method!")
  }
}
GraspMesh::GraspMesh(std::shared_ptr<StaticGeom> mesh,scalarD theta,const Dirs* dss,sizeType subsampleTo)
{
  reset(mesh,theta);
  if(dss)
    generateContactsPixelLevel(*dss);
  else if(subsampleTo>=0)
    generateContactsPixelLevel(subsampleTo);
  else {
    ASSERT_MSG(false,"Define your sample method!")
  }
}
GraspMesh::GraspMesh(const std::string& path,scalarD theta,const Dirs* dss,sizeType subsampleTo)
{
  ASSERT_MSGV(exists(path+"/mesh.obj") || exists(path+"/mesh.off") || exists(path+"/mesh.geom"),
              "Cannot find file: %s or %s or %s!",(path+"/mesh.obj").c_str(),(path+"/mesh.off").c_str(),(path+"/mesh.geom").c_str())
  if(exists(path+"/mesh.obj")) {
    std::ifstream is(path+"/mesh.obj");
    ASSERT_MSGV(_mesh.read(is,false,false),"Cannot read mesh from: %s",path.c_str());
    _mesh.smooth();
    *this=GraspMesh(_mesh,theta,dss,subsampleTo);
  } else if(exists(path+"/mesh.off")) {
    std::ifstream is(path+"/mesh.off");
    GraspVolumeMesh::readOFF(is,_mesh);
    _mesh.smooth();
    *this=GraspMesh(_mesh,theta,dss,subsampleTo);
  } else if(exists(path+"/mesh.geom")) {
    std::shared_ptr<StaticGeom> geom(new StaticGeom);
    geom->Serializable::read(path+"/mesh.geom");
    *this=GraspMesh(geom,theta,dss,subsampleTo);
  } else {
    ASSERT_MSGV(false,"Cannot find mesh files in folder: %s!",path.c_str())
  }
}
std::string GraspMesh::type() const
{
  return typeid(GraspMesh).name();
}
bool GraspMesh::read(std::istream& is,IOData* dat)
{
  registerType<DeformableBody>(dat);
  readBinaryData(_theta,is);
  _mesh.readBinary(is);
  readBinaryData(_testPts,is);
  readBinaryData(_contactIds,is);
  readBinaryData(_contacts,is);
  readBinaryData(_inNormals,is);
  readBinaryData(_wFromF,is);
  readBinaryData(_gFromW,is);
  readBinaryData(_sigmaFromF,is);
  readBinaryData(_dss,is);
  //FEM
  _sol=NULL;
  readBinaryData(_body,is,dat);
  readBinaryData(_bodyEnergyCoefF,is);
  //just clear collCellId and build during runtime
  _collCellId.clear();
  return is.good();
}
bool GraspMesh::write(std::ostream& os,IOData* dat) const
{
  registerType<DeformableBody>(dat);
  writeBinaryData(_theta,os);
  _mesh.writeBinary(os);
  writeBinaryData(_testPts,os);
  writeBinaryData(_contactIds,os);
  writeBinaryData(_contacts,os);
  writeBinaryData(_inNormals,os);
  writeBinaryData(_wFromF,os);
  writeBinaryData(_gFromW,os);
  writeBinaryData(_sigmaFromF,os);
  writeBinaryData(_dss,os);
  //FEM
  writeBinaryData(_body,os,dat);
  writeBinaryData(_bodyEnergyCoefF,os);
  return os.good();
}
std::shared_ptr<SerializableBase> GraspMesh::copy() const
{
  std::shared_ptr<GraspMesh> mesh(new GraspMesh);
  *mesh=*this;
  return mesh;
}
//processing
void GraspMesh::readContactPoints(const std::string& path)
{
  INFO("Reading Contacts")
  ASSERT_MSGV(exists(path+"/contact_points.txt"),"Cannot find file: %s!",
              (path+"/contact_points.txt").c_str())
  _contactIds.clear();
  _contacts.clear();
  _inNormals.clear();
  ParticleSetN pSet;
  std::string line;
  std::ifstream is(path+"/contact_points.txt");
  while(std::getline(is,line)) {
    //read contact
    ParticleN<scalar> p;
    std::istringstream iss(line);
    iss >> p._pos[0] >> p._pos[1] >> p._pos[2];
    pSet.addParticle(p);
  }
  detectContactIds(pSet,_contactIds,_contacts,_inNormals);
  assembleContacts();
}
void GraspMesh::subSampleTriangleLevel(IDVEC& ids,Pts& contacts,Pts& inNormals,sizeType subsampleTo) const
{
  ids.clear();
  contacts.clear();
  inNormals.clear();
  for(sizeType i=0; i<(sizeType)_mesh.getI().size(); i++) {
    Vec3i I=_mesh.getI(i);
    ids.push_back(i);
    contacts.push_back((_mesh.getV(I[0])+_mesh.getV(I[1])+_mesh.getV(I[2])).cast<scalarD>()/3);
    inNormals.push_back(-_mesh.getTN(i).cast<scalarD>());
  }
  //fill up raw sample
  ParallelPoissonDiskSampling sampler(3);
  ParticleSetN pSet;
  pSet.resize(contacts.size());
  for(sizeType i=0; i<(sizeType)contacts.size(); i++) {
    pSet[i]._pos=contacts[i].cast<scalar>();
    pSet[i]._normal=-inNormals[i].cast<scalar>();
  }
  if(subsampleTo<=0)
    sampler.getPSet()=pSet;
  else subSampleInner(sampler,pSet,subsampleTo);
  //detect contact ids
  detectContactIds(sampler.getPSet(),ids,contacts,inNormals);
}
void GraspMesh::subSamplePixelLevel(IDVEC& ids,Pts& contacts,Pts& inNormals,sizeType subsampleTo) const
{
#define DENSITY 100
  if(subsampleTo<=0) {
    subSampleTriangleLevel(ids,contacts,inNormals,subsampleTo);
    return;
  }
  ObjMesh meshS;
  scalar sumArea=0;
  scalar avgEdgeLen=avgEdgeLength();
  scalar area=avgEdgeLen*avgEdgeLen*M_PI;
  for(sizeType i=0; i<(sizeType)_mesh.getI().size(); i++)
    sumArea+=std::abs<scalar>(_mesh.getArea(i));
  ParallelPoissonDiskSampling sampler(3);
  sampler.setRadius(avgEdgeLen);
  sampler.setDensity(DENSITY*subsampleTo*area/sumArea);
  _mesh.cast<scalar>(meshS);
  //fill up raw sample
  sampler.generateRawSample(meshS);
  subSampleInner(sampler,sampler.getPSet(),subsampleTo);
  //detect contact ids
  detectContactIds(sampler.getPSet(),ids,contacts,inNormals);
#undef DENSITY
}
void GraspMesh::generateSigmaCoefFEM(scalar sizeF,scalar poisson,bool writeVTK,const std::string& dir)
{
  disableTiming();
  //generate mesh
  if(!exists(dir+"/tmp.mesh")) {
    GraspVolumeMesh::objToOFF(_mesh,dir+"/tmp.OFF");
    INFOV("Begin Generating Volume Mesh, size=%lf!",avgEdgeLength()*sizeF)
    GraspVolumeMesh::OFFToMesh(dir+"/tmp.OFF",dir+"/tmp.mesh",avgEdgeLength()*sizeF);
    INFO("End Generating Volume Mesh!")
  }
  //read tmp.ABQ
  std::ifstream t(dir+"/tmp.mesh");
  std::string str((std::istreambuf_iterator<char>(t)),std::istreambuf_iterator<char>());
  //build ptree
  tinyxml2::XMLDocument pt;
  pt.InsertEndChild(pt.NewElement("root"));
  put<std::string>(*(pt.RootElement()),"deformableBodyMesh",str);
  put<scalarD>(*(pt.RootElement()),"poisson",poisson);
  put<scalarD>(*(pt.RootElement()),"young",1);
  //build deformable body
  _body.reset(new DeformableBody);
  _body->assemble(*(pt.RootElement()));

  //make surface mesh consistent
  Coli tetss;
  _mesh.getV().resize(_body->U0().size()/3);
  for(sizeType i=0; i<(sizeType)_mesh.getV().size(); i++)
    _mesh.getV(i)=_body->U0().segment<3>(i*3);
  Mat3Xi triss=_body->getMuRange(Vec2d(-ScalarUtil<scalarD>::scalar_max(),ScalarUtil<scalarD>::scalar_max()),&tetss);
  _mesh.getI().resize(triss.cols());
  for(sizeType i=0; i<triss.cols(); i++)
    _mesh.getI(i)=triss.col(i);
  _mesh.smooth();
  Vec3d ctr=reset(_mesh,_theta,true);
  if(!_dss.empty())
    generateContactsTriangleLevel(_dss);
  else generateContactsTriangleLevel((sizeType)_contacts.size());
  for(sizeType i=0; i<_body->U0().size(); i+=3)
    _body->U0().segment<3>(i)-=ctr;
  //test write
  if(writeVTK) {
    _body->writeVTK(_body->U0(),dir+"/tmp_vmesh.vtk");
    _mesh.writeVTK(dir+"/tmp_smesh.vtk",true);
  }

  STrips trips;
  Mat3Xd coefAll=Mat3Xd::Zero(3,12);
  Mat3Xd coefTorqueAll=Mat3Xd::Zero(3,12);
  //build internal force matrix: internal body force
  for(sizeType i=0; i<_body->tss().cols(); i++) {
    Mat12d coef=Mat12d::Zero(12,12);
    Mat3Xd coefTorque=Mat3Xd::Zero(3,12);
    //force
    bodyForceCoef(_body->U0().segment<3>(_body->tss()(0,i)*3),
                  _body->U0().segment<3>(_body->tss()(1,i)*3),
                  _body->U0().segment<3>(_body->tss()(2,i)*3),
                  _body->U0().segment<3>(_body->tss()(3,i)*3),
                  _body->vol()[i],coef);
    coefAll+=coef.block<3,12>(0,0);
    coefAll+=coef.block<3,12>(3,0);
    coefAll+=coef.block<3,12>(6,0);
    coefAll+=coef.block<3,12>(9,0);
    //torque
    bodyForceTorqueCoef(_body->U0().segment<3>(_body->tss()(0,i)*3),
                        _body->U0().segment<3>(_body->tss()(1,i)*3),
                        _body->U0().segment<3>(_body->tss()(2,i)*3),
                        _body->U0().segment<3>(_body->tss()(3,i)*3),
                        _body->vol()[i],coefTorque);
    coefTorqueAll+=coefTorque;
    //assemble
    addBlock(trips,_body->tss()(0,i)*3,0,coef.block<3,12>(0,0));
    addBlock(trips,_body->tss()(1,i)*3,0,coef.block<3,12>(3,0));
    addBlock(trips,_body->tss()(2,i)*3,0,coef.block<3,12>(6,0));
    addBlock(trips,_body->tss()(3,i)*3,0,coef.block<3,12>(9,0));
  }
  debugMatrices(coefAll,coefTorqueAll);

  //build external force matrix: external contact force
  SMat bodyEnergyCoef;
  for(sizeType i=0; i<(sizeType)_contacts.size(); i++) {
    const Vec3i& I=_mesh.getI(_contactIds[i]);
    for(sizeType d=0; d<3; d++)
      addBlockId(trips,I[d]*3,12+i*3,3,1/3.0f);
  }
  bodyEnergyCoef.resize(_body->U0().size(),12+_contacts.size()*3);
  bodyEnergyCoef.setFromTriplets(trips.begin(),trips.end());
  _bodyEnergyCoefF=bodyEnergyCoef.block(0,0,_body->U0().size(),12)*toSparse<scalarD,Matd>(_gFromW*_wFromF)+bodyEnergyCoef.block(0,12,_body->U0().size(),_contacts.size()*3);
  debugNullSpace();

  //assemble sigma matrix: prepare
  //factorize
  _sol=LinearSolverTraits<SMat>::getLUSolver();
  ASSERT_MSG(_sol->recompute(buildFEMKMat(),0,false),"Factorization failed!")
  //assemble sigma matrix: solve column-by-column to avoid memory leakage
  _sigmaFromF.assign(tetss.size(),Matd::Zero(9,_bodyEnergyCoefF.cols()));
  for(sizeType c=0; c<_bodyEnergyCoefF.cols(); c++) {
    Cold invKCoefF=_sol->solve(concat<Cold,Cold>(_bodyEnergyCoefF.col(c),Cold::Zero(6)));
    if(c%SOLVER_REP_INTERVAL==0) {
      INFOV("Assembling sigma column %d/%d",c+1,_bodyEnergyCoefF.cols())
    }
    Cold sigma=_body->sigma()*(_body->G()*invKCoefF.segment(0,_body->G().cols()));
    for(sizeType i=0; i<tetss.size(); i++)
      _sigmaFromF[i].col(c)=sigma.segment<9>(tetss[i]*9);
  }
  INFOV("Assembling sigma column %d/%d",_bodyEnergyCoefF.cols(),_bodyEnergyCoefF.cols())
}
const DeformableBody& GraspMesh::body() const
{
  ASSERT_MSG(_body,"DeformableBody has not been defined!")
  return *_body;
}
GraspMesh::SMat GraspMesh::buildFEMKMat() const
{
  Cold F;
  SMat K,KCons;
  _body->buildKM(_body->U0(),&F,&K,NULL);
  //factor out null space
  STrips trips;
  KCons.resize(6,K.cols());
  for(sizeType i=0; i<_body->U0().size(); i+=3) {
    Vec3d v=_body->U0().segment<3>(i);
    addBlockId(trips,0,i,3,1);
    addBlock(trips,3,i,cross<scalarD>(v));
  }
  KCons.setFromTriplets(trips.begin(),trips.end());
  INFOV("Nullspace of K: %f",(K*KCons.transpose()).toDense().cwiseAbs().maxCoeff())
  return buildKKT(K,KCons,0);
}
//debug
void GraspMesh::debugRandomSigma(const std::string& path,const IDSET& ids) const
{
  //generate F
  Cold F=Cold::Random((sizeType)ids.size()*3);
  //compute sigma
  std::vector<scalarD> H=getH(F,ids);
  _mesh.writeVTK(path,true,false,false,NULL,&H);
  //displacement
  std::string pathVTK=std::experimental::filesystem::v1::path(path).replace_extension(".vtk").string();
  writeDisplacementVTK(pathVTK,F,ids,1.0f);
}
void GraspMesh::debugMatrices(const Mat3Xd& coefAll,const Mat3Xd& coefTorqueAll) const
{
  DEFINE_NUMERIC_DELTA
  Cold F=Cold::Random(_wFromF.cols());
  Vec6d wF=_wFromF*F;
  Vec12d gW=_gFromW*wF;
  Vec6d wG=concat(coefAll*gW,coefTorqueAll*gW);
  DEBUG_GRADIENT("wG",wF.norm(),(wF-wG).norm())
}
void GraspMesh::debugNullSpace() const
{
  Mat3Xd F=Mat3Xd::Zero(3,_body->U0().size());
  Mat3Xd T=Mat3Xd::Zero(3,_body->U0().size());
  for(sizeType i=0; i<_body->U0().size(); i+=3) {
    F.block<3,3>(0,i).setIdentity();
    T.block<3,3>(0,i)=cross<scalarD>(_body->U0().segment<3>(i));
  }
  Mat3Xd Ff=F*_bodyEnergyCoefF;
  INFOV("Ff maxCoef: %f",Ff.cwiseAbs().maxCoeff())
  Mat3Xd Tf=T*_bodyEnergyCoefF;
  INFOV("Tf maxCoef: %f",Tf.cwiseAbs().maxCoeff())
}
//getter
GraspMesh::IDSET GraspMesh::allIds() const
{
  IDSET ids;
  for(sizeType i=0; i<(sizeType)_contactIds.size(); i++)
    ids.insert(i);
  return ids;
}
GraspMesh::IDSET GraspMesh::allIdsDss() const
{
  IDSET tMinIds;
  ASSERT(!_dss.empty())
  for(sizeType lid=0; lid<(sizeType)_dss.size(); lid++) {
    //find lineSeg
    scalarD len=_mesh.getBB().getExtent().norm();
    Vec3d d=_dss[lid].segment<3>(0);
    Vec3d d0=_dss[lid].segment<3>(3);
    Vec3d dN=d.normalized(),d0N=d0-d0.dot(dN)*dN;
    LineSegTpl<scalarD> l(-dN*len+d0N,dN*len+d0N);
    scalarD tMin=ScalarUtil<scalarD>::scalar_max(),t;
    sizeType tMinId=-1;
    //find min point
    for(sizeType i=0; i<(sizeType)_mesh.getI().size(); i++) {
      Vec3i I=_mesh.getI(i);
      TriangleTpl<scalarD> tri(_mesh.getV(I[0]).cast<scalarD>(),_mesh.getV(I[1]).cast<scalarD>(),_mesh.getV(I[2]).cast<scalarD>());
      if(tri.intersect(l,t,false,NULL) && t<tMin) {
        tMinId=i;
        tMin=t;
      }
    }
    //add min point
    tMinIds.insert(tMinId);
  }
  //return ids
  IDSET ids;
  for(sizeType i=0; i<(sizeType)_contactIds.size(); i++)
    if(tMinIds.find(_contactIds[i])!=tMinIds.end())
      ids.insert(i);
  return ids;
}
GraspMesh::IDSET GraspMesh::randomIds(sizeType nr) const
{
  std::vector<sizeType> ids;
  for(sizeType i=0; i<(sizeType)contacts().size(); i++)
    ids.push_back(i);
  std::random_shuffle(ids.begin(),ids.end());
  ASSERT(nr<=(sizeType)contacts().size())

  ids.resize(nr);
  IDSET idsSet(ids.begin(),ids.end());
  return idsSet;
}
void GraspMesh::getH(std::vector<scalarD>& H,const Cold& F,const IDSET& ids) const
{
  ASSERT_MSG(_sigmaFromF.size()==_mesh.getI().size(),"Sigma array not computed!")
  H.resize((sizeType)_sigmaFromF.size());
  Vec9d sigma;
  Vec3d sigmaEV;
  OMP_PARALLEL_FOR_I(OMP_PRI(sigma,sigmaEV))
  for(sizeType i=0; i<(sizeType)_sigmaFromF.size(); i++) {
    sizeType off=0;
    sigma.setZero();
    for(IDSET::const_iterator beg=ids.begin(),end=ids.end(); beg!=end; beg++,off+=3)
      sigma+=(_sigmaFromF[i].block<9,3>(0,*beg*3)*F.segment<3>(off));
    dsyevc3<scalarD>((scalarD(*)[3])sigma.data(),sigmaEV.data());
    H[i]=sigmaEV.cwiseAbs().maxCoeff();
  }
}
std::vector<scalarD> GraspMesh::getHRef(const Cold& F,const IDSET& ids) const
{
  ASSERT_MSG(_sigmaFromF.size()==_mesh.getI().size(),"Sigma array not computed!")
  std::vector<scalarD> H((sizeType)_sigmaFromF.size());
  for(sizeType i=0; i<(sizeType)_sigmaFromF.size(); i++) {
    sizeType off=0;
    Vec9d sigma=Vec9d::Zero();
    for(IDSET::const_iterator beg=ids.begin(),end=ids.end(); beg!=end; beg++,off+=3)
      sigma+=(_sigmaFromF[i].block<9,3>(0,*beg*3)*F.segment<3>(off));
    Eigen::SelfAdjointEigenSolver<Mat3d> eig(Eigen::Map<Mat3d>(sigma.data()));
    H[i]=eig.eigenvalues().cwiseAbs().maxCoeff();
  }
  return H;
}
std::vector<scalarD> GraspMesh::getH(const Cold& F,const IDSET& ids) const
{
  std::vector<scalarD> H;
  getH(H,F,ids);
  return H;
}
Cold GraspMesh::getDisplacement(const Cold& F,const IDSET& ids) const
{
  if(!_sol) {
    const_cast<std::shared_ptr<LinearSolverInterface>&>(_sol)=LinearSolverTraits<SMat>::getLUSolver();
    ASSERT_MSG(_sol->recompute(buildFEMKMat(),0,false),"Factorization failed!")
  }

  sizeType off=0;
  Cold f=Cold::Zero(_bodyEnergyCoefF.cols());
  for(IDSET::const_iterator beg=ids.begin(),end=ids.end(); beg!=end; beg++,off+=3)
    f.segment<3>(*beg*3)=F.segment<3>(off);

  Cold rhs=concat<Cold,Cold>(_bodyEnergyCoefF*f,Cold::Zero(6));
  return _sol->solve(rhs).col(0).segment(0,_body->U0().size());
}
sizeType GraspMesh::collCellId(sizeType id) const
{
  return _collCellId[id];
}
const Vec3d& GraspMesh::inNormal(sizeType id) const
{
  return _inNormals[id];
}
const Vec3d GraspMesh::tangent1(sizeType id) const
{
  sizeType minId;
  inNormal(id).cwiseAbs().minCoeff(&minId);
  return inNormal(id).cross(Vec3d::Unit(minId)).normalized();
}
const Vec3d GraspMesh::tangent2(sizeType id) const
{
  return inNormal(id).cross(tangent1(id));
}
Eigen::Block<const Matd,6,3> GraspMesh::getG(sizeType id) const
{
  return _wFromF.block<6,3>(0,id*3);
}
scalar GraspMesh::avgEdgeLength() const
{
  ObjMeshD::EdgeMap ess;
  _mesh.buildEdge(ess);
  scalarD avgEdgeLength=0,nrEdge=0;
  for(auto beg=ess._ess.begin(),end=ess._ess.end(); beg!=end; beg++) {
    avgEdgeLength+=(_mesh.getV(beg->first.first)-_mesh.getV(beg->first.second)).norm();
    nrEdge+=1;
  }
  return avgEdgeLength/nrEdge;
}
const ObjMeshD& GraspMesh::mesh() const
{
  return _mesh;
}
const GraspMesh::Pts& GraspMesh::contacts() const
{
  return _contacts;
}
scalarD GraspMesh::theta() const
{
  return _theta;
}
sizeType GraspMesh::nrSigma() const
{
  return (sizeType)_sigmaFromF.size();
}
Matd GraspMesh::sigmaFromF(sizeType id,const IDSET& ids) const
{
  sizeType off=0;
  Matd ret=Matd::Zero(9,(sizeType)ids.size()*3);
  for(IDSET::const_iterator beg=ids.begin(),end=ids.end(); beg!=end; beg++) {
    ret.block<9,3>(0,off)=_sigmaFromF[id].block<9,3>(0,*beg*3);
    off+=3;
  }
  return ret;
}
//io
Cold GraspMesh::getNormalF(const IDSET& ids) const
{
  Cold F=Cold::Zero((sizeType)ids.size()*3);
  Mat3Xd G=Mat3Xd::Zero(3,(sizeType)ids.size()*3);
  sizeType off=0;
  for(IDSET::const_iterator beg=ids.begin(),end=ids.end(); beg!=end; beg++,off+=3) {
    F.segment<3>(off)=inNormal(*beg);
    G.block<3,3>(0,off)=getG(*beg).block<3,3>(3,0);
  }
  F-=G.transpose()*(G*G.transpose()).inverse()*G*F;
  return F;
}
void GraspMesh::writeNormalHVTK(const std::string& path,const IDSET& ids) const
{
  std::vector<scalarD> H;
  getH(H,getNormalF(ids),ids);
  _mesh.writeVTK(path,true,false,false,NULL,&H);
}
void GraspMesh::writeNormalDisplacementVTK(const std::string& path,const IDSET& ids,scalar coef) const
{
  Cold disp=getDisplacement(getNormalF(ids),ids);
  disp*=avgEdgeLength()*coef/disp.cwiseAbs().maxCoeff();
  _body->writeVTK(_body->U0(),path,&disp);
}
void GraspMesh::writeDisplacementVTK(const std::string& path,const Cold& F,const IDSET& ids,scalar coef) const
{
  ObjMeshD mesh=_mesh;
  Cold disp=getDisplacement(F,ids);
  disp*=avgEdgeLength()*coef/disp.cwiseAbs().maxCoeff();
  for(sizeType i=0; i<disp.size(); i+=3)
    mesh.getV(i/3)+=disp.segment<3>(i);
  mesh.writeVTK(path,true);
}
void GraspMesh::writeTestPtsVTK(const std::string& path,scalar coef,sizeType slice) const
{
  ObjMesh s;
  ObjMesh pts;
  MakeMesh::makeSphere3D(s,avgEdgeLength()*coef,slice);
  for(sizeType i=0; i<(sizeType)_testPts.size(); i++) {
    ObjMesh sMoved=s;
    sMoved.getPos()=_testPts[i].cast<scalar>();
    sMoved.applyTrans();
    pts.addMesh(sMoved);
  }
  pts.writeVTK(path,true);
}
void GraspMesh::writeContactPtsVTK(const std::string& path,const IDSET* ids,scalar coef,sizeType slice) const
{
  ObjMesh s,pts;
  std::vector<scalar> cellIds;
  MakeMesh::makeSphere3D(s,avgEdgeLength()*coef,slice);
  for(sizeType i=0; i<(sizeType)_contacts.size(); i++)
    if(!ids || ids->find(i)!=ids->end()) {
      ObjMesh sMoved=s;
      sMoved.getPos()=_contacts[i].cast<scalar>();
      sMoved.applyTrans(Vec3::Zero());
      pts.addMesh(sMoved);
      //collCellIds
      ASSERT(_collCellId.size()==_contacts.size() || _collCellId.empty())
      if(_collCellId.size()==_contacts.size())
        for(sizeType j=0; j<(sizeType)sMoved.getI().size(); j++)
          cellIds.push_back(_collCellId[i]);
    }
  //cellIds
  if(cellIds.empty())
    pts.writeVTK(path,true);
  else pts.writeVTK(path,true,false,false,NULL,&cellIds);
}
void GraspMesh::writeFrictionConesPtsVTK(const std::string& path,const IDSET* ids,scalar coef,sizeType slice) const
{
  //create cone
  ObjMesh s;
  s.getV().push_back(Vec3::Zero());
  scalar avgEdgeLen=avgEdgeLength();
  for(sizeType i=0; i<slice; i++) {
    scalar angle=i*M_PI*2/slice;
    s.getV().push_back(Vec3(cos(angle)*_theta,sin(angle)*_theta,1)*avgEdgeLen*coef);
  }
  s.getV().push_back(Vec3::UnitZ()*avgEdgeLen*coef);
  for(sizeType i=0; i<slice; i++) {
    s.getI().push_back(Vec3i(0,1+((i+1)%slice),1+i));
    s.getI().push_back(Vec3i(1+((i+1)%slice),slice+1,1+i));
  }
  s.smooth();

  //move to contact position
  ObjMesh pts;
  for(sizeType i=0; i<(sizeType)_contacts.size(); i++)
    if(!ids || ids->find(i)!=ids->end()) {
      ObjMesh sMoved=s;
      sMoved.getPos()=_contacts[i].cast<scalar>();
      sMoved.getT()=ScalarUtil<scalar>::ScalarQuat::FromTwoVectors(Vec3::UnitZ(),-_inNormals[i].cast<scalar>()).toRotationMatrix();
      sMoved.applyTrans(Vec3::Zero());
      pts.addMesh(sMoved);
    }
  pts.writeVTK(path,true);
}
//helper
void GraspMesh::subSampleInner(ParallelPoissonDiskSampling& sampler,ParticleSetN pSet,sizeType subsampleTo) const
{
  scalar avgEdgeLen=avgEdgeLength();
  if(pSet.size()<=subsampleTo) {
    sampler.getPSet()=pSet;
    return;
  }
  //sample: find low
  scalar low=avgEdgeLen;
  do {
    low*=0.5f;
    sampler.getPSet()=pSet;
    sampler.setRadius(low);
    sampler.sample();
  } while(sampler.getPSet().size()<=subsampleTo);
  //sample: find high
  scalar high=avgEdgeLen;
  do {
    high+=avgEdgeLen;
    sampler.getPSet()=pSet;
    sampler.setRadius(high);
    sampler.sample();
  } while(sampler.getPSet().size()>=subsampleTo);
  //binary search
  while(high-low>1E-4f*avgEdgeLen) {
    scalar mid=(low+high)/2;
    sampler.getPSet()=pSet;
    sampler.setRadius(mid);
    sampler.sample();
    if(sampler.getPSet().size()==subsampleTo)
      break;
    else if(sampler.getPSet().size()>subsampleTo)
      low=mid;
    else high=mid;
  }
}
void GraspMesh::detectContactIds(const ParticleSetN& pSet,IDVEC& ids,Pts& contacts,Pts& inNormals) const
{
  ids.assign(pSet.size(),-1);
  contacts.resize(pSet.size());
  inNormals.resize(pSet.size());
  std::vector<scalarD> dist(pSet.size(),ScalarUtil<scalarD>::scalar_max());
  OMP_PARALLEL_FOR_
  for(sizeType i=0; i<pSet.size(); i++) {
    for(sizeType t=0; t<(sizeType)_mesh.getI().size(); t++) {
      const Vec3i& I=_mesh.getI()[t];
      TriangleTpl<scalarD> tri(_mesh.getV(I[0]),_mesh.getV(I[1]),_mesh.getV(I[2]));
      Vec3d bary=tri.bary(pSet[i]._pos.cast<scalarD>());
      bary[0]=std::min<scalarD>(std::max<scalarD>(bary[0],0),1);
      bary[1]=std::min<scalarD>(std::max<scalarD>(bary[1],0),1-bary[0]);
      bary[2]=std::min<scalarD>(std::max<scalarD>(bary[2],0),1-bary[0]-bary[1]);
      Vec3d ptRef=tri._a*bary[0]+tri._b*bary[1]+tri._c*bary[2];
      if((ptRef-pSet[i]._pos.cast<scalarD>()).norm()<dist[i]) {
        dist[i]=(ptRef-pSet[i]._pos.cast<scalarD>()).norm();
        contacts[i]=pSet[i]._pos.cast<scalarD>();
        inNormals[i]=-_mesh.getTN(t);
        ids[i]=t;
      }
    }
  }
  for(sizeType i=0; i<pSet.size(); i++) {
    INFOV("DistErr%ld: %lf!",i,dist[i])
  }
}
Vec3d GraspMesh::reset(std::shared_ptr<StaticGeom>,scalarD theta)
{
  _theta=theta;
  //ensure uniform dist
  _mesh.makeUniform();
  if(_mesh.getVolume()<0)
    _mesh.insideOut();
  //recenter
  Vec3d ctr=_mesh.getVolumeCentroid();
  _mesh.getPos()=-ctr;
  _mesh.applyTrans();
  _mesh.smooth();
  return ctr;
}
Vec3d GraspMesh::reset(const ObjMeshD& mesh,scalarD theta,bool keepOriginalTopology)
{
  _theta=theta;
  _mesh=mesh;
  //ensure uniform dist
  if(!keepOriginalTopology)
    _mesh.makeUnique();
  _mesh.makeUniform();
  if(_mesh.getVolume()<0)
    _mesh.insideOut();
  //ensure uniform dist again
  _mesh.makeUniform();
  if(_mesh.getVolume()<0)
    _mesh.insideOut();
  //recenter
  Vec3d ctr=_mesh.getVolumeCentroid();
  _mesh.getPos()=-ctr;
  _mesh.applyTrans();
  _mesh.smooth();
  return ctr;
}
void GraspMesh::generateContactsTriangleLevel(sizeType subsampleTo)
{
  //add contact
  _dss.clear();
  {
    IDVEC ids;
    Pts contacts;
    Pts inNormals;
    subSampleTriangleLevel(ids,contacts,inNormals,subsampleTo);
    _contactIds=ids;
    _contacts=contacts;
    _inNormals=inNormals;
  }
  assembleContacts();
}
void GraspMesh::generateContactsTriangleLevel(const Dirs& dss)
{
  _dss=dss;
  IDSET tMinIds;
  for(sizeType lid=0; lid<(sizeType)_dss.size(); lid++) {
    //find lineSeg
    scalarD len=_mesh.getBB().getExtent().norm()*2;
    Vec3d d=_dss[lid].segment<3>(0);
    Vec3d d0=_dss[lid].segment<3>(3);
    Vec3d dN=d.normalized(),d0N=d0-d0.dot(dN)*dN;
    LineSegTpl<scalarD> l(-dN*len+d0N,dN*len+d0N);
    scalarD tMin=ScalarUtil<scalarD>::scalar_max(),t;
    sizeType tMinId=-1;
    //find min point
    for(sizeType i=0; i<(sizeType)_mesh.getI().size(); i++) {
      Vec3i I=_mesh.getI(i);
      TriangleTpl<scalarD> tri(_mesh.getV(I[0]).cast<scalarD>(),_mesh.getV(I[1]).cast<scalarD>(),_mesh.getV(I[2]).cast<scalarD>());
      if(tri.intersect(l,t,false,NULL) && t<tMin) {
        tMinId=i;
        tMin=t;
      }
    }
    //add min point
    tMinIds.insert(tMinId);
    ASSERT_MSGV(tMinId>=0,"We cannot find intersection with: d=(%f,%f,%f) d0=(%f,%f,%f)",d[0],d[1],d[2],d0[0],d0[1],d0[2])
  }
  //get one-ring (this is for two point contact)
  IDSET idsOneRing;
  if((sizeType)tMinIds.size()<3) {
    ObjMeshD::EdgeMap ess;
    _mesh.buildEdge(ess);
    for(auto beg=ess._ess.begin(),end=ess._ess.end(); beg!=end; beg++) {
      bool valid=false;
      for(sizeType i=0; i<(sizeType)beg->second._tris.size(); i++)
        if(tMinIds.find(beg->second._tris[i])!=tMinIds.end())
          valid=true;
      if(valid)
        for(sizeType i=0; i<(sizeType)beg->second._tris.size(); i++)
          idsOneRing.insert(beg->second._tris[i]);
    }
  } else {
    idsOneRing.insert(tMinIds.begin(),tMinIds.end());
  }
  //add contact
  _contactIds.clear();
  _contacts.clear();
  _inNormals.clear();
  for(IDSET::const_iterator beg=idsOneRing.begin(),end=idsOneRing.end(); beg!=end; beg++) {
    Vec3i I=_mesh.getI(*beg);
    _contactIds.push_back(*beg);
    _contacts.push_back((_mesh.getV(I[0])+_mesh.getV(I[1])+_mesh.getV(I[2])).cast<scalarD>()/3);
    _inNormals.push_back(-_mesh.getTN(*beg).cast<scalarD>());
  }
  assembleContacts();
}
void GraspMesh::generateContactsPixelLevel(sizeType subsampleTo)
{
  //add contact
  _dss.clear();
  {
    IDVEC ids;
    Pts contacts;
    Pts inNormals;
    subSamplePixelLevel(ids,contacts,inNormals,subsampleTo);
    _contactIds=ids;
    _contacts=contacts;
    _inNormals=inNormals;
  }
  assembleContacts();
}
void GraspMesh::generateContactsPixelLevel(const Dirs& dss)
{
  _dss=dss;
  _contactIds.clear();
  _contacts.clear();
  _inNormals.clear();
  for(sizeType lid=0; lid<(sizeType)_dss.size(); lid++) {
    //find lineSeg
    scalarD len=_mesh.getBB().getExtent().norm()*2;
    Vec3d d=_dss[lid].segment<3>(0);
    Vec3d d0=_dss[lid].segment<3>(3);
    Vec3d dN=d.normalized(),d0N=d0-d0.dot(dN)*dN;
    LineSegTpl<scalarD> l(-dN*len+d0N,dN*len+d0N);
    scalarD tMin=ScalarUtil<scalarD>::scalar_max(),t;
    sizeType tMinId=-1;
    //find min point
    for(sizeType i=0; i<(sizeType)_mesh.getI().size(); i++) {
      Vec3i I=_mesh.getI(i);
      TriangleTpl<scalarD> tri(_mesh.getV(I[0]).cast<scalarD>(),_mesh.getV(I[1]).cast<scalarD>(),_mesh.getV(I[2]).cast<scalarD>());
      if(tri.intersect(l,t,false,NULL) && t<tMin) {
        tMinId=i;
        tMin=t;
      }
    }
    //add min point
    ASSERT_MSGV(tMinId>=0,"We cannot find intersection with: d=(%f,%f,%f) d0=(%f,%f,%f)",d[0],d[1],d[2],d0[0],d0[1],d0[2])
    _contactIds.push_back(tMinId);
    _contacts.push_back(l._x+(l._y-l._x)*tMin);
    _inNormals.push_back(-_mesh.getTN(tMinId));
  }
  assembleContacts();
}
void GraspMesh::assembleContacts()
{
  _sol=NULL;
  //compute wFromF
  _wFromF.setZero(6,_contacts.size()*3);
  for(sizeType i=0; i<(sizeType)_contacts.size(); i++) {
    //Vec3i I=_mesh.getI(_contactIds[i]);
    _wFromF.block<3,3>(0,i*3)=-Mat3d::Identity();
    //this is consistent with piecewise constant force model used by FEM/BEM
    //_wFromF.block<3,3>(3,i*3)-=cross<scalarD>(_mesh.getV(I[0]))/3;
    //_wFromF.block<3,3>(3,i*3)-=cross<scalarD>(_mesh.getV(I[1]))/3;
    //_wFromF.block<3,3>(3,i*3)-=cross<scalarD>(_mesh.getV(I[2]))/3;
    _wFromF.block<3,3>(3,i*3)-=cross<scalarD>(_contacts[i]);
  }
  //compute inertia tensor
  ObjMesh meshScalar;
  _mesh.cast<scalar>(meshScalar);
  RigidBodyMass m(meshScalar);
  Mat3d I=m.getMassCOM().block<3,3>(3,3).cast<scalarD>();
  //I: (yy+zz  -xy   -xz )
  //   ( -xy  xx+zz  -yz )
  //   ( -xz   -yz  xx+yy)
  scalarD intXY=-I(1,0);
  scalarD intXZ=-I(2,0);
  scalarD intYZ=-I(1,2);
  scalarD intXX=I.trace()/2-I(0,0);
  scalarD intYY=I.trace()/2-I(1,1);
  scalarD intZZ=I.trace()/2-I(2,2);
  //compute M
  Mat3d MI;
  MI << intXX,intXY,intXZ,
  intXY      ,intYY,intYZ,
  intXZ      ,intYZ,intZZ;
  Mat9d M=kronecker(MI,3);
  Mat3Xd T=Mat3Xd::Zero(3,9);
  //T: ( 0  -xz  xy)( 0  -yz  yy)( 0  -zz  yz)
  //   ( xz  0  -xx)( yz  0  -xy)( zz  0  -xz)
  //   (-xy  xx  0 )(-yy  xy  0 )(-yz  xz  0 )
  T <<        0,-intXZ, intXY,     0,-intYZ, intYY,     0,-intZZ, intYZ,
  intXZ,      0,-intXX, intYZ,     0,-intXY, intZZ,     0,-intXZ,
  -intXY       , intXX,     0,-intYY, intXY,     0,-intYZ, intXZ,     0;
  Mat9d invM=M.inverse();
  //compute gFromW
  _gFromW.setZero(12,6);
  _gFromW.block<3,3>(0,0)=Mat3d::Identity()/_mesh.getVolume();
  _gFromW.block<9,3>(3,3)=invM*T.transpose()*(T*invM*T.transpose()).inverse();
}
