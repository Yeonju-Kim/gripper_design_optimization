#include "DeformableBody.h"
#include "SparseUtils.h"
#include "Utils.h"
#include "DebugGradient.h"
#include "EigenInterface.h"
#include <CommonFile/CollisionDetection.h>

USE_PRJ_NAMESPACE

//DeformableObjective
DeformableBody::DeformableObjective::DeformableObjective(const DeformableBody& defo,const Cold& FLine):_defo(defo),_FLine(FLine) {}
int DeformableBody::DeformableObjective::operator()(const Vec& x,Vec& fvec,SMat* fjac,bool)
{
  Cold X=_defo._U0+_defo._GC*x;
  _defo.buildFK(NULL,_FLine,X,&fvec,fjac);
  fvec=(_defo._GC.transpose()*fvec).eval();
  if(fjac)
    *fjac=_defo._GC.transpose()*(*fjac*_defo._GC).eval();
  return 0;
}
int DeformableBody::DeformableObjective::inputs() const
{
  return _defo._GC.cols();
}
int DeformableBody::DeformableObjective::values() const
{
  return _defo._GC.cols();
}
//DeformableBody
EIGEN_DEVICE_FUNC DeformableBody::DeformableBody():Serializable(typeid(DeformableBody).name()) {}
void DeformableBody::assemble(const tinyxml2::XMLElement& pt)
{
  //initialize
  Vec4i tet;
  Vec3d pos;
  sizeType nr,one;
  std::string line;
  //read deformable object
  std::istringstream is(get<std::string>(pt,"deformableBodyMesh"));
  bool CGAL=get<bool>(pt,"isCGALMesh",true);
  //vertices
  while(CGAL) {
    getline(is,line);
    if(beginsWith(line,"Vertices"))
      break;
    if(!is.good() || is.eof())
      return;
  }
  getline(is,line);
  std::istringstream(line) >> nr;
  _U0.resize(nr*3);
  for(sizeType i=0; i<nr; i++) {
    getline(is,line);
    std::istringstream iss(line);
    iss >> pos[0] >> pos[1] >> pos[2];
    _U0.segment<3>(i*3)=pos;
  }
  //indices
  while(CGAL) {
    getline(is,line);
    if(beginsWith(line,"Tetrahedra"))
      break;
    if(!is.good() || is.eof())
      return;
  }
  getline(is,line);
  std::istringstream(line) >> nr;
  _tss.resize(4,nr);
  for(sizeType i=0; i<nr; i++) {
    getline(is,line);
    std::istringstream iss(line);
    if(CGAL)
      iss >> tet[0] >> tet[1] >> tet[2] >> tet[3];
    else iss >> one >> tet[0] >> tet[1] >> tet[2] >> tet[3];
    _tss.col(i)=tet;
  }
  if(_tss.minCoeff()==1)
    _tss.array()-=1;
  //constraints
  if(hasAttribute(pt,"fixed")) {
    sizeType nrFixed=get<sizeType>(pt,"fixed.nrFixed");
    _GC=buildGC(nrNode(),parsePtree<Coli>(pt,"fixed.fixedIds",nrFixed));
  }
  //assemble
  assembleMaterial(pt);
  //surface
  _sss=getMuRange(Vec2d(0,ScalarUtil<scalarD>::scalar_max()));
  //read lines
  for(const tinyxml2::XMLElement* v=pt.FirstChildElement(); v; v=v->NextSiblingElement()) {
    if(std::string(v->Name()) == "line") {
      sizeType nrAttachment=get<sizeType>(*v,"nrAttachment");
      _lines.push_back(parsePtree<Coli>(*v,"attachment",nrAttachment));
    }
  }
}
void DeformableBody::assembleMaterial(const tinyxml2::XMLElement& pt)
{
  STrips trips;
  _invDss.resize(3,nrTet()*3);
  _mu.setConstant(nrTet(),ScalarUtil<scalarD>::scalar_max());
  _lambda.setConstant(nrTet(),ScalarUtil<scalarD>::scalar_max());
  _vol.resize(nrTet());
  _G.resize(nrTet()*9,nrNode()*3);
  for(sizeType i=0; i<nrTet(); i++) {
    Mat3X4d c;
    for(sizeType j=0; j<4; j++)
      c.col(j)=_U0.segment<3>(_tss(j,i)*3);
    Mat3d invD,F;
    buildF(c,c,F,invD);
    _invDss.block<3,3>(0,i*3)=invD;
    for(sizeType k=0; k<3; k++)
      for(sizeType j=0; j<3; j++) {
        addBlockId(trips,i*9+j*3,_tss(k+1,i)*3,3, invD(k,j));
        addBlockId(trips,i*9+j*3,_tss(0  ,i)*3,3,-invD(k,j));
      }
    //homogeneous material
    if(hasAttribute(pt,"young") && hasAttribute(pt,"poisson")) {
      scalarD young=get<scalarF>(pt,"young");
      scalarD poisson=get<scalarF>(pt,"poisson");
      _mu[i]=young/(2.0f*(1.0f+poisson));
      _lambda[i]=poisson*young/((1.0f+poisson)*(1.0f-2.0f*poisson));
    } else if(hasAttribute(pt,"mu") && hasAttribute(pt,"lambda")) {
      _mu[i]=get<scalarF>(pt,"mu");
      _lambda[i]=get<scalarF>(pt,"lambda");
    }
    TetrahedronTpl<scalarD> tet(c.col(0),c.col(1),c.col(2),c.col(3));
    if(tet._swap)
      std::swap(_tss(2,i),_tss(3,i));
    _vol[i]=tet.volume();
  }
  _G.setFromTriplets(trips.begin(),trips.end());
  //sigma matrix
  trips.clear();
  _sigma.resize(nrTet()*9,nrTet()*9);
  for(sizeType i=0; i<nrTet(); i++) {
    //mu term
    for(sizeType r=0; r<3; r++)
      for(sizeType c=0; c<3; c++) {
        trips.push_back(STrip(i*9+r+c*3,i*9+r+c*3,_mu[i]));
        trips.push_back(STrip(i*9+r+c*3,i*9+c+r*3,_mu[i]));
      }
    //lambda term
    for(sizeType r=0; r<3; r++)
      for(sizeType c=0; c<3; c++)
        trips.push_back(STrip(i*9+r*4,i*9+c*4,_lambda[i]));
  }
  _sigma.setFromTriplets(trips.begin(),trips.end());
  //debugG(0.1f);
  //non-homogeneous material
  if(hasAttribute(pt,"youngs") && hasAttribute(pt,"poissons")) {
    Cold youngs=parsePtree<Cold>(pt,"youngs",nrTet());
    Cold poissons=parsePtree<Cold>(pt,"poissons",nrTet());
    for(sizeType i=0; i<nrTet(); i++) {
      _mu[i]=youngs[i]/(2.0f*(1.0f+poissons[i]));
      _lambda[i]=poissons[i]*youngs[i]/((1.0f+poissons[i])*(1.0f-2.0f*poissons[i]));
    }
  } else if(hasAttribute(pt,"mus") && hasAttribute(pt,"lambdas")) {
    _mu=parsePtree<Cold>(pt,"mus",nrTet());
    _lambda=parsePtree<Cold>(pt,"lambdas",nrTet());
  }
  _isLinear=get<bool>(pt,"isLinear",true);
  ASSERT(_mu.maxCoeff() != ScalarUtil<scalarD>::scalar_max())
  ASSERT(_lambda.maxCoeff() != ScalarUtil<scalarD>::scalar_max())
}
bool DeformableBody::read(std::istream& is)
{
  readBinaryData(_G,is);
  readBinaryData(_sigma,is);
  readBinaryData(_GC,is);
  readBinaryData(_U0,is);
  readBinaryData(_sss,is);
  readBinaryData(_tss,is);
  readBinaryData(_invDss,is);
  readBinaryData(_lines,is);
  //material
  readBinaryData(_mu,is);
  readBinaryData(_lambda,is);
  readBinaryData(_vol,is);
  readBinaryData(_isLinear,is);
  return is.good();
}
bool DeformableBody::write(std::ostream& os) const
{
  writeBinaryData(_G,os);
  writeBinaryData(_sigma,os);
  writeBinaryData(_GC,os);
  writeBinaryData(_U0,os);
  writeBinaryData(_sss,os);
  writeBinaryData(_tss,os);
  writeBinaryData(_invDss,os);
  writeBinaryData(_lines,os);
  //material
  writeBinaryData(_mu,os);
  writeBinaryData(_lambda,os);
  writeBinaryData(_vol,os);
  writeBinaryData(_isLinear,os);
  return os.good();
}
std::shared_ptr<SerializableBase> DeformableBody::copy() const
{
  return std::shared_ptr<SerializableBase>(new DeformableBody);
}
void DeformableBody::addLine(const Coli& attachment)
{
  _lines.push_back(attachment);
}
void DeformableBody::updateLinesDirs(const Cold& x,LinesDirs& linesDirs) const
{
  linesDirs.resize(_lines.size());
  for(sizeType l=0; l<(sizeType)linesDirs.size(); l++) {
    Vec3d abN;
    linesDirs[l].clear();
    const Coli& attachment=_lines[l];
    for(sizeType i=0; i<attachment.size()-1; i++) {
      sizeType offA=attachment[i]*3;
      sizeType offB=attachment[i+1]*3;
      Vec3d a=x.segment<3>(offA);
      Vec3d b=x.segment<3>(offB);
      scalarD len=std::max<scalarD>((a-b).norm(),1E-15f);
      abN=(a-b)/len;
      linesDirs[l].push_back(abN);
    }
  }
}
scalarD DeformableBody::buildKM(const Cold& x,Cold* F,SMat* K,SMat* M) const
{
  //compute FK due to deformable body
  Mat9d hess;
  scalarD E=0;
  STrips trips;
  Cold G=_G*x,FT=Cold::Zero(_tss.cols()*9);
  OMP_PARALLEL_FOR_I(OMP_PRI(hess) OMP_ADD(E))
  for(sizeType i=0; i<_tss.cols(); i++) {
    Eigen::Map<const Mat3d> FMap(G.data()+i*9);
    Eigen::Map<Vec9d> FTMap(FT.data()+i*9);
    if(_isLinear)
      E+=evalFLinear(FMap,_lambda[i],_mu[i],_vol[i],1,FTMap,K ? &hess : NULL);
    else E+=evalFKNonHK(FMap,_lambda[i],_mu[i],_vol[i],0,0,FTMap,K ? &hess : NULL);
    addBlock(trips,i*9,i*9,hess);
  }
  if(F)
    *F=_G.transpose()*FT;
  if(K) {
    SMat H;
    H.resize(_tss.cols()*9,_tss.cols()*9);
    H.setFromTriplets(trips.begin(),trips.end());
    *K=_G.transpose()*(H*_G);
  }
  if(M) {
    trips.clear();
    for(sizeType i=0; i<_tss.cols(); i++)
      for(sizeType j=0; j<4; j++)
        for(sizeType k=0; k<4; k++)
          addBlockId(trips,_tss(j,i)*3,_tss(k,i)*3,3,j==k?_vol[i]/10:_vol[i]/20);
    M->resize(_U0.size(),_U0.size());
    M->setFromTriplets(trips.begin(),trips.end());
  }
  return E;
}
scalarD DeformableBody::buildFK(const LinesDirs* linesDirs,const Cold& FLine,const Cold& x,Cold* F,SMat* K) const
{
  scalarD E=buildKM(x,F,K,NULL);
  //compute FK due to line actuator
  STrips trips;
  for(sizeType i=0; i<(sizeType)_lines.size(); i++)
    E+=buildFKLine(linesDirs ? &(linesDirs->at(i)) : NULL,_lines[i],FLine[i],x,F,K ? &trips : NULL);
  if(K) {
    SMat H;
    H.resize(x.size(),x.size());
    H.setFromTriplets(trips.begin(),trips.end());
    *K+=H;
  }
  return E;
}
scalarD DeformableBody::buildFKLine(const LineDirs* lineDirs,const Coli& attachment,scalarD f,const Cold& x,Cold* F,STrips* trips) const
{
  Vec3d abN;
  scalarD E=0;
  for(sizeType i=0; i<attachment.size()-1; i++) {
    sizeType offA=attachment[i]*3;
    sizeType offB=attachment[i+1]*3;
    Vec3d a=x.segment<3>(offA);
    Vec3d b=x.segment<3>(offB);
    if(lineDirs) {
      abN=lineDirs->at(i);
    } else {
      scalarD len=std::max<scalarD>((a-b).norm(),1E-15f);
      abN=(a-b)/len;
      if(trips) {
        Mat3d H=(Mat3d::Identity()-abN*abN.transpose())*f/len;
        addBlock(*trips,offA,offA, H);
        addBlock(*trips,offA,offB,-H);
        addBlock(*trips,offB,offA,-H);
        addBlock(*trips,offB,offB, H);
      }
    }
    E+=abN.dot(a-b)*f;
    Vec3d fab=abN*f;
    if(F) {
      F->segment<3>(offA)+=fab;
      F->segment<3>(offB)-=fab;
    }
  }
  return E;
}
bool DeformableBody::solveNewton(const Cold& FLine,Cold& x,scalarD tolX,scalarD tolG,sizeType maxIter,bool useCB) const
{
  SMat MC,K;
  scalarD nu=1;
  scalarD mu=1E3f;
  scalarD maxMu=1E8f;
  scalarD minMu=1E-8f;
  sizeType maxIterI=10;
  Cold xC=_GC.transpose()*(x-_U0);
  buildKM(_U0,NULL,NULL,&MC);
  MC=_GC.transpose()*(MC*_GC).eval();
  std::shared_ptr<LinearSolverInterface> sol=LinearSolverTraits<SMat>::getCholSolver();
  for(sizeType iter=0; iter<maxIter;) {
    //internal iteration
    Cold xC0=xC,xC1=xC,F,F0;
    sizeType iterI=0;
    bool succ=true;
    for(; iterI<maxIterI; iterI++,iter++) {
      buildFK(NULL,FLine,_GC*xC1+_U0,&F,&K);
      if(!sol->recompute(SMat(_GC.transpose()*SMat(K*_GC)+(MC*mu)),0,false)) {
        succ=false;
        break;
      }
      F0=_GC.transpose()*F;
      Cold NDX=sol->solve(F0+(MC*mu)*(xC1-xC0));
      if(NDX.cwiseAbs().maxCoeff() < tolX)
        break;
      xC1-=NDX;
    }
    //update parameter
    if(!succ || iterI==maxIterI) {
      mu=std::min<scalarD>(mu*10,maxMu);
      if(mu == maxMu)
        return false;
      nu=1;
    } else {
      xC=xC1;
      mu=std::max<scalarD>(mu*0.75f*nu,minMu);
      nu=std::max<scalarD>(nu*0.5f,minMu);
      //termination
      if(useCB) {
        INFOV("DeltaX: %f",(xC0-xC1).cwiseAbs().maxCoeff())
      }
      if(F0.cwiseAbs().maxCoeff() < tolG && (xC1-xC0).cwiseAbs().maxCoeff() < tolX)
        break;
    }
  }
  x=_U0+_GC*xC;
  return true;
}
Matd DeformableBody::buildDeriv(const Cold& FLine,const Cold& x) const
{
  SMat K;
  buildFK(NULL,FLine,x,NULL,&K);
  Matd F=buildDeriv(x);

  EigenCholInterface sol;
  ASSERT_MSG(sol.recompute(SMat(_GC.transpose()*(K*_GC)),0,false),"Derivative computation failed!")
  return -_GC*sol.solve(_GC.transpose()*F);
}
Matd DeformableBody::buildDeriv(const Cold& x) const
{
  Matd F=Matd::Zero(nrNode()*3,nrLine());
  for(sizeType i=0; i<(sizeType)_lines.size(); i++) {
    Cold FI=Cold::Zero(nrNode()*3);
    buildFKLine(NULL,_lines[i],1,x,&FI,NULL);
    F.col(i)=FI;
  }
  return F;
}
const DeformableBody::SMat& DeformableBody::G() const
{
  return _G;
}
const DeformableBody::SMat& DeformableBody::sigma() const
{
  return _sigma;
}
const Mat4Xi& DeformableBody::tss() const
{
  return _tss;
}
Mat4Xi& DeformableBody::tss()
{
  return _tss;
}
const Cold& DeformableBody::vol() const
{
  return _vol;
}
const Cold& DeformableBody::U0() const
{
  return _U0;
}
Cold& DeformableBody::U0()
{
  return _U0;
}
scalarD DeformableBody::minMu() const
{
  return _mu.minCoeff();
}
scalarD DeformableBody::maxMu() const
{
  return _mu.maxCoeff();
}
sizeType DeformableBody::nrNode() const
{
  return (sizeType)_U0.size()/3;
}
sizeType DeformableBody::nrTet() const
{
  return _tss.cols();
}
sizeType DeformableBody::nrLine() const
{
  return (sizeType)_lines.size();
}
const DeformableBody::SMat& DeformableBody::getGC() const
{
  return _GC;
}
void DeformableBody::debugG(scalarD scale) const
{
  Mat3d F,d;
  Mat3X4d c,c0;
  DEFINE_NUMERIC_DELTA
  Cold U=_U0+Cold::Random(_U0.size())*scale,GU=_G*U;
  for(sizeType i=0; i<nrTet(); i++) {
    for(sizeType j=0; j<4; j++) {
      c .col(j)=U.segment<3>(_tss(j,i)*3);
      c0.col(j)=_U0.segment<3>(_tss(j,i)*3);
    }
    buildF(c,c0,F,d);
    Mat3d GUI=Eigen::Map<const Mat3d>(GU.data()+i*9);
    DEBUG_GRADIENT("DefoGrad",F.norm(),(F-GUI).norm())
  }
  exit(EXIT_SUCCESS);
}
void DeformableBody::debugFK(scalarD scale,sizeType maxIt) const
{
  SMat K;
  Cold F,F2,FLine=Cold::Random(_lines.size())*0.5f+Cold::Constant(_lines.size(),0.5f);
  INFO("-------------------------------------------------------------DeformableBody::debugFK")
  DEFINE_NUMERIC_DELTA
  for(sizeType i=0; i<maxIt; i++) {
    Cold x=Cold::Random(_U0.size())*scale+_U0;
    Cold dx=Cold::Random(_U0.size());
    buildFK(NULL,FLine,x,&F,&K);
    buildFK(NULL,FLine,x+dx*DELTA,&F2,NULL);
    DEBUG_GRADIENT("HessianFK",(K*dx).norm(),(K*dx-(F2-F)/DELTA).norm())
  }
  LinesDirs linesDirs;
  for(sizeType i=0; i<maxIt; i++) {
    Cold x=Cold::Random(_U0.size())*scale+_U0;
    Cold dx=Cold::Random(_U0.size());
    updateLinesDirs(x,linesDirs);
    scalarD E=buildFK(&linesDirs,FLine,x,&F,&K);
    scalarD E2=buildFK(&linesDirs,FLine,x+dx*DELTA,&F2,NULL);
    DEBUG_GRADIENT("GradientFK",F.dot(dx),(E2-E)/DELTA-F.dot(dx))
    DEBUG_GRADIENT("HessianFK",(K*dx).norm(),(K*dx-(F2-F)/DELTA).norm())
  }
  exit(EXIT_SUCCESS);
}
void DeformableBody::debugDeriv(scalarD scale,sizeType maxIt) const
{
  DEFINE_NUMERIC_DELTA
  INFO("-------------------------------------------------------------DeformableBody::debugDeriv")
  for(sizeType i=0; i<maxIt; i++) {
    Cold FLine=(Cold::Random(_lines.size())*0.5f+Cold::Constant(_lines.size(),0.5f))*scale;
    Cold dFLine=Cold::Random(_lines.size());
    Cold x=U0(),x2=U0();
    solveNewton(FLine,x,ScalarUtil<scalarD>::scalar_eps(),ScalarUtil<scalarD>::scalar_eps(),1E4,false);
    Matd dxdFLine=buildDeriv(FLine,x);
    solveNewton(FLine+dFLine*DELTA,x2,ScalarUtil<scalarD>::scalar_eps(),ScalarUtil<scalarD>::scalar_eps(),1E4,false);
    DEBUG_GRADIENT("Deriv",(dxdFLine*dFLine).norm(),(dxdFLine*dFLine-(x2-x)/DELTA).norm())
  }
  exit(EXIT_SUCCESS);
}
//writeMesh
void DeformableBody::getH(const Cold& U,std::vector<scalarD>& H) const
{
  H.resize((sizeType)_sigma.rows()/9);
  Cold sigmas=_sigma*_G*U;
  Vec9d sigma;
  Vec3d sigmaEV;
  OMP_PARALLEL_FOR_I(OMP_PRI(sigma,sigmaEV))
  for(sizeType i=0; i<(sizeType)H.size(); i++) {
    sigma=sigmas.segment<9>(i*9);
    dsyevc3<scalarD>((scalarD(*)[3])sigma.data(),sigmaEV.data());
    H[i]=sigmaEV.cwiseAbs().maxCoeff();
  }
}
void DeformableBody::writeVTK(const Cold& U,const std::string& path,const Cold* D,const std::vector<scalarD>* color) const
{
  std::vector<Vec3d,Eigen::aligned_allocator<Vec3d>> vss;
  std::vector<Vec4i,Eigen::aligned_allocator<Vec4i>> tss;
  for(sizeType i=0; i<U.size(); i+=3)
    vss.push_back(U.segment<3>(i));
  for(sizeType i=0; i<_tss.cols(); i++)
    tss.push_back(_tss.col(i));

  VTKWriter<scalarD> os("tet",path,true);
  os.appendPoints(vss.begin(),vss.end());
  os.appendCells(tss.begin(),tss.end(),VTKWriter<scalarD>::TETRA);
  if(D) {
    std::vector<scalarD> H;
    getH(*D,H);
    os.appendCustomData("H",H.begin(),H.end());
  }
  if(color) {
    os.appendCustomData("color",color->begin(),color->end());
  }
}
void DeformableBody::writeLineVTK(const Cold& U,const std::string& path) const
{
  VTKWriter<scalarD> os("line",path,true);
  std::vector<Vec3d,Eigen::aligned_allocator<Vec3d> > vss;
  for(sizeType i=0; i<(sizeType)_lines.size(); i++)
    for(sizeType j=0; j<_lines[i].size()-1; j++) {
      vss.push_back(U.segment<3>(_lines[i][j]*3));
      vss.push_back(U.segment<3>(_lines[i][j+1]*3));
    }
  os.appendPoints(vss.begin(),vss.end());
  os.appendCells(VTKWriter<scalarD>::IteratorIndex<Vec3i>(0,2,0),
                 VTKWriter<scalarD>::IteratorIndex<Vec3i>(vss.size()/2,2,0),
                 VTKWriter<scalarD>::LINE);
  os.appendCells(VTKWriter<scalarD>::IteratorIndex<Vec3i>(0,0,0),
                 VTKWriter<scalarD>::IteratorIndex<Vec3i>(vss.size(),0,0),
                 VTKWriter<scalarD>::POINT);
}
Vec3d DeformableBody::getVolumeCentroid(const Cold& U) const
{
  scalarD V=0;
  Vec3d ret=Vec3d::Zero();
  for(sizeType i=0; i<_tss.cols(); i++) {
    Vec3d TC=(U.segment<3>(_tss(0,i)*3)+
              U.segment<3>(_tss(1,i)*3)+
              U.segment<3>(_tss(2,i)*3)+
              U.segment<3>(_tss(3,i)*3))/4;
    ret+=_vol[i]*TC;
    V+=_vol[i];
  }
  return ret/V;
}
Mat3Xi DeformableBody::getMuRange(const Vec2d& range,Coli* tss) const
{
  //surface
  std::unordered_map<Vec3i,sizeType,Hash> sm;
  for(sizeType i=0; i<_tss.cols(); i++)
    if(_mu[i]>=range[0] && _mu[i]<=range[1])
      for(sizeType d=0; d<4; d++) {
        Vec3i id(_tss((d+0)%4,i),_tss((d+1)%4,i),_tss((d+2)%4,i));
        std::sort(id.data()+0,id.data()+id.size());
        if(sm.find(id) == sm.end())
          sm[id]=i;
        else sm.erase(id);
      }
  sizeType k=0;
  Mat3Xi sss;
  sss.resize(3,sm.size());
  if(tss)
    tss->resize(sm.size());
  for(std::unordered_map<Vec3i,sizeType,Hash>::const_iterator beg=sm.begin(),end=sm.end(); beg!=end; beg++) {
    sss.col(k)=beg->first.transpose();
    if(tss)
      (*tss)[k]=beg->second;
    k++;
  }
  return sss;
}
ObjMesh DeformableBody::getMeshMuRange(const Cold& U,const Vec2d& range) const
{
  Mat3Xi sss=getMuRange(range);
  return getMesh(U,&sss);
}
ObjMesh DeformableBody::getMesh(const Cold& U,const Mat3Xi* sss) const
{
  ObjMesh mesh;
  for(sizeType i=0; i<U.size(); i+=3)
    mesh.getV().push_back(U.segment<3>(i).cast<scalar>());
  if(!sss)
    sss=&_sss;
  for(sizeType i=0; i<sss->cols(); i++)
    mesh.getI().push_back(sss->col(i));
  mesh.smooth();
  mesh.makeUniform();
  mesh.smooth();
  return mesh;
}
ObjMesh DeformableBody::getMesh() const
{
  return getMesh(_U0);
}
//utility
scalarD DeformableBody::evalFKNonHK(Eigen::Map<const Mat3d> f,scalarD lambda,scalarD mu,scalarD V,scalarD dimMask,scalarD minVol,Eigen::Map<Vec9d> grad,Mat9d* hess)
{
  scalarD E;
  //input
  //scalarD V;
  //scalarD dimMask;
  //Mat3d f;
  //scalarD lambda;
  //scalarD mu;

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
  scalarD tt20;
  scalarD tt21;
  scalarD tt22;
  scalarD tt23;
  scalarD tt24;
  scalarD tt25;
  scalarD tt26;
  scalarD tt27;
  scalarD tt28;
  scalarD tt29;
  scalarD tt30;
  scalarD tt31;
  scalarD tt32;
  scalarD tt33;
  scalarD tt34;
  scalarD tt35;
  scalarD tt36;
  scalarD tt37;
  scalarD tt38;
  scalarD tt39;
  scalarD tt40;
  scalarD tt41;
  scalarD tt42;
  scalarD tt43;
  scalarD tt44;
  scalarD tt45;
  scalarD tt46;
  scalarD tt47;
  scalarD tt48;
  scalarD tt49;
  scalarD tt50;
  scalarD tt51;
  scalarD tt52;
  scalarD tt53;
  scalarD tt54;
  scalarD tt55;
  scalarD tt56;
  scalarD tt57;
  scalarD tt58;
  scalarD tt59;
  scalarD tt60;

  tt1=dimMask+f(2,2);
  tt2=f(1,0)*f(2,1)-f(1,1)*f(2,0);
  tt3=f(1,1)*tt1-f(1,2)*f(2,1);
  tt4=f(0,0)*tt3-f(0,1)*(f(1,0)*tt1-f(1,2)*f(2,0))+f(0,2)*tt2;
  tt5=log(std::max(tt4,minVol));
  tt6=1/tt4;
  tt7=f(0,2)*f(2,1)-f(0,1)*tt1;
  tt8=f(0,1)*f(1,2)-f(0,2)*f(1,1);
  tt9=f(1,2)*f(2,0)-f(1,0)*tt1;
  tt10=f(0,0)*tt1-f(0,2)*f(2,0);
  tt11=f(0,2)*f(1,0)-f(0,0)*f(1,2);
  tt12=f(0,1)*f(2,0)-f(0,0)*f(2,1);
  tt13=f(0,0)*f(1,1)-f(0,1)*f(1,0);
  E=V*(0.5*pow(tt5,2)*lambda+0.5*mu*(-2*tt5+pow(tt1,2)+pow(f(2,1),2)+pow(f(2,0),2)+pow(f(1,2),2)+pow(f(1,1),2)+pow(f(1,0),2)+pow(f(0,2),2)+pow(f(0,1),2)+pow(f(0,0),2)-3));
  grad[0]=V*(1.0*tt3*tt6*tt5*lambda+0.5*mu*(2*f(0,0)-2*tt3*tt6));
  grad[1]=V*(1.0*tt7*tt6*tt5*lambda+0.5*mu*(2*f(1,0)-2*tt7*tt6));
  grad[2]=V*(1.0*tt8*tt6*tt5*lambda+0.5*mu*(2*f(2,0)-2*tt8*tt6));
  grad[3]=V*(1.0*tt9*tt6*tt5*lambda+0.5*mu*(2*f(0,1)-2*tt9*tt6));
  grad[4]=V*(1.0*tt10*tt6*tt5*lambda+0.5*mu*(2*f(1,1)-2*tt10*tt6));
  grad[5]=V*(1.0*tt11*tt6*tt5*lambda+0.5*mu*(2*f(2,1)-2*tt11*tt6));
  grad[6]=V*(1.0*tt2*tt6*tt5*lambda+0.5*mu*(2*f(0,2)-2*tt2*tt6));
  grad[7]=V*(1.0*tt12*tt6*tt5*lambda+0.5*mu*(2*f(1,2)-2*tt12*tt6));
  grad[8]=V*(1.0*tt13*tt6*tt5*lambda+0.5*mu*(2*tt1-2*tt13*tt6));
  if(!hess)return E;

  tt14=pow(tt3,2);
  tt15=1/pow(tt4,2);
  tt16=V*(-1.0*tt7*tt3*tt15*tt5*lambda+1.0*tt7*tt3*tt15*lambda+1.0*mu*tt7*tt3*tt15);
  tt17=V*(-1.0*tt8*tt3*tt15*tt5*lambda+1.0*tt8*tt3*tt15*lambda+1.0*tt8*mu*tt3*tt15);
  tt18=V*(-1.0*tt9*tt3*tt15*tt5*lambda+1.0*tt9*tt3*tt15*lambda+1.0*mu*tt9*tt3*tt15);
  tt19=V*(1.0*tt1*tt6*tt5*lambda-1.0*tt10*tt3*tt15*tt5*lambda+1.0*tt10*tt3*tt15*lambda+0.5*mu*(2*tt10*tt3*tt15-2*tt1*tt6));
  tt20=V*(-1.0*f(1,2)*tt6*tt5*lambda-1.0*tt11*tt3*tt15*tt5*lambda+1.0*tt11*tt3*tt15*lambda+0.5*mu*(2*f(1,2)*tt6+2*tt11*tt3*tt15));
  tt21=V*(-1.0*tt2*tt3*tt15*tt5*lambda+1.0*tt2*tt3*tt15*lambda+1.0*tt2*mu*tt3*tt15);
  tt22=V*(-1.0*f(2,1)*tt6*tt5*lambda-1.0*tt12*tt3*tt15*tt5*lambda+1.0*tt12*tt3*tt15*lambda+0.5*mu*(2*f(2,1)*tt6+2*tt12*tt3*tt15));
  tt23=V*(1.0*f(1,1)*tt6*tt5*lambda-1.0*tt13*tt3*tt15*tt5*lambda+1.0*tt13*tt3*tt15*lambda+0.5*mu*(2*tt13*tt3*tt15-2*f(1,1)*tt6));
  tt24=pow(tt7,2);
  tt25=V*(-1.0*tt8*tt7*tt15*tt5*lambda+1.0*tt8*tt7*tt15*lambda+1.0*tt8*mu*tt7*tt15);
  tt26=-dimMask-f(2,2);
  tt27=V*(1.0*tt26*tt6*tt5*lambda-1.0*tt7*tt9*tt15*tt5*lambda+1.0*tt7*tt9*tt15*lambda+0.5*mu*(2*tt7*tt9*tt15-2*tt26*tt6));
  tt28=V*(-1.0*tt10*tt7*tt15*tt5*lambda+1.0*tt10*tt7*tt15*lambda+1.0*mu*tt10*tt7*tt15);
  tt29=V*(1.0*f(0,2)*tt6*tt5*lambda-1.0*tt11*tt7*tt15*tt5*lambda+1.0*tt11*tt7*tt15*lambda+0.5*mu*(2*tt11*tt7*tt15-2*f(0,2)*tt6));
  tt30=V*(1.0*f(2,1)*tt6*tt5*lambda-1.0*tt2*tt7*tt15*tt5*lambda+1.0*tt2*tt7*tt15*lambda+0.5*mu*(2*tt2*tt7*tt15-2*f(2,1)*tt6));
  tt31=V*(-1.0*tt12*tt7*tt15*tt5*lambda+1.0*tt12*tt7*tt15*lambda+1.0*tt12*mu*tt7*tt15);
  tt32=V*(-1.0*f(0,1)*tt6*tt5*lambda-1.0*tt13*tt7*tt15*tt5*lambda+1.0*tt13*tt7*tt15*lambda+0.5*mu*(2*f(0,1)*tt6+2*tt13*tt7*tt15));
  tt33=pow(tt8,2);
  tt34=V*(1.0*f(1,2)*tt6*tt5*lambda-1.0*tt8*tt9*tt15*tt5*lambda+1.0*tt8*tt9*tt15*lambda+0.5*mu*(2*tt8*tt9*tt15-2*f(1,2)*tt6));
  tt35=V*(-1.0*f(0,2)*tt6*tt5*lambda-1.0*tt8*tt10*tt15*tt5*lambda+1.0*tt8*tt10*tt15*lambda+0.5*mu*(2*f(0,2)*tt6+2*tt8*tt10*tt15));
  tt36=V*(-1.0*tt11*tt8*tt15*tt5*lambda+1.0*tt11*tt8*tt15*lambda+1.0*tt11*tt8*mu*tt15);
  tt37=V*(-1.0*f(1,1)*tt6*tt5*lambda-1.0*tt8*tt2*tt15*tt5*lambda+1.0*tt8*tt2*tt15*lambda+0.5*mu*(2*f(1,1)*tt6+2*tt8*tt2*tt15));
  tt38=V*(1.0*f(0,1)*tt6*tt5*lambda-1.0*tt8*tt12*tt15*tt5*lambda+1.0*tt8*tt12*tt15*lambda+0.5*mu*(2*tt8*tt12*tt15-2*f(0,1)*tt6));
  tt39=V*(-1.0*tt13*tt8*tt15*tt5*lambda+1.0*tt13*tt8*tt15*lambda+1.0*tt13*tt8*mu*tt15);
  tt40=pow(tt9,2);
  tt41=V*(-1.0*tt10*tt9*tt15*tt5*lambda+1.0*tt10*tt9*tt15*lambda+1.0*mu*tt10*tt9*tt15);
  tt42=V*(-1.0*tt11*tt9*tt15*tt5*lambda+1.0*tt11*tt9*tt15*lambda+1.0*tt11*mu*tt9*tt15);
  tt43=V*(-1.0*tt2*tt9*tt15*tt5*lambda+1.0*tt2*tt9*tt15*lambda+1.0*tt2*mu*tt9*tt15);
  tt44=V*(1.0*f(2,0)*tt6*tt5*lambda-1.0*tt12*tt9*tt15*tt5*lambda+1.0*tt12*tt9*tt15*lambda+0.5*mu*(2*tt12*tt9*tt15-2*f(2,0)*tt6));
  tt45=V*(-1.0*f(1,0)*tt6*tt5*lambda-1.0*tt13*tt9*tt15*tt5*lambda+1.0*tt13*tt9*tt15*lambda+0.5*mu*(2*f(1,0)*tt6+2*tt13*tt9*tt15));
  tt46=pow(tt10,2);
  tt47=V*(-1.0*tt11*tt10*tt15*tt5*lambda+1.0*tt11*tt10*tt15*lambda+1.0*tt11*mu*tt10*tt15);
  tt48=V*(-1.0*f(2,0)*tt6*tt5*lambda-1.0*tt2*tt10*tt15*tt5*lambda+1.0*tt2*tt10*tt15*lambda+0.5*mu*(2*f(2,0)*tt6+2*tt2*tt10*tt15));
  tt49=V*(-1.0*tt12*tt10*tt15*tt5*lambda+1.0*tt12*tt10*tt15*lambda+1.0*tt12*mu*tt10*tt15);
  tt50=V*(1.0*f(0,0)*tt6*tt5*lambda-1.0*tt13*tt10*tt15*tt5*lambda+1.0*tt13*tt10*tt15*lambda+0.5*mu*(2*tt13*tt10*tt15-2*f(0,0)*tt6));
  tt51=pow(tt11,2);
  tt52=V*(1.0*f(1,0)*tt6*tt5*lambda-1.0*tt11*tt2*tt15*tt5*lambda+1.0*tt11*tt2*tt15*lambda+0.5*mu*(2*tt11*tt2*tt15-2*f(1,0)*tt6));
  tt53=V*(-1.0*f(0,0)*tt6*tt5*lambda-1.0*tt11*tt12*tt15*tt5*lambda+1.0*tt11*tt12*tt15*lambda+0.5*mu*(2*f(0,0)*tt6+2*tt11*tt12*tt15));
  tt54=V*(-1.0*tt13*tt11*tt15*tt5*lambda+1.0*tt13*tt11*tt15*lambda+1.0*tt13*tt11*mu*tt15);
  tt55=pow(tt2,2);
  tt56=V*(-1.0*tt12*tt2*tt15*tt5*lambda+1.0*tt12*tt2*tt15*lambda+1.0*tt12*tt2*mu*tt15);
  tt57=V*(-1.0*tt13*tt2*tt15*tt5*lambda+1.0*tt13*tt2*tt15*lambda+1.0*tt13*tt2*mu*tt15);
  tt58=pow(tt12,2);
  tt59=V*(-1.0*tt13*tt12*tt15*tt5*lambda+1.0*tt13*tt12*tt15*lambda+1.0*tt13*tt12*mu*tt15);
  tt60=pow(tt13,2);
  (*hess)(0,0)=V*(-1.0*tt14*tt15*tt5*lambda+1.0*tt14*tt15*lambda+0.5*mu*(2*tt14*tt15+2));
  (*hess)(0,1)=tt16;
  (*hess)(0,2)=tt17;
  (*hess)(0,3)=tt18;
  (*hess)(0,4)=tt19;
  (*hess)(0,5)=tt20;
  (*hess)(0,6)=tt21;
  (*hess)(0,7)=tt22;
  (*hess)(0,8)=tt23;
  (*hess)(1,0)=tt16;
  (*hess)(1,1)=V*(-1.0*tt24*tt15*tt5*lambda+1.0*tt24*tt15*lambda+0.5*mu*(2*tt24*tt15+2));
  (*hess)(1,2)=tt25;
  (*hess)(1,3)=tt27;
  (*hess)(1,4)=tt28;
  (*hess)(1,5)=tt29;
  (*hess)(1,6)=tt30;
  (*hess)(1,7)=tt31;
  (*hess)(1,8)=tt32;
  (*hess)(2,0)=tt17;
  (*hess)(2,1)=tt25;
  (*hess)(2,2)=V*(-1.0*tt33*tt15*tt5*lambda+1.0*tt33*tt15*lambda+0.5*mu*(2*tt33*tt15+2));
  (*hess)(2,3)=tt34;
  (*hess)(2,4)=tt35;
  (*hess)(2,5)=tt36;
  (*hess)(2,6)=tt37;
  (*hess)(2,7)=tt38;
  (*hess)(2,8)=tt39;
  (*hess)(3,0)=tt18;
  (*hess)(3,1)=tt27;
  (*hess)(3,2)=tt34;
  (*hess)(3,3)=V*(-1.0*tt40*tt15*tt5*lambda+1.0*tt40*tt15*lambda+0.5*mu*(2*tt40*tt15+2));
  (*hess)(3,4)=tt41;
  (*hess)(3,5)=tt42;
  (*hess)(3,6)=tt43;
  (*hess)(3,7)=tt44;
  (*hess)(3,8)=tt45;
  (*hess)(4,0)=tt19;
  (*hess)(4,1)=tt28;
  (*hess)(4,2)=tt35;
  (*hess)(4,3)=tt41;
  (*hess)(4,4)=V*(-1.0*tt46*tt15*tt5*lambda+1.0*tt46*tt15*lambda+0.5*mu*(2*tt46*tt15+2));
  (*hess)(4,5)=tt47;
  (*hess)(4,6)=tt48;
  (*hess)(4,7)=tt49;
  (*hess)(4,8)=tt50;
  (*hess)(5,0)=tt20;
  (*hess)(5,1)=tt29;
  (*hess)(5,2)=tt36;
  (*hess)(5,3)=tt42;
  (*hess)(5,4)=tt47;
  (*hess)(5,5)=V*(-1.0*tt51*tt15*tt5*lambda+1.0*tt51*tt15*lambda+0.5*mu*(2*tt51*tt15+2));
  (*hess)(5,6)=tt52;
  (*hess)(5,7)=tt53;
  (*hess)(5,8)=tt54;
  (*hess)(6,0)=tt21;
  (*hess)(6,1)=tt30;
  (*hess)(6,2)=tt37;
  (*hess)(6,3)=tt43;
  (*hess)(6,4)=tt48;
  (*hess)(6,5)=tt52;
  (*hess)(6,6)=V*(-1.0*tt55*tt15*tt5*lambda+1.0*tt55*tt15*lambda+0.5*mu*(2*tt55*tt15+2));
  (*hess)(6,7)=tt56;
  (*hess)(6,8)=tt57;
  (*hess)(7,0)=tt22;
  (*hess)(7,1)=tt31;
  (*hess)(7,2)=tt38;
  (*hess)(7,3)=tt44;
  (*hess)(7,4)=tt49;
  (*hess)(7,5)=tt53;
  (*hess)(7,6)=tt56;
  (*hess)(7,7)=V*(-1.0*tt58*tt15*tt5*lambda+1.0*tt58*tt15*lambda+0.5*mu*(2*tt58*tt15+2));
  (*hess)(7,8)=tt59;
  (*hess)(8,0)=tt23;
  (*hess)(8,1)=tt32;
  (*hess)(8,2)=tt39;
  (*hess)(8,3)=tt45;
  (*hess)(8,4)=tt50;
  (*hess)(8,5)=tt54;
  (*hess)(8,6)=tt57;
  (*hess)(8,7)=tt59;
  (*hess)(8,8)=V*(-1.0*tt60*tt15*tt5*lambda+1.0*tt60*tt15*lambda+0.5*mu*(2*tt60*tt15+2));
  return E;
}
scalarD DeformableBody::evalFLinear(Eigen::Map<const Mat3d> f,scalarD lambda,scalarD mu,scalarD V,scalarD dimMask,Eigen::Map<Vec9d> grad,Mat9d* hess)
{
  scalarD E;
  //input
  //scalarD V;
  //scalarD dimMask;
  //Mat3d f;
  //scalarD lambda;
  //scalarD mu;

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

  tt1=1.0*f(0,0);
  tt2=tt1-1;
  tt3=f(1,0)+f(0,1);
  tt4=1.0*f(1,1);
  tt5=tt4-1;
  tt6=f(2,0)+f(0,2);
  tt7=f(2,1)+f(1,2);
  tt8=1.0*f(2,2);
  tt9=-dimMask;
  tt10=tt9+tt8;
  tt11=tt9+tt8+tt4+tt1-2;
  tt12=1.0*tt11*lambda;
  tt13=1.0*tt3*mu*V;
  tt14=1.0*tt6*mu*V;
  tt15=1.0*tt7*mu*V;
  E=V*(pow(tt11,2)*lambda/2.0+mu*(pow(tt10,2)+0.5*pow(tt7,2)+0.5*pow(tt6,2)+pow(tt5,2)+0.5*pow(tt3,2)+pow(tt2,2)));
  grad[0]=V*(tt12+2.0*tt2*mu);
  grad[1]=tt13;
  grad[2]=tt14;
  grad[3]=tt13;
  grad[4]=V*(tt12+2.0*tt5*mu);
  grad[5]=tt15;
  grad[6]=tt14;
  grad[7]=tt15;
  grad[8]=V*(tt12+2.0*mu*tt10);
  if(!hess)return E;

  tt16=V*(1.0*lambda+2.0*mu);
  tt17=1.0*V*lambda;
  tt18=1.0*mu*V;
  (*hess)(0,0)=tt16;
  (*hess)(0,1)=0;
  (*hess)(0,2)=0;
  (*hess)(0,3)=0;
  (*hess)(0,4)=tt17;
  (*hess)(0,5)=0;
  (*hess)(0,6)=0;
  (*hess)(0,7)=0;
  (*hess)(0,8)=tt17;
  (*hess)(1,0)=0;
  (*hess)(1,1)=tt18;
  (*hess)(1,2)=0;
  (*hess)(1,3)=tt18;
  (*hess)(1,4)=0;
  (*hess)(1,5)=0;
  (*hess)(1,6)=0;
  (*hess)(1,7)=0;
  (*hess)(1,8)=0;
  (*hess)(2,0)=0;
  (*hess)(2,1)=0;
  (*hess)(2,2)=tt18;
  (*hess)(2,3)=0;
  (*hess)(2,4)=0;
  (*hess)(2,5)=0;
  (*hess)(2,6)=tt18;
  (*hess)(2,7)=0;
  (*hess)(2,8)=0;
  (*hess)(3,0)=0;
  (*hess)(3,1)=tt18;
  (*hess)(3,2)=0;
  (*hess)(3,3)=tt18;
  (*hess)(3,4)=0;
  (*hess)(3,5)=0;
  (*hess)(3,6)=0;
  (*hess)(3,7)=0;
  (*hess)(3,8)=0;
  (*hess)(4,0)=tt17;
  (*hess)(4,1)=0;
  (*hess)(4,2)=0;
  (*hess)(4,3)=0;
  (*hess)(4,4)=tt16;
  (*hess)(4,5)=0;
  (*hess)(4,6)=0;
  (*hess)(4,7)=0;
  (*hess)(4,8)=tt17;
  (*hess)(5,0)=0;
  (*hess)(5,1)=0;
  (*hess)(5,2)=0;
  (*hess)(5,3)=0;
  (*hess)(5,4)=0;
  (*hess)(5,5)=tt18;
  (*hess)(5,6)=0;
  (*hess)(5,7)=tt18;
  (*hess)(5,8)=0;
  (*hess)(6,0)=0;
  (*hess)(6,1)=0;
  (*hess)(6,2)=tt18;
  (*hess)(6,3)=0;
  (*hess)(6,4)=0;
  (*hess)(6,5)=0;
  (*hess)(6,6)=tt18;
  (*hess)(6,7)=0;
  (*hess)(6,8)=0;
  (*hess)(7,0)=0;
  (*hess)(7,1)=0;
  (*hess)(7,2)=0;
  (*hess)(7,3)=0;
  (*hess)(7,4)=0;
  (*hess)(7,5)=tt18;
  (*hess)(7,6)=0;
  (*hess)(7,7)=tt18;
  (*hess)(7,8)=0;
  (*hess)(8,0)=tt17;
  (*hess)(8,1)=0;
  (*hess)(8,2)=0;
  (*hess)(8,3)=0;
  (*hess)(8,4)=tt17;
  (*hess)(8,5)=0;
  (*hess)(8,6)=0;
  (*hess)(8,7)=0;
  (*hess)(8,8)=tt16;
  return E;
}
void DeformableBody::buildF(const Mat3X4d& c,const Mat3X4d& c0,Mat3d& F,Mat3d& d)
{
  d.col(0)=c0.col(1)-c0.col(0);
  d.col(1)=c0.col(2)-c0.col(0);
  d.col(2)=c0.col(3)-c0.col(0);
  d=d.inverse().eval();
  ASSERT(isFinite(d))

  Mat3d DX=Mat3d::Zero();
  DX.col(0)=c.col(1)-c.col(0);
  DX.col(1)=c.col(2)-c.col(0);
  DX.col(2)=c.col(3)-c.col(0);
  F=DX*d;
}
void DeformableBody::calcGComp3D(Mat9X12d& GComp,const Mat3d& d)
{
  //F=DH*invDM
  GComp.setZero();
  for(sizeType r=0; r<3; r++)
    for(sizeType c=0; c<3; c++)
      for(sizeType i=0; i<3; i++) {
        //DH(r,i)*invDM(i,c)=(x(i+1)(r)-x1(r))*invDM(i,c)
        GComp(c*3+r,(i+1)*3+r)+=d(i,c);
        GComp(c*3+r,r)-=d(i,c);
      }
}
DeformableBody::SMat DeformableBody::buildGC(sizeType nrC,const Coli& cons)
{
  SMat ret;
  STrips trips;
  std::set<sizeType> consSet;
  for(sizeType i=0; i<cons.size(); i++)
    consSet.insert(cons[i]);
  for(sizeType i=0,j=0; i<nrC; i++)
    if(consSet.find(i) == consSet.end())
      addBlockId(trips,i*3,(j++)*3,3,1);
  ret.resize(nrC*3,nrC*3-consSet.size()*3);
  ret.setFromTriplets(trips.begin(),trips.end());
  return ret;
}
