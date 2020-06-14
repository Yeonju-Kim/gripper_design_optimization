#ifndef DEFORMABLE_BODY_H
#define DEFORMABLE_BODY_H

#include "Utils.h"
#include "Objective.h"
#include <CommonFile/ObjMesh.h>

PRJ_BEGIN

class DeformableBody : public Serializable
{
public:
  using Serializable::read;
  using Serializable::write;
  typedef Objective<scalarD>::SMat SMat;
  typedef Objective<scalarD>::STrip STrip;
  typedef Objective<scalarD>::STrips STrips;
  typedef std::vector<Vec3d,Eigen::aligned_allocator<Vec3d> > LineDirs;
  typedef std::vector<LineDirs> LinesDirs;
  class DeformableObjective : public Objective<scalarD>
  {
  public:
    using Objective<scalarD>::operator();
    DeformableObjective(const DeformableBody& defo,const Cold& FLine);
    virtual int operator()(const Vec& x,Vec& fvec,SMat* fjac,bool modifiableOrSameX);
    virtual int inputs() const;
    virtual int values() const;
    const DeformableBody& _defo;
    const Cold& _FLine;
  };
  EIGEN_DEVICE_FUNC DeformableBody();
  void assemble(const tinyxml2::XMLElement& pt);
  void assembleMaterial(const tinyxml2::XMLElement& pt);
  bool read(std::istream& is) override;
  bool write(std::ostream& os) const override;
  std::shared_ptr<SerializableBase> copy() const override;
  void addLine(const Coli& attachment);
  void updateLinesDirs(const Cold& x,LinesDirs& linesDirs) const;
  scalarD buildKM(const Cold& x,Cold* F,SMat* K,SMat* M) const;
  scalarD buildFK(const LinesDirs* linesDirs,const Cold& FLine,const Cold& x,Cold* F,SMat* K) const;
  scalarD buildFKLine(const LineDirs* linesDirs,const Coli& attachment,scalarD f,const Cold& x,Cold* F,STrips* trips) const;
  bool solveNewton(const Cold& FLine,Cold& x,scalarD tolX=1E-8f,scalarD tolG=1E-5f,sizeType maxIter=1E4,bool useCB=true) const;
  Matd buildDeriv(const Cold& FLine,const Cold& x) const;
  Matd buildDeriv(const Cold& x) const;
  const SMat& G() const;
  const SMat& sigma() const;
  const Mat4Xi& tss() const;
  Mat4Xi& tss();
  const Cold& vol() const;
  const Cold& U0() const;
  Cold& U0();
  scalarD minMu() const;
  scalarD maxMu() const;
  sizeType nrNode() const;
  sizeType nrTet() const;
  sizeType nrLine() const;
  const SMat& getGC() const;
  void debugG(scalarD scale) const;
  void debugFK(scalarD scale,sizeType maxIt=10) const;
  void debugDeriv(scalarD scale,sizeType maxIt=10) const;
  //writeVTK
  void getH(const Cold& U,std::vector<scalarD>& H) const;
  void writeVTK(const Cold& U,const std::string& path,const Cold* D=NULL,const std::vector<scalarD>* color=NULL) const;
  void writeLineVTK(const Cold& U,const std::string& path) const;
  Vec3d getVolumeCentroid(const Cold& U) const;
  Mat3Xi getMuRange(const Vec2d& range,Coli* tss=NULL) const;
  ObjMesh getMeshMuRange(const Cold& U,const Vec2d& range) const;
  ObjMesh getMesh(const Cold& U,const Mat3Xi* sss=NULL) const;
  ObjMesh getMesh() const;
  //utility
  static scalarD evalFKNonHK(Eigen::Map<const Mat3d> F,scalarD lambda,scalarD mu,scalarD V,scalarD dimMask,scalarD minVol,Eigen::Map<Vec9d> grad,Mat9d* hess);
  static scalarD evalFLinear(Eigen::Map<const Mat3d> f,scalarD lambda,scalarD mu,scalarD V,scalarD dimMask,Eigen::Map<Vec9d> grad,Mat9d* hess);
  static void buildF(const Mat3X4d& c,const Mat3X4d& c0,Mat3d& F,Mat3d& d);
  static void calcGComp3D(Mat9X12d& GComp,const Mat3d& d);
  static SMat buildGC(sizeType nrC,const Coli& cons);
protected:
  SMat _G,_sigma,_GC;
  Cold _U0;
  Mat3Xi _sss;
  Mat4Xi _tss;
  Mat3Xd _invDss;
  std::vector<Coli,Eigen::aligned_allocator<Coli> > _lines;
  //material
  Cold _mu,_lambda,_vol;
  bool _isLinear;
};

PRJ_END

#endif
