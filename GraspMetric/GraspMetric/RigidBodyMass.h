#ifndef RIGID_BODY_MASS_H
#define RIGID_BODY_MASS_H

#include <CommonFile/ObjMesh.h>

PRJ_BEGIN

template <typename T>
struct RigidBodyMass {
  typedef typename ObjMeshTraits<T>::Type OBJMESH;
  typedef typename ScalarUtil<T>::ScalarVec3 VEC3;
  typedef typename ScalarUtil<T>::ScalarMat6 MAT6;
  typedef typename ScalarUtil<T>::ScalarMat3 MAT3;
  RigidBodyMass(const OBJMESH& mesh);
  VEC3 getCtr() const;
  MAT6 getMass() const;
  MAT6 getMassCOM() const;
  T getM() const;
  VEC3 getMC() const;
  MAT3 getMCCT() const;
private:
  MAT6 _mat,_matCOM;
  VEC3 _ctr;
  MAT3 _MCCT;
};

PRJ_END

#endif
