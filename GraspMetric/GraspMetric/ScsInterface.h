#ifndef SCS_INTERFACE_H
#define SCS_INTERFACE_H
#ifdef SCS_SUPPORT

#include "CommonFile/MathBasic.h"
#include "Objective.h"
#include <memory>
#undef EPS
#include "scs.h"

PRJ_BEGIN

class ScsInterface
{
public:
  typedef Objective<scalarD>::Vec Vec;
  typedef Objective<scalarD>::SMat SMat;
  typedef Objective<scalarD>::STrip STrip;
  typedef Objective<scalarD>::STrips STrips;
  enum CONE_TYPE
  {
    EQUALITY,
    INEQUALITY,
    SECOND_ORDER,
    SEMI_DEFINITE,
  };
  struct CONE
  {
    bool operator==(const CONE& other) const {
      return _cid==other._cid;
    }
    bool operator<(const CONE& other) const {
      return _cid<other._cid;
    }
    sizeType _cid;
    SMat _A;
    Vec _b;
  };
  typedef std::map<CONE_TYPE,std::vector<std::shared_ptr<CONE>>> CONES;
  typedef std::map<sizeType,std::shared_ptr<CONE>> CONE_INDICES;
  ScsInterface();
  //b-A*x \in K
  void add(sizeType cid,const Vec& b);
  sizeType constraintLinear(const Matd& coef,sizeType off,const Vec& b,CONE_TYPE type);
  sizeType constraintLinear(const Vec& coef,sizeType off,scalarD b,CONE_TYPE type);
  sizeType constraintSecondOrder(const Matd& coef,sizeType off,const Vec& b,CONE_TYPE type);
  sizeType constraintSecondOrder(const SMat& coef,sizeType off,const Vec& b,CONE_TYPE type);
  sizeType constraintLinearMatrixInequality(const Matd& coef,sizeType off,const Matd& c0,CONE_TYPE type);
  sizeType constraint(const SMat& A,const Vec& b,CONE_TYPE type);
  sizeType solve(const Vec& c,Vec& x,bool report=false);
  void clearCones();
protected:
  void getCones(SMat& A,Vec& b,sizeType n);
  static Vec toLower(Matd m);
  sizeType SCRows(sizeType entries) const;
  CONE_INDICES _coneIndices;
  CONES _cones;
  ScsCone* _k;
  ScsData* _d;
  ScsSolution* _sol;
  ScsInfo _info;
  sizeType _cid;
};

PRJ_END

#endif
#endif
