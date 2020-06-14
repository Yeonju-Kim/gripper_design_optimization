#ifndef DEBUG_GRADIENT_H
#define DEBUG_GRADIENT_H

#include "CommonFile/Config.h"

//#define BB_DEBUG_MODE
#ifdef BB_DEBUG_MODE
struct CheckVar {
  CheckVar();
  static void report();
  bool BB_DEBUG_PARENT_CHECK;
  bool BB_DEBUG_FEASIBILITY_CHECK;
  bool BB_DEBUG_NO_WARMSTART_CHECK;
  bool BB_DEBUG_INTERNAL_WARMSTART_CHECK;
  bool BB_DEBUG_LEAF_WARMSTART_CHECK;
  static CheckVar _instance;
};
#endif

//numeric delta
#define DEFINE_NUMERIC_DELTA_T(T) \
T DELTA=0;  \
if(sizeof(T)==4)    \
  DELTA=1E-5f;  \
else if(sizeof(T)==8)   \
  DELTA=1E-9;   \
else {  \
  ASSERT(sizeof(T)==16) \
  DELTA=1E-15;  \
}
#define DEFINE_NUMERIC_DELTA DEFINE_NUMERIC_DELTA_T(scalar)

//gradient debug
#define DEBUG_GRADIENT(NAME,A,B) \
if(std::abs(B) > std::sqrt(DELTA)) { \
  std::cout << NAME << ": " << A << " Err: " << B << std::endl; \
} else {  \
  std::cout << NAME << ": " << A << " Err: " << B << std::endl;  \
}
#define DEBUG_GRADIENT_REL(NAME,A,B) \
if(std::abs(B) > std::sqrt(DELTA)*std::abs(A)) { \
  std::cout << NAME << ": " << A << " Err: " << B << std::endl; \
} else {  \
  std::cout << NAME << ": " << A << " Err: " << B << std::endl;  \
}

#endif
