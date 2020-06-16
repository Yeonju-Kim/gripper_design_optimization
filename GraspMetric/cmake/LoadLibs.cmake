#LIBRARIES
SET(USE_QUADMATH OFF)
SET(USE_CGAL OFF)
SET(USE_QHULL ON)
SET(USE_MOSEK OFF)
SET(USE_SUITE_SPARSE OFF)
SET(USE_SCS OFF)

IF(USE_QUADMATH)
  #QUADMATH
  FIND_PACKAGE(QUADMATH QUIET)
  IF(QUADMATH_FOUND)
    LIST(APPEND ALL_LIBRARIES ${QUADMATH_LIBRARIES})
    MESSAGE(STATUS "Found QUADMATH @ ${QUADMATH_LIBRARIES}")
    IF(USE_QUADMATH)
      ADD_DEFINITIONS(-DQUADMATH_SUPPORT)
    ENDIF()
    ADD_DEFINITIONS(-DFOUND_QUADMATH)
  ELSE(QUADMATH_FOUND)
    MESSAGE(WARNING "Cannot find QUADMATH, compiling without it!")
  ENDIF(QUADMATH_FOUND)
ENDIF()

#EIGEN3
IF(DEFINED ENV{EIGEN3_INCLUDE_DIR})
  MESSAGE(STATUS "Found Custom EIGEN3 @ $ENV{EIGEN3_INCLUDE_DIR}")
  INCLUDE_DIRECTORIES($ENV{EIGEN3_INCLUDE_DIR})
ELSE()
  FIND_PACKAGE(Eigen3 QUIET REQUIRED)
  IF(EIGEN3_FOUND)
    MESSAGE(STATUS "Found EIGEN3 @ ${EIGEN3_INCLUDE_DIR}")
  ELSE(EIGEN3_FOUND)
    MESSAGE(SEND_ERROR "Cannot find EIGEN3!")
  ENDIF(EIGEN3_FOUND)
  INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
ENDIF()

#TinyXML2
FIND_PACKAGE(TINYXML2 QUIET)
IF(TINYXML2_FOUND)
  INCLUDE_DIRECTORIES(${TINYXML2_INCLUDE_DIRS})
  MESSAGE(STATUS "Found TinyXML2 @ ${TINYXML2_INCLUDE_DIRS}")
  LIST(APPEND ALL_LIBRARIES ${TINYXML2_LIBRARIES})
ELSE(TINYXML2_FOUND)
  MESSAGE(SEND_ERROR "Cannot find TinyXML2!")
ENDIF(TINYXML2_FOUND)

IF(USE_CGAL)
  #CGAL
  FIND_PACKAGE(CGAL QUIET)
  IF(CGAL_FOUND)
    MESSAGE(STATUS "Found CGAL @ ${CGAL_INCLUDE_DIR}")
    INCLUDE_DIRECTORIES(${CGAL_INCLUDE_DIR})
    LIST(APPEND ALL_LIBRARIES ${CGAL_LIBRARIES})
    ADD_DEFINITIONS(-DCGAL_SUPPORT)
  ELSE(CGAL_FOUND)
    MESSAGE(WARNING "Cannot find CGAL, compiling without it!")
  ENDIF(CGAL_FOUND)

  #GMP
  FIND_PACKAGE(GMP QUIET)
  IF(GMP_FOUND)
    MESSAGE(STATUS "Found GMP @ ${GMP_INCLUDES}")
    INCLUDE_DIRECTORIES(${GMP_INCLUDES})
    LIST(APPEND ALL_LIBRARIES ${GMP_LIBRARIES})
  ELSE(GMP_FOUND)
    MESSAGE(SEND_ERROR "Cannot find GMP!")
  ENDIF(GMP_FOUND)

  #FIND MPFR
  FIND_PACKAGE(MPFR QUIET REQUIRED)
  IF(MPFR_FOUND)
    INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/ThirdParty")
    MESSAGE(STATUS "Found MPFR Libraries @ ${MPFR_LIBRARIES}")
    MESSAGE(STATUS "Found MPFR @ ${MPFR_INCLUDES}")
    LIST(APPEND ALL_LIBRARIES ${MPFR_LIBRARIES})
  ELSEIF(MPFR_FOUND)
    MESSAGE(SEND_ERROR "Cannot find MPFR!")
  ENDIF(MPFR_FOUND)
ENDIF(USE_CGAL)

IF(USE_QHULL)
  #QHull
  FIND_PACKAGE(QHull QUIET)
  IF(QHULL_FOUND)
    MESSAGE(STATUS "Found QHull @ ${QHULL_INCLUDE_DIRS}")
    INCLUDE_DIRECTORIES(${QHULL_INCLUDE_DIRS})
    LIST(APPEND ALL_STATIC_LIBRARIES ${QHULL_LIBRARIES})
    ADD_DEFINITIONS(-DQHULL_SUPPORT)
  ELSE(QHULL_FOUND)
    MESSAGE(WARNING "Cannot find QHull, compiling without it!")
  ENDIF(QHULL_FOUND)
ENDIF(USE_QHULL)

IF(USE_MOSEK)
  #MOSEK
  MACRO(ADD_MOSEK_PATH)
    LIST(APPEND ALL_LIBRARIES "${MOSEK_ROOT}/bin/libmosek64.so")
    INCLUDE_DIRECTORIES("${MOSEK_ROOT}/src/fusion_cxx")
    INCLUDE_DIRECTORIES("${MOSEK_ROOT}/h")
    FILE(GLOB_RECURSE headerFusion ${MOSEK_ROOT}/src/fusion_cxx/*.h ${MOSEK_ROOT}/src/fusion_cxx/*.hpp ${MOSEK_ROOT}/src/fusion_cxx/*.hh)
    FILE(GLOB_RECURSE sourceFusion ${MOSEK_ROOT}/src/fusion_cxx/*.cpp ${MOSEK_ROOT}/src/fusion_cxx/*.cpp ${MOSEK_ROOT}/src/fusion_cxx/*.cc)
  ENDMACRO()
  SET(USE_MOSEK_9 ON CACHE STRING "Default to use Mosek9")
  IF(EXISTS "/home/$ENV{USER}/mosek/9.0/tools/platform/linux64x86" AND ${USE_MOSEK_9})
    SET(MOSEK_ROOT "/home/$ENV{USER}/mosek/9.0/tools/platform/linux64x86")
    MESSAGE(STATUS "Using mosek 9.0!")
    ADD_DEFINITIONS(-DMOSEK_SUPPORT)
    ADD_DEFINITIONS(-DUSE_MOSEK_9)
    ADD_MOSEK_PATH()
  ELSEIF(EXISTS "/home/$ENV{USER}/mosek/8/tools/platform/linux64x86")
    SET(MOSEK_ROOT "/home/$ENV{USER}/mosek/8/tools/platform/linux64x86")
    MESSAGE(STATUS "Using mosek 8.0!")
    ADD_DEFINITIONS(-DMOSEK_SUPPORT)
    ADD_DEFINITIONS(-DUSE_MOSEK_8)
    ADD_MOSEK_PATH()
  ELSEIF(EXISTS "/nas/longleaf/apps")
    SET(MOSEK_ROOT "/nas/longleaf/apps/mosek/9.0.97/mosek/9.0/tools/platform/linux64x86")
    MESSAGE(STATUS "Using mosek 9.0!")
    ADD_DEFINITIONS(-DMOSEK_SUPPORT)
    ADD_DEFINITIONS(-DUSE_MOSEK_9)
    ADD_MOSEK_PATH()
  ELSE()
    MESSAGE(WARNING "Cannot find mosek, compiling without it!")
  ENDIF()
ENDIF(USE_MOSEK)

IF(USE_SUITE_SPARSE)
  #SuiteSparse
  IF(NOT SUITE_SPARSE_ROOT)
    SET(SUITE_SPARSE_ROOT "${PROJECT_SOURCE_DIR}/../SuiteSparse")
  ENDIF()
  #component Cholmod
  FIND_PACKAGE(Cholmod QUIET REQUIRED)
  IF(CHOLMOD_FOUND)
    INCLUDE_DIRECTORIES(${CHOLMOD_INCLUDE_DIR})
    MESSAGE(STATUS "Found Cholmod @ ${CHOLMOD_INCLUDE_DIR}")
    MESSAGE(STATUS "Found Cholmod Libraries @ ${CHOLMOD_LIBRARIES}")
    LIST(APPEND ALL_LIBRARIES ${CHOLMOD_LIBRARIES})
    ADD_DEFINITIONS(-DCHOLMOD_SUPPORT)
  ELSE(CHOLMOD_FOUND)
    MESSAGE(WARNING "Cannot find Cholmod, compiling without it!")
  ENDIF(CHOLMOD_FOUND)
  #component Umfpack
  FIND_PACKAGE(Umfpack QUIET REQUIRED)
  IF(UMFPACK_FOUND)
    INCLUDE_DIRECTORIES(${UMFPACK_INCLUDES})
    MESSAGE(STATUS "Found Umfpack @ ${UMFPACK_INCLUDES}")
    MESSAGE(STATUS "Found Umfpack Libraries @ ${UMFPACK_LIBRARIES}")
    LIST(APPEND ALL_LIBRARIES ${UMFPACK_LIBRARIES})
    ADD_DEFINITIONS(-DUMFPACK_SUPPORT)
  ELSE(UMFPACK_FOUND)
    MESSAGE(WARNING "Cannot find Umfpack, compiling without it!")
  ENDIF(UMFPACK_FOUND)
ENDIF(USE_SUITE_SPARSE)

IF(USE_SCS)
  #SCS
  IF(NOT EXISTS ${PROJECT_BINARY_DIR}/scs-master)
    EXECUTE_PROCESS(COMMAND cp -rf scs-master ${PROJECT_BINARY_DIR} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    EXECUTE_PROCESS(COMMAND make DESTDIR=. install WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/scs-master)
  ENDIF()
  INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR}/scs-master/include)
  INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR}/scs-master/linsys)
  LIST(APPEND ALL_STATIC_LIBRARIES ${PROJECT_BINARY_DIR}/scs-master/out/libscsdir.a)
  ADD_DEFINITIONS(-DSCS_SUPPORT)

  #BLAS/LAPACK
  FIND_PACKAGE(BLAS QUIET)
  FIND_PACKAGE(LAPACK QUIET)
  IF(BLAS_FOUND AND LAPACK_FOUND)
    MESSAGE(STATUS "Found BLAS/LAPACK @ ${BLAS_LIBRARIES};${LAPACK_LIBRARIES}")
    LIST(APPEND ALL_LIBRARIES ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
  ELSE(BLAS_FOUND AND LAPACK_FOUND)
    MESSAGE(SEND_ERROR "Cannot find BLAS/LAPACK!")
  ENDIF(BLAS_FOUND AND LAPACK_FOUND)
ENDIF(USE_SCS)
