#ifndef MOSEK_INTERFACE_H
#define MOSEK_INTERFACE_H

#ifdef MOSEK_SUPPORT
#include "Objective.h"
#include <functional>
#include <fusion.h>

PRJ_BEGIN

//Mosek
#define DECL_MOSEK_TYPES    \
typedef mosek::fusion::Expr ExprM;  \
typedef mosek::fusion::Model ModelM;  \
typedef mosek::fusion::Matrix MatrixM;  \
typedef mosek::fusion::Domain DomainM;  \
typedef mosek::fusion::Variable VariableM;  \
typedef mosek::fusion::Constraint ConstraintM;  \
typedef mosek::fusion::Expression ExpressionM;  \
typedef mosek::fusion::ObjectiveSense ObjectiveSenseM;
struct MosekInterface : public Objective<scalarD>
{
  DECL_MOSEK_TYPES
  //solver
  static bool trySolve(ModelM& m,std::string& str);
  static bool trySolve(ModelM& m);
  static void ensureSolve(ModelM& m,std::string& str);
  static void ensureSolve(ModelM& m);
  //to mosek matrix
  static std::shared_ptr<monty::ndarray<double,1>> toMosek(const Vec& v);
  static std::shared_ptr<monty::ndarray<double,1>> toMosek(std::vector<double>& vals);
  static std::shared_ptr<monty::ndarray<double,1>> toMosek(double* dat,sizeType sz);
  static std::shared_ptr<monty::ndarray<double,2>> toMosek(double* dat,sizeType szr,sizeType szc);
  static monty::rc_ptr<MatrixM> toMosek(const Matd& m);
  static monty::rc_ptr<MatrixM> toMosek(const SMat& m);
  static monty::rc_ptr<MatrixM> toMosek(int nrR,int nrC,std::vector<int>& rows,std::vector<int>& cols,std::vector<double>& vals);
  static monty::rc_ptr<MatrixM> toMosek(int nrR,int nrC,int nrE,int* rows,int* cols,double* vals);
  static void debugToMosek();
  //to mosek expression
  static monty::rc_ptr<ExpressionM> toMosekE(const Vec& v);
  static monty::rc_ptr<ExpressionM> toMosekE(std::vector<double>& vals);
  static monty::rc_ptr<ExpressionM> toMosekE(double* dat,sizeType sz);
  //to eigen matrix
  static Vec fromMosek(VariableM& v);
  //initial guess
  static void toMosek(VariableM& v,const Vec& vLevel);
  //index
  static monty::rc_ptr<ExpressionM> index(monty::rc_ptr<VariableM> v,const std::vector<sizeType>& ids);
  //linear constraint
  static void addCI(ModelM& m,const SMat& CI,const Vec& CI0,const Coli& CIType,monty::rc_ptr<VariableM> v);
};

PRJ_END
#endif

#endif
