#include "MosekInterface.h"
#ifdef MOSEK_SUPPORT

USE_PRJ_NAMESPACE

//solver
bool MosekInterface::trySolve(ModelM& m,std::string& str)
{
  //solve
  m.setLogHandler([&](const std::string& msg) {
    str+=msg;
  });
  m.solve();
  //test result
  return m.getPrimalSolutionStatus()==mosek::fusion::SolutionStatus::Optimal;
}
bool MosekInterface::trySolve(ModelM& m)
{
  std::string str;
  return trySolve(m,str);
}
void MosekInterface::ensureSolve(ModelM& m,std::string& str)
{
  if(!trySolve(m,str)) {
    m.dispose();
    std::cout << str << std::endl;
    exit(EXIT_FAILURE);
  }
}
void MosekInterface::ensureSolve(ModelM& m)
{
  std::string str;
  ensureSolve(m,str);
}
//to Mosek matrix
std::shared_ptr<monty::ndarray<double,1>> MosekInterface::toMosek(const Vec& v)
{
  Eigen::Matrix<double,-1,1> vd=v.cast<double>();
  return toMosek(vd.data(),vd.size());
}
std::shared_ptr<monty::ndarray<double,1>> MosekInterface::toMosek(std::vector<double>& vals)
{
  return toMosek(&vals[0],vals.size());
}
std::shared_ptr<monty::ndarray<double,1>> MosekInterface::toMosek(double* dat,sizeType sz)
{
  return std::shared_ptr<monty::ndarray<double,1>>(new monty::ndarray<double,1>(dat,monty::shape_t<1>(sz)));
}
std::shared_ptr<monty::ndarray<double,2>> MosekInterface::toMosek(double* dat,sizeType szr,sizeType szc)
{
  return std::shared_ptr<monty::ndarray<double,2>>(new monty::ndarray<double,2>(dat,monty::shape_t<2>(szr,szc)));
}
monty::rc_ptr<MosekInterface::MatrixM> MosekInterface::toMosek(const Matd& m)
{
  Eigen::Matrix<double,-1,-1,Eigen::RowMajor> mD=m.cast<double>();
  return MatrixM::dense(toMosek((double*)(mD.data()),m.rows(),m.cols()));
}
monty::rc_ptr<MosekInterface::MatrixM> MosekInterface::toMosek(const SMat& m)
{
  std::vector<int> rows,cols;
  std::vector<double> vals;
  for(sizeType k=0; k<m.outerSize(); ++k)
    for(Objective<scalarD>::SMat::InnerIterator it(m,k); it; ++it) {
      rows.push_back(it.row());
      cols.push_back(it.col());
      vals.push_back(it.value());
    }
  return toMosek(m.rows(),m.cols(),rows,cols,vals);
}
monty::rc_ptr<MosekInterface::MatrixM> MosekInterface::toMosek(int nrR,int nrC,std::vector<int>& rows,std::vector<int>& cols,std::vector<double>& vals)
{
  return toMosek(nrR,nrC,(int)rows.size(),&rows[0],&cols[0],&vals[0]);
}
monty::rc_ptr<MosekInterface::MatrixM> MosekInterface::toMosek(int nrR,int nrC,int nrE,int* rows,int* cols,double* vals)
{
  std::shared_ptr<monty::ndarray<int,1>> ndarrRows(new monty::ndarray<int,1>(rows,monty::shape_t<1>(nrE)));
  std::shared_ptr<monty::ndarray<int,1>> ndarrCols(new monty::ndarray<int,1>(cols,monty::shape_t<1>(nrE)));
  std::shared_ptr<monty::ndarray<double,1>> ndarrVals(new monty::ndarray<double,1>(vals,monty::shape_t<1>(nrE)));
  return MatrixM::sparse(nrR,nrC,ndarrRows,ndarrCols,ndarrVals);
}
void MosekInterface::debugToMosek()
{
  Mat3X4d m=Mat3X4d::Random();
  std::cout << m << std::endl << std::endl;

  monty::rc_ptr<MosekInterface::MatrixM> mM=toMosek(Matd(m));
  for(sizeType r=0; r<mM->numRows(); r++) {
    for(sizeType c=0; c<mM->numColumns(); c++)
      std::cout << mM->get(r,c) << " ";
    std::cout << std::endl;
  }
}
//to mosek expression
monty::rc_ptr<MosekInterface::ExpressionM> MosekInterface::toMosekE(const Vec& v)
{
  Eigen::Matrix<double,-1,1> vd=v.cast<double>();
  return toMosekE(vd.data(),vd.size());
}
monty::rc_ptr<MosekInterface::ExpressionM> MosekInterface::toMosekE(std::vector<double>& vals)
{
  return toMosekE(&vals[0],vals.size());
}
monty::rc_ptr<MosekInterface::ExpressionM> MosekInterface::toMosekE(double* dat,sizeType sz)
{
  std::shared_ptr<monty::ndarray<double,1>> ndarr(new monty::ndarray<double,1>(dat,monty::shape_t<1>(sz)));
  return ExprM::constTerm(ndarr);
}
//to eigen matrix
MosekInterface::Vec MosekInterface::fromMosek(VariableM& v)
{
#ifdef USE_MOSEK_9
  Vec ret=Vec::Zero(v.getSize());
#endif
#ifdef USE_MOSEK_8
  Vec ret=Vec::Zero(v.size());
#endif
  for(sizeType i=0; i<ret.size(); i++)
    ret[i]=v.level()->operator[](i);
  return ret;
}
//initial guess
void MosekInterface::toMosek(VariableM& v,const Vec& vLevel)
{
  Eigen::Matrix<double,-1,1> vLevelD=vLevel.template cast<double>();
  std::shared_ptr<monty::ndarray<double,1>> ndarr(new monty::ndarray<double,1>(vLevelD.data(),monty::shape_t<1>(vLevelD.size())));
  v.setLevel(ndarr);
}
//index
monty::rc_ptr<MosekInterface::ExpressionM> MosekInterface::index(monty::rc_ptr<VariableM> v,const std::vector<sizeType>& ids)
{
  monty::rc_ptr<ExpressionM> ret;
  for(sizeType i=1; i<(sizeType)ids.size(); i++)
    if(i==1)
      ret=ExprM::vstack(v->index(ids[0]),v->index(ids[1]));
    else ret=ExprM::vstack(ret,v->index(ids[i]));
  return ret;
}
//linear constraint
void MosekInterface::addCI(ModelM& m,const SMat& CI,const Vec& CI0,const Coli& CIType,monty::rc_ptr<VariableM> v)
{
  std::vector<std::vector<int>> rowss(CI.rows()),colss(CI.rows());
  std::vector<std::vector<double>> valss(CI.rows());
  for(sizeType k=0; k<CI.outerSize(); ++k)
    for(Objective<scalarD>::SMat::InnerIterator it(CI,k); it; ++it) {
      rowss[it.row()].push_back(0);
      colss[it.row()].push_back(it.col());
      valss[it.row()].push_back(it.value());
    }
  for(sizeType i=0; i<CI.rows(); i++) {
    std::string name="CI"+std::to_string(i);
    monty::rc_ptr<MatrixM> CIM=toMosek(1,CI.cols(),rowss[i],colss[i],valss[i]);
    monty::rc_ptr<ExpressionM> LHS=ExprM::add(ExprM::mul(CIM,v),CI0[i]);
    if(CIType[i]==0)
      m.constraint(name,LHS,DomainM::equalsTo(0));
    else if(CIType[i]>0)
      m.constraint(name,LHS,DomainM::greaterThan(0));
    else m.constraint(name,LHS,DomainM::lessThan(0));
  }
}

#endif
