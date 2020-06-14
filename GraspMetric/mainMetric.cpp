#include "GraspMetric/GraspMesh.h"
#include "GraspMetric/Support.h"
#include "GraspMetric/Metric.h"
#include "GraspMetric/Utils.h"
#include "CommonFile/Timing.h"

USE_PRJ_NAMESPACE

double sizeF=1;
bool twoPoint,useSCS;
Eigen::Matrix<double,6,1> V;
template <typename T,sizeType D>
std::string to_string(const Eigen::Matrix<T,D,1>& d,std::string sep="_")
{
  std::string ret;
  for(sizeType i=0; i<d.size(); i++) {
    if(i>0)
      ret+=sep;
    if(d[i]==(int)d[i])
      ret+=std::to_string((int)d[i]);
    else ret+=std::to_string(d[i]);
  }
  return ret;
}
void readDirs(GraspMesh::Dirs& dss,std::string& dssStr,int argc,char** argv)
{
  twoPoint=false;
  useSCS=false;
  for(sizeType i=4; i<argc; i++) {
    if(i>4)
      dssStr+="_";
    if(std::string(argv[i])=="twoPoint") {
      twoPoint=true;
      dssStr+="twoPoint";
    } else if(beginsWith(argv[i],"w")) {
      std::sscanf(argv[i],"w,%lf,%lf,%lf,%lf,%lf,%lf",&V[0],&V[1],&V[2],&V[3],&V[4],&V[5]);
      dssStr+="w,"+to_string<double,6>(V,",");
    } else if(beginsWith(argv[i],"ab")) {
      Eigen::Matrix<double,6,1> df;
      std::sscanf(argv[i],"ab,%lf,%lf,%lf,%lf,%lf,%lf",&df[0],&df[1],&df[2],&df[3],&df[4],&df[5]);
      df.segment<3>(0)-=df.segment<3>(3);
      dssStr+="ab,"+to_string<double,-1>(df,",");
      dss.push_back(df);
    } else if(beginsWith(argv[i],"useSCS")) {
      dssStr+="useSCS";
      useSCS=true;
    } else if(beginsWith(argv[i],"sizeF")) {
      std::sscanf(argv[i],"sizeF,%lf",&sizeF);
      if(sizeF==(int)sizeF)
        dssStr+="sizeF,"+std::to_string((int)sizeF);
      else dssStr+="sizeF,"+std::to_string(sizeF);
    } else {
      Eigen::Matrix<int,6,1> d;
      std::sscanf(argv[i],"%d,%d,%d,%d,%d,%d",&d[0],&d[1],&d[2],&d[3],&d[4],&d[5]);
      dssStr+=to_string<int,-1>(d,",");
      dss.push_back(d.cast<scalarD>());
    }
  }
}
int main(int argc,char** argv)
{
  twoPoint=false;
  V.setOnes();
  ASSERT_MSG(argc>4,"Usage: [main] mesh_dir friction_coef poisson")
  std::string dir=argv[1];
  scalarD fri=atof(argv[2]);
  scalarD poi=atof(argv[3]);
  GraspMesh::Dirs dss;
  std::string dssStr;
  readDirs(dss,dssStr,argc,argv);
  ASSERT_MSGV(fri>0 && fri<1,"Incorrect friction_coef: %f!",fri)
  std::string configuration=std::to_string(fri)+"_"+std::to_string(poi)+"_"+dssStr;
  std::string mesh_name=dir+"/mesh_"+configuration+".FEM";
  INFOV("mesh_dir=%s friction_coef=%f poisson=%f twoPoint=%s",dir.c_str(),fri,poi,twoPoint?"true":"false")
  INFOV("directions: %s",dssStr.c_str())

  //mesh
  std::shared_ptr<GraspMesh> m;
  if(!exists(mesh_name)) {
    m.reset(new GraspMesh(dir,fri,&dss,0));
    if(poi>0 && poi<0.5f)
      m->generateSigmaCoefFEM(sizeF,poi,true,dir);
    m->SerializableBase::write(mesh_name);
  } else {
    m.reset(new GraspMesh);
    m->SerializableBase::read(mesh_name);
  }
  GraspMesh::IDSET idsAll=m->allIdsDss();
  if(poi>0 && poi<0.5f) {
    m->writeNormalHVTK(dir+"/meshH_"+configuration+".vtk",idsAll);
    m->writeNormalDisplacementVTK(dir+"/meshD_"+configuration+".vtk",idsAll);
  }
  m->writeContactPtsVTK(dir+"/contact_"+configuration+".vtk",&idsAll);
  m->writeFrictionConesPtsVTK(dir+"/cone_"+configuration+".vtk",&idsAll);

  if(useSCS)
    Support::_useSCS=true;
  scalarD Q1,timeQ1;
  scalarD QInf,timeQInf;
  scalarD QSM=0,timeQSM=0;  //may not be computed
  scalarD QMSV,timeQMSV;
  scalarD QVEW,timeQVEW;
  scalarD QG11,timeQG11;
  {
    //metric Q1
    std::shared_ptr<Q1Metric> metricQ1(new Q1Metric(m,V.cast<scalarD>().asDiagonal(),1E-3f));
    TBEG("Q1");
    Q1=metricQ1->computeMetric(NULL,NULL,m->allIds(),false,twoPoint?Vec3d(dss[0].segment<3>(0)):Vec3d(Vec3d::Zero()));
    timeQ1=TENDV();
    INFOV("Q1=%f TimeQ1=%f!",Q1,timeQ1)
    metricQ1->printIterationLog();
  }
  {
    //metric QInf
    std::shared_ptr<QInfMetric> metricQInf(new QInfMetric(m,V.cast<scalarD>().asDiagonal(),1E-3f));
    TBEG("QInf");
    QInf=metricQInf->computeMetric(NULL,NULL,m->allIds(),false,twoPoint?Vec3d(dss[0].segment<3>(0)):Vec3d(Vec3d::Zero()));
    timeQInf=TENDV();
    INFOV("QInf=%f TimeQInf=%f!",QInf,timeQInf)
    metricQInf->printIterationLog();
  }
  if(poi>0 && poi<0.5f) {
    //metric QSM
    std::shared_ptr<QSMMetric> metricQSM(new QSMMetric(m,V.cast<scalarD>().asDiagonal(),1E-3f));
    metricQSM->setProgressive();
    metricQSM->simplifyConstraintPoisson(50,false);
    metricQSM->simplifyConstraintRandom(1000,50,true);
    //metricQSM->debugSolver();
    TBEG("QSM");
    QSM=metricQSM->computeMetric(NULL,NULL,m->allIds(),false,twoPoint?Vec3d(dss[0].segment<3>(0)):Vec3d(Vec3d::Zero()));
    timeQSM=TENDV();
    INFOV("QSM=%f TimeQSM=%f!",QSM,timeQSM)
    metricQSM->printIterationLog();
  }
  {
    //metric QMSV
    std::shared_ptr<QMSVMetric> metricQMSV(new QMSVMetric(m));
    TBEG("QMSV");
    QMSV=metricQMSV->computeMetric(m->allIds());
    timeQMSV=TENDV();
    INFOV("QMSV=%f TimeQMSV=%f!",QMSV,timeQMSV)
  }
  {
    //metric QVEW
    std::shared_ptr<QVEWMetric> metricQVEW(new QVEWMetric(m));
    TBEG("QVEW");
    QVEW=metricQVEW->computeMetric(m->allIds());
    timeQVEW=TENDV();
    INFOV("QVEW=%f TimeQVEW=%f!",QVEW,timeQVEW)
  }
  {
    //metric QVEW
    std::shared_ptr<QG11Metric> metricQG11(new QG11Metric(m));
    TBEG("QG11");
    QG11=metricQG11->computeMetric(m->allIds());
    timeQG11=TENDV();
    INFOV("QG11=%f TimeQG11=%f!",QG11,timeQG11)
  }
  std::ofstream os(dir+"/mesh_"+configuration+".Q");
  os << "Q1: " << Q1 << " QInf: " << QInf << " QSM: " << QSM << " QMSV: " << QMSV << " QVEW: " << QVEW << " QG11: " << QG11 << std::endl;
  os << "TimeQ1: " << timeQ1 << " TimeQInf: " << timeQInf << " TimeQSM: " << timeQSM << " TimeQMSV: " << timeQMSV << " TimeQVEW: " << timeQVEW << " TimeQG11: " << timeQG11 << std::endl;
  return 0;
}
