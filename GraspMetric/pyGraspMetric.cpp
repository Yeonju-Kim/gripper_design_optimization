#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <GraspMetric/pyGraspMetric.h>
#include <sstream>

USE_PRJ_NAMESPACE

namespace py=pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T,std::shared_ptr<T>);
#define PYTHON_FUNC(C,NAME) .def(STRINGIFY_OMP(NAME),&C::NAME)
#define PYTHON_SFUNC(C,NAME) .def_static(STRINGIFY_OMP(NAME),&C::NAME)
#define PYTHON_OFUNC(C,NAME,SIG) .def(STRINGIFY_OMP(NAME),SIG &C::NAME)
#define PYTHON_OSFUNC(C,NAME,SIG) .def_static(STRINGIFY_OMP(NAME),SIG &C::NAME)
#define PYTHON_MEMBER_READ(C,NAME) .def_readonly(STRINGIFY_OMP(NAME),&C::NAME)
#define PYTHON_MEMBER_READWRITE(C,NAME) .def_readwrite(STRINGIFY_OMP(NAME),&C::NAME,&C::NAME)
#define PYTHON_FIELD_READWRITE(C,NAME) .def_readwrite(STRINGIFY_OMP(NAME),&C::NAME)

#define EIGEN_OP_VEC(T) \
.def(py::init<>())  \
.def(py::init([](const T& other) {return T(other);}))  \
.def(py::init([](py::list lst) {T v;v.resize(lst.size());sizeType id=0;for(auto item:lst)v[id++]=item.cast<T::Scalar>();return v;}))  \
.def("__deepcopy__",[](const T& v,py::dict){return T(v);})   \
.def("__add__",[](const T& v,const T& w){return T(v+w);},py::is_operator()) \
.def("__sub__",[](const T& v,const T& w){return T(v-w);},py::is_operator()) \
.def("__mul__",[](const T& v,const T::Scalar& c){return T(v*c);},py::is_operator()) \
.def("__div__",[](const T& v,const T::Scalar& c){return T(v/c);},py::is_operator()) \
.def("fromList",[](T& v,py::list lst){v.resize(lst.size());sizeType id=0;for(auto item:lst)v[id++]=item.cast<T::Scalar>();}) \
.def("toList",[](const T& v){py::list li;for(sizeType i=0;i<v.size();i++)li.append(v[i]);return li;})  \
.def("castFrom",[](T& v,const T& v2){v=v2.template cast<T::Scalar>();}) \
.def("castToDouble",[](const T& v){return v.template cast<double>().eval();}) \
.def("setZero",[](T& v){v.setZero();})  \
.def("setOnes",[](T& v){v.setOnes();})  \
.def("setConstant",[](T& v,T::Scalar val){v.setConstant(val);})  \
.def("resize",[](T& v,sizeType sz){v.resize(sz);})   \
.def("dot",[](const T& v,const T& w){return v.dot(w);}) \
.def("size",[](const T& v){return v.size();})   \
.def("__len__",[](const T& v){return v.size();})   \
.def("__getitem__",[](const T& v,sizeType i){return v[i];}) \
.def("__setitem__",[](T& v,sizeType i,T::Scalar val){v[i]=val;})    \
.def("__repr__",[](const T& v){std::ostringstream os;os<<v.unaryExpr([&](const typename T::Scalar& in) {return (scalar)in;});return os.str();});

#define EIGEN_OP_MAT(T) \
.def(py::init<>())  \
.def(py::init([](const T& other) {return T(other);}))  \
.def(py::init([](py::list lst) {T v;v.resize(lst.size(),lst.begin()->template cast<py::list>().size());sizeType id=0;for(auto item:lst){sizeType id2=0;for(auto item2:item)v(id,id2++)=item2.template cast<T::Scalar>();id++;}return v;}))  \
.def("__deepcopy__",[](const T& v,py::dict){return T(v);})   \
.def("__add__",[](const T& v,const T& w){return T(v+w);},py::is_operator()) \
.def("__sub__",[](const T& v,const T& w){return T(v-w);},py::is_operator()) \
.def("__mul__",[](const T& v,const T::Scalar& c){return T(v*c);},py::is_operator()) \
.def("__div__",[](const T& v,const T::Scalar& c){return T(v/c);},py::is_operator()) \
.def("fromList",[](T& v,py::list lst){v.resize(lst.size(),lst.begin()->template cast<py::list>().size());sizeType id=0;for(auto item:lst){sizeType id2=0;for(auto item2:item)v(id,id2++)=item2.template cast<T::Scalar>();id++;}}) \
.def("toList",[](const T& v){py::list li;for(sizeType i=0;i<v.rows();i++){py::list lic;for(sizeType j=0;j<v.cols();j++)lic.append(v(i,j));li.append(lic);}return li;})  \
.def("castFrom",[](T& v,const T& v2){v=v2.template cast<T::Scalar>();}) \
.def("castToDouble",[](const T& v){return v.template cast<double>().eval();}) \
.def("setZero",[](T& v){v.setZero();})  \
.def("setOnes",[](T& v){v.setOnes();})  \
.def("setConstant",[](T& v,T::Scalar val){v.setConstant(val);})  \
.def("resize",[](T& v,sizeType r,sizeType c){v.resize(r,c);})   \
.def("rows",[](const T& v){return v.rows();})   \
.def("cols",[](const T& v){return v.cols();})   \
.def("__len__",[](const T& v){return v.size();})   \
.def("__getitem__",[](const T& v,sizeType i,sizeType j){return v(i,j);}) \
.def("__getitem__",[](const T& v,py::tuple t){return v(t[0].template cast<sizeType>(),t[1].template cast<sizeType>());}) \
.def("__setitem__",[](T& v,sizeType i,sizeType j,T::Scalar val){v(i,j)=val;})    \
.def("__setitem__",[](T& v,py::tuple t,T::Scalar val){v(t[0].template cast<sizeType>(),t[1].template cast<sizeType>())=val;})    \
.def("__repr__",[](const T& v){std::ostringstream os;os<<v.unaryExpr([&](const typename T::Scalar& in) {return (scalar)in;});return os.str();});

PYBIND11_MODULE(pyGraspMetric,m)
{
  //-----------------------------------------------------------------basic types
  py::class_<Vec3f>(m,"Vec3f")
  .def(py::init<float,float,float>())
  EIGEN_OP_VEC(Vec3f)
  py::class_<Vec3d>(m,"Vec3d")
  .def(py::init<double,double,double>())
  EIGEN_OP_VEC(Vec3d)
  py::class_<Mat6f>(m,"Mat6f")
  EIGEN_OP_MAT(Mat6f)
  py::class_<Mat6d>(m,"Mat6d")
  EIGEN_OP_MAT(Mat6d)
  m.def("Q1",computeQ1);
  m.def("QInf",computeQInf);
  m.def("QMSV",computeQMSV);
  m.def("QVEW",computeQVEW);
  m.def("QG11",computeQG11);
}
