from compile_objects import auto_download
from compile_gripper import Link,Gripper
from controller import Controller
from compile_world import World
import mujoco_py as mjc
import math,numpy as np
import scipy,random

class Metric:
    #Note that all metrics are supposed to be maximized
    def __init__(self,OBJECT_DEPENDENT):
        self.OBJECT_DEPENDENT=OBJECT_DEPENDENT
    
    def __reduce__(self):
        return (self.__class__,(self.OBJECT_DEPENDENT,))
    
    def compute(self):
        raise RuntimeError('This is abstract super-class, use sub-class!')

class MassMetric(Metric):
    #this is the mass of the gripper
    def __init__(self):
        Metric.__init__(self,False)
    
    def __reduce__(self):
        return (self.__class__,())
    
    def compute(self,controller):
        if isinstance(controller,Controller):
            self.sim=controller.sim
        else: self.sim=controller
        
        model=self.sim.model
        bid=model.body_names.index('base')
        mass=model.body_subtreemass[bid]
        return 1./mass

class SizeMetric(Metric):
    #this is the surface area of the bounding box
    def __init__(self):
        Metric.__init__(self,False)
    
    def __reduce__(self):
        return (self.__class__,())
    
    def compute(self,controller):
        if isinstance(controller,Controller):
            self.sim=controller.sim
        else: self.sim=controller
        
        #compute bounding box
        vmin=[ 1000., 1000., 1000.]
        vmax=[-1000.,-1000.,-1000.]
        model=self.sim.model
        data=self.sim.data
        self.sim.forward()
        bids=self.body_names('base')
        for bid in bids:
            geom_adr=model.body_geomadr[bid]
            geom_num=model.body_geomnum[bid]
            for gid in range(geom_adr,geom_adr+geom_num):
                xpos=data.geom_xpos[gid,:]
                xori=data.geom_xmat[gid,:]
                mid=model.geom_dataid[gid]
                if mid<0:
                    continue
                vid0=model.mesh_vertadr[mid]
                vid1=vid0+model.mesh_vertnum[mid]
                for vid in range(vid0,vid1):
                    v=xori.reshape((3,3))@(model.mesh_vert[vid])+xpos
                    for d in range(3):
                        vmin[d]=min(v[d],vmin[d])
                        vmax[d]=max(v[d],vmax[d])
        #print(vmin,vmax)
        
        #compute surface area
        surface_area=0
        for d in range(3):
            ext=[]
            for d2 in range(3):
                if d2!=d:
                    ext.append(vmax[d2]-vmin[d2])
            surface_area+=ext[0]*ext[1]*2
        return 1./surface_area
        
    def body_names(self,rootName):
        bids=[]
        model=self.sim.model
        for i in range(model.nbody):
            if model.body_names[i]==rootName:
                bids.append(i)
            else:
                bid=i
                while bid>0:
                    bid=model.body_parentid[bid]
                    if model.body_names[bid]==rootName:
                        bids.append(i)
                        break
        return bids

class Q1Metric(Metric):
    USE_NATIVE_CPP=True
    #this is the grasp quality measured after close
    def __init__(self,FRICTION=0.7,metric=[1.]*6):
        Metric.__init__(self,True)
        self.FRICTION=FRICTION
        self.metric=metric
    
    def __reduce__(self):
        return (self.__class__,(self.FRICTION,))
    
    def compute(self,controller,callback=False):
        try:
            if not Q1Metric.USE_NATIVE_CPP:
                raise RuntimeError("Use python not cpp!")
            import pyGraspMetric as gm
            mMatrix=gm.Mat6d()
            mMatrix.setZero()
            for d in range(6):
                mMatrix[d,d]=self.metric[d]
                
            contact_poses=[gm.Vec3d(cp[0],cp[1],cp[2]) for cp in controller.contact_poses]
            contact_normals=[gm.Vec3d(cn[0],cn[1],cn[2]) for cn in controller.contact_normals]
            return gm.Q1(self.FRICTION,contact_poses,contact_normals,mMatrix,callback)
        except:
            return self.compute_python(controller,callback)
      
    def compute_python(self,controller,callback):
        #compute G
        nf=len(controller.contact_normals)
        G=np.zeros((nf,6,3),dtype=np.float64)
        pss=np.array(controller.contact_poses)
        nss=np.array(controller.contact_normals)
        for d in range(3):
            row,col=(d+2)%3,(d+1)%3
            G[:,d,d]=-1
            G[:,3+row,col]=-pss[:,d]
            G[:,3+col,row]= pss[:,d]
        #metric scaling
        metric=np.identity(6,dtype=np.float64)
        for d in range(6):
            metric[d,d]=self.metric[d]
        metricSqrt=scipy.linalg.sqrtm(metric)
        #force-closedness
        pss=[]
        for d in range(6):
            for val in [1.,-1.]:
                D=[val if i==d else 0. for i in range(6)]
                sp,w=self.support(D,metricSqrt,G,nss)
                if sp>0:
                    pss.append(w.tolist())
                else: return 0.
        #convex hull
        iter=0
        eps=1e-3
        hull=scipy.spatial.ConvexHull(np.array(pss),incremental=True)   
        while True:
            D,sp=self.blocking(hull)
            sp2,w=self.support(np.array(D),metricSqrt,G,nss)
            if callback:
                print("Iter %d: Q=%f!"%(iter,sp))
            if sp2-sp>eps*abs(sp):
                hull.add_points(np.array([w.tolist()]))
            else: return sp
            iter+=1
        
    def blocking(self,hull):
        D,sp=(None,None)
        for i in range(hull.equations.shape[0]):
            coef=1./np.linalg.norm(hull.equations[i,0:6])
            spi=-hull.equations[i,6]*coef
            if sp is None or spi<sp:
                D=hull.equations[i,0:6]
                sp=spi
        return D.tolist(),sp
        
    def support(self,d,metricSqrt,G,nss):
        ret=0.
        wOut=np.zeros((6,))
        dm=metricSqrt@d
        for i in range(G.shape[0]):
            n=nss[i,:]
            dmG=G[i,:,:].T@dm
            wPerp=dmG.dot(n)
            dmGn=wPerp*n
            dmGt=dmG-wPerp*n
            wPara=np.linalg.norm(dmGt)
            val=wPerp+self.FRICTION*wPara
            if val>ret:
                ret=val
                wOut=G[i,:,:]@(n+dmGt*self.FRICTION/max(wPara,1e-6))
        return ret,metricSqrt@wOut 

class QInfMetric(Q1Metric):
    #this is the grasp quality measured after close
    def __init__(self,FRICTION=0.7,metric=[1.]*6):
        Q1Metric.__init__(self,FRICTION,metric)
    
    def __reduce__(self):
        return (self.__class__,(self.FRICTION,))
    
    def compute(self,controller,callback=False):
        try:
            if not Q1Metric.USE_NATIVE_CPP:
                raise RuntimeError("Use python not cpp!")
            import pyGraspMetric as gm
            mMatrix=gm.Mat6d()
            mMatrix.setZero()
            for d in range(6):
                mMatrix[d,d]=self.metric[d]
                
            contact_poses=[gm.Vec3d(cp[0],cp[1],cp[2]) for cp in controller.contact_poses]
            contact_normals=[gm.Vec3d(cn[0],cn[1],cn[2]) for cn in controller.contact_normals]
            return gm.QInf(self.FRICTION,contact_poses,contact_normals,mMatrix,callback)
        except:
            return self.compute_python(controller,callback)
        
    def support(self,d,metricSqrt,G,nss):
        ret=0.
        wOut=np.zeros((6,))
        dm=metricSqrt@d
        for i in range(G.shape[0]):
            n=nss[i,:]
            dmG=G[i,:,:].T@dm
            wPerp=dmG.dot(n)
            dmGn=wPerp*n
            dmGt=dmG-wPerp*n
            wPara=np.linalg.norm(dmGt)
            val=wPerp+self.FRICTION*wPara
            if val>0.:
                ret+=val
                wOut+=G[i,:,:]@(n+dmGt*self.FRICTION/max(wPara,1e-6))
        return ret,metricSqrt@wOut 

class QMSVMetric(Q1Metric):
    #this is the grasp quality measured after close
    def __init__(self):
        Q1Metric.__init__(self)
    
    def __reduce__(self):
        return (self.__class__,(self.FRICTION,))
    
    def compute(self,controller,callback=False):
        try:
            if not Q1Metric.USE_NATIVE_CPP:
                raise RuntimeError("Use python not cpp!")
            import pyGraspMetric as gm
            mMatrix=gm.Mat6d()
            mMatrix.setZero()
            for d in range(6):
                mMatrix[d,d]=1.0
                
            contact_poses=[gm.Vec3d(cp[0],cp[1],cp[2]) for cp in controller.contact_poses]
            contact_normals=[gm.Vec3d(cn[0],cn[1],cn[2]) for cn in controller.contact_normals]
            return gm.QMSV(self.FRICTION,contact_poses,contact_normals)
        except:
            nf=len(controller.contact_normals)
            G=np.zeros((nf,6,3),dtype=np.float64)
            pss=np.array(controller.contact_poses)
            for d in range(3):
                row,col=(d+2)%3,(d+1)%3
                G[:,d,d]=1
                G[:,3+row,col]= pss[:,d]
                G[:,3+col,row]=-pss[:,d]
            GGT=np.sum(G@np.swapaxes(G,1,2),axis=0)
            eigs,_=np.linalg.eig(GGT)
            return eigs.min()

class QVEWMetric(Q1Metric):
    #this is the grasp quality measured after close
    def __init__(self):
        Q1Metric.__init__(self)
    
    def __reduce__(self):
        return (self.__class__,(self.FRICTION,))
    
    def compute(self,controller,callback=False):
        try:
            if not Q1Metric.USE_NATIVE_CPP:
                raise RuntimeError("Use python not cpp!")
            import pyGraspMetric as gm
            mMatrix=gm.Mat6d()
            mMatrix.setZero()
            for d in range(6):
                mMatrix[d,d]=1.0
                
            contact_poses=[gm.Vec3d(cp[0],cp[1],cp[2]) for cp in controller.contact_poses]
            contact_normals=[gm.Vec3d(cn[0],cn[1],cn[2]) for cn in controller.contact_normals]
            return gm.QVEW(self.FRICTION,contact_poses,contact_normals)
        except:
            nf=len(controller.contact_normals)
            G=np.zeros((nf,6,3),dtype=np.float64)
            pss=np.array(controller.contact_poses)
            for d in range(3):
                row,col=(d+2)%3,(d+1)%3
                G[:,d,d]=1
                G[:,3+row,col]= pss[:,d]
                G[:,3+col,row]=-pss[:,d]
            GGT=np.sum(G@np.swapaxes(G,1,2),axis=0)
            return math.sqrt(np.linalg.det(GGT))

class QG11Metric(Q1Metric):
    #this is the grasp quality measured after close
    def __init__(self):
        Q1Metric.__init__(self)
    
    def __reduce__(self):
        return (self.__class__,(self.FRICTION,))
    
    def compute(self,controller,callback=False):
        try:
            if not Q1Metric.USE_NATIVE_CPP:
                raise RuntimeError("Use python not cpp!")
            import pyGraspMetric as gm
            mMatrix=gm.Mat6d()
            mMatrix.setZero()
            for d in range(6):
                mMatrix[d,d]=1.0
                
            contact_poses=[gm.Vec3d(cp[0],cp[1],cp[2]) for cp in controller.contact_poses]
            contact_normals=[gm.Vec3d(cn[0],cn[1],cn[2]) for cn in controller.contact_normals]
            return gm.QG11(self.FRICTION,contact_poses,contact_normals)
        except:
            nf=len(controller.contact_normals)
            G=np.zeros((nf,6,3),dtype=np.float64)
            pss=np.array(controller.contact_poses)
            for d in range(3):
                row,col=(d+2)%3,(d+1)%3
                G[:,d,d]=1
                G[:,3+row,col]= pss[:,d]
                G[:,3+col,row]=-pss[:,d]
            GGT=np.sum(G@np.swapaxes(G,1,2),axis=0)
            eigs,_=np.linalg.eig(GGT)
            return eigs.min()/max(eigs.max(),1.e-9)

class LiftMetric(Metric):
    #this metric measures whether the gripper can close, and then lift, and finally shake
    def __init__(self):
        Metric.__init__(self,True)
    
    def __reduce__(self):
        return (self.__class__,())
    
    def compute(self,controller):
        score=0.0
        if controller.closed:
            score+=1
        if controller.lifted:
            score+=1
        if controller.shaked:
            score+=1
        return score

class ElapsedMetric(Metric):
    #this metric measures how much time can the current gripper grasp the object
    def __init__(self):
        Metric.__init__(self,True)
    
    def __reduce__(self):
        return (self.__class__,())
    
    def compute(self,controller):
        dt=controller.sim.model.opt.timestep
        return controller.elapsed*dt

if __name__=='__main__':
    #create gripper
    gripper=Gripper()
    design={'base_off':0.2,'finger_length':0.15,'finger_width':0.3,'finger_curvature':4.,'num_finger':3,'hinge_rad':0.025}
    policy={'id':0,'angle':[0.,math.pi*0.95/2,math.pi/2]}
    link=gripper.get_robot(**design)

    #create world    
    world=World()
    from dataset_cup import get_dataset_cup
    world.compile_simulator(objects=get_dataset_cup(True),link=link)
    world.test_object(0)
    
    #create controller
    controller=Controller(world)
    controller.reset(**policy)
    while not controller.step():pass
    
    #compute mass metric
    print('MassMetric=',MassMetric().compute(controller))
    print('SizeMetric=',SizeMetric().compute(controller))
    for q in [Q1Metric,QInfMetric]:
        mm=[random.uniform(1.,2.) for i in range(6)]
        Q1Metric.USE_NATIVE_CPP=False
        mp=q(metric=mm).compute(controller)
        Q1Metric.USE_NATIVE_CPP=True
        mc=q(metric=mm).compute(controller)
        print((str(q.__name__)+'CPP=%f, '+str(q.__name__)+'PYTHON=%f')%(mc,mp))
    for q in [QMSVMetric,QVEWMetric,QG11Metric]:
        Q1Metric.USE_NATIVE_CPP=False
        mp=q().compute(controller)
        Q1Metric.USE_NATIVE_CPP=True
        mc=q().compute(controller)
        print((str(q.__name__)+'CPP=%f, '+str(q.__name__)+'PYTHON=%f')%(mc,mp))
    print('LiftMetric=',LiftMetric().compute(controller))
    print('ElapsedMetric=',ElapsedMetric().compute(controller))