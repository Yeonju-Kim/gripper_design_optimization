from compile_objects import auto_download
from compile_gripper import Link,Gripper
from compile_world import World
from controller import Controller
import pyGraspMetric as gm
import mujoco_py as mjc

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
                    v=xori.reshape((3,3)).dot(model.mesh_vert[vid])+xpos
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
    #this is the grasp quality measured after close
    def __init__(self,FRICTION=0.7):
        Metric.__init__(self,True)
        self.FRICTION=FRICTION
    
    def __reduce__(self):
        return (self.__class__,(self.FRICTION,))
    
    def compute(self,controller,callback=False):
        self.controller=controller
        self.mMatrix=gm.Mat6d()
        self.mMatrix.setZero()
        for d in range(6):
            self.mMatrix[d,d]=1.0
            
        contact_poses=[gm.Vec3d(cp[0],cp[1],cp[2]) for cp in self.controller.contact_poses]
        contact_normals=[gm.Vec3d(cn[0],cn[1],cn[2]) for cn in self.controller.contact_normals]
        return gm.Q1(self.FRICTION,contact_poses,contact_normals,self.mMatrix,callback)
        
class QInfMetric(Q1Metric):
    #this is the grasp quality measured after close
    def __init__(self,FRICTION=0.7):
        Metric.__init__(self,True)
        self.FRICTION=FRICTION
    
    def __reduce__(self):
        return (self.__class__,(self.FRICTION,))
    
    def compute(self,controller,callback=False):
        self.controller=controller
        self.mMatrix=gm.Mat6d()
        self.mMatrix.setZero()
        for d in range(6):
            self.mMatrix[d,d]=1.0
            
        contact_poses=[gm.Vec3d(cp[0],cp[1],cp[2]) for cp in self.controller.contact_poses]
        contact_normals=[gm.Vec3d(cn[0],cn[1],cn[2]) for cn in self.controller.contact_normals]
        return gm.QInf(self.FRICTION,contact_poses,contact_normals,self.mMatrix,callback)
        
class QMSVMetric(Q1Metric):
    #this is the grasp quality measured after close
    def __init__(self,FRICTION=0.7):
        Metric.__init__(self,True)
        self.FRICTION=FRICTION
    
    def __reduce__(self):
        return (self.__class__,(self.FRICTION,))
    
    def compute(self,controller,callback=False):
        self.controller=controller
        self.mMatrix=gm.Mat6d()
        self.mMatrix.setZero()
        for d in range(6):
            self.mMatrix[d,d]=1.0
            
        contact_poses=[gm.Vec3d(cp[0],cp[1],cp[2]) for cp in self.controller.contact_poses]
        contact_normals=[gm.Vec3d(cn[0],cn[1],cn[2]) for cn in self.controller.contact_normals]
        return gm.QMSV(self.FRICTION,contact_poses,contact_normals)
   
class QVEWMetric(Q1Metric):
    #this is the grasp quality measured after close
    def __init__(self,FRICTION=0.7):
        Metric.__init__(self,True)
        self.FRICTION=FRICTION
    
    def __reduce__(self):
        return (self.__class__,(self.FRICTION,))
    
    def compute(self,controller,callback=False):
        self.controller=controller
        self.mMatrix=gm.Mat6d()
        self.mMatrix.setZero()
        for d in range(6):
            self.mMatrix[d,d]=1.0
            
        contact_poses=[gm.Vec3d(cp[0],cp[1],cp[2]) for cp in self.controller.contact_poses]
        contact_normals=[gm.Vec3d(cn[0],cn[1],cn[2]) for cn in self.controller.contact_normals]
        return gm.QVEW(self.FRICTION,contact_poses,contact_normals)
     
class QG11Metric(Q1Metric):
    #this is the grasp quality measured after close
    def __init__(self,FRICTION=0.7):
        Metric.__init__(self,True)
        self.FRICTION=FRICTION
    
    def __reduce__(self):
        return (self.__class__,(self.FRICTION,))
    
    def compute(self,controller,callback=False):
        self.controller=controller
        self.mMatrix=gm.Mat6d()
        self.mMatrix.setZero()
        for d in range(6):
            self.mMatrix[d,d]=1.0
            
        contact_poses=[gm.Vec3d(cp[0],cp[1],cp[2]) for cp in self.controller.contact_poses]
        contact_normals=[gm.Vec3d(cn[0],cn[1],cn[2]) for cn in self.controller.contact_normals]
        return gm.QG11(self.FRICTION,contact_poses,contact_normals)
        
class LiftMetric(Metric):
    #this metric measures whether the gripper can close, and then lift, and finally shake
    def __init__(self):
        Metric.__init__(self,True)
    
    def __reduce__(self):
        return (self.__class__,())
    
    def compute(self,controller):
        self.controller=controller
        
        score=0.0
        if self.controller.closed:
            score+=1
        if self.controller.lifted:
            score+=1
        if self.controller.shaked:
            score+=1
        return score

class ElapsedMetric(Metric):
    #this metric measures how much time can the current gripper grasp the object
    def __init__(self):
        Metric.__init__(self,True)
    
    def __reduce__(self):
        return (self.__class__,())
    
    def compute(self,controller):
        self.controller=controller
        
        dt=self.controller.sim.model.opt.timestep
        return self.controller.elapsed*dt
    
if __name__=='__main__':
    #create gripper
    gripper=Gripper()
    link=gripper.get_robot(base_off=0.3,finger_width=0.4,finger_curvature=2)

    #create world    
    world=World()
    from dataset_cup import get_dataset_cup
    world.compile_simulator(objects=get_dataset_cup(True),link=link)
    world.test_object(0)
    
    #create controller
    controller=Controller(world)
    controller.reset(0,[0.1,0.,5.],-0.1)
    while not controller.step():
        pass
    
    #compute mass metric
    print('MassMetric=',MassMetric().compute(controller))
    print('SizeMetric=',SizeMetric().compute(controller))
    print('Q1Metric=',Q1Metric().compute(controller))
    print('QInfMetric=',QInfMetric().compute(controller))
    print('QMSVMetric=',QMSVMetric().compute(controller))
    print('QVEWMetric=',QVEWMetric().compute(controller))
    print('QG11Metric=',QG11Metric().compute(controller))
    print('LiftMetric=',LiftMetric().compute(controller))
    print('ElapsedMetric=',ElapsedMetric().compute(controller))