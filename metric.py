from compile_objects import auto_download
from compile_gripper import Link,Gripper
from compile_world import World
from controller import Controller
import pyGraspMetric as gm
import mujoco_py as mjc

class Metric:
    #Note that all metrics are supposed to be maximized
    def __init__(self):
        pass
    
    def compute(self):
        raise RuntimeError('This is abstract super-class, use sub-class!')
    
class MassMetric(Metric):
    #this is the mass of the gripper
    OBJECT_DEPENDENT=False
    
    def __init__(self,controller,link=None):
        self.controller=controller
        self.target_link=link
        if link is None:
            self.target_link=self.controller.link
        
    def compute(self):
        mass=self.get_mass(self.target_link)
        return 1./mass
    
    def get_mass(self,link):
        model=self.controller.sim.model
        bid=model.body_names.index(link.name)
        ret=model.body_mass[bid]
        for c in link.children:
            ret+=self.get_mass(c)
        return ret
        
class SizeMetric(Metric):
    #this is the surface area of the bounding box
    OBJECT_DEPENDENT=False
    
    def __init__(self,controller,link=None):
        self.controller=controller
        self.target_link=link
        if link is None:
            self.target_link=self.controller.link
        
    def compute(self):
        #compute bounding box
        vmin=[ 1000., 1000., 1000.]
        vmax=[-1000.,-1000.,-1000.]
        self.controller.sim.set_state(self.controller.init_state)
        self.controller.sim.forward()
        model=self.controller.sim.model
        data=self.controller.sim.data
        names=self.body_names(self.target_link)
        for bname in names:
            bid=model.body_names.index(bname)
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
        
    def body_names(self,link):
        ret=[link.name]
        for c in link.children:
            ret+=self.body_names(c)
        return ret
        
class Q1Metric(Metric):
    #this is the grasp quality measured after close
    OBJECT_DEPENDENT=True
    FRICTION=0.7
    
    def __init__(self,controller):
        self.controller=controller
        self.mMatrix=gm.Mat6d()
        self.mMatrix.setZero()
        for d in range(6):
            self.mMatrix[d,d]=1.0
        
    def compute(self,callback=False):
        contact_poses=[gm.Vec3d(cp[0],cp[1],cp[2]) for cp in self.controller.contact_poses]
        contact_normals=[gm.Vec3d(cn[0],cn[1],cn[2]) for cn in self.controller.contact_normals]
        return gm.Q1(Q1Metric.FRICTION,contact_poses,contact_normals,self.mMatrix,callback)
        
class QInfMetric(Q1Metric):
    #this is the grasp quality measured after close
    OBJECT_DEPENDENT=True
    
    def __init__(self,controller):
        Q1Metric.__init__(self,controller)
        
    def compute(self,callback=False):
        contact_poses=[gm.Vec3d(cp[0],cp[1],cp[2]) for cp in self.controller.contact_poses]
        contact_normals=[gm.Vec3d(cn[0],cn[1],cn[2]) for cn in self.controller.contact_normals]
        return gm.QInf(Q1Metric.FRICTION,contact_poses,contact_normals,self.mMatrix,callback)
        
class QMSVMetric(Q1Metric):
    #this is the grasp quality measured after close
    OBJECT_DEPENDENT=True
    
    def __init__(self,controller):
        Q1Metric.__init__(self,controller)
        
    def compute(self,callback=False):
        contact_poses=[gm.Vec3d(cp[0],cp[1],cp[2]) for cp in self.controller.contact_poses]
        contact_normals=[gm.Vec3d(cn[0],cn[1],cn[2]) for cn in self.controller.contact_normals]
        return gm.QMSV(Q1Metric.FRICTION,contact_poses,contact_normals)
   
class QVEWMetric(Q1Metric):
    #this is the grasp quality measured after close
    OBJECT_DEPENDENT=True
    
    def __init__(self,controller):
        Q1Metric.__init__(self,controller)
        
    def compute(self,callback=False):
        contact_poses=[gm.Vec3d(cp[0],cp[1],cp[2]) for cp in self.controller.contact_poses]
        contact_normals=[gm.Vec3d(cn[0],cn[1],cn[2]) for cn in self.controller.contact_normals]
        return gm.QVEW(Q1Metric.FRICTION,contact_poses,contact_normals)
     
class QG11Metric(Q1Metric):
    #this is the grasp quality measured after close
    OBJECT_DEPENDENT=True
    
    def __init__(self,controller):
        Q1Metric.__init__(self,controller)
        
    def compute(self,callback=False):
        contact_poses=[gm.Vec3d(cp[0],cp[1],cp[2]) for cp in self.controller.contact_poses]
        contact_normals=[gm.Vec3d(cn[0],cn[1],cn[2]) for cn in self.controller.contact_normals]
        return gm.QG11(Q1Metric.FRICTION,contact_poses,contact_normals)
        
class LiftMetric(Metric):
    #this metric measures whether the gripper can close, and then lift, and finally shake
    OBJECT_DEPENDENT=True
    
    def __init__(self,controller):
        self.controller=controller
        
    def compute(self):
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
    OBJECT_DEPENDENT=True
    
    def __init__(self,controller):
        self.controller=controller
        
    def compute(self):
        dt=self.controller.sim.model.opt.timestep
        return self.controller.elapsed*dt
    
if __name__=='__main__':
    auto_download()
    
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
    print('MassMetric=',MassMetric(controller).compute())
    print('SizeMetric=',SizeMetric(controller).compute())
    print('MassMetric=',MassMetric(controller,link.children[0]).compute())
    print('SizeMetric=',SizeMetric(controller,link.children[0]).compute())
    print('Q1Metric=',Q1Metric(controller).compute())
    print('QInfMetric=',QInfMetric(controller).compute())
    print('QMSVMetric=',QMSVMetric(controller).compute())
    print('QVEWMetric=',QVEWMetric(controller).compute())
    print('QG11Metric=',QG11Metric(controller).compute())
    print('LiftMetric=',LiftMetric(controller).compute())
    print('ElapsedMetric=',ElapsedMetric(controller).compute())