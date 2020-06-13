from compile_objects import auto_download
from compile_gripper import Gripper,as_mesh
from compile_world import World
from controller import Controller
import mujoco_py as mjc

class Metric:
    def __init__(self,link,controller):
        self.link=link
        self.controller=controller
    
    def compute(self):
        raise RuntimeError('This is abstract super-class, use sub-class!')
    
class MassMetric(Metric):
    #this is the mass of the gripper
    
    def __init__(self,link,controller):
        Metric.__init__(self,link,controller)
        
    def compute(self):
        mesh=as_mesh(self.link.get_mesh())
        return mesh.mass
        
class SizeMetric(Metric):
    #this is the surface area of the bounding box
    
    def __init__(self,link,controller):
        Metric.__init__(self,link,controller)
        
    def compute(self):
        mesh=as_mesh(self.link.get_mesh())
        vmin=mesh.bounds[0]
        vmax=mesh.bounds[1]
        surface_area=0
        for d in range(3):
            ext=[]
            for d2 in range(3):
                if d2!=d:
                    ext.append(vmax[d2]-vmin[d2])
            surface_area+=ext[0]*ext[1]*2
        return surface_area
        
class Q1Metric(Metric):
    #this is the grasp quality measured after close
    
    def __init__(self,link,controller):
        Metric.__init__(self,link,controller)
        
    def compute(self):
        pass
        
class LiftMetric(Metric):
    #this metric measures whether the gripper can close, and then lift, and finally shake
    
    def __init__(self,link,controller):
        Metric.__init__(self,link,controller)
        
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
    
    def __init__(self,link,controller):
        Metric.__init__(self,link,controller)
        
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
    world.compile_simulator(object_file_name='data/ObjectNet3D/CAD/off/cup/[0-9][0-9].off',link=link)
    world.test_object(0)
    
    #create controller
    controller=Controller(world)
    controller.reset_approach_dir([0.1,0.,5.],-0.1)
    while not controller.step():
        pass
    
    #compute mass metric
    print('MassMetric=',MassMetric(link,controller).compute())
    print('SizeMetric=',SizeMetric(link,controller).compute())
    #print('Q1Metric=',Q1Metric(link,controller).compute())
    print('LiftMetric=',LiftMetric(link,controller).compute())
    print('ElapsedMetric=',ElapsedMetric(link,controller).compute())