from compile_objects import auto_download
from compile_gripper import Gripper
from compile_world import World
import mujoco_py as mjc
import trimesh as tm
import numpy as np
import math

class Controller:
    def __init__(self,world,*,approach_vel=1.0,close_vel=1.0,lift_height=0.5,shake_frequency=1.0,shake_amplitude=0.5,shake_times=2):
        self.world=world
        self.sim=self.world.sim
        if self.world.link is None:
            raise RuntimeError('To initialize a controller, one must have link defined')
        else: 
            self.link=self.world.link
        self.link.get_ctrl_address(self.sim)
        self.approach_vel=approach_vel
        self.close_vel=close_vel
        self.lift_height=lift_height
        self.shake_frequency=shake_frequency
        self.shake_amplitude=shake_amplitude
        self.shake_times=shake_times
    
    def reset_approach_dir(self,initial_pos,axial_rotation):
        #we assume the gripper is always approaching from initial_pos to [0,0,0]
        state=self.sim.get_state()
        
        v0=np.array([0,0,1],dtype=np.float64)
        v1=np.array([-initial_pos[0],-initial_pos[1],-initial_pos[2]],dtype=np.float64)
        v1*=1/np.linalg.norm(v1)
        R1=tm.transformations.rotation_matrix(tm.transformations.angle_between_vectors(v0,v1),tm.transformations.vector_product(v0,v1))
        R0=tm.transformations.rotation_matrix(angle=axial_rotation,direction=[0,0,1])
        R=np.matmul(R1,R0)[0:3,0:3]
        x,y,z=tm.transformations.euler_from_matrix(R,'rxyz')
        self.target_rot=[x,y,z]
        self.target_pos=initial_pos
        self.target_vel=[v*self.approach_vel for v in v1.tolist()]
        self.target_dir=v1
        
        self.link.set_PD_target(self.target_pos+self.target_rot+[math.pi/2,0.0],state=state)
        self.sim.set_state(state)
        self.approached=False
        self.closed=False
        self.lifted=False
        self.shaked=False
    
    def approach(self):
        state=self.sim.get_state()
        pos=np.array([state.qpos[self.link.joint_ids[d]] for d in range(3)])
        pos=pos.dot(self.target_dir)*self.target_dir
        x=pos.tolist()+self.target_rot+[math.pi/2,0.0]
        vx=self.target_vel+[0,0,0]+[0,0]
        self.link.set_PD_target(x,vx)
        self.link.define_ctrl(self.sim,state.qpos,state.qvel)
    
    def close(self):
        pass
    
    def lift(self):
        pass
    
    def shake(self,frequency,dir=[3,0,0]):
        pass
    
    def step(self,will_close=False,will_lift=False,will_shake=False):
        if not self.approached:
            self.approach()
            self.sim.step()
            return False
        elif will_close and not self.closed:
            self.close()
            self.sim.step()
            return False
        elif will_lift and not self.lifted:
            self.lift()
            self.sim.step()
            return False
        elif will_shake and not self.shaked:
            self.shake()
            self.sim.step()
            return False
        return True
    
if __name__=='__main__':
    auto_download()
    
    #create gripper
    gripper=Gripper()
    link=gripper.get_robot(base_off=0.3,finger_width=0.4,finger_curvature=2)

    #create world    
    world=World()
    world.compile_simulator(object_file_name='data/ObjectNet3D/CAD/off/cup/[0-9][0-9].off',link=link)
    
    #create controller
    controller=Controller(world)
    controller.reset_approach_dir([1,0,9],0.4)
    
    #create viewer
    viewer=mjc.MjViewer(world.sim)
    world.test_object(1)
    while not controller.step():
        viewer.render()