from compile_gripper import Gripper
from compile_world import World
from compile_objects import *
import mujoco_py as mjc
import trimesh as tm
import numpy as np
import math

class Controller:
    def __init__(self,world,*,approach_vel=0.2,thres_vel=1e-1,lift_vel=1.0,lift_height=1.5,shake_vel=[0.4,0,0],shake_times=10):
        self.world=world
        self.sim=self.world.sim
        if self.world.link is None:
            raise RuntimeError('To initialize a controller, one must have link defined')
        else: 
            self.link=self.world.link
        self.link.get_ctrl_address(self.sim)
        #param
        self.approach_vel=approach_vel
        self.thres_vel=thres_vel
        #lift
        self.lift_vel=lift_vel
        self.lift_height=lift_height
        #shake
        self.shake_vel=shake_vel
        self.shake_times=shake_times
        #object id
        self.link_ids=[]
        self.get_link_ids(self.link)
        self.floor_id=self.sim.model.geom_names.index('floor')
    
    def get_link_ids(self,link):
        for idn,n in enumerate(self.sim.model.geom_names):
            if n.startswith(link.name):
                self.link_ids.append(idn)
        for c in link.children:
            self.get_link_ids(c)
    
    def reset(self,id,initial_pos,axial_rotation):
        #we assume the gripper is always approaching from initial_pos to [0,0,0]
        self.world.test_object(id)
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
        self.shake_count=0
        self.elapsed=0.0
        self.contact_poses=[]
        self.contact_normals=[]
    
    def record_contacts(self):
        #get object pos
        state=self.sim.get_state()
        addr=self.world.addrs[self.world.target_object_id]
        objpos=[state.qpos[addr[0]],state.qpos[addr[1]],state.qpos[addr[2]]]
        #get contacts
        self.contact_poses=[]
        self.contact_normals=[]
        for ic in range(self.sim.data.ncon):
            c=self.sim.data.contact[ic]
            if c.geom1 in self.link_ids and c.geom2 not in self.link_ids:
                if c.geom2 in self.world.target_geom_ids:
                    self.contact_poses.append([cp-op for cp,op in zip(c.pos,objpos)])
                    self.contact_normals.append(c.frame[0:3].tolist())
            elif c.geom1 not in self.link_ids and c.geom2 in self.link_ids:
                if c.geom1 in self.world.target_geom_ids:
                    self.contact_poses.append([cp-op for cp,op in zip(c.pos,objpos)])
                    #to compute Q_* metric, we assume normals are point inward i.e. n*p<0
                    self.contact_normals.append([-n for n in c.frame[0:3].tolist()])
    
    def contact_state(self):
        floor_contact=False
        obj_contact=False
        for ic in range(self.sim.data.ncon):
            c=self.sim.data.contact[ic]
            if c.geom1 in self.link_ids and c.geom2 not in self.link_ids:
                if c.geom2==self.floor_id:
                    floor_contact=True
                else: obj_contact=True
            elif c.geom1 not in self.link_ids and c.geom2 in self.link_ids:
                if c.geom1==self.floor_id:
                    floor_contact=True
                else: obj_contact=True
        return floor_contact,obj_contact
    
    def approach(self):
        state=self.sim.get_state()
        pos=np.array([state.qpos[self.link.joint_ids[d]] for d in range(3)])
        pos=(pos.dot(self.target_dir)*self.target_dir).tolist()
        x=[p+v for p,v in zip(pos,self.target_vel)]+self.target_rot+[math.pi/2,0]
        vx=self.target_vel+[0,0,0]+[0,0]
        self.link.set_PD_target(x,vx)
        self.link.define_ctrl(self.sim,state.qpos,state.qvel)
        
        #return succeed or failed
        self.sim.step()
        fc,oc=self.contact_state()
        if fc:  #if floor contact, immediately return false
            return False
        if oc:
            self.approached=True
            state=self.sim.get_state()
            self.x_approached=self.link.fetch_q(state.qpos)
            return True
        return True
    
    def close(self):
        state=self.sim.get_state()
        qpos=self.link.fetch_q(state.qpos)
        x=self.x_approached[0:6]+[qpos[6]-self.approach_vel,qpos[7]-self.approach_vel]
        vx=[0 for i in range(6)]+[-self.approach_vel,-self.approach_vel]
        self.link.set_PD_target(x,vx)
        self.link.define_ctrl(self.sim,state.qpos,state.qvel)
    
        #return succeed or failed (actually close will always succeed)
        self.sim.step()
        self.elapsed+=1
        fc,_=self.contact_state()
        if fc:  #if floor contact, immediately return false
            return False
        state=self.sim.get_state()
        maxVel=max([abs(q) for q in self.link.fetch_q(state.qvel)])
        if maxVel<self.thres_vel:  #we assume closed when the velocity is small enough
            self.x_closed=self.link.fetch_q(state.qpos)
            self.record_contacts()  #this will be used to compute Q_* metric later
            self.closed=True
        return True
        
    def lift(self):
        state=self.sim.get_state()
        height=self.link.fetch_q(state.qpos)[2]
        x=self.x_closed[0:8]
        vx=[0 for i in range(6)]+[-self.lift_vel,-self.lift_vel]
        vx[2]=self.lift_vel
        x[2]=height+vx[2]
        self.link.set_PD_target(x,vx)
        self.link.define_ctrl(self.sim,state.qpos,state.qvel)
        
        self.sim.step()
        self.elapsed+=1
        fc,oc=self.contact_state()
        if fc:# or not oc:  #if floor contact or no object contact, immediately return false
            return False
        
        #return succeed or failed based on whether gripper is still in contact with object
        state=self.sim.get_state()
        if height>self.lift_height:
            if oc:
                self.x_lifted=self.link.fetch_q(state.qpos)
                self.lifted=True
                return True
            else:
                return False
        return True
    
    def shake(self):
        sgn=1 if self.shake_count%2==0 else -1
        state=self.sim.get_state()
        pos=np.array([state.qpos[self.link.joint_ids[d]] for d in range(3)])
        svel=np.array(self.shake_vel)
        len=np.linalg.norm(svel)
        svel*=1/len
        
        x=self.x_lifted[0:8]
        pos0=np.array(x[0:3])-np.array(x[0:3]).dot(svel)*svel
        pos0+=svel*(pos.dot(svel)+len*sgn)
        x[0:3]=pos0.tolist()
        
        vx=[0 for i in range(6)]+[-self.lift_vel,-self.lift_vel]
        vx[0:3]=[v*sgn for v in self.shake_vel]
        self.link.set_PD_target(x,vx)
        self.link.define_ctrl(self.sim,state.qpos,state.qvel)
    
        self.sim.step()
        self.elapsed+=1
        fc,oc=self.contact_state()
        if fc or not oc:  #if floor contact or no object contact, immediately return false
            return False
        
        #return succeed or failed based on whether gripper is still in contact with object
        state=self.sim.get_state()
        off=(np.array(pos[0:3])-np.array(x[0:3])).dot(svel)
        if abs(off)>len:
            self.shake_count+=1
            if self.shake_count>=self.shake_times:
                if oc:
                    self.x_shaked=self.link.fetch_q(state.qpos)
                    self.shaked=True
                    return True
                else:
                    return False
        return True
    
    def step(self,will_close=True,will_lift=True,will_shake=True):
        #this function return whether an experiment has finished
        if not self.approached:
            if not self.approach():
                return True
            return False
        elif will_close and not self.closed:
            if not self.close():
                return True
            return False
        elif will_lift and not self.lifted:
            if not self.lift():
                return True
            return False
        elif will_shake and not self.shaked:
            if not self.shake():
                return True
            return False
        return True
    
if __name__=='__main__':
    #create gripper
    gripper=Gripper()
    link=gripper.get_robot(base_off=0.2,finger_length=0.15,finger_width=0.2,finger_curvature=4.)

    #create world    
    world=World()
    use_surrogate=True
    if use_surrogate:
        objs=[surrogate_object_01('01'),    \
              surrogate_object_02('02'),    \
              surrogate_object_03('03'),    \
              surrogate_object_04('04'),    \
              surrogate_object_05('05'),    \
              surrogate_object_06('06'),    \
              surrogate_object_07('07'),    \
              surrogate_object_08('08'),    \
              surrogate_object_09('09'),    \
              surrogate_object_10('10')]
    else:
        auto_download()
        objs=glob.glob('data/ObjectNet3D/CAD/off/cup/[0-9][0-9].off')
    world.compile_simulator(objects=objs,link=link)
    viewer=mjc.MjViewer(world.sim)
    
    #create controller
    controller=Controller(world)
    
    id=0
    while True:
        controller.reset(id,[0.1,0.,3.],-0.1)
        while not controller.step():
            viewer.render()
        id=(id+1)%len(controller.world.names)