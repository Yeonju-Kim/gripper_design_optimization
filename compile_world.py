from compile_gripper import Link,Gripper,set_simulator_option
from compile_objects import *
import mujoco_py as mjc
import lxml.etree as ET
import os

class World:
    def __init__(self):
        pass
        
    def compile(self,objects,link,path,damping,damping_gripper,scale_obj):
        root=ET.Element('mujoco')
        sz=ET.SubElement(root,'size')
        sz.set('njmax','8000')
        sz.set('nconmax','4000')
        set_simulator_option(root)
        asset=ET.SubElement(root,'asset')
        body=ET.SubElement(root,'worldbody')
        actuator=ET.SubElement(root,'actuator')
        create_fog(root)
        create_skybox(asset)
        create_floor(asset,body)
        create_light(body)
        
        #object
        self.compile_objects(objects,asset,body,damping,scale_obj)
        
        #link
        if link is not None:
            link.compile_gripper(body,asset,actuator,damping_gripper)
            self.link=link
        else: self.link=None
        return root

    def compile_objects(self,objects,asset,body,damping,scale_obj):
        #<texture name="texgeom" type="cube" builtin="flat" mark="cross" width="127" height="127" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/>  
        #<material name='geom' texture="texgeom" texuniform="true"/>
        create_material(asset)
        #object
        if objects is not None:
            self.names=[]
            for iobj,objInfo in enumerate(objects):
                #<body name='obj' pos='0 -.08 1'>
                    #<geom name='obj' type='capsule' size='.05 .075'/>
                    #<joint name='ballz' type='slide' axis='0 0 1' limited='false' damping='.01'/>
                    #<joint name='bally' type='slide' axis='0 1 0' limited='false' damping='.01'/>
                    #<joint name='ballx' type='hinge' axis='1 0 0' limited='false' damping='.01'/>
                #</body>
                self.names.append(str(iobj))
                obj=ET.SubElement(body,'body')
                compile_body(str(iobj),obj,asset,objInfo,material='geom')
                #joint
                for p in range(2):
                    for i in range(3):
                        joint=ET.SubElement(obj,'joint')
                        axis=''
                        for d in range(3):
                            axis+='1' if d==i else '0'
                            if d<2:axis+=' '
                        joint.set('axis',axis)
                        joint.set('limited','false')
                        joint.set('name',self.names[-1]+str('t' if p==0 else 'r')+str(i))
                        joint.set('type','slide' if p==0 else 'hinge')
                        joint.set('damping',str(damping))
        else: 
            self.names=None 
            self.fullnames=None

    def compile_simulator(self,objects=None,link=None,path='data',*,damping=10,damping_gripper=1000,scale_obj=2):
        if not os.path.exists(path):
            os.mkdir(path)
        root=self.compile(objects=objects,link=link,path=path,damping=damping,damping_gripper=damping_gripper,scale_obj=scale_obj)
        open(path+'/world_PID='+str(os.getpid())+'.xml','w').write(ET.tostring(root,pretty_print=True).decode())
        model=mjc.load_model_from_path(path+'/world_PID='+str(os.getpid())+'.xml')
        self.sim=mjc.MjSim(model)
        self.get_sim_info()
        self.test_object(0)
        
    def get_minZ(self,body_name):
        minZ=10000.0
        model=self.sim.model
        bid=model.body_names.index(body_name)
        geom_adr=model.body_geomadr[bid]
        geom_num=model.body_geomnum[bid]
        for gid in range(geom_adr,geom_adr+geom_num):
            mid=model.geom_dataid[gid]
            if mid<0:
                continue
            vid0=model.mesh_vertadr[mid]
            vid1=vid0+model.mesh_vertnum[mid]
            for vid in range(vid0,vid1):
                minZ=min(model.mesh_vert[vid][2],minZ)
        if minZ>=10000.0:
            minZ=0.
        return minZ-0.1
        
    def get_sim_info(self):
        if self.names is not None:
            self.addrs=[]
            self.COMs=[]
            model=self.sim.model
            for name in self.names:
                #q address
                addr=[]
                for p in range(2):
                    for d in range(3):
                        addr.append(self.sim.model.get_joint_qpos_addr(name+str('t' if p==0 else 'r')+str(d)))
                self.addrs.append(addr)
                #mesh z coordinates
                self.COMs.append(self.get_minZ(name))
        else: 
            self.addrs=[]

    def test_object(self,id,sep_dist=[3.,0.]):
        self.sim.reset()
        if self.names is None:
            return
        off=1
        state=self.sim.get_state()
        for addr,COM in zip(self.addrs,self.COMs):
            #pos
            state.qpos[addr[Gripper.X]]=0 if off-1==id else sep_dist[0]*off
            state.qpos[addr[Gripper.Y]]=0 if off-1==id else sep_dist[1]*off
            state.qpos[addr[Gripper.Z]]=-COM
            state.qpos[addr[3+Gripper.X]]=0
            state.qpos[addr[3+Gripper.Y]]=0
            state.qpos[addr[3+Gripper.Z]]=0
            #vel
            state.qvel[addr[Gripper.X]]=0
            state.qvel[addr[Gripper.Y]]=0
            state.qvel[addr[Gripper.Z]]=0
            state.qvel[addr[3+Gripper.X]]=0
            state.qvel[addr[3+Gripper.Y]]=0
            state.qvel[addr[3+Gripper.Z]]=0
            off+=1
        self.sim.set_state(state)
        self.target_object_id=id
        
        model=self.sim.model
        bid=model.body_names.index(self.names[id])
        geom_adr=model.body_geomadr[bid]
        geom_num=model.body_geomnum[bid]
        self.target_geom_ids=range(geom_adr,geom_adr+geom_num)

if __name__=='__main__':
    #create gripper
    gripper=Gripper()
    link=gripper.get_robot(base_off=0.3,finger_width=0.4,finger_curvature=-2.)

    #create world 
    world=World()
    from dataset_cup import get_dataset_cup
    from dataset_canonical import get_dataset_canonical
    world.compile_simulator(objects=get_dataset_canonical(),link=None)#link)
    
    #create viewer
    viewer=mjc.MjViewer(world.sim)
    world.test_object(0)
    while True:
        world.sim.step()
        viewer.render()