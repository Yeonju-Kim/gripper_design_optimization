from compile_objects import auto_download,compile_objects,get_COM
from compile_gripper import Link,Gripper,set_simulator_option
import mujoco_py as mjc
import lxml.etree as ET

def create_fog(mujoco):
    #<visual>
    #    <map fogstart="3" fogend="5" force="0.1" znear="0.1"/>
    #    <quality shadowsize="2048"/>
    #    <global offwidth="800" offheight="800"/>
    #</visual>
    visual=ET.SubElement(mujoco,'visual')
    #map
    map=ET.SubElement(visual,'map')
    map.set('fogstart','3')
    map.set('fogend','5')
    map.set('force','0.1')
    map.set('znear','0.1')
    #quality
    quality=ET.SubElement(visual,'quality')
    quality.set('shadowsize','2048')
    #global
    Global=ET.SubElement(visual,'global')
    Global.set('offwidth','800')
    Global.set('offheight','800')
    
def create_skybox(asset):
    #<texture type="skybox" builtin="gradient" width="128" height="128" rgb1=".4 .6 .8" rgb2="0 0 0"/>  
    tex=ET.SubElement(asset,'texture')
    tex.set('type','skybox')
    tex.set('builtin','gradient')
    tex.set('width','128')
    tex.set('height','128')
    tex.set('rgb1','.4 .6 .8')
    tex.set('rgb2','0 0 0')
    
def create_floor(asset,worldbody,pos=[0,0,0],size=[100,100,0.1]):
    #<texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512"/>  
    #<material name='MatPlane' reflectance='0.5' texture="texplane" texrepeat="1 1" texuniform="true"/>
    #<geom name='floor' pos='0 -1 0' size='1.5 3 .125' type='plane' material="MatPlane" condim='3'/>
    #texture
    texture=ET.SubElement(asset,'texture')
    texture.set('name','texplane')
    texture.set('type','2d')
    texture.set('builtin','checker')
    texture.set('rgb1','.2 .3 .4')
    texture.set('rgb2','.1 .15 .2')
    texture.set('width','512')
    texture.set('height','512')
    #material
    material=ET.SubElement(asset,'material')
    material.set('name','MatPlane')
    material.set('reflectance','0.5')
    material.set('texture','texplane')
    material.set('texrepeat','1 1')
    material.set('texuniform','true')
    #geom
    geom=ET.SubElement(worldbody,'geom')
    geom.set('name','floor')
    geom.set('pos',str(pos[0])+' '+str(pos[1])+' '+str(pos[2]))
    geom.set('size',str(size[0])+' '+str(size[1])+' '+str(size[2]))
    geom.set('type','plane')
    geom.set('material','MatPlane')
    geom.set('condim','3')

def create_light(worldbody,pos=[0,0,10],dir=[0,0,-1]):
    #<light directional='false' diffuse='.8 .8 .8' specular='0.3 0.3 0.3' pos='0 0 4.0' dir='0 -.15 -1'/>
    light=ET.SubElement(worldbody,'light')
    light.set('directional','false')
    light.set('diffuse','.8 .8 .8')
    light.set('specular','.3 .3 .3')
    light.set('pos',str(pos[0])+' '+str(pos[1])+' '+str(pos[2]))
    light.set('dir',str(dir[0])+' '+str(dir[1])+' '+str(dir[2]))

class World:
    def __init__(self):
        pass
        
    def compile(self,object_file_name,link,path,damping,damping_gripper,scale_obj):
        root=ET.Element('mujoco')
        set_simulator_option(root)
        asset=ET.SubElement(root,'asset')
        body=ET.SubElement(root,'worldbody')
        actuator=ET.SubElement(root,'actuator')
        create_fog(root)
        create_skybox(asset)
        create_floor(asset,body)
        create_light(body)
        
        #object
        self.compile_objects(object_file_name,asset,body,damping,scale_obj)
        
        #link
        if link is not None:
            link.compile_gripper(body,asset,actuator,path,damping_gripper)
            self.link=link
        else: self.link=None
        return root

    def compile_objects(self,object_file_name,asset,body,damping,scale_obj):
        #<texture name="texgeom" type="cube" builtin="flat" mark="cross" width="127" height="127" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/>  
        #<material name='geom' texture="texgeom" texuniform="true"/>
        #texture
        texture=ET.SubElement(asset,'texture')
        texture.set('name','texgeom')
        texture.set('type','cube')
        texture.set('builtin','flat')
        texture.set('mark','cross')
        texture.set('width','127')
        texture.set('height','127')
        texture.set('rgb1','.8 .6 .4')
        texture.set('rgb2','.8 .6 .4')
        texture.set('markrgb','1 1 1')
        texture.set('random','0.01')
        #material
        material=ET.SubElement(asset,'material')
        material.set('name','geom')
        material.set('texture','texgeom')
        material.set('texuniform','true')
        #object
        if object_file_name is not None:
            self.names,self.fullnames=compile_objects(asset,object_file_name,scale_obj)
            for name,fullname in zip(self.names,self.fullnames):
                #<body name='obj' pos='0 -.08 1'>
                    #<geom name='obj' type='capsule' size='.05 .075'/>
                    #<joint name='ballz' type='slide' axis='0 0 1' limited='false' damping='.01'/>
                    #<joint name='bally' type='slide' axis='0 1 0' limited='false' damping='.01'/>
                    #<joint name='ballx' type='hinge' axis='1 0 0' limited='false' damping='.01'/>
                #</body>
                obj=ET.SubElement(body,'body')
                obj.set('name',name)
                #geom
                geom=ET.SubElement(obj,'geom')
                geom.set('name',name)
                geom.set('mesh',name)
                geom.set('type','mesh')
                geom.set('material','geom')
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
                        joint.set('name',name+str('t' if p==0 else 'r')+str(i))
                        joint.set('type','slide' if p==0 else 'hinge')
                        joint.set('damping',str(damping))
        else: 
            self.names=None 
            self.fullnames=None

    def compile_simulator(self,object_file_name=None,link=None,path='data/gripper',*,damping=10,damping_gripper=1000,scale_obj=2):
        root=self.compile(object_file_name=object_file_name,link=link,path=path,damping=damping,damping_gripper=damping_gripper,scale_obj=scale_obj)
        open(path+'/world.xml','w').write(ET.tostring(root,pretty_print=True).decode())
        model=mjc.load_model_from_path(path+'/world.xml')
        self.sim=mjc.MjSim(model)
        self.get_sim_info()
        self.test_object(0)
        
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
                minZ=10000.0
                id=model.mesh_names.index(name)
                vid0=model.mesh_vertadr[id]
                vid1=vid0+model.mesh_vertnum[id]
                for vid in range(vid0,vid1):
                    minZ=min(model.mesh_vert[vid][2],minZ)
                self.COMs.append(minZ)
        else: 
            self.addrs=[]

    def test_object(self,id,sep_dist=[5.,0.]):
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

if __name__=='__main__':
    auto_download()
    
    #create gripper
    gripper=Gripper()
    link=gripper.get_robot(base_off=0.3,finger_width=0.4,finger_curvature=2)

    #create world    
    world=World()
    world.compile_simulator(object_file_name='data/ObjectNet3D/CAD/off/cup/[0-9][0-9].off',link=link)
    
    #create viewer
    viewer=mjc.MjViewer(world.sim)
    world.test_object(1)
    while True:
        world.sim.step()
        viewer.render()