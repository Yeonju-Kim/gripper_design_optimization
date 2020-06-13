from compile_objects import auto_download
import mujoco_py as mjc
import lxml.etree as ET
import glob,ntpath,os
import trimesh as tm
import numpy as np
import math

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, tm.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = tm.util.concatenate(
                tuple(tm.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(mesh, tm.Trimesh))
        mesh = scene_or_mesh
    return mesh

def set_simulator_option(root,dt=0.001,withparent=False):
    #<option timestep='0.002' iterations="50" solver="PGS">
    #    <flag energy="enable"/>
    #</option>
    option=ET.SubElement(root,'option')
    option.set('timestep',str(dt))
    option.set('iterations','50')
    option.set('solver','PGS')
    flag=ET.SubElement(option,'flag')
    flag.set('filterparent','disable' if withparent else 'enable')
    flag.set('energy','enable')

class Link:
    def __init__(self,geom,name,nameMesh,trans=None,parent=None,affine=None):
        self.geom=geom
        self.name=name
        self.nameMesh=nameMesh
        if trans is None:
            trans=np.array([[1.,0.,0.,0.],
                            [0.,1.,0.,0.],
                            [0.,0.,1.,0.],
                            [0.,0.,0.,1.]])
        self.trans=trans
        self.children=[]
        self.parent=parent
        if self.parent is not None:
            self.parent.children.append(self)
        if affine is None:
            self.affine=1
        else: self.affine=affine
        self.id=None
      
    def assign_id(self,affine=1.0):
        depth=self.depth()
        if depth==0:
            self.id=0
        elif depth==1:
            self.id=6
        elif depth==2:
            self.id=7
        else:
            #this is connected DOF
            self.id=7
            self.affine=affine
        for c in self.children:
            c.assign_id(affine=affine)

    def depth(self):
        if self.parent is None:
            return 0
        else: return self.parent.depth()+1

    def total_DOF(self):
        ret=self.id+self.num_DOF()
        for c in self.children:
            ret=max(ret,c.total_DOF())
        return ret

    def num_DOF(self):
        return 6 if self.parent is None else 1

    def get_trans(self,x,T0=None):
        if self.parent is None:
            DOF=x[self.id:self.id+6]
            T=tm.transformations.translation_matrix(DOF[0:3])
            angle=np.linalg.norm(np.array(DOF[3:6]))
            if angle<Gripper.EPS:
                DOF[3]=1
            R=tm.transformations.rotation_matrix(angle=angle,direction=DOF[3:6])
            T=np.matmul(T,R)
        else:
            T=tm.transformations.rotation_matrix(angle=x[self.id]*self.affine,direction=[0,1,0])
        T=np.matmul(self.trans,T)
        if T0 is not None:
            T=np.matmul(T0,T)
        return T

    def get_mesh(self,x=None,T0=None,scene=None):
        if x is None:
            x=[0.0 for i in range(self.total_DOF())]
        if scene is None:
            scene=tm.Scene()
        T=self.get_trans(x,T0)
        for g in self.geom.geometry.items():
            gcpy=g[1].copy()
            gcpy.apply_transform(T)
            scene.add_geometry(gcpy)
        for c in self.children:
            c.get_mesh(x,T,scene)
        return scene

    def get_pos(self):
        return str(self.trans[0,3])+' '+str(self.trans[1,3])+' '+str(self.trans[2,3])

    def get_quat(self):
        q=tm.transformations.quaternion_from_matrix(self.trans[0:3,0:3])
        return str(q[0])+' '+str(q[1])+' '+str(q[2])+' '+str(q[3])

    def compile_gripper(self,body,asset,actuator,path,damping=1000.0,gear=1):
        b=ET.SubElement(body,'body')
        b.set('pos',self.get_pos())
        b.set('quat',self.get_quat())
        b.set('name',self.name)
        #geom
        geom=ET.SubElement(b,'geom')
        geom.set('mesh',self.nameMesh)
        geom.set('name',self.name)
        geom.set('type','mesh')
        #joint
        self.ctrl_names=[]
        if self.parent is None:
            for p in range(2):
                for i in range(3):
                    joint=ET.SubElement(b,'joint')
                    axis=''
                    for d in range(3):
                        axis+='1' if d==i else '0'
                        if d<2:axis+=' '
                    jointName=self.name+str('t' if p==0 else 'r')+str(i)
                    joint.set('axis',axis)
                    joint.set('limited','false')
                    joint.set('name',jointName)
                    joint.set('type','slide' if p==0 else 'hinge')
                    joint.set('damping',str(damping))
                    if actuator is not None:
                        motor=ET.SubElement(actuator,'motor')
                        motor.set('gear',str(gear))
                        motor.set('name',jointName)
                        motor.set('joint',jointName)
                        self.ctrl_names.append(jointName)
        else:
            joint=ET.SubElement(b,'joint')
            joint.set('axis','0 1 0')
            joint.set('range',str(-math.pi/2)+' '+str(math.pi/2))
            joint.set('name',self.name)
            joint.set('type','hinge')
            joint.set('damping',str(damping))
            if actuator is not None:
                motor=ET.SubElement(actuator,'motor')
                motor.set('gear',str(gear))
                motor.set('name',self.name)
                motor.set('joint',self.name)
                self.ctrl_names.append(self.name)
        #asset
        if not os.path.exists(path):
            os.mkdir(path)
        if not path.endswith('/'):
            path+='/'
        if self.parent is None:
            as_mesh(self.geom).export(path+'base.stl')
            mesh=ET.SubElement(asset,'mesh')
            mesh.set('file','base.stl')
            mesh.set('name',self.nameMesh)
        elif self.finger_id()==0 and self.parent.parent is None:
            as_mesh(self.geom).export(path+'finger.stl')
            mesh=ET.SubElement(asset,'mesh')
            mesh.set('file','finger.stl')
            mesh.set('name',self.nameMesh)
        elif self.finger_id()==0 and len(self.children)==0:
            as_mesh(self.geom).export(path+'fingerTop.stl')
            mesh=ET.SubElement(asset,'mesh')
            mesh.set('file','fingerTop.stl')
            mesh.set('name',self.nameMesh)
        #children
        for c in self.children:
            c.compile_gripper(b,asset,actuator=actuator,path=path,gear=gear)

    def finger_id(self):
        if self.parent.parent is None:
            return self.parent.children.index(self)
        else: return self.parent.finger_id()
    
    def set_PD_target(self,qpos,qvel):
        self.PTarget=[]
        self.DTarget=[]
        for jid in self.joint_ids:
            self.PTarget.append(qpos[jid])
            self.DTarget.append(qvel[jid])
        for c in self.children:
            c.set_PD_target(qpos,qvel)
        
    def set_PD_target(self,x):
        self.PTarget=x[self.id:self.id+self.num_DOF()]
        self.DTarget=[0 for P in self.PTarget]
        for c in self.children:
            c.set_PD_target(x)
        
    def define_ctrl(self,sim,qpos,qvel,pcoef=15000.0,dcoef=100.0):
        for cid,jid,PT,DT in zip(self.ctrl_ids,self.joint_ids,self.PTarget,self.DTarget):
            sim.data.ctrl[cid]=(PT-qpos[jid])*pcoef+(DT-qvel[jid])*dcoef
        for c in self.children:
            c.define_ctrl(sim,qpos,qvel,pcoef,dcoef)
        
    def get_ctrl_address(self,sim):
        self.ctrl_ids=[]
        self.joint_ids=[]
        for name in self.ctrl_names:
            self.ctrl_ids.append(sim.model.actuator_names.index(name))
            self.joint_ids.append(sim.model.get_joint_qpos_addr(name))
        for c in self.children:
            c.get_ctrl_address(sim)
    
class Gripper:
    X,Y,Z=(0,1,2)
    EPS=1e-6
    def __init__(self,*,base_radius=0.5,finger_length=0.3,finger_width=0.2,thick=0.1,hinge_rad=0.02,hinge_thick=0.02):
        self.base=tm.creation.cylinder(base_radius,thick,sections=32)
        self.finger=tm.creation.box([thick,finger_width,finger_length])
        #subdivide finger
        for i in range(4):
            faces=[]
            for fid,f in enumerate(self.finger.faces):
                v0=self.finger.vertices[f[0]]
                v1=self.finger.vertices[f[1]]
                v2=self.finger.vertices[f[2]]
                if abs(v0[Gripper.X]-v1[Gripper.X])<Gripper.EPS and abs(v0[Gripper.X]-v2[Gripper.X])<Gripper.EPS:
                    faces.append(fid)
                if abs(v0[Gripper.Z]-v1[Gripper.Z])<Gripper.EPS and abs(v0[Gripper.Z]-v2[Gripper.Z])<Gripper.EPS:
                    faces.append(fid)
            self.finger=self.finger.subdivide(faces)
        #hinge
        self.hinge=tm.creation.cylinder(hinge_rad,hinge_thick)
        self.hinge.apply_transform(tm.transformations.rotation_matrix(angle=math.pi/2,direction=[1,0,0]))
            
    def finger_width(self):
        return self.finger.bounds[1][Gripper.Y]-self.finger.bounds[0][Gripper.Y]
    
    def finger_length(self):
        return self.finger.bounds[1][Gripper.Z]-self.finger.bounds[0][Gripper.Z]
            
    def hinge_radius(self):
        return (self.hinge.bounds[1][Gripper.Z]-self.hinge.bounds[0][Gripper.Z])/2*1.1
            
    def hinge_thick(self):
        return self.hinge.bounds[1][Gripper.Y]-self.hinge.bounds[0][Gripper.Y]
            
    def base_rad(self):
        return self.base.bounds[1][Gripper.X]-self.base.bounds[0][Gripper.X]
       
    def thick(self):
        return self.base.bounds[1][Gripper.Z]-self.base.bounds[0][Gripper.Z]
            
    def get_finger(self,top_hinge=True,bot_hinge=True,*,finger_width=None,finger_length=None,finger_curvature=None):
        if finger_width is None:
            scaleY=1
            finger_width=self.finger_width()
        scaleY=finger_width/self.finger_width()
        
        if finger_length is None:
            scaleZ=1
            finger_length=self.finger_length()
        scaleZ=finger_length/self.finger_length()
        
        if finger_curvature is None:
            finger_curvature=0
        
        scene=tm.Scene()
        finger=self.finger.copy().apply_scale([1,scaleY,scaleZ])
        for v in finger.vertices:
            if v[Gripper.Y]>-finger_width/2+Gripper.EPS and v[Gripper.Y]<finger_width/2-Gripper.EPS:
                v[Gripper.X]+=finger_curvature*(finger_width*finger_width/4-v[Gripper.Y]*v[Gripper.Y])
        finger.update_vertices([True for i in range(len(finger.vertices))])
        scene.add_geometry(finger)
        
        if top_hinge:
            hinge=self.hinge.copy()
            hinge.apply_translation([0,( finger_width-self.hinge_thick())/2,finger_length/2+self.hinge_radius()])
            scene.add_geometry(hinge)
            hinge=self.hinge.copy()
            hinge.apply_translation([0,(-finger_width+self.hinge_thick())/2,finger_length/2+self.hinge_radius()])
            scene.add_geometry(hinge)
            
        if bot_hinge:
            hinge=self.hinge.copy()
            hinge.apply_translation([0,( finger_width-self.hinge_thick()*3.1)/2,-finger_length/2-self.hinge_radius()])
            scene.add_geometry(hinge)
            hinge=self.hinge.copy()
            hinge.apply_translation([0,(-finger_width+self.hinge_thick()*3.1)/2,-finger_length/2-self.hinge_radius()])
            scene.add_geometry(hinge)
            
        scene.apply_translation([0,0,finger_length/2+self.hinge_radius()])
        return scene,tm.transformations.translation_matrix([0,0,finger_length+self.hinge_radius()*2])
            
    def get_base(self,base_off=0.4,num_finger=3,*,finger_width=None,base_rad=None):
        if finger_width is None:
            finger_width=self.finger_width()
            
        if base_rad is None:
            scaleXY=1
            base_rad=self.base_rad()
        scaleXY=base_rad/self.base_rad()
        
        scene=tm.Scene()
        base=self.base.copy()
        base.apply_scale([scaleXY,scaleXY,1]) 
        scene.add_geometry(base)
        trans=[]
        for i in range(num_finger):
            R=tm.transformations.rotation_matrix(angle=math.pi*2*i/num_finger,direction=[0,0,1])
            #left
            T=tm.transformations.translation_matrix([base_off,( finger_width-self.hinge_thick())/2,self.thick()/2+self.hinge_radius()])
            hinge=self.hinge.copy()
            hinge.apply_transform(np.matmul(R,T))
            scene.add_geometry(hinge)
            #right
            T=tm.transformations.translation_matrix([base_off,(-finger_width+self.hinge_thick())/2,self.thick()/2+self.hinge_radius()])
            hinge=self.hinge.copy()
            hinge.apply_transform(np.matmul(R,T))
            scene.add_geometry(hinge)
            #trans
            T=tm.transformations.translation_matrix([base_off,0,self.thick()/2+self.hinge_radius()])
            trans.append(np.matmul(R,T))
        return scene,trans
            
    def get_robot(self,base_off=0.4,num_finger=3,num_segment=3,*,   \
                  finger_width=None,finger_length=None,finger_curvature=None,base_rad=None):
        fingerTop,transF=self.get_finger(top_hinge=False,finger_width=finger_width,finger_length=finger_length,finger_curvature=finger_curvature)
        finger,transF=self.get_finger(finger_width=finger_width,finger_length=finger_length,finger_curvature=finger_curvature)
        base,transB=self.get_base(base_off=base_off,num_finger=num_finger,finger_width=finger_width,base_rad=base_rad)
        
        root=Link(base,'base','base')
        for fid,TB in enumerate(transB):
            parent=root
            for s in range(num_segment):
                name='finger_'+str(fid)+'_'+str(s)
                nameMesh='fingerTop' if s==num_segment-1 else 'finger'
                parent=Link(fingerTop if s==num_segment-1 else finger,name,nameMesh,TB if s==0 else transF,parent)
        root.assign_id()
        return root
        
if __name__=='__main__':
    auto_download()
    
    gripper=Gripper()
    path='data/gripper'
    root=ET.Element('mujoco')
    set_simulator_option(root)
    asset=ET.SubElement(root,'asset')
    body=ET.SubElement(root,'worldbody')
    actuator=ET.SubElement(root,'actuator')
    link=gripper.get_robot(base_off=0.3,finger_width=0.4,finger_curvature=2)
    link.compile_gripper(body,asset,actuator,path)
    
    open(path+'/gripper.xml','w').write(ET.tostring(root,pretty_print=True).decode())
    model=mjc.load_model_from_path(path+'/gripper.xml')
    sim=mjc.MjSim(model)
    link.get_ctrl_address(sim)
    viewer=mjc.MjViewer(sim)
    
    state=sim.get_state()
    link.set_PD_target([0.0 for i in range(6)]+[1.9,0.9])
    while True:
        state=sim.get_state()
        link.define_ctrl(sim,state.qpos,state.qvel)
        sim.step()
        viewer.render()