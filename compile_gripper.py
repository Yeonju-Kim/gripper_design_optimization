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
        mesh = scene_or_mesh
        assert(isinstance(mesh, tm.Trimesh))
    return mesh

def set_simulator_option(root,dt=0.001,withparent=True):
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
    WRITTEN_NAMES=None
    
    def __init__(self,geom,name,trans=None,parent=None,affine=None):
        self.geom=geom
        self.name=name
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

    def compile_gripper(self,body,asset,actuator,path,damping=1000.0,gear=1,*,name_suffix=''):
        b=ET.SubElement(body,'body')
        b.set('pos',self.get_pos())
        b.set('quat',self.get_quat())
        b.set('name',self.name)
        #geom
        if self.parent is None:
            Link.WRITTEN_NAMES=set()
        self.add_geom(b,asset,self.geom,path,name_suffix)
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
        #children
        for c in self.children:
            c.compile_gripper(b,asset,actuator=actuator,path=path,gear=gear,name_suffix=name_suffix)

    def add_geom(self,b,asset,g,path,name_suffix):
        if isinstance(g,list):
            for gi in g:
                self.add_geom(b,asset,gi,path,name_suffix)
        else:
            assert isinstance(g,dict)
            if 'mesh' in g:
                #asset
                entity=g['mesh']
                nameMesh=g['name']
                if nameMesh not in Link.WRITTEN_NAMES:
                    if not os.path.exists(path):
                        os.mkdir(path)
                    if not path.endswith('/'):
                        path+='/'
                    as_mesh(entity).export(path+nameMesh+name_suffix+'.stl')
                    mesh=ET.SubElement(asset,'mesh')
                    mesh.set('file',nameMesh+name_suffix+'.stl')
                    mesh.set('name',nameMesh)
                    Link.WRITTEN_NAMES.add(nameMesh)
                #substitute name
                g={'type':'mesh','mesh':nameMesh,'name':self.name}
            #geom
            geom=ET.SubElement(b,'geom')
            for k,v in g.items():
                geom.set(k,str(v))

    def finger_id(self):
        if self.parent.parent is None:
            return self.parent.children.index(self)
        else: return self.parent.finger_id()
     
    def set_PD_target(self,x,vx=None,state=None):
        #set PD target to the provided DOF
        #if state!=None, then MuJoCo's state is cleared to match PD target (reset simulator)
        self.PTarget=x[self.id:self.id+self.num_DOF()]
        self.DTarget=[0 for P in self.PTarget] if vx is None else vx[self.id:self.id+self.num_DOF()]
        if state is not None:
            if self.parent is None:
                #this is base, set it to initial_pos and approach dir
                for off,jid in enumerate(self.joint_ids):
                    state.qpos[jid]=x[off+self.id]
            else:
                #this is non-base, set it to 0
                for off,jid in enumerate(self.joint_ids):
                    state.qpos[jid]=0.0
            for jid in self.joint_ids:
                state.qvel[jid]=0.0
        for c in self.children:
            c.set_PD_target(x,vx=vx,state=state)
        
    def define_ctrl(self,sim,qpos,qvel,pcoef=15000.0,dcoef=1000.0):
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
            
    def fetch_q(self,qvec):
        ret=[qvec[jid] for jid in self.joint_ids]
        for c in self.children:
            ret+=c.fetch_q(qvec)
        return ret
    
class Gripper:
    X,Y,Z=(0,1,2)
    EPS=1e-6
    def __init__(self,*,base_radius=0.5,finger_length=0.3,finger_width=0.2,thick=0.1,hinge_rad=0.02,hinge_thick=0.02):
        self.base=Gripper.cylinder_create(base_radius,thick)
        self.base=Gripper.cylinder_transform(self.base,[0,0,-thick/2])
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
        #finger
        #slice=16
        #self.finger=[]
        #for i in range(slice):
        #    l,r=(-finger_width/2,finger_width/2)
        #    alpha0,alpha1=(i/slice,(i+1)/slice)
        #    a,b=(l*(1-alpha0)+r*alpha0,l*(1-alpha1)+r*alpha1)
        #    self.finger.append(Gripper.box_create([thick,(a,b),finger_length]))
        #hinge
        self.hinge=Gripper.cylinder_create(hinge_rad,hinge_thick)
        self.hinge=Gripper.cylinder_transform(self.hinge,[0,0,-hinge_thick/2])
        self.hinge=Gripper.cylinder_transform(self.hinge,tm.transformations.rotation_matrix(angle=math.pi/2,direction=[1,0,0]))
    
    def cylinder_create(rad,height):
        return {'type':'cylinder','size':rad,'fromto':'0 0 0 0 0 %f'%height}
     
    def cylinder_scale_xy(c,scaleXY):
        ret=dict()
        for k,v in c.items():
            if k=='radius':
                ret[k]=v*scaleXY
            else: ret[k]=v
        return ret
    
    def cylinder_transform(c,T):
        ret=c.copy()
        ft=[float(val) for val in ret['fromto'].split(' ')]
        if not isinstance(T,np.ndarray):
            T=np.array(T)
        if T.shape==(3,4) or T.shape==(4,4):
            f=T[0:3,0:3].dot(ft[0:3])+T[0:3,3]
            t=T[0:3,0:3].dot(ft[3:6])+T[0:3,3]
        else:
            assert T.shape==(3,)
            f=ft[0:3]+T
            t=ft[3:6]+T
        ret['fromto']='%f %f %f %f %f %f'%tuple(f.tolist()+t.tolist())
        return ret
    
    def mesh_transform(sg,T):
        if 'mesh' in sg:
            assert isinstance(sg['mesh'],tm.Scene) or isinstance(sg['mesh'],tm.Trimesh)
            ret=sg.copy()
            if isinstance(T,list):
                ret['mesh'].apply_translation(T)
            else: ret['mesh'].apply_transform(T)
            return ret
        else:
            assert 'vertex' in sg
            vss=[float(val) for val in ret['vertex'].split(' ')]
            vssr=[]
            for i in range(len(vss)//3):
                v=vss[i*3:i*3+3]
                if not isinstance(T,np.ndarray):
                    T=np.array(T)
                if T.shape==(3,4) or T.shape==(4,4):
                    v=T[0:3,0:3].dot(v)+T[0:3,3]
                else:
                    assert T.shape==(3,)
                    v+=T
                vssr+=v.tolist()
            vert=''
            for v in vssr:
                vert+=str(v)+' '
                
            #transform
            ret=sg.copy()
            ret['vertex']=vert[0:len(vert)-1]
            return ret
    
    def scene_transform(s,T):
        ret=[]
        for sg in s:
            if sg['type']=='mesh':
                ret.append(Gripper.mesh_transform(sg,T))
            elif sg['type']=='cylinder':
                ret.append(Gripper.cylinder_transform(sg,T))
            else: 
                raise RuntimeError('Unknown geometry type for: %s!'%str(sg))
        return ret
       
    def finger_width(self):
        return self.finger.bounds[1][Gripper.Y]-self.finger.bounds[0][Gripper.Y]
    
    def finger_length(self):
        return self.finger.bounds[1][Gripper.Z]-self.finger.bounds[0][Gripper.Z]
            
    def hinge_radius(self):
        return self.hinge['size']*1.1
            
    def hinge_thick(self):
        ft=[float(val) for val in self.hinge['fromto'].split(' ')]
        return np.linalg.norm(np.array(ft[0:3])-np.array(ft[3:6]))
            
    def base_rad(self):
        return self.base['size']
       
    def thick(self):
        ft=[float(val) for val in self.base['fromto'].split(' ')]
        return np.linalg.norm(np.array(ft[0:3])-np.array(ft[3:6]))
            
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
        
        finger=self.finger.copy().apply_scale([1,scaleY,scaleZ])
        for v in finger.vertices:
            if v[Gripper.Y]>-finger_width/2+Gripper.EPS and v[Gripper.Y]<finger_width/2-Gripper.EPS:
                v[Gripper.X]+=finger_curvature*(finger_width*finger_width/4-v[Gripper.Y]*v[Gripper.Y])
        finger.update_vertices([True for i in range(len(finger.vertices))])
        scene=[{'type':'mesh','mesh':finger,'name':'finger'}]
        
        if top_hinge:
            scene.append(Gripper.cylinder_transform(self.hinge,[0,( finger_width-self.hinge_thick())/2,finger_length/2+self.hinge_radius()]))
            scene.append(Gripper.cylinder_transform(self.hinge,[0,(-finger_width+self.hinge_thick())/2,finger_length/2+self.hinge_radius()]))
            
        if bot_hinge:
            scene.append(Gripper.cylinder_transform(self.hinge,[0,( finger_width-self.hinge_thick()*3.1)/2,-finger_length/2-self.hinge_radius()]))
            scene.append(Gripper.cylinder_transform(self.hinge,[0,(-finger_width+self.hinge_thick()*3.1)/2,-finger_length/2-self.hinge_radius()]))
            
        scene=Gripper.scene_transform(scene,[0,0,finger_length/2+self.hinge_radius()])
        return scene,tm.transformations.translation_matrix([0,0,finger_length+self.hinge_radius()*2])
            
    def get_base(self,base_off=0.4,num_finger=3,*,finger_width=None,base_rad=None):
        if finger_width is None:
            finger_width=self.finger_width()
            
        if base_rad is None:
            scaleXY=1
            base_rad=self.base_rad()
        scaleXY=base_rad/self.base_rad()
        
        scene=[Gripper.cylinder_scale_xy(self.base,scaleXY)]
        trans=[]
        for i in range(num_finger):
            R=tm.transformations.rotation_matrix(angle=math.pi*2*i/num_finger,direction=[0,0,1])
            #left
            T=tm.transformations.translation_matrix([base_off,( finger_width-self.hinge_thick())/2,self.thick()/2+self.hinge_radius()])
            scene.append(Gripper.cylinder_transform(self.hinge,np.matmul(R,T)))
            #right
            T=tm.transformations.translation_matrix([base_off,(-finger_width+self.hinge_thick())/2,self.thick()/2+self.hinge_radius()])
            scene.append(Gripper.cylinder_transform(self.hinge,np.matmul(R,T)))
            #trans
            T=tm.transformations.translation_matrix([base_off,0,self.thick()/2+self.hinge_radius()])
            trans.append(np.matmul(R,T))
        return scene,trans
            
    def get_robot(self,base_off=0.4,num_finger=3,num_segment=3,*,   \
                  finger_width=None,finger_length=None,finger_curvature=None,base_rad=None):
        fingerTop,transF=self.get_finger(top_hinge=False,finger_width=finger_width,finger_length=finger_length,finger_curvature=finger_curvature)
        finger,transF=self.get_finger(finger_width=finger_width,finger_length=finger_length,finger_curvature=finger_curvature)
        base,transB=self.get_base(base_off=base_off,num_finger=num_finger,finger_width=finger_width,base_rad=base_rad)
        
        root=Link(base,'base')
        for fid,TB in enumerate(transB):
            parent=root
            for s in range(num_segment):
                name='finger_'+str(fid)+'_'+str(s)
                geom=fingerTop if s==num_segment-1 else finger
                parent=Link(geom,name,TB if s==0 else transF,parent)
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
    link.set_PD_target([0.0 for i in range(6)]+[0.5,-0.2])
    while True:
        state=sim.get_state()
        link.define_ctrl(sim,state.qpos,state.qvel)
        sim.step()
        viewer.render()