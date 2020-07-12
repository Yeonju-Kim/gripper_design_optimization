from compile_objects import *
import mujoco_py as mjc
import lxml.etree as ET
import glob,ntpath,os
import trimesh as tm
import numpy as np
import math

class Link:
    WRITTEN_NAMES=None
    DUMMY_NAMES=None
    
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

    def get_pos(self):
        return str(self.trans[0,3])+' '+str(self.trans[1,3])+' '+str(self.trans[2,3])

    def get_quat(self):
        q=tm.transformations.quaternion_from_matrix(self.trans[0:3,0:3])
        return str(q[0])+' '+str(q[1])+' '+str(q[2])+' '+str(q[3])

    def compile_gripper(self,body,asset,actuator,damping=1000.0,gear=1):
        b=ET.SubElement(body,'body')
        b.set('pos',self.get_pos())
        b.set('quat',self.get_quat())
        b.set('name',self.name)
        #geom
        if self.parent is None:
            Link.WRITTEN_NAMES=set()
        Link.DUMMY_NAMES=set()
        self.add_geom(b,asset,self.geom)
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
            c.compile_gripper(b,asset,actuator=actuator,gear=gear)

    def add_geom(self,b,asset,g):
        if isinstance(g,list):
            for gi in g:
                self.add_geom(b,asset,gi)
        else:
            assert isinstance(g,dict)
            if 'vertex' in g:
                verts=g['vertex']
                nameMesh=g['name']
                if nameMesh not in Link.WRITTEN_NAMES:
                    mesh=ET.SubElement(asset,'mesh')
                    mesh.set('vertex',verts)
                    mesh.set('name',nameMesh)
                    Link.WRITTEN_NAMES.add(nameMesh)
                #substitute name
                g={'type':'mesh','mesh':nameMesh,'name':self.name+':'+nameMesh}
            #geom
            geom=ET.SubElement(b,'geom')
            for k,v in g.items():
                geom.set(k,str(v))
            if 'name' not in g:
                geom.set('name',self.name+':dummy'+str(len(Link.DUMMY_NAMES)))
                Link.DUMMY_NAMES.add(len(Link.DUMMY_NAMES))

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
    def __init__(self,*,base_radius=0.25,finger_length=0.15,finger_width=0.2,thick=0.1,hinge_rad=0.02,hinge_thick=0.02):
        self.base=cylinder_create(base_radius,thick)
        self.base=geom_transform(self.base,[0,0,-thick/2])
        #finger
        slice=16
        self.finger=[]
        for i in range(slice):
            l,r=(-finger_width/2,finger_width/2)
            alpha0,alpha1=(i/slice,(i+1)/slice)
            a,b=(l*(1-alpha0)+r*alpha0,l*(1-alpha1)+r*alpha1)
            self.finger.append(box_create([thick,(a,b),finger_length],'fingerSeg%d'%i))
        #hinge
        self.hinge=cylinder_create(hinge_rad,hinge_thick)
        self.hinge=geom_transform(self.hinge,[0,0,-hinge_thick/2])
        self.hinge=geom_transform(self.hinge,tm.transformations.rotation_matrix(angle=math.pi/2,direction=[1,0,0]))
    
    def mesh_vtrans(sg,vtrans):
        if 'mesh' in sg:
            raise RuntimeError('mesh_vtrans not supported for Trimesh!')
        else:
            assert 'vertex' in sg
            vss=[float(val) for val in sg['vertex'].split(' ')]
            vssr=[]
            for i in range(len(vss)//3):
                vssr+=vtrans(vss[i*3:i*3+3])
            vert=''
            for v in vssr:
                vert+=str(v)+' '
                
            #transform
            ret=sg.copy()
            ret['vertex']=vert[0:len(vert)-1]
            return ret
    
    def finger_width(self):
        l,r=(100.,-100.)
        for f in self.finger:
            vss=[float(val) for val in f['vertex'].split(' ')]
            for i in range(len(vss)//3):
                val=vss[i*3+Gripper.Y]
                l,r=(min(l,val),max(r,val))
        return r-l
    
    def finger_length(self):
        l,r=(100.,-100.)
        for f in self.finger:
            vss=[float(val) for val in f['vertex'].split(' ')]
            for i in range(len(vss)//3):
                val=vss[i*3+Gripper.Z]
                l,r=(min(l,val),max(r,val))
        return r-l
            
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
        
        finger=[geom_scale(f.copy(),[1,scaleY,scaleZ]) for f in self.finger]
        def vtrans(v):
            if v[Gripper.Y]>-finger_width/2+Gripper.EPS and v[Gripper.Y]<finger_width/2-Gripper.EPS:
                v[Gripper.X]+=finger_curvature*(finger_width*finger_width/4-v[Gripper.Y]*v[Gripper.Y])
            return v
        scene=[Gripper.mesh_vtrans(f.copy(),vtrans) for f in finger]
        
        if top_hinge:
            scene.append(geom_transform(self.hinge,[0,( finger_width-self.hinge_thick())/2,finger_length/2+self.hinge_radius()]))
            scene.append(geom_transform(self.hinge,[0,(-finger_width+self.hinge_thick())/2,finger_length/2+self.hinge_radius()]))
            
        if bot_hinge:
            scene.append(geom_transform(self.hinge,[0,( finger_width-self.hinge_thick()*3.1)/2,-finger_length/2-self.hinge_radius()]))
            scene.append(geom_transform(self.hinge,[0,(-finger_width+self.hinge_thick()*3.1)/2,-finger_length/2-self.hinge_radius()]))
            
        scene=scene_transform(scene,[0,0,finger_length/2+self.hinge_radius()])
        return scene,tm.transformations.translation_matrix([0,0,finger_length+self.hinge_radius()*2])
            
    def get_base(self,base_off=0.4,num_finger=3,*,finger_width=None,base_rad=None):
        if finger_width is None:
            finger_width=self.finger_width()
            
        if base_rad is None:
            scaleXY=1
            base_rad=self.base_rad()
        scaleXY=base_rad/self.base_rad()
        
        scene=[prim_scale_xy(self.base,scaleXY)]
        trans=[]
        for i in range(num_finger):
            R=tm.transformations.rotation_matrix(angle=math.pi*2*i/num_finger,direction=[0,0,1])
            #left
            T=tm.transformations.translation_matrix([base_off,( finger_width-self.hinge_thick())/2,self.thick()/2+self.hinge_radius()])
            scene.append(geom_transform(self.hinge,np.matmul(R,T)))
            #right
            T=tm.transformations.translation_matrix([base_off,(-finger_width+self.hinge_thick())/2,self.thick()/2+self.hinge_radius()])
            scene.append(geom_transform(self.hinge,np.matmul(R,T)))
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
    gripper=Gripper()
    path='data'
    root=ET.Element('mujoco')
    set_simulator_option(root)
    asset=ET.SubElement(root,'asset')
    body=ET.SubElement(root,'worldbody')
    actuator=ET.SubElement(root,'actuator')
    link=gripper.get_robot(base_off=0.2,finger_width=0.2,finger_length=0.5,finger_curvature=-3.,num_finger=4)
    link.compile_gripper(body,asset,actuator,path)
    
    open(path+'/gripper.xml','w').write(ET.tostring(root,pretty_print=True).decode())
    model=mjc.load_model_from_path(path+'/gripper.xml')
    sim=mjc.MjSim(model)
    link.get_ctrl_address(sim)
    viewer=mjc.MjViewer(sim)
    
    state=sim.get_state()
    link.set_PD_target([0.0 for i in range(6)]+[0.5,-0.1])
    while True:
        state=sim.get_state()
        link.define_ctrl(sim,state.qpos,state.qvel)
        sim.step()
        viewer.render()