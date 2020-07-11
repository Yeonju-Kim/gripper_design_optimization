import glob,ntpath,os,random,math
import mujoco_py as mjc
import lxml.etree as ET
import trimesh as tm
import numpy as np

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

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def path_leaf_no_extension(path):
    name=path_leaf(path)
    if name.find('.')!=-1:
        return os.path.splitext(name)[0]
    else: return name

def auto_download():
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('data/ObjectNet3D_cads.zip'):
        os.system('wget -P data ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_cads.zip')
    if not os.path.exists('data/ObjectNet3D'):
        os.system('unzip data/ObjectNet3D_cads.zip -d data')

def ensure_stl(obj,force=False):
    if (not obj.endswith('stl') and not obj.endswith('STL')) or force:
        ret=os.path.splitext(obj)[0]+'.stl'
        if not os.path.exists(ret) or force:
            mesh=tm.exchange.load.load(obj)
            mesh.export(ret)
        return ret
    else: return obj
    
def get_COM(obj):
    mesh=tm.exchange.load.load(obj)
    return mesh.center_mass,mesh.bounds

def compile_body(name,b,asset,g,DUMMY_NAMES=set(),force=False,material=None,collision=True):
    b.set('name',name)
    if isinstance(g,str):
        g=ensure_stl(g,force)
        gname=path_leaf_no_extension(g)
        mesh=ET.SubElement(asset,'mesh')
        mesh.set('file',os.path.abspath(g))
        mesh.set('name',gname)
        compile_body(name,b,asset,{'type':'mesh','mesh':gname,'name':name},DUMMY_NAMES=DUMMY_NAMES,force=force,material=material,collision=collision)
    elif isinstance(g,list):
        for gi in g:
            DUMMY_NAMES=compile_body(name,b,asset,gi,DUMMY_NAMES=DUMMY_NAMES,force=force,material=material,collision=collision)
        return DUMMY_NAMES
    else:
        assert isinstance(g,dict)
        if 'vertex' in g:
            verts=g['vertex']
            nameMesh=g['name']
            mesh=ET.SubElement(asset,'mesh')
            mesh.set('vertex',verts)
            mesh.set('name',nameMesh)
            #substitute name
            g={'type':'mesh','mesh':nameMesh,'name':name+':'+nameMesh}
        #geom
        geom=ET.SubElement(b,'geom')
        for k,v in g.items():
            geom.set(k,str(v))
        if 'name' not in g:
            geom.set('name',name+':dummy'+str(len(DUMMY_NAMES)))
            DUMMY_NAMES.add(len(DUMMY_NAMES))
        if material is not None:
            geom.set('material',material)
        if not collision:
            geom.set('contype','0')
            geom.set('conaffinity','0')
        return DUMMY_NAMES

def prim_transform(c,T):
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
    assert 'vertex' in sg
    vss=[float(val) for val in sg['vertex'].split(' ')]
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
    
def geom_transform(obj,T):
    if 'fromto' in obj:
        return prim_transform(obj,T)
    else: 
        assert 'vertex' in obj
        return mesh_transform(obj,T)

def scene_transform(s,T):
    ret=[]
    for sg in s:
        ret.append(geom_transform(sg,T))
    return ret

def prim_scale_xy(c,scaleXY):
    ret=dict()
    for k,v in c.items():
        if k=='radius' or k=='size':
            ret[k]=v*scaleXY
        else: ret[k]=v
    return ret
      
def prim_scale(c,T):
    ret=prim_transform(c,tm.transformations.scale_matrix(T[0]))
    ret=prim_scale_xy(ret,T[0])
    return ret
      
def mesh_scale(sg,T):
    assert 'vertex' in sg
    vss=[float(val) for val in sg['vertex'].split(' ')]
    vssr=[]
    for i in range(len(vss)//3):
        v=vss[i*3:i*3+3]
        vssr+=[v[0]*T[0],v[1]*T[1],v[2]*T[2]]
    vert=''
    for v in vssr:
        vert+=str(v)+' '
        
    #transform
    ret=sg.copy()
    ret['vertex']=vert[0:len(vert)-1]
    return ret
   
def geom_scale(obj,T):
    if 'fromto' in obj:
        return prim_scale(obj,T)
    else: 
        assert 'vertex' in obj
        return mesh_scale(obj,T)
       
def scene_scale(s,T):
    ret=[]
    for sg in s:
        ret.append(geom_scale(sg,T))
    return ret
               
def hollow_prism_create(vss,axis,slice=32,name=None):
    ret=[]
    for i in range(slice):
        R0=tm.transformations.rotation_matrix(angle=math.pi*2*i/slice,direction=axis)
        R1=tm.transformations.rotation_matrix(angle=math.pi*2*(i+1)/slice,direction=axis)
        vssi=[R0[0:3,0:3].dot(v).tolist() for v in vss]
        vssi+=[R1[0:3,0:3].dot(v).tolist() for v in vss]
        
        vert=''
        for v in vssi:
            vert+=str(v[0])+' '+str(v[1])+' '+str(v[2])+' '
        g={'type':'mesh','vertex':vert[0:len(vert)-1]}
        if name is not None:
            g['name']=name+'Seg'+str(i)
        ret.append(g)
    return ret

def prism_create(f,t,dss,slice=32,name=None,x0=None):
    f=np.array(list(f))
    t=np.array(list(t))
    
    z=t-f
    z/=np.linalg.norm(z)
    if x0 is not None:
        x=np.array(x0,dtype=float)
        x/=np.linalg.norm(x)
        y=np.cross(z,x)
    else:
        while True:
            x=[random.uniform(-1.,1.) for i in range(3)]
            y=np.cross(x,z)
            if np.linalg.norm(y)<1e-6:
                continue
            y/=np.linalg.norm(y)
            x=np.cross(y,z)
            break
    
    vss=[]
    for dz,dist in dss:
        dz*=np.linalg.norm(t-f)
        for i in range(slice):
            R=tm.transformations.rotation_matrix(angle=math.pi*2*i/slice,direction=z)
            if dist>0.:
                vss+=(f+z*dz+R[0:3,0:3].dot(x*dist)).tolist()
        if dist==0.:
            vss+=(f+z*dz).tolist()
        
    vert=''
    for v in vss:
        vert+=str(v)+' '
    ret={'type':'geom','vertex':vert[0:len(vert)-1]}
    if name is not None:
        ret['name']=name
    return ret
    
def cone_create(f,t,rf,rt,slice=32,name=None,x0=None):
    return prism_create(f,t,[(0.,rf),(1.,rt)],slice,name,x0=x0)

def box_create(ext,name=None):
    vssr=[]
    for d in range(3):
        if isinstance(ext[d],float) or isinstance(ext[d],int):
            ext[d]=(-ext[d]/2,ext[d]/2)
    for id in range(8):
        vssr.append(ext[0][0] if (id&1)==0 else ext[0][1])
        vssr.append(ext[1][0] if (id&2)==0 else ext[1][1])
        vssr.append(ext[2][0] if (id&4)==0 else ext[2][1])
    vert=''
    for v in vssr:
        vert+=str(v)+' '
        
    #transform
    ret={'type':'mesh','vertex':vert[0:len(vert)-1]}
    if name is not None:
        ret['name']=name
    return ret
    
def cylinder_create(rad,height=None,fromto=None,name=None):
    ret={'type':'cylinder','size':rad}
    if height is not None:
        ret['fromto']='0 0 0 0 0 %f'%height
    else:
        ret['fromto']='%f %f %f %f %f %f'%(fromto[0],fromto[1],fromto[2],fromto[3],fromto[4],fromto[5])
    if name is not None:
        ret['name']=name
    return ret
  
def capsule_create(rad,height=None,fromto=None,name=None):
    ret={'type':'capsule','size':rad}
    if height is not None:
        ret['fromto']='0 0 0 0 0 %f'%height
    else:
        ret['fromto']='%f %f %f %f %f %f'%(fromto[0],fromto[1],fromto[2],fromto[3],fromto[4],fromto[5])
    if name is not None:
        ret['name']=name
    return ret
  