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

def compile_body(name,b,asset,g,DUMMY_NAMES=set(),force=False,material=None):
    b.set('name',name)
    if isinstance(g,str):
        g=ensure_stl(g,force)
        gname=path_leaf_no_extension(g)
        mesh=ET.SubElement(asset,'mesh')
        mesh.set('file',os.path.abspath(g))
        mesh.set('name',gname)
        compile_body(name,b,asset,{'type':'mesh','mesh':gname,'name':name})
    elif isinstance(g,list):
        for gi in g:
            DUMMY_NAMES=compile_body(name,b,asset,gi,DUMMY_NAMES=DUMMY_NAMES,force=force,material=material)
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
        return DUMMY_NAMES

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
    
def cone_create(f,t,rf,rt,slice=32,name=None):
    return prism_create(f,t,[(0.,rf),(1.,rt)],slice,name)

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
  
def surrogate_object_01(name):
    ret=[]
    t=0.02
    rad=0.025
    ret.append(cone_create(f=[0,0,-0.312],t=[0,0,-0.29],rf=0.2,rt=0.221,name=name+'Base0'))
    ret.append(hollow_prism_create(vss=[(0.221-t,0,-0.29),(0.221,0,-0.29),(0.221-t,0,0.313),(0.221,0,0.312)],axis=(0,0,1),name=name+'Body0'))
    sliceY=6
    for i in range(sliceY):
        phi0=-math.pi/2+math.pi*i/sliceY
        pt0=(0.221+0.16*math.cos(phi0),0,0.16*math.sin(phi0)+0.01)
        phi1=-math.pi/2+math.pi*(i+1)/sliceY
        pt1=(0.221+0.16*math.cos(phi1),0,0.16*math.sin(phi1)+0.01)
        ret.append(capsule_create(rad,fromto=list(pt0)+list(pt1),name=name+'HandleSeg'+str(i)))
    return ret
  
def surrogate_object_02(name):
    ret=[]
    ret.append(cone_create(f=[0,0,-0.204],t=[0,0,-0.16],rf=0.16,rt=0.21,name=name+'Base0'))
    ret.append(hollow_prism_create(vss=[(0.13,0,-0.16),(0.21,0,-0.16),(0.22,0,-0.05),(0.26,0,-0.05)],axis=(0,0,1),name=name+'Body0'))
    ret.append(hollow_prism_create(vss=[(0.22,0,-0.05),(0.26,0,-0.05),(0.265,0,0.204),(0.29,0,0.204)],axis=(0,0,1),name=name+'Body1'))
    ret.append(capsule_create(rad=0.025,fromto=[0.28,0,0.14,0.37,0,0.11],name=name+'Handle0'))
    ret.append(capsule_create(rad=0.025,fromto=[0.37,0,0.11,0.4,0,0.04],name=name+'Handle1'))
    ret.append(capsule_create(rad=0.025,fromto=[0.4,0,0.04,0.37,0,-0.05],name=name+'Handle2'))
    ret.append(capsule_create(rad=0.025,fromto=[0.37,0,-0.05,0.24,0,-0.1],name=name+'Handle3'))
    return ret
  
def surrogate_object_03(name):
    ret=[]
    t=0.02
    ret.append(box_create([(-0.245  ,-0.245+t),(-0.245  , 0.245  ),(-0.276  , 0.276   )],name=name+'Left'))
    ret.append(box_create([( 0.245-t, 0.245  ),(-0.245  , 0.245  ),(-0.276  , 0.276   )],name=name+'Right'))
    ret.append(box_create([(-0.245  , 0.245  ),(-0.245  ,-0.245+t),(-0.276  , 0.276   )],name=name+'Front'))
    ret.append(box_create([(-0.245  , 0.245  ),( 0.245-t, 0.245  ),(-0.276  , 0.276   )],name=name+'Back'))
    ret.append(box_create([(-0.245  , 0.245  ),(-0.245  , 0.245  ),(-0.276  ,-0.276+t)],name=name+'Base0'))
    ret.append(box_create([( 0.245  , 0.425  ),(-0.03   , 0.03   ),( 0.220-t, 0.220  )],name=name+'Handle1'))
    ret.append(box_create([( 0.425-t, 0.425  ),(-0.03   , 0.03   ),(-0.160  , 0.220  )],name=name+'Handle2'))
    ret.append(box_create([( 0.245  , 0.425  ),(-0.03   , 0.03   ),(-0.160  ,-0.160+t)],name=name+'Handle3'))
    return ret
   
def surrogate_object_04(name):
    ret=[]
    ret.append(cylinder_create(rad=0.27,fromto=[0,0,-0.238,0,0,-0.23],name=name+'Base0'))
    ret.append(hollow_prism_create(vss=[(0.31,0,0),(0.245,0,0),(0.26,0,0.238),(0.28,0,0.238),(0.26,0,-0.238),(0.28,0,-0.238)],axis=(0,0,1),name=name+'Body0'))
    return ret
   
def surrogate_object_05(name):
    ret=[]
    ret.append(prism_create(f=(0,0,-0.422),t=(0,0,0.199),dss=[(0.,0.11),(0.9,0.2),(1.,0.2)],name=name+'Body0'))
    ret.append(cylinder_create(rad=0.21,fromto=[0,0,0.17,0,0,0.199],name=name+'Cap0'))
    return ret

def surrogate_object_06(name):
    ret=[]
    t=0.02
    r=0.25
    ret.append(prism_create(f=(0,0,-0.353),t=(0,0,-0.353+t),dss=[(0.,r),(1.,r)],name=name+'Base0',slice=6,x0=[1,0,0]))
    ret+=hollow_prism_create(vss=[(r-t,0,-0.353+t),(r,0,-0.353+t),(r-t,0,0.353),(r,0,0.353)],axis=(0,0,1),name=name+'Body0',slice=6)
    return ret

def surrogate_object_07(name):
    ret=[]
    ret.append(cylinder_create(rad=0.188,fromto=[0,0,-0.375,0,0,-0.328],name=name+'Base0'))
    ret.append(cone_create(f=[0,0,-0.328],t=[0,0,-0.25],rf=0.1,rt=0.,name=name+'Base1'))
    ret.append(cylinder_create(rad=0.05,fromto=[0,0,-0.328,0,0,-0.13],name=name+'Base2'))
    ret.append(cone_create(f=[0,0,-0.14],t=[0,0,-0.115],rf=0.07,rt=0.105,name=name+'Base3'))
    ret+=hollow_prism_create(vss=[(0.08,0,-0.115),(0.105,0,-0.115),(0.209,0,0.375),(0.234,0,0.375)],axis=[0,0,1],name=name+'Body0')
    return ret

def surrogate_object_08(name):
    ret=[]
    t=0.01
    t2=0.005
    ret.append(prism_create(f=(0,0,-0.47),t=(0,0,-0.4),dss=[(0.,0.095),(0.2,0.096),(1.,0.)],name=name+'Base0'))
    ret.append(cylinder_create(rad=0.013,fromto=[0,0,-0.46,0,0,-0.01],name=name+'Base1'))
    ret.append(prism_create(f=(0,0,-0.01),t=(0,0,0.25),dss=[(0.,0.03),(0.5,0.062),(1.,0.06)],name=name+'Body0'))
    ret.append(hollow_prism_create(vss=[(0.06-t,0,0.25),(0.06,0,0.25),(0.07-t,0,0.35),(0.07,0,0.35)],axis=(0,0,1),name=name+'Body1'))
    ret.append(hollow_prism_create(vss=[(0.07-t,0,0.35),(0.07,0,0.35),(0.12-t2,0,0.47),(0.12,0,0.47)],axis=(0,0,1),name=name+'Body2'))
    return ret

def surrogate_object_09(name):
    ret=[]
    ret.append(prism_create(f=(0,0,-0.443),t=(0,0,-0.415),dss=[(0.,0.145),(0.3,0.145),(1.,0.)],name=name+'Base0'))
    ret.append(cone_create(f=(0,0,-0.435),t=(0,0,0.1),rf=0.02,rt=0,name=name+'Base1'))
    ret.append(cone_create(f=(0,0,-0.435),t=(0,0,0.1),rf=0.0,rt=0.02,name=name+'Base2'))
    ret.append(cone_create(f=(0,0,0.1),t=(0,0,0.14),rf=0.02,rt=0.07,name=name+'Base3'))
    ret.append(hollow_prism_create(vss=[(0.04,0,0.14),(0.07,0,0.14),(0.125,0,0.25),(0.145,0,0.25)],axis=(0,0,1),name=name+'Body0'))
    ret.append(hollow_prism_create(vss=[(0.125,0,0.25),(0.145,0,0.25),(0.147,0,0.35),(0.164,0,0.35)],axis=(0,0,1),name=name+'Body1'))
    ret.append(hollow_prism_create(vss=[(0.147,0,0.35),(0.164,0,0.35),(0.137,0,0.443),(0.152,0,0.443)],axis=(0,0,1),name=name+'Body2'))
    return ret

def surrogate_object_10(name):
    ret=[]
    t=0.02
    ret.append(prism_create(f=(0,0,-0.274),t=(0,0,-0.225),dss=[(0.,0.13),(0.2,0.122),(1.,0.)],name=name+'Base0'))
    ret.append(cylinder_create(rad=0.028,fromto=[0,0,-0.274,0,0,0.045],name=name+'Base1'))
    ret.append(hollow_prism_create(vss=[(0.028,0,0.045-t),(0.028,0,0.045),(0.122,0,0.072-t),(0.122,0,0.072)],axis=(0,0,1),name=name+'Body0'))
    ret.append(hollow_prism_create(vss=[(0.122,0,0.072-t),(0.122,0,0.072),(0.24-t,0,0.15),(0.24,0,0.15)],axis=(0,0,1),name=name+'Body1'))
    ret.append(hollow_prism_create(vss=[(0.24-t,0,0.15),(0.24,0,0.15),(0.296-t,0,0.274),(0.296,0,0.274)],axis=(0,0,1),name=name+'Body2'))
    return ret

def compare_debug(id,sur,move_x):
    path='data/'
    root=ET.Element('mujoco')
    set_simulator_option(root)
    asset=ET.SubElement(root,'asset')
    body=ET.SubElement(root,'worldbody')
    object_list=glob.glob('data/ObjectNet3D/CAD/off/cup/[0-9][0-9].off')
    
    b=ET.SubElement(body,'body')
    compile_body('Real',b,asset,object_list[id])
    print('Comparing for %s!'%object_list[id])
    
    b=ET.SubElement(body,'body')
    compile_body('Surrogate',b,asset,sur('Surrogate'))
    joint=ET.SubElement(b,'joint')
    joint.set('axis','1 0 0')
    joint.set('type','slide')
    
    open(path+'/compare.xml','w').write(ET.tostring(root,pretty_print=True).decode())
    model=mjc.load_model_from_path(path+'/compare.xml')
    sim=mjc.MjSim(model)
    viewer=mjc.MjViewer(sim)
    
    state=sim.get_state()
    state.qpos[0]=move_x
    sim.set_state(state)
    while True:
        sim.step()
        viewer.render()

if __name__=='__main__':
    auto_download()
    compare_debug(0,surrogate_object_05,1.)
    #compare_debug(1,surrogate_object_03,1.)
    #compare_debug(2,surrogate_object_07,1.)
    #compare_debug(3,surrogate_object_10,1.)
    #compare_debug(4,surrogate_object_02,1.)
    #compare_debug(5,surrogate_object_08,1.)
    #compare_debug(6,surrogate_object_04,1.)
    #compare_debug(7,surrogate_object_01,1.)
    #compare_debug(8,surrogate_object_09,1.)
    #compare_debug(9,surrogate_object_06,1.)