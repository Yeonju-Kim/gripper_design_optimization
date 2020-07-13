from compile_objects import *

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
    t=0.02
    ret.append(hollow_prism_create(vss=[(0.11,0,-0.422),(0.11-t,0,-0.422),(0.2,0,0.199),(0.2-t,0,0.199)],axis=(0,0,1),name=name+'Body0'))
    ret.append(cylinder_create(rad=0.11,fromto=[0,0,-0.422,0,0,-0.422+t],name=name+'Base0'))
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

def get_dataset_cup(use_surrogate):
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
    return objs

def compare_debug(sur,sep,groundtruth=False):
    path='data/'
    root=ET.Element('mujoco')
    set_simulator_option(root)
    asset=ET.SubElement(root,'asset')
    body=ET.SubElement(root,'worldbody')
    if groundtruth:
        auto_download()
        object_list=glob.glob('data/ObjectNet3D/CAD/off/cup/[0-9][0-9].off')
    create_fog(root)
    create_skybox(asset)
    create_floor(asset,body,pos=[0,0,-.5])
    create_light(body)
    create_material(asset)
    
    for id in range(len(sur)):
        if groundtruth:
            b=ET.SubElement(body,'body')
            compile_body('Real'+str(id),b,asset,object_list[id],collision=False,material='geom')
            joint=ET.SubElement(b,'joint')
            joint.set('axis','1 0 0')
            joint.set('type','slide')
            joint=ET.SubElement(b,'joint')
            joint.set('axis','0 1 0')
            joint.set('type','slide')
        
        b=ET.SubElement(body,'body')
        compile_body('Surrogate'+str(id),b,asset,sur[id]('Surrogate'+str(id)),collision=False,material='geom')
        joint=ET.SubElement(b,'joint')
        joint.set('axis','1 0 0')
        joint.set('type','slide')
        joint=ET.SubElement(b,'joint')
        joint.set('axis','0 1 0')
        joint.set('type','slide')
    
    open(path+'/compare.xml','w').write(ET.tostring(root,pretty_print=True).decode())
    model=mjc.load_model_from_path(path+'/compare.xml')
    sim=mjc.MjSim(model)
    viewer=mjc.MjViewer(sim)
    
    state=sim.get_state()
    for id in range(len(sur)):
        if groundtruth:
            state.qpos[id*4+0]=id*sep
            state.qpos[id*4+1]=0
            state.qpos[id*4+2]=id*sep
            state.qpos[id*4+3]=sep
        else:
            state.qpos[id*2+0]=id*sep
            state.qpos[id*2+1]=0
    sim.set_state(state)
    while True:
        sim.step()
        viewer.render()

if __name__=='__main__':
    compare_debug([surrogate_object_05, \
                   surrogate_object_03, \
                   surrogate_object_07, \
                   surrogate_object_10, \
                   surrogate_object_02, \
                   surrogate_object_08, \
                   surrogate_object_04, \
                   surrogate_object_01, \
                   surrogate_object_09, \
                   surrogate_object_06],1.)