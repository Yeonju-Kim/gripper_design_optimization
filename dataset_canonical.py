from compile_objects import *

def canonical_pole(name,base,rad,len,thick=0.02):
    ret=[]
    ret.append(box_create([base,base,thick],name=name+'Base0'))
    ret.append(cylinder_create(rad=rad,fromto=[0,0,0,0,0,len],name=name+'Body0'))
    return name,ret

def pole_thin():
    return canonical_pole('poleThin',0.3,0.02,0.35)

def pole_mid():
    return canonical_pole('poleMid',0.3,0.03,0.35)

def pole_thick():
    return canonical_pole('poleThick',0.3,0.06,0.35)

def canonical_handle(name,base,rad,height,nr,thick=0.02):
    ret=[]
    ret.append(cylinder_create(rad=base,fromto=[0,0,0,0,0,thick],name=name+'Base0'))
    for i in range(nr):
        ret.append(capsule_create(rad=rad,fromto=[-base+rad,0,rad+thick,-base+rad,0,height],name=name+str(i)+'Pole0'))
        ret.append(capsule_create(rad=rad,fromto=[ base-rad,0,rad+thick, base-rad,0,height],name=name+str(i)+'Pole1'))
        ret.append(capsule_create(rad=rad,fromto=[-base+rad,0,height, base-rad,0,height],name=name+str(i)+'Handle'))
        ret[-3:]=scene_transform(ret[-3:],tm.transformations.rotation_matrix(angle=math.pi*i/nr,direction=[0,0,1]))
    return name,ret
    
def single_handle():
    return canonical_handle('singleHandle',0.15,0.02,0.35,1)

def two_handle():
    return canonical_handle('twoHandle',0.15,0.02,0.35,2)

def three_handle():
    return canonical_handle('threeHandle',0.15,0.02,0.35,3)

def canonical_board(name,base,rad,height,nr,thick=0.02):
    ret=[]
    ret.append(cylinder_create(rad=base,fromto=[0,0,0,0,0,thick],name=name+'Base0'))
    for i in range(nr):
        ret.append(box_create([base*2,rad*2,(thick,height)],name=name+str(i)+'Board'))
        ret[-1:]=scene_transform(ret[-1:],tm.transformations.rotation_matrix(angle=math.pi*i/nr,direction=[0,0,1]))
    return name,ret

def single_board():
    return canonical_board('singleBoard',0.15,0.01,0.35,1)

def two_board():
    return canonical_board('twoBoard',0.15,0.01,0.35,2)

def three_board():
    return canonical_board('threeBoard',0.15,0.01,0.35,3)

def canonical_cup(name,base,vss,thick=0.02):
    ret=[]
    t=0.02
    ret.append(cylinder_create(rad=base,fromto=[0,0,0,0,0,thick],name=name+'Base0'))
    vss=[(base,0,0.01)]+[(base+v[0],v[1],thick+v[2]) for v in vss]
    for i in range(len(vss)-1):
        a=vss[i]
        b=vss[i+1]
        c=(a[0]-t,a[1],a[2])
        d=(b[0]-t,b[1],b[2])
        ret+=hollow_prism_create(vss=[a,b,c,d],axis=(0,0,1),name=name+'Body'+str(i))
    return name,ret

def cup1():
    return canonical_cup('cup1',0.12,[(0.05,0,0.35)])

def cup2():
    return canonical_cup('cup2',0.12,[(0,0,0.35)])

def cup3():
    return canonical_cup('cup3',0.20,[(-0.07,0,0.35)])

def canonical_valley(name,base,vss,basey=None,thick=0.02):
    ret=[]
    t=0.02
    if basey is None:
        basey=base
    def negx(verts):
        if isinstance(verts,list):
            return [negx(v) for v in verts]
        else: return (-verts[0],verts[1],verts[2])
    def addy(verts,dy):
        if isinstance(verts,list):
            return [addy(v,dy) for v in verts]
        else: return (verts[0],verts[1]+dy,verts[2])
    ret.append(box_create(ext=[base*2,basey*2,(0,thick)],name=name+'Base0'))
    vss=[(base,0,0.01)]+[(base+v[0],v[1],thick+v[2]) for v in vss]
    for i in range(len(vss)-1):
        for side in range(2):
            a=vss[i]
            b=vss[i+1]
            c=(a[0]-t,a[1],a[2])
            d=(b[0]-t,b[1],b[2])
            verts=addy([a,b,c,d],-basey)+addy([a,b,c,d],basey)
            if side==1:
                verts=negx(verts)
            #verts
            vert=''
            for v in verts:
                vert+=str(v[0])+' '+str(v[1])+' '+str(v[2])+' '
            ret.append({'type':'mesh','vertex':vert[0:len(vert)-1],'name':name+'Body'+str(i*2+side)})
    return name,ret

def valley1():
    return canonical_valley('valley1',0.15,[(0.05,0,0.35)])

def valley2():
    return canonical_valley('valley2',0.15,[(0,0,0.35)])

def valley3():
    return canonical_valley('valley3',0.15,[(-0.05,0,0.35)])

def valley4():
    return canonical_valley('valley4',0.15,[(-0.14,0,0.35)],0.1)

def canonical_table(name,base,rad,height,slice,thick=0.02):
    ret=[]
    ret.append(cylinder_create(rad=base,fromto=[0,0,0,0,0,thick],name=name+'Base0'))
    ret.append(cylinder_create(rad=thick,fromto=[0,0,0,0,0,height],name=name+'Base1'))
    ret.append(cone_create(f=(0,0,height-thick),t=(0,0,height),rf=rad,rt=rad,x0=(1,0,0),name=name+'Body0',slice=slice))
    return name,ret

def table1():
    return canonical_table('table1',0.05,0.15,0.35,32)

def table2():
    return canonical_table('table2',0.05,0.2,0.35,4)

def visualize_debug(obj,sep):
    path='data/'
    root=ET.Element('mujoco')
    set_simulator_option(root)
    asset=ET.SubElement(root,'asset')
    body=ET.SubElement(root,'worldbody')
    create_fog(root)
    create_skybox(asset)
    create_floor(asset,body)
    create_light(body)
    create_material(asset)
    
    for oFunc in obj:
        b=ET.SubElement(body,'body')
        name,geom=oFunc()
        compile_body(name,b,asset,geom,collision=False,material='geom')
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
    res=int(math.ceil(math.sqrt(float(len(obj)))))
    for i,_ in enumerate(obj):
        state.qpos[i*2+0]=sep*(i%res)
        state.qpos[i*2+1]=sep*(i//res)
    sim.set_state(state)
    while True:
        sim.step()
        viewer.render()

def get_dataset_canonical(scale=3.):
    scale=[scale for i in range(3)]
    return [scene_scale(pole_thin()[1],scale),
            scene_scale(pole_thick()[1],scale),
            scene_scale(single_handle()[1],scale),
            scene_scale(two_handle()[1],scale),
            scene_scale(single_board()[1],scale),
            scene_scale(two_board()[1],scale),
            scene_scale(cup1()[1],scale),
            scene_scale(cup3()[1],scale),
            scene_scale(valley1()[1],scale),
            scene_scale(valley3()[1],scale),
            scene_scale(valley4()[1],scale),
            scene_scale(table1()[1],scale),
            scene_scale(table2()[1],scale)]

if __name__=='__main__':
    visualize_debug([pole_thin,pole_thick,  \
                     single_handle,two_handle,  \
                     single_board,two_board,    \
                     cup1,cup3, \
                     valley1,valley3,valley4,   \
                     table1,table2],.5)