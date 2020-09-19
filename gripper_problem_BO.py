from compile_objects import *
from compile_gripper import Gripper
from compile_world import World
from problem_BO import ProblemBO
from controller import Controller
from metric import *
import numpy as np
import math,copy,os,shutil,time,multiprocessing

class GripperProblemBO(ProblemBO):
    NUMBER_PROCESS=max(1,multiprocessing.cpu_count()//2)
    ONE_OBJECT_PER_WORLD=True
    DEFAULT_PATH='data'
    #design space can be policy-related or gripper-related:
    #policy-related variables are: theta,phi,beta
    #all other variables are gripper-related (e.g. finger_length)
    #
    #metric can be object-dependent or object-independent:
    #object-dependent metrics will be first maximized for each object, then taken mean over all objects
    #object-independent metrics will be computed for each gripper
    def __init__(self,*,design_space,metrics,objects,policy_space):
        if not os.path.exists(GripperProblemBO.DEFAULT_PATH):
            os.mkdir(GripperProblemBO.DEFAULT_PATH)
        print('Initializing Domain, multi-threaded evaluation using %d processes!'%GripperProblemBO.NUMBER_PROCESS)
        self.gripper=Gripper()
        self.objects=objects
        
        self.policy_names=['theta'  ,'phi'         ,'beta' ,'init_pose0','init_pose1','approach_coef0','approach_coef1','init_dist','grasp_dir']
        self.policy_vmin= [0.       ,math.pi/4     ,0.     ,-math.pi/2  ,-math.pi/2  ,-1.             ,-1.             ,2.         ,-1.        ]
        self.policy_vmax= [math.pi*2,math.pi/2*0.99,math.pi, math.pi/2  , math.pi/2  , 1.             , 1.             ,3.5        , 1.        ]
        self.policy_init= [0.       ,math.pi/2*0.99,0.     , math.pi/2  , 0.         , 1.             , 1.             ,3.5        ,None       ]
        #*0.99 to phi will avoid Gimbal lock of Euler angles
        
        #metric
        self.metrics=metrics
    
        ProblemBO.__init__(self,design_space,policy_space)
        
    def compute_metrics(self,points,parallel=True,remove_tmp=True,visualize=False):
        #create temporary file path
        if not os.path.exists(GripperProblemBO.DEFAULT_PATH):
            os.mkdir(GripperProblemBO.DEFAULT_PATH)
            
        #intialize result pool
        gripper_metrics=[None for pt_id in points]
        object_metrics=[[None for policy_id in self.policies] for pt_id in points]
        if parallel:
            from concurrent.futures import ProcessPoolExecutor
            pool=ProcessPoolExecutor(max_workers=GripperProblemBO.NUMBER_PROCESS)
        
        #iterate over all points
        for pt_id in range(len(points)):
            #compute object-independent metrics first
            link,gripper_metric=self.compute_gripper_dependent_metrics(points[pt_id])
            gripper_metrics[pt_id]=gripper_metric
            
            #compute object-dependent metrics in parallel
            for policy_id in range(len(self.policies)):
                if parallel:
                    object_metrics[pt_id][policy_id]=pool.submit(self.compute_object_dependent_metrics,link,points[pt_id],self.policies[policy_id],visualize)
                else: object_metrics[pt_id][policy_id]=self.compute_object_dependent_metrics(link,points[pt_id],self.policies[policy_id],visualize)
                
        #join-all & get results
        if parallel:
            pool.shutdown(wait=True)
            for pt_id in range(len(points)):
                for policy_id in range(len(self.policies)):
                    object_metrics[pt_id][policy_id]=object_metrics[pt_id][policy_id].result()
            
        #remove temporary file path
        if os.path.exists(GripperProblemBO.DEFAULT_PATH) and remove_tmp:
            shutil.rmtree(GripperProblemBO.DEFAULT_PATH)
        return gripper_metrics,object_metrics
        
    def compute_gripper_dependent_metrics(self,pt):
        args=self.args0
        for a,b,n,id,v in zip(self.vmin,self.vmax,self.vname,self.vpolicyid,pt):
            if id==-1:
                assert v>=a and v<=b
                args[n]=v
        
        root=ET.Element('mujoco')
        set_simulator_option(root)
        asset=ET.SubElement(root,'asset')
        body=ET.SubElement(root,'worldbody')
        actuator=ET.SubElement(root,'actuator')
        link=self.gripper.get_robot(**args)
        link.compile_gripper(body,asset,actuator)
        
        if not os.path.exists(GripperProblemBO.DEFAULT_PATH):
            os.mkdir(GripperProblemBO.DEFAULT_PATH)
        open(GripperProblemBO.DEFAULT_PATH+'/gripper.xml','w').write(ET.tostring(root,pretty_print=True).decode())
        model=mjc.load_model_from_path(GripperProblemBO.DEFAULT_PATH+'/gripper.xml')
        return link,[0. if m.OBJECT_DEPENDENT else m.compute(mjc.MjSim(model)) for m in self.metrics]
    
    def compute_object_dependent_metrics(self,link,pt,policy,visualize):
        #create designed gripper
        for id,v in zip(self.vpolicyid,pt):
            if id>=0:
                policy[id]=v
        for k,v in self.mimic:
            policy[k[0]]=[p*v for p in pt[k[1]]] if isinstance(pt[k[1]],list) else pt[k[1]]*v
        
        #compile to MuJoCo
        if not GripperProblemBO.ONE_OBJECT_PER_WORLD:
            world=World()
            world.compile_simulator(path=GripperProblemBO.DEFAULT_PATH,objects=self.objects,link=link)
            ctrl=Controller(world)
                
        #then compute object-dependent metrics:
        score_obj=[]
        for id in range(len(self.objects)):
            #compile to MuJoCo
            if GripperProblemBO.ONE_OBJECT_PER_WORLD:
                world=World()
                world.compile_simulator(path=GripperProblemBO.DEFAULT_PATH,objects=self.objects[id:id+1],link=link)
                ctrl=Controller(world)
            policyid=[p[id] if isinstance(p,list) else p for p in policy]
            ctrl.reset(0 if GripperProblemBO.ONE_OBJECT_PER_WORLD else id,  \
                       angle=policyid[0:3],init_pose=policyid[3:5], \
                       approach_coef=policyid[5:7],init_dist=policyid[7],   \
                       grasp_dir=policyid[8] if len(policyid)>8 else None)
            viewer=mjc.MjViewer(world.sim) if visualize else None
            while not ctrl.step():
                if viewer is not None:
                    viewer.render()
            score_obj.append([m.compute(ctrl) if m.OBJECT_DEPENDENT else 0. for m in self.metrics])
        return score_obj

    def init_pos(self,theta,phi):
        x=math.cos(theta)*math.cos(phi)*self.init_dist
        y=math.sin(theta)*math.cos(phi)*self.init_dist
        z=math.sin(phi)*self.init_dist
        return [x,y,z]

    def plot_solution(self,point,metric_id):
        _,object_metrics=self.compute_metrics([point])
        object_metrics=np.array(object_metrics)[0,:,:,metric_id].argmax(axis=0)
        
        #compile to MuJoCo
        world=World()
        link,_=self.compute_gripper_dependent_metrics(point)
        world.compile_simulator(path=GripperProblemBO.DEFAULT_PATH,objects=self.objects,link=link)
        ctrl=Controller(world)

        viewer=mjc.MjViewer(world.sim)
        while True:
            for id in range(len(world.names)):
                policy=self.policies[object_metrics[id]]
                for id,v in zip(self.vpolicyid,pt):
                    if id>=0:
                        policy[id]=v
                for k,v in self.mimic:
                    policy[k[0]]=[p*v for p in pt[k[1]]] if isinstance(pt[k[1]],list) else pt[k[1]]*v
                policyid=[p[id] if isinstance(p,list) else p for p in policy]
                ctrl.reset(0 if GripperProblemBO.ONE_OBJECT_PER_WORLD else id,  \
                           angle=policyid[0:3],init_pose=policyid[3:5], \
                           approach_coef=policyid[5:7],init_dist=policyid[7],   \
                           grasp_dir=policyid[8] if len(policyid)>8 else None)
                while not ctrl.step():
                    viewer.render()

    def name(self):
        return 'GripperProblemBO'

if __name__=='__main__':
    from dataset_cup import get_dataset_cup
    nObj=3
    #case I: only optimize gripper
    domain=GripperProblemBO(design_space=[('finger_length',(0.2,0.5)),('finger_curvature',(-2,2))],
                            metrics=[MassMetric(),Q1Metric()],
                            objects=get_dataset_cup(True)[:nObj],
                            policy_space=[('theta',10),('phi',5),('beta',10),('init_dist',3.)])
    print(domain)
    
    #case II: optimize gripper as well as policy
    domain=GripperProblemBO(design_space=[('base_rad',0.25),('base_off',0.2),('finger_length',(0.2,0.5)),('finger_curvature',(-2,2))],
                            metrics=[SizeMetric(),ElapsedMetric()],
                            objects=get_dataset_cup(True)[:nObj],
                            policy_space=[('theta',None),('phi',('theta',1.5)),('beta',10),('init_dist',3.)])
    print(domain)
    
    #case II: another case
    domain=GripperProblemBO(design_space=[('base_rad',0.25),('base_off',0.2),('finger_length',(0.2,0.5)),('finger_curvature',(-2,2))],
                            metrics=[SizeMetric(),ElapsedMetric()],
                            objects=get_dataset_cup(True)[:nObj],
                            policy_space=[('theta',None),('phi',None),('beta',10),('init_dist',3.),('grasp_dir',1.)])
    print(domain)
    
    #test evaluating two points
    print(domain.eval([[0.4,0.,0.1,1.0],[0.21,0.5,0.1,1.0]],parallel=True,visualize=False))
    print(domain.eval([[0.4,0.,0.1,1.0],[0.21,0.5,0.1,1.0]],parallel=True,visualize=False,mode='MAX_POLICY'))
    #test evaluating two points, with the first point using different policy for each object
    print(domain.eval([[0.4,0.,np.linspace(0.1,6.0,10).tolist(),1.0],[0.21,0.5,0.1,1.0]]))
    #test evaluating a single point
    for ONE_OBJECT_PER_WORLD in [True,False]:
        GripperProblemBO.ONE_OBJECT_PER_WORLD=ONE_OBJECT_PER_WORLD
        start=time.time()
        ret=domain.eval([[0.2,0.,0.1,math.pi/2*0.9]])
        end=time.time()
        print("ONE_OBJECT_PER_WORLD=%d, time=%s, result=%s"%(ONE_OBJECT_PER_WORLD,str(end-start),str(ret)))
    