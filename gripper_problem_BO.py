from compile_objects import auto_download,compile_objects
from compile_gripper import Gripper
from compile_world import World
from problem_BO import ProblemBO
from controller import Controller
import multiprocessing
from metric import *
import numpy as np
import math,copy,os,shutil

class GripperProblemBO(ProblemBO):
    NUMBER_PROCESS=max(1,multiprocessing.cpu_count()//2)
    DEFAULT_PATH='data/gripper_tmp'
    #design space can be policy-related or gripper-related:
    #policy-related variables are: theta,phi,beta
    #all other variables are gripper-related (e.g. finger_length)
    #
    #metric can be object-dependent or object-independent:
    #object-dependent metrics will be first maximized for each object, then taken mean over all objects
    #object-independent metrics will be computed for each gripper
    def __init__(self,*,design_space,metrics,object_file_name,policy_space=[10,5,10,3.]):
        print('Initializing Domain, multi-threaded evaluation using %d processes!'%GripperProblemBO.NUMBER_PROCESS)
        self.gripper=Gripper()
        self.object_file_name=object_file_name
        
        #vmin/vmax/vname
        self.vmin=[]
        self.vmax=[]
        self.vname=[]
        for designParam in design_space.split('|'):
            designParam=designParam.split(':')
            if len(designParam)!=2:
                raise RuntimeError('Incorrect format for design_space!')
            minmax=designParam[1].split(',')
            if len(minmax)!=2:
                raise RuntimeError('Incorrect format for design_space!')
            self.vmin.append(float(minmax[0]))
            self.vmax.append(float(minmax[1]))
            self.vname.append(designParam[0])
            
        #policy
        coordinates=[]
        policy_names=['theta','phi','beta']
        self.policy_vmin=[0,math.pi/4,0.]
        self.policy_vmax=[math.pi*2,math.pi/2*0.99,math.pi*2]    #*0.99 to phi will avoid Gimbal lock of Euler angles
        for d in range(3):
            if policy_space[d] is not None:
                css=[]
                for i in range(policy_space[d]):
                    alpha=(i+0.5)/policy_space[d]
                    css.append(self.policy_vmin[d]*(1-alpha)+self.policy_vmax[d]*alpha)
                coordinates.append(np.array(css))
            else: 
                coordinates.append(np.linspace(0.,0.,1))
                self.vmin.append(self.policy_vmin[d])
                self.vmax.append(self.policy_vmax[d])
                self.vname.append(policy_names[d])
        self.policies=np.array([dimi.flatten() for dimi in np.meshgrid(*coordinates)]).T.tolist()
        self.init_dist=policy_space[3]
            
        #metric
        self.metrics=[globals()[metricName] for metricName in metrics.split('|')]
        
        #we need to compile objects once to invoke format conversion
        import lxml.etree as ET
        compile_objects(ET.Element('asset'),self.object_file_name)
    
    def eval(self,points):
        #gripper_metrics[pt_id][metric_id]
        #object_metrics[pt_id][policy_id][object_id][metric_id]
        gripper_metrics,object_metrics=self.compute_metrics(points)
        #mean over objects, max over policies
        combined_metrics=np.array(gripper_metrics)+np.array(object_metrics).max(axis=1).mean(axis=1)
        return combined_metrics.tolist()
    
    def compute_metrics(self,points,parallel=True,remove_tmp=True):
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
                    object_metrics[pt_id][policy_id]=pool.submit(self.compute_object_dependent_metrics,link,points[pt_id],self.policies[policy_id])
                else: object_metrics[pt_id][policy_id]=self.compute_object_dependent_metrics(link,points[pt_id],self.policies[policy_id])
                
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
        args={}
        for a,b,n,v in zip(self.vmin,self.vmax,self.vname,pt):
            if n=='theta':pass
            elif n=='phi':pass
            elif n=='beta':pass
            else:
                assert v>=a and v<=b
                args[n]=v
        link=self.gripper.get_robot(**args)
        return link,[0. if m.OBJECT_DEPENDENT else m(link).compute() for m in self.metrics]
    
    def compute_object_dependent_metrics(self,link,pt,policy):
        #create designed gripper
        for n,v in zip(self.vname,pt):
            if n=='theta':
                policy[0]=v
            elif n=='phi':
                policy[1]=v
            elif n=='beta':
                policy[2]=v
        
        #compile to MuJoCo
        world=World()
        world.compile_simulator(path=GripperProblemBO.DEFAULT_PATH,object_file_name=self.object_file_name,link=link)
        ctrl=Controller(world)
        
        #then compute object-dependent metrics:
        score_obj=[]
        for id in range(len(world.names)):
            #we support different policy for each object if: isinstance(policy[0/1/2],list)==True
            #we also support same policy over all objects if: isinstance(policy[0/1/2],list)==False
            theta=policy[0][id] if isinstance(policy[0],list) else policy[0]
            assert theta>=self.policy_vmin[0] and theta<=self.policy_vmax[0]
            
            phi=policy[1][id] if isinstance(policy[1],list) else policy[1]
            assert phi>=self.policy_vmin[1] and phi<=self.policy_vmax[1]
            
            beta=policy[2][id] if isinstance(policy[2],list) else policy[2]
            assert beta>=self.policy_vmin[2] and beta<=self.policy_vmax[2]
            
            #print('Using policy: (%f,%f,%f) for object %d!'%(theta,phi,beta,id))
            ctrl.reset(id,self.init_pos(theta,phi),beta)
            while not ctrl.step():pass
            #print('Experimented!')
            score_obj.append([m(ctrl).compute() if m.OBJECT_DEPENDENT else 0. for m in self.metrics])
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
        world.compile_simulator(path=GripperProblemBO.DEFAULT_PATH,object_file_name=self.object_file_name,link=link)
        ctrl=Controller(world)

        viewer=mjc.MjViewer(world.sim)
        while True:
            for id in range(len(world.names)):
                policy=self.policies[object_metrics[id]]
                for n,v in zip(self.vname,point):
                    if n=='theta':
                        policy[0]=v
                    elif n=='phi':
                        policy[1]=v
                    elif n=='beta':
                        policy[2]=v
                        
                theta=policy[0]
                phi=policy[1]
                beta=policy[2]
                ctrl.reset(id,self.init_pos(theta,phi),beta)
                while not ctrl.step():
                    viewer.render()

    def __str__(self):
        import glob
        ret='ProblemBO:\n'
        for o in glob.glob(self.object_file_name):
            ret+='Object: %s\n'%o
        for a,b,n in zip(self.vmin,self.vmax,self.vname):
            ret+='Param %20s: [%10f,%10f]\n'%(n,a,b)
        ret+='Evaluating %d policies per point\n'%len(self.policies)
        return ret

    def name(self):
        return 'GripperProblemBO'

if __name__=='__main__':
    auto_download()
    
    #case I: only optimize gripper
    domain=GripperProblemBO(design_space='finger_length:0.2,0.5|finger_curvature:-2,2',metrics='MassMetric|Q1Metric',
                            object_file_name='data/ObjectNet3D/CAD/off/cup/[0-9][0-9].off',
                            policy_space=[10,5,10,3.])
    print(domain)
    
    #case II: optimize gripper as well as policy
    domain=GripperProblemBO(design_space='finger_length:0.2,0.5|finger_curvature:-2,2',metrics='SizeMetric|ElapsedMetric',
                            object_file_name='data/ObjectNet3D/CAD/off/cup/[0-9][0-9].off',
                            policy_space=[None,None,5,3.])
    print(domain)
    
    #test evaluating a single point
    print(domain.eval([[0.4,0.,0.1,1.0]]))
    #test evaluating two points
    print(domain.eval([[0.4,0.,0.1,1.0],[0.21,0.5,0.1,1.0]]))
    #test evaluating two points, with the first point using different policy for each object
    print(domain.eval([[0.4,0.,np.linspace(0.1,6.0,10).tolist(),1.0],[0.21,0.5,0.1,1.0]]))
    