from compile_objects import auto_download
from compile_gripper import Gripper
from compile_world import World
from controller import Controller
from metric import *
import numpy as np
import math,copy

class Domain:
    NUMBER_THREAD=4
    PARALLEL_EVAL=True
    #design space can be policy-related or gripper-related:
    #policy-related variables are: theta,phi,beta
    #all other variables are gripper-related (e.g. finger_length)
    #
    #metric can be object-dependent or object-independent:
    #object-dependent metrics will be first maximized for each object, then taken mean over all objects
    #object-independent metrics will be computed for each gripper
    def __init__(self,*,design_space,metrics,object_file_name,policy_space=[10,5,10,3.]):
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
        policy_vmin=[0,math.pi/4,0.]
        policy_vmax=[math.pi*2,math.pi/2,math.pi*2]
        for d in range(3):
            if policy_space[d] is not None:
                css=[]
                for i in range(policy_space[d]):
                    alpha=(i+0.5)/policy_space[d]
                    css.append(policy_vmin[d]*(1-alpha)+policy_vmax[d]*alpha)
                coordinates.append(np.array(css))
            else: 
                coordinates.append(np.linspace(0.,0.,1))
                self.vmin.append(policy_vmin[d])
                self.vmax.append(policy_vmax[d])
                self.vname.append(policy_names[d])
        self.policies=np.array([dimi.flatten() for dimi in np.meshgrid(*coordinates)]).T.tolist()
        self.init_dist=policy_space[3]
            
        #metric
        self.metrics=[globals()[metricName] for metricName in metrics.split('|')]
    
    def eval(self,points):
        scores=[[] for pt in points]
        if Domain.PARALLEL_EVAL:
            from concurrent.futures import ThreadPoolExecutor
            pool=ThreadPoolExecutor(max_workers=Domain.NUMBER_THREAD)
            def job(id):
                scores[id]=self.eval_single(points[id])
            for id in range(len(points)):
                pool.submit(job,id)
            pool.shutdown(wait=True)
        else:
            for ipt,pt in enumerate(points):
                scores[ipt]=self.eval_single(pt)
        return scores
    
    def eval_single(self,pt):
        #create policy
        policies=copy.deepcopy(self.policies)
        
        #create designed gripper
        args={}
        for a,b,n,v in zip(self.vmin,self.vmax,self.vname,pt):
            assert v>=a and v<=b
            if n=='theta':
                for i in range(len(policies)):
                    policies[i][0]=v
            elif n=='phi':
                for i in range(len(policies)):
                    policies[i][1]=v
            elif n=='beta':
                for i in range(len(policies)):
                    policies[i][2]=v
            else: args[n]=v
        link=self.gripper.get_robot(base_off=0.3,finger_width=0.4,finger_curvature=2)#(**args)
        
        #compile to MuJoCo
        world=World()
        world.compile_simulator(object_file_name=self.object_file_name,link=link)
        ctrl=Controller(world)
        
        #first compute object-independent metrics:
        score=[0. if m.OBJECT_DEPENDENT else m(ctrl).compute() for m in self.metrics]
        
        #then compute object-dependent metrics:
        for id in range(len(world.names)):
            score_obj=[0. for m in self.metrics]
            for p in policies:
                ctrl.reset(id,self.init_pos(p[0],p[1]),p[2])
                while not ctrl.step():pass
                print('Experimented!')
                score_obj=[max(so,m(ctrl).compute()) if m.OBJECT_DEPENDENT else 0. for so,m in zip(score_obj,self.metrics)]
            score=[so+s for so,s in zip(score_obj,score)]
        #take mean over objects
        return [s/len(world.names) if m.OBJECT_DEPENDENT else s for s,m in zip(score,self.metrics)]

    def init_pos(self,theta,phi):
        x=math.cos(theta)*math.cos(phi)*self.init_dist
        y=math.sin(theta)*math.cos(phi)*self.init_dist
        z=math.sin(phi)*self.init_dist
        return [x,y,z]

    def __str__(self):
        import glob
        ret='Domain:\n'
        for o in glob.glob(self.object_file_name):
            ret+='Object: %s\n'%o
        for a,b,n in zip(self.vmin,self.vmax,self.vname):
            ret+='Param %20s: [%10f,%10f]\n'%(n,a,b)
        ret+='Evaluating %d policies per point\n'%len(self.policies)
        return ret

if __name__=='__main__':
    auto_download()
    
    #case I: only optimize gripper
    domain=Domain(design_space='finger_length:0.2,0.5|finger_curvature:-2,2',metrics='ElapsedMetric',
                  object_file_name='data/ObjectNet3D/CAD/off/cup/[0-9][0-9].off',
                  policy_space=[10,5,10,3.])
    print(domain)
    
    #case II: optimize gripper as well as policy
    domain=Domain(design_space='finger_length:0.2,0.5|finger_curvature:-2,2',metrics='ElapsedMetric',
                  object_file_name='data/ObjectNet3D/CAD/off/cup/[0-9][0-9].off',
                  policy_space=[None,None,5,3.])
    print(domain)
    print(domain.eval([[0.4,0.,0.1,1.0]]))