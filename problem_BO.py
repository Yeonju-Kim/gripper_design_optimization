from metric import Metric
import numpy as np
import math

class ProblemBO:
    def __init__(self,design_space,policy_space):
        from collections import OrderedDict
        design_space=OrderedDict(design_space)
        policy_space=OrderedDict(policy_space)
        
        #vmin/vmax/vname/args0
        self.vmin=[]
        self.vmax=[]
        self.vname=[]
        self.vpolicyid=[]
        self.mimic={}
        self.args0={}
        for designParam,minmax in design_space.items():
            if not isinstance(minmax,tuple) and not isinstance(minmax,float):
                raise RuntimeError('Incorrect format for design_space!')
            elif isinstance(minmax,tuple):
                self.vmin.append(float(minmax[0]))
                self.vmax.append(float(minmax[1]))
                self.vname.append(designParam)
                self.vpolicyid.append(-1)
            else: self.args0[designParam]=float(minmax)
            
        #policy
        coordinates=[]
        for d in range(len(self.policy_names)):
            if self.policy_names[d] in policy_space:
                var=policy_space[self.policy_names[d]]
                if isinstance(var,int):
                    coordinates.append(np.linspace(self.policy_vmin[d],self.policy_vmax[d],var))
                    print('%s=%s'%(self.policy_names[d],str(coordinates[-1].tolist())))
                elif isinstance(var,float):
                    coordinates.append(np.linspace(var,var,1))
                    print('%s=%s'%(self.policy_names[d],str(coordinates[-1].tolist())))
                elif var is None:
                    self.vmin.append(self.policy_vmin[d])
                    self.vmax.append(self.policy_vmax[d])
                    self.vname.append(self.policy_names[d])
                    self.vpolicyid.append(d)
                    coordinates.append(np.linspace(0.,0.,1))
                    print('%s=(var%d,%f,%f)'%(self.policy_names[d],len(self.vmin)-1,self.vmin[-1],self.vmax[-1]))
                elif isinstance(var,tuple) and isinstance(var[0],str) and isinstance(var[1],float):
                    id=self.vname.index(var[0])
                    minv=self.policy_vmin[d]/var[1]
                    maxv=self.policy_vmax[d]/var[1]
                    if maxv<minv:
                        tmp=minv
                        minv=maxv
                        maxv=tmp
                    self.vmin[id]=max(self.vmin[id],minv)
                    self.vmax[id]=min(self.vmax[id],maxv)
                    self.vname[id]+=':'+self.policy_names[d]+'='+self.vname[id]+'*'+str(var[1])
                    self.mimic[(d,id)]=var[1]
                    coordinates.append(np.linspace(0.,0.,1))
                    print('%s=(var%d,%f,%f)'%(self.policy_names[d],id,self.vmin[id],self.vmax[id]))
            elif self.policy_init[d] is not None: 
                coordinates.append(np.linspace(self.policy_init[d],self.policy_init[d],1))
        self.policies=np.array([dimi.flatten() for dimi in np.meshgrid(*coordinates)]).T.tolist()
            
    def eval(self,points,parallel=True,remove_tmp=True,visualize=False,avgObject=True):
        #gripper_metrics[pt_id][metric_id]
        #object_metrics[pt_id][policy_id][object_id][metric_id]
        gripper_metrics,object_metrics=self.compute_metrics(points,parallel=parallel,remove_tmp=remove_tmp,visualize=visualize)
        
        if avgObject:
            #for normal BO: mean over objects, max over policies
            combined_metrics=np.array(gripper_metrics)+np.array(object_metrics).max(axis=1).mean(axis=1)
            return combined_metrics.tolist()
        else:
            #for actor-critic BO: max over policies
            #in this case, we preserve a placeholder in gripper_metrics for object_dependent_meterics with all 0, vice versa
            return gripper_metrics,np.array(object_metrics).max(axis=1).tolist()
    
    def name(self):
        raise RuntimeError('This is abstract super-class, use sub-class!')
    
    def __str__(self):
        import glob
        ret='ProblemBO:\n'
        for o in self.objects:
            if isinstance(o,str):
                var=str(o)
            elif isinstance(o,list): 
                var='Composite(#geom=%d)'%len(o)
            else: var='Composite(#geom=1)'
            ret+='Object: %s\n'%var
        for a,b,n in zip(self.vmin,self.vmax,self.vname):
            ret+='Param %20s: [%10f,%10f]\n'%(n,a,b)
        ret+='Evaluating %d policies per point\n'%len(self.policies)
        return ret

class Test1DMetric(Metric):
    NOISE=0.1
    def __init__(self):
        Metric.__init__(self,OBJECT_DEPENDENT=False)
    
    def compute(self,pt):
        x=pt[0]
        from numpy.random import normal
        noise=normal(loc=0,scale=Test1DMetric.NOISE)
        return (x**2*math.sin(5*math.pi*x)**6.0)+noise

class ReversedTest1DMetric(Metric):
    NOISE=0.1
    def __init__(self):
        Metric.__init__(self,OBJECT_DEPENDENT=False)
    
    def compute(self,pt):
        x=1.-pt[0]
        from numpy.random import normal
        noise=normal(loc=0,scale=Test1DMetric.NOISE)
        return (x**2*math.sin(5*math.pi*x)**6.0)+noise

class Test2DMetric(Metric):
    NOISE=0.1
    def __init__(self):
        Metric.__init__(self,OBJECT_DEPENDENT=False)
    
    def compute(self,pt):
        x=pt[0]
        y=pt[1]
        from numpy.random import normal
        noise=normal(loc=0,scale=Test2DMetric.NOISE)
        return (x**2*math.sin(5*math.pi*x)**6.0*
                y**2*math.cos(5*math.pi*y)**6.0)+noise

class ReversedTest2DMetric(Metric):
    NOISE=0.1
    def __init__(self):
        Metric.__init__(self,OBJECT_DEPENDENT=False)
    
    def compute(self,pt):
        x=1.-pt[0]
        y=1.-pt[1]
        from numpy.random import normal
        noise=normal(loc=0,scale=Test2DMetric.NOISE)
        return (x**2*math.sin(5*math.pi*x)**6.0*
                y**2*math.cos(5*math.pi*y)**6.0)+noise

class MeanTest2DMetric(Metric):
    NOISE=0.1
    def __init__(self):
        Metric.__init__(self,OBJECT_DEPENDENT=False)
    
    def compute(self,pt):
        x=1.-pt[0]
        y=pt[1]
        from numpy.random import normal
        noise=normal(loc=0,scale=Test2DMetric.NOISE)
        return (x**2*math.sin(5*math.pi*x)**6.0*
                y**2*math.cos(5*math.pi*y)**6.0)+noise

class Test1D1MProblemBO(ProblemBO):
    def __init__(self):
        self.vmin=[0.]
        self.vmax=[1.]
        self.vnames=['x']
        self.metrics=[Test1DMetric()]
        
    def eval(self,points):
        return [[m.compute(pt) for m in self.metrics] for pt in points]
    
    def name(self):
        return 'Test1D1MProblemBO'

class Test1D2MProblemBO(ProblemBO):
    def __init__(self):
        self.vmin=[0.]
        self.vmax=[1.]
        self.vnames=['x']
        self.metrics=[Test1DMetric(),ReversedTest1DMetric()]
        
    def eval(self,points):
        return [[m.compute(pt) for m in self.metrics] for pt in points]
    
    def name(self):
        return 'Test1D2MProblemBO'
    
class Test2D1MProblemBO(ProblemBO):
    def __init__(self):
        self.vmin=[0.,0.]
        self.vmax=[1.,1.]
        self.vnames=['x','y']
        self.metrics=[Test2DMetric()]
        
    def eval(self,points):
        return [[m.compute(pt) for m in self.metrics] for pt in points]
    
    def name(self):
        return 'Test2D1MProblemBO'
    
class Test2D2MProblemBO(ProblemBO):
    def __init__(self):
        self.vmin=[0.,0.]
        self.vmax=[1.,1.]
        self.vnames=['x','y']
        self.metrics=[Test2DMetric(),ReversedTest2DMetric()]
        
    def eval(self,points):
        return [[m.compute(pt) for m in self.metrics] for pt in points]
    
    def name(self):
        return 'Test2D2MProblemBO'

class Test2D3MProblemBO(ProblemBO):
    def __init__(self):
        self.vmin=[0.,0.]
        self.vmax=[1.,1.]
        self.vnames=['x','y']
        self.metrics=[Test2DMetric(),ReversedTest2DMetric(),MeanTest2DMetric()]
        
    def eval(self,points):
        return [[m.compute(pt) for m in self.metrics] for pt in points]
    
    def name(self):
        return 'Test2D3MProblemBO'
    
if __name__=='__main__':
    print(Test1D1MProblemBO().eval([[0.1],[0.2],[0.3]]))
    print(Test1D2MProblemBO().eval([[0.1],[0.2],[0.3]]))
    print(Test2D1MProblemBO().eval([[0.1,0.7],[0.5,0.2],[0.3,0.6]]))
    print(Test2D2MProblemBO().eval([[0.1,0.7],[0.5,0.2],[0.3,0.6]]))
    print(Test2D3MProblemBO().eval([[0.1,0.7],[0.5,0.2],[0.3,0.6]]))