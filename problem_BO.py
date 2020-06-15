from metric import Metric
import numpy as np
import math

class ProblemBO:
    def __init__(self):
        pass
    
    def eval(self,points):
        raise RuntimeError('This is abstract super-class, use sub-class!')
    
    def name(self):
        raise RuntimeError('This is abstract super-class, use sub-class!')
    
class Test1DMetric(Metric):
    NOISE=0.1
    
    def __init__(self,pt):
        self.pt=pt
        
    def compute(self):
        x=self.pt[0]
        from numpy.random import normal
        noise=normal(loc=0,scale=Test1DMetric.NOISE)
        return (x**2*math.sin(5*math.pi*x)**6.0)+noise

class Test2DMetric(Metric):
    NOISE=0.1
    
    def __init__(self,pt):
        self.pt=pt
        
    def compute(self):
        x=self.pt[0]
        y=self.pt[1]
        from numpy.random import normal
        noise=normal(loc=0,scale=Test2DMetric.NOISE)
        return (x**2*math.sin(5*math.pi*x)**6.0*
                y**2*math.cos(5*math.pi*y)**6.0)+noise

class Test1D1MProblemBO(ProblemBO):
    def __init__(self):
        self.vmin=[0.]
        self.vmax=[1.]
        self.vnames=['x']
        self.metrics=[Test1DMetric]
        
    def eval(self,points):
        return [[m(pt).compute() for m in self.metrics] for pt in points]
    
    def name(self):
        return 'Test1D1MProblemBO'
    
class Test2D1MProblemBO(ProblemBO):
    def __init__(self):
        self.vmin=[0.,0.]
        self.vmax=[1.,1.]
        self.vnames=['x']
        self.metrics=[Test2DMetric]
        
    def eval(self,points):
        return [[m(pt).compute() for m in self.metrics] for pt in points]
    
    def name(self):
        return 'Test2D1MProblemBO'
    
class HyperVolumeTransformedProblemBO(ProblemBO):
    def __init__(self,inner):
        self.inner=inner
        self.vmin=inner.vmin
        self.vmax=inner.vmax
        self.metrics=['HyperVolumeMetric']
        
    def eval(self,points):
        metrics=self.inner.eval(points)
        return [[np.prod(m)] for m in metrics]
        
    def name(self):
        return 'HyperVolumeTransformed('+self.inner.name()+')'
    
if __name__=='__main__':
    print(Test1D1MProblemBO().eval([[0.1],[0.2],[0.3]]))
    print(Test2D1MProblemBO().eval([[0.1,0.7],[0.5,0.2],[0.3,0.6]]))