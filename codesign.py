import numpy as np
import math

class DesignBase:
    """Base class for a design.  Created by DesignSpace.makeDesign."""
    def __init__(self,parameters):
        self.parameters = parameters

class DesignSpace:
    """Defines a design space.  The default implementation is a box, with 
    bounds initialized on construction. 
    
    Users: implement me.  Usually the ``makeDesign`` method should be implemented
    in your subclass.
    """
    def __init__(self,xmin=None,xmax=None):
        self.xmin = xmin
        self.xmax = xmax
        assert (xmax is not None) == (xmin is not None)
        if xmin is not None:
            assert len(xmin)==len(xmax),"Must provide a pair of equal-length vectors"
            for (a,b) in zip(xmin,xmax):
                assert a <= b,"Bounding box must have minimum lower than maximum"

    def bound(self):
        """Returns a pair (xmin,xmax) defining the minimum and maximum 
        extents of the design space."""
        if self.xmin is None:
            raise NotImplementedError()
        return self.xmin,self.xmax
    
    def dimension(self):
        """Returns # of dimensions of the space"""
        try:
            bmin,bmax = self.bound()
            return len(bmin)
        except NotImplementedError:
            raise
        
    def randomSample(self):
        """Randomly samples a design"""
        try:
            bmin,bmax = self.bound()
            return np.random.uniform(bmin,bmax)
        except NotImplementedError:
            raise
    
    def grid(self,N):
        """Generates a grid containing approximately N samples"""
        try:
            bmin,bmax = self.bound()
            dims = len([1 for a,b in zip(bmin,bmax) if b > a])
            Ndivs = int(math.floor(pow(N,1.0/dims)))+1
            divarray = []
            for a,b in zip(bmin,bmax):
                if b > a:
                    divarray.append(np.linspace(a,b,Ndivs))
                else:
                    divarray.append([a])
            items = np.meshgrid(*divarray)
            return zip(*[v.flatten() for v in items])
        except NotImplementedError:
            raise
    
    def makeDesign(self,parameters):
        """Create an object representing the actual design instantiated
        by the given parameters.  Default just returns an empty DesignBase
        object."""
        return DesignBase(parameters)
    
    
class BehaviorSpace:
    """Defines a behavior space.  The default implementation is a box, with 
    bounds initialized on construction. To implement non-Euclidean behavior
    spaces, you must subclass.  Also, you may want to implement randomSample,
    isValid,  and evaluationData for maximum performance. """
    def __init__(self,xmin=None,xmax=None):
        self.xmin = xmin
        self.xmax = xmax
        assert (xmax is not None) == (xmin is not None)
        if xmin is not None:
            assert len(xmin)==len(xmax),"Must provide a pair of equal-length vectors"
            for (a,b) in zip(xmin,xmax):
                assert a <= b,"Bounding box must have minimum lower than maximum"

    def bound(self):
        """Returns a pair (bmin,bmax) defining the minimum and maximum 
        extents of the behavior space."""
        if self.xmin is None:
            raise NotImplementedError()
        return self.xmin,self.xmax
    
    def dimension(self):
        """Returns # of dimensions of the space"""
        try:
            bmin,bmax = self.bound()
            return len(bmin)
        except NotImplementedError:
            raise
        
    def randomSample(self):
        """Randomly samples a behavior for a given design and environment.
        The result is not required to be tested for validity.
        
        Args:
            design (DesignBase): the design for which the behavior should be
                applied.
            environment (user-defined): the environment in which the behavior
                should be applied.
        """
        try:
            bmin,bmax = self.bound()
            return np.random.uniform(bmin,bmax)
        except NotImplementedError:
            raise
        
    def isValid(self,design,environment,behavior):
        """Returns true if the behavior is valid for the given design and
        environment.
        
        Args:
            design (DesignBase): the design for which the behavior should be
                tested.
            environment (user-defined): the environment in which the behavior
                should be tested.
            behavior (array-like): the set of behavior parameters that should
                be tested for the design and environment.
        """
        return True
    
    def evaluationData(self,design,environment,behavior):
        """Calculates some data for evaluating a behavior that might be
        shared between metrics.  For example, a simulation trace."""

        return None
    
    def grid(self,N):
        """Generates a grid containing approximately N samples"""
        try:
            bmin,bmax = self.bound()
            dims = len([1 for a,b in zip(bmin,bmax) if b > a])
            Ndivs = int(math.floor(pow(N,1.0/dims)))+1
            divarray = []
            for a,b in zip(bmin,bmax):
                if b > a:
                    divarray.append(np.linspace(a,b,Ndivs))
                else:
                    divarray.append([a])
            items = np.meshgrid(*divarray)
            return zip(*[v.flatten() for v in items])
        except NotImplementedError:
            raise


class DesignMetric:
    """Evaluates a design numerically.  Users: implement me.

    Can return a scalar or an array upon __call__(design). Larger is better.
    
    Default just calls a 1-parameter function provided to constructor.
    """
    def __init__(self,func=None,name=None):
        self.func = func
        self.name = name
        if func is not None:
            assert callable(func)
    
    def __str__(self):
        return self.name if self.name is not None else self.__class__.__name__
            
    def __call__(self,design):
        """Overload me to implement the metric evaluation."""
        if self.func is not None:
            return self.func(design)
        raise NotImplementedError()


class BehaviorMetric:
    """Evaluates a behavior numerically.  Users: implement me.
    
    Can return a scalar or an array upon 
    __call__(design,environment,behavior,data).  Larger is better.  
    
    Here, data is the result returned from
    ``BehaviorSpace.evaluationData(design,environment,behavior)``
    
    An array result means that the metric is *linked*.  TODO: implement
    linked metrics.
    
    Default just calls a 1-parameter function provided to constructor.
    """
    def __init__(self,func=None,name=None):
        self.func = func
        self.name = name
        if func is not None:
            assert callable(func)
        
    def __str__(self):
        return self.name if self.name is not None else self.__class__.__name__
        
    def __call__(self,design,environment,behavior,data):
        raise NotImplementedError()


class DesignVisualizerBase:
    """A base class for /visualizing results.  Users: implement me."""
    def showDesign(self,design):
        raise NotImplementedError()
        
    def showDesigns(self,designs):
        raise NotImplementedError()
    
    def showEnvironment(self,environment):
        raise NotImplementedError()
        
    def showEnvironments(self,environments):
        raise NotImplementedError()
        
    def showDesignBehavior(self,design,environment,behavior):
        raise NotImplementedError()
        
    def showDesignBehaviors(self,design,environments,behaviors):
        raise NotImplementedError()
    
    def showDesignBehaviorMetric(self,design,environment,behavior,metric):
        self.showDesignBehavior(design,environment,behavior)
        
    def showDesignBehaviorMetrics(self,design,environments,behaviors,metrics):
        self.showDesignBehaviors(design,environments,behaviors)


class CodesignProblem:
    """Specifies a behavior-dependent co-design problem.
    
    Attributes:
        decisionSpace (DesignSpace):
        behaviorSpace (BehaviorSpace):
        environments (list of user-defined objects:
        designMetrics (list of DesignMetric)
        behaviorMetrics (list of BehaviorMetric)
        environmentWeights (np.ndarray): array of shape (m,len(environments))
            dictating the map from behavior metrics to design metric space.
            Default measures average performance
        environmentWeightLabels (list of str): list of length m giving each
            sub-metric's name, one for each row of environmentWeights.
    """
    def __init__(self,designSpace,behaviorSpace=None,environments=None,
        designMetrics=None,behaviorMetrics=None,name=None,mean=False):
        self.designSpace = designSpace
        self.behaviorSpace = behaviorSpace
        self.environments = environments if environments is not None else []
        self.designMetrics = designMetrics if designMetrics is not None else []
        self.behaviorMetrics = behaviorMetrics if behaviorMetrics is not None else []
        self.metricWeights = np.eye(len(self.environments))
        self.metricWeightsLabels = []
        if len(self.environments) >= 1 and mean:
            self.metricWeights = np.full((1, len(self.environments)),1.0/len(self.environments))
            self.metricWeightsLabels = ['mean']
        self.name = name


    def metricVector(self,design,behaviors,full=False):
        """Maps a design and behaviors to metric space.  If full=True,
        then all behavior metrics are returned, not just the ones processed
        by multiplication with the environmentWeights matrix. """
        assert len(behaviors) == len(self.environments)
        desMetrics = [f(design) for f in self.designMetrics]
        behMetrics = []
        for g in self.behaviorMetrics:
            gvec = []
            for b,e in zip(behaviors,self.environments):
                if not self.behaviorSpace.isValid(design,e,b):
                    gvec.append(-float('inf'))
                else:
                    data = self.behaviorSpace.evaluationData(design,e,b)
                    gvec.append(g(design,e,b,data))
            if full:
                behMetrics += gvec
            else:
                behMetrics += self.metricWeights.dot(np.stack(gvec))
        return np.stack(desMetrics+behMetrics)
