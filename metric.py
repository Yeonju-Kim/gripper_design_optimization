from controller import Controller
from compile_gripper import Link

class Metric:
    def __init__(self,link,controller):
        pass
    
    def compute(self):
        raise RuntimeError('This is abstract super-class, use sub-class!')
    
class MassMetric(Metric):
    def __init__(self,link,controller):
        Metric.__init__(self,link,controller)
        
    def compute(self):
        pass
        
class SizeMetric(Metric):
    def __init__(self,link,controller):
        Metric.__init__(self,link,controller)
        
    def compute(self):
        pass
        
class Q1Metric(Metric):
    def __init__(self,link,controller):
        Metric.__init__(self,link,controller)
        
    def compute(self):
        pass
        
class LiftMetric(Metric):
    def __init__(self,link,controller):
        Metric.__init__(self,link,controller)
        
    def compute(self):
        pass
        