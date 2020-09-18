from multi_objective_BO_GPUCB import *

class ActorCritic:
    def __init__(self,kernel):
        
        #gp_critic maps from points to scores
        self.gp_critic=
        self.points=[]
        self.scores=[]
        
        #gp_actor maps from points to policies
        self.gp_actor=[]
        self.policies=[]