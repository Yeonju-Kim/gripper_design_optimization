from multi_objective_BO_GPUCB import *

class ActorCritic:
    def __init__(self,BO,kernel,npolicy):
        self.BO=BO
        self.points=[]
        #the last npolicy variables of each point is policy
        self.npolicy=npolicy
        self.ndesign=len(self.BO.problemBO.vmin)-npolicy
        #gp_critic maps from points to scores
        self.critic=GaussianProcessScaled(kernel=kernel,n_restarts_optimizer=25,alpha=0.0001)
        self.scores=[]
        #gp_actor maps from points to policies
        self.actor=GaussianProcessScaled(kernel=kernel,n_restarts_optimizer=25,alpha=0.0001)
        self.policies=[]
        
    def add_points(self,points,scores):
        self.points+=points
        self.scores+=scores
        spoints=self.BO.scale_01(self.points)
        self.critic.fit(spoints,self.scores)
        
        #optimize policy
        for p in points:
            def obj(x,user_data):
                design_policy=p[:self.ndesign]+x
                value=self.critic.predict([design_policy])
                return -value[0],0
            policy,score,ierror=DIRECT.solve(obj,self.BO.problemBO.vmin[self.ndesign],   \
                                             self.BO.problemBO.vmax[self.ndesign],       \
                                             logfilename='../direct.txt',algmethod=1)
            self.policies.append(policy)
        self.actor.fit(spoints,self.policies)
        
    def estimate_best_policy(self,design):
        return self.actor.predict([design])[0]
    
    def estimate_best_design_policy(self,design):
        return design+self.estimate_best_policy(design)