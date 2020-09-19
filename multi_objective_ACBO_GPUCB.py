from multi_objective_BO_GPUCB import *

class ActorCritic:
    def __init__(self,BO,kernel,npolicy):
        self.BO=BO
        self.points=[]
        #the last npolicy variables of each point is policy
        self.npolicy=npolicy
        self.ndesign=len(self.BO.problemBO.vpolicyid)-npolicy
        #gp_critic maps from points to scores
        self.critic=GaussianProcessScaled(kernel=kernel,n_restarts_optimizer=25,alpha=0.0001)
        self.scores=[]
        #gp_actor maps from points to policies
        self.actor=GaussianProcessScaled(kernel=kernel,n_restarts_optimizer=25,alpha=0.0001)
        self.policies=[]
        
    def add_points(self,points,scores):
        self.points+=points
        self.scores+=scores
        self.critic.fit(self.BO.scale_01(self.points),self.scores)
        
        #optimize policy
        for p in points:
            def obj(x,user_data):
                design_policy=p[:self.ndesign]+x
                value=self.critic.predict([design_policy])
                return -value[0],0
            policy,score,ierror=DIRECT.solve(obj,self.BO.problemBO.vmin[self.ndesign:],   \
                                             self.BO.problemBO.vmax[self.ndesign:],       \
                                             logfilename='../direct.txt',algmethod=1)
            self.policies.append(policy)
        self.actor.fit(self.BO.scale_01([pt[:self.ndesign] for pt in self.points]),self.policies)
        
    def estimate_best_policy(self,design):
        return self.actor.predict([design])[0]
    
    def estimate_best_design_policy(self,design):
        return design+self.estimate_best_policy(design)

    def estimate_best_score(self,design,return_std=False):
        return self.critic.predict([self.estimate_best_design_policy(design)],return_std=return_std)

    def load(self,points,scores,policies):
        self.points=points
        self.scores=scores
        self.policies=policies
        self.add_points([],[])

    def save(self):
        return [self.points,self.scores,self.policies]

class MultiObjectiveACBOGPUCB(MultiObjectiveBOGPUCB):
    def __init__(self,problemBO,kappa=10.,nu=None,length_scale=1.):
        if nu is not None:
            kernel=Matern(nu=nu,length_scale=length_scale)
        else: kernel=RBF(length_scale=length_scale)
        
        #create npolicy
        self.npolicy=0
        for pid in problemBO.vpolicyid:
            if pid>=0:
                self.npolicy+=1
        self.ndesign=problemBO.vpolicyid-self.npolicy
        
        #create GP
        self.gpOI=GaussianProcessScaled(kernel=kernel,n_restarts_optimizer=25,alpha=0.0001)
        self.gpOD=[]
        for o in problemBO.objects:
            self.gpOD.append([ActorCritic(self,kernel,self.npolicy) for m in problemBO.metrics if m.OBJECT_DEPENDENT])
        
        self.problemBO=problemBO
        self.kappa=kappa
        if len(self.problemBO.metrics)==1:
            raise RuntimeError('MultiObjectiveBO passed with single metric!')
        
    def init(self,num_grid):
        coordinates=[np.linspace(vminVal,vmaxVal,num_grid) for vminVal,vmaxVal in zip(self.problemBO.vmin,self.problemBO.vmax)]
        if os.path.exists('init.dat'):
            self.load('init.dat')
        else:
            self.pointsOI=[]
            self.scoresOI=[]
            #scoresOI indexes: [pt_id][metric_id]
            #scoresOD indexes: [pt_id][object_id][metric_id]
            points=np.array([dimi.flatten() for dimi in np.meshgrid(*coordinates)]).T.tolist()
            scoresOI,scoresOD=self.problemBO.eval(self.points,mode='MAX_POLICY')
            self.update_gp(points,scoresOI,scoresOD)
            
    def iterate(self):
        def obj(x,user_data):
            return -self.acquisition(x),0
        design,acquisition_val,ierror=DIRECT.solve(obj,self.problemBO.vmin[:self.ndesign],
                                                  self.problemBO.vmax[:self.ndesign],
                                                  logfilename='../direct.txt',algmethod=1)
        
        #recover solution point
        points=[]
        for m in problemBO.metrics:
            if m.OBJECT_DEPENDENT:
                policies=[o.estimate_best_policy(design) for io,o in enumerate(problemBO.objects)]
                points.append(design+np.array(policies).T.tolist())
        scoresOI,scoresOD=self.problemBO.eval(points,mode='MAX_POLICY')
        self.update_gp(points,scoresOI,scoresOD)
    
    def acquisition(self,x,user_data=None):
        #GP-UCB
        vol=1.
        sigmaSum=0.
        
        #object dependent
        mu,sigma=self.gpOI.predict(self.scale_01([x]),return_std=True)
        vol*=np.product(mu[0])
        
        #object independent
        im=0
        for m in problemBO.metrics:
            if m.OBJECT_DEPENDENT:
                muOIAvg,sigmaOIAvg=0.,0.
                for io,o in enumerate(problemBO.objects):
                    muOI,sigmaOI=self.gpOD[io][im].estimate_best_score(x,return_std=True)
                    muOIAvg+=muOI[0]
                    sigmaOIAvg+=sigmaOI[0]
                muOIAvg/=len(problemBO.objects)
                sigmaOIAvg/=len(problemBO.objects)
                #accmulate
                mu*=muOIAvg
                sigma+=sigmaOIAvg
                im+=1
        return vol+sigmaSum*self.kappa
                    
    def run(self, num_grid=5, num_iter=100):
        self.init(num_grid)
        for i in range(num_iter):
            print("Multi-Objective ACBO Iter=%d!"%i)
            self.iterate()
        self.reconstruct_scores()
    
    def reconstruct_scores(self):
        self.points=self.pointsOI
        self.scores=[]
        for ip,p in enumerate(self.points):
            score=[]
            imOI=0
            imOD=0
            for m in self.problemBO.metrics:
                if not m.OBJECT_DEPENDENT:
                    score.append(self.scoresOI[ip][imOI])
                    imOI+=1
                else:
                    meanScore=0.
                    for io,o in enumerate(self.problemBO.objects):
                        meanScore+=self.gpOD[io][imOD].estimate_best_score(p)
                    score.append(meanScore/len(self.problemBO.objects))
                    imOD+=1
            self.scores.append(score)
    
    def load(self,filename):
        data=pickle.load(open(filename,'rb'))
        self.pointsOI=data[0]
        self.scoresOI=data[1]
        self.gpOI.fit(self.scale_01(self.pointsOI),self.scoresOI)
        
        offset=2
        for o in problemBO.objects:
            im=0
            for m in problemBO.metrics:
                if m.OBJECT_DEPENDENT:
                    self.gpOD[io][im].load(data[offset+0],data[offset+1],data[offset+2])
                    im+=1
                    offset+=3
        self.reconstruct_scores()
        
    def save(self,filename):
        data=[self.pointsOI,self.scoresOI]
        for o in problemBO.objects:
            im=0
            for m in problemBO.metrics:
                if m.OBJECT_DEPENDENT:
                    data+=self.gpOD[io][im].save()
                    im+=1
        pickle.dump(data,open(filename,'wb'))
        
    def update_gp(self,points,scoresOI,scoresOD):
        #scoresOI indexes: [pt_id][metric_id]
        self.pointsOI+=[pt[:self.ndesign] for pt in points]
        for p in points:
            self.scoresOI.append([scoresOI[im] for im,m in enumerate(problemBO.metrics) if not m.OBJECT_DEPENDENT])
        self.gpOI.fit(self.scale_01(self.pointsOI),self.scoresOI)
        
        #scoresOD indexes: [pt_id][object_id][metric_id]
        for io,o in enumerate(problemBO.objects):
            im=0
            for m in problemBO.metrics:
                if m.OBJECT_DEPENDENT:
                    self.gpOD[io][im].add_points(points,[scoresOD[ip][io][im] for ip,p in enumerate(points)])
                    im+=1
                    
    def name(self):
        return 'MACBO-GP-UCB('+self.problemBO.name()+')'
                 