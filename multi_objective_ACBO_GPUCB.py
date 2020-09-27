from multi_objective_BO_GPUCB import *
import scipy.optimize

class ActorCritic:
    def __init__(self,BO,kernel,npolicy,ndesign,localOpt=True):
        self.BO=BO
        self.points=[]
        #the last npolicy variables of each point is policy
        self.npolicy=npolicy
        self.ndesign=ndesign
        #gp_critic maps from points to scores
        self.critic=GaussianProcessScaled(kernel=kernel,n_restarts_optimizer=25,alpha=0.0001)
        self.scores=[]
        #gp_actor maps from points to policies
        self.actor=GaussianProcessScaled(kernel=kernel,n_restarts_optimizer=25,alpha=0.0001)
        self.policies=[]
        #old policy local optimization
        self.localOpt=localOpt
        
    def add_points(self,points,scores):
        #update critic
        self.critic.fit(self.points+points,self.scores+scores)
        
        #optimize policy for old points: local
        def obj_local(x):
            x=x.tolist()
            points=[p[:self.ndesign]+x[i*self.npolicy:i*self.npolicy+self.npolicy] for i,p in enumerate(self.points)]
            return -self.critic.predict(points).sum()
        if self.localOpt and len(self.policies)>0:
            x=[]
            bounds=[]
            for p in self.policies:
                x+=self.sigmoid_transform(p)
                bounds+=[(a,b) for a,b in zip(self.BO.problemBO.vmin[self.ndesign:],self.BO.problemBO.vmax[self.ndesign:])]
            x=scipy.optimize.minimize(obj_local,np.array(x),method='L-BFGS-B',bounds=bounds).x
            self.policies=[self.logit_transform(x[i*self.npolicy:i*self.npolicy+self.npolicy]) for i,p in enumerate(self.policies)]
            
        #store points
        self.points+=points
        self.scores+=scores
        
        #optimize policy for new points: global
        for p in points:
            def obj_global(x,user_data):
                design_policy=p[:self.ndesign]+x.tolist()
                value=self.critic.predict([design_policy])
                return -value[0],0
            policy,score,ierror=DIRECT.solve(obj_global,self.BO.problemBO.vmin[self.ndesign:],  \
                                             self.BO.problemBO.vmax[self.ndesign:],       \
                                             logfilename='../direct.txt',algmethod=1)
            self.policies.append(self.logit_transform(policy.tolist()))
        self.actor.fit([pt[:self.ndesign] for pt in self.points],self.policies)
        
    def estimate_best_policy(self,design):
        return self.sigmoid_transform(self.actor.predict([design])[0].tolist())
    
    def logit_transform(self,policy,margin=0.01):
        ret=[]
        for ip,p in enumerate(policy):
            a=self.BO.problemBO.vmin[self.ndesign+ip]
            b=self.BO.problemBO.vmax[self.ndesign+ip]
            margin=(b-a)*margin
            a-=margin
            b+=margin
            val=(p-a)/(b-a)
            ret.append(math.log(val)-math.log(1-val))
        return ret
    
    def sigmoid_transform(self,policy,margin=0.01):
        ret=[]
        for ip,p in enumerate(policy):
            a=self.BO.problemBO.vmin[self.ndesign+ip]
            b=self.BO.problemBO.vmax[self.ndesign+ip]
            margin=(b-a)*margin
            a-=margin
            b+=margin
            val=1./(1.+1./math.exp(p))
            ret.append(val*(b-a)+a)
        return ret
    
    def estimate_best_design_policy(self,design):
        return design+self.estimate_best_policy(design)

    def estimate_best_score(self,design,return_std=False):
        design_policy=self.estimate_best_design_policy(design)
        return self.critic.predict([design_policy],return_std=return_std)

    def load(self,points,scores,policies):
        self.points=points
        self.scores=scores
        self.policies=policies
        self.add_points([],[])

    def save(self):
        return [self.points,self.scores,self.policies]

class MultiObjectiveACBOGPUCB(MultiObjectiveBOGPUCB):
    def __init__(self,problemBO,kappa=10.,nu=None,length_scale=1.,localOpt=True):
        if nu is not None:
            kernel=Matern(nu=nu,length_scale=length_scale)
        else: kernel=RBF(length_scale=length_scale)
        
        #create npolicy
        self.npolicy=0
        for pid in problemBO.vpolicyid:
            if pid>=0:
                self.npolicy+=1
        self.ndesign=len(problemBO.vpolicyid)-self.npolicy
        
        #create GP
        self.gpOI=GaussianProcessScaled(kernel=kernel,n_restarts_optimizer=25,alpha=0.0001)
        self.gpOD=[]
        for o in problemBO.objects:
            self.gpOD.append([ActorCritic(self,kernel,self.npolicy,self.ndesign,localOpt=localOpt) for m in problemBO.metrics if m.OBJECT_DEPENDENT])
        
        self.problemBO=problemBO
        self.kappa=kappa
        if len(self.problemBO.metrics)==1:
            raise RuntimeError('MultiObjectiveBO passed with single metric!')
        
    def init(self,num_grid,log_path):
        coordinates=[np.linspace(vminVal,vmaxVal,num_grid) for vminVal,vmaxVal in zip(self.problemBO.vmin,self.problemBO.vmax)]
        if log_path is not None and os.path.exists(log_path+'/init.dat'):
            self.load(log_path+'/init.dat')
        else:
            self.pointsOI=[]
            self.scoresOI=[]
            #scoresOI indexes: [pt_id][metric_id]
            #scoresOD indexes: [pt_id][object_id][metric_id]
            points=np.array([dimi.flatten() for dimi in np.meshgrid(*coordinates)]).T.tolist()
            scoresOI,scoresOD=self.problemBO.eval(points,mode='MAX_POLICY')
            self.update_gp(points,scoresOI,scoresOD)
            if log_path is not None:
                self.save(log_path+'/init.dat')
            
    def iterate(self):
        def obj(x,user_data):
            return -self.acquisition(x),0
        design,acquisition_val,ierror=DIRECT.solve(obj,self.problemBO.vmin[:self.ndesign],
                                                  self.problemBO.vmax[:self.ndesign],
                                                  logfilename='../direct.txt',algmethod=1)
        design=design.tolist()
        
        #recover solution point
        points=[]
        offOD=0
        for m in self.problemBO.metrics:
            if m.OBJECT_DEPENDENT:
                policies=[self.gpOD[io][offOD].estimate_best_policy(design) for io,o in enumerate(self.problemBO.objects)]
                points.append(design+np.array(policies).T.tolist())
                offOD+=1
        scoresOI,scoresOD=self.problemBO.eval(points,mode='MAX_POLICY')
        self.update_gp(points,scoresOI,scoresOD)
    
    def get_best(self):
        kappa_tmp=self.kappa
        self.kappa=0.
        def obj(x,user_data):
            return -self.acquisition(x),0
        design,acquisition_val,ierror=DIRECT.solve(obj,self.problemBO.vmin[:self.ndesign],
                                                   self.problemBO.vmax[:self.ndesign],
                                                   logfilename='../direct.txt',algmethod=1)
        design=design.tolist()
        self.kappa=kappa_tmp
        
        #recover solution point
        points=[]
        offOD=0
        for m in self.problemBO.metrics:
            if m.OBJECT_DEPENDENT:
                policies=[self.gpOD[io][offOD].estimate_best_policy(design) for io,o in enumerate(self.problemBO.objects)]
                points.append(design+np.array(policies).T.tolist())
                offOD+=1
        return points
    
    def get_best_on_metric(self,id):
        #find index
        offOD=0
        offOI=0
        for im,m in enumerate(self.problemBO.metrics):
            if m.OBJECT_DEPENDENT:
                if im==id:
                    break
                offOD+=1
            else:
                if im==id:
                    break
                offOI+=1
                
        #optimize
        def obj(x,user_data):
            x=x.tolist()
            if self.problemBO.metrics[id].OBJECT_DEPENDENT:
                muOIAvg=0.
                for io,o in enumerate(self.problemBO.objects):
                    muOIAvg+=self.gpOD[io][offOD].estimate_best_score(x)
                muOIAvg/=len(self.problemBO.objects)
                return -muOIAvg,0
            else:
                return -self.gpOI.predict([x])[0][offOI],0
        design,acquisition_val,ierror=DIRECT.solve(obj,self.problemBO.vmin[:self.ndesign],
                                                   self.problemBO.vmax[:self.ndesign],
                                                   logfilename='../direct.txt',algmethod=1)
        design=design.tolist()
        
        #recover solution point
        points=[]
        offOD=0
        for m in self.problemBO.metrics:
            if m.OBJECT_DEPENDENT:
                policies=[self.gpOD[io][offOD].estimate_best_policy(design) for io,o in enumerate(self.problemBO.objects)]
                points.append(design+np.array(policies).T.tolist())
                offOD+=1
        return points
    
    def acquisition(self,x,user_data=None):
        #GP-UCB
        vol=1.
        sigmaSum=0.
        x=x.tolist()
        
        #object dependent
        mu,sigma=self.gpOI.predict([x],return_std=True)
        vol*=np.product(mu[0])
        sigmaSum+=np.sum(sigma[0])
        
        #object independent
        im=0
        for m in self.problemBO.metrics:
            if m.OBJECT_DEPENDENT:
                muOIAvg,sigmaOIAvg=0.,0.
                for io,o in enumerate(self.problemBO.objects):
                    muOI,sigmaOI=self.gpOD[io][im].estimate_best_score(x,return_std=True)
                    muOIAvg+=muOI[0]
                    sigmaOIAvg+=sigmaOI[0]
                muOIAvg/=len(self.problemBO.objects)
                sigmaOIAvg/=len(self.problemBO.objects)
                #accmulate
                vol*=muOIAvg
                sigmaSum+=sigmaOIAvg
                im+=1
        return vol+sigmaSum*self.kappa
                    
    def run(self, num_grid=5, num_iter=100, log_path=None, log_interval=100, keep_latest=5):
        if log_path is not None and not os.path.exists(log_path):
            os.mkdir(log_path)
        if num_grid>0:
            self.init(num_grid, log_path)
        i=self.load_log(log_path,log_interval,keep_latest)
        while i<=num_iter:
            print("Multi-Objective ACBO Iter=%d!"%i)
            self.iterate()
            self.save_log(i,log_path,log_interval,keep_latest)
            i+=1
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
        self.gpOI.fit(self.pointsOI,self.scoresOI)
        data=data[2:]
        
        for io,o in enumerate(self.problemBO.objects):
            offOD=0
            for m in self.problemBO.metrics:
                if m.OBJECT_DEPENDENT:
                    self.gpOD[io][offOD].load(data[0],data[1],data[2])
                    data=data[3:]
                    offOD+=1
        self.reconstruct_scores()
        
    def save(self,filename):
        data=[self.pointsOI,self.scoresOI]
        for io,o in enumerate(self.problemBO.objects):
            offOD=0
            for m in self.problemBO.metrics:
                if m.OBJECT_DEPENDENT:
                    data+=self.gpOD[io][offOD].save()
                    offOD+=1
        pickle.dump(data,open(filename,'wb'))
        
    def update_gp(self,points,scoresOI,scoresOD):
        #scoresOI indexes: [pt_id][metric_id]
        self.pointsOI+=[pt[:self.ndesign] for pt in points]
        for score in scoresOI:
            self.scoresOI.append([score[im] for im,m in enumerate(self.problemBO.metrics) if not m.OBJECT_DEPENDENT])
        self.gpOI.fit(self.pointsOI,self.scoresOI)
        
        #scoresOD indexes: [pt_id][object_id][metric_id]
        for io,o in enumerate(self.problemBO.objects):
            offOD=0
            for im,m in enumerate(self.problemBO.metrics):
                if m.OBJECT_DEPENDENT:
                    self.gpOD[io][offOD].add_points(self.points_object(points,io),[scoresOD[ip][io][im] for ip,p in enumerate(points)])
                    offOD+=1
                    
    def points_object(self,points,io):
        ret=[]
        for p in points:
            ret.append([pi[io] if isinstance(pi,list) else pi for pi in p])
        return ret
                    
    def name(self):
        return 'MACBO-GP-UCB('+self.problemBO.name()+')'
                 
if __name__=='__main__':
    from reach_problem_BO import *
    objects=[(-0.5,1.0),(0.0,1.0),(0.5,1.0)]
    obstacles=[Circle((-0.35,0.5),0.2),Circle((0.35,0.5),0.2)]
    reach=ReachProblemBO(objects=objects,obstacles=obstacles,policy_space=[('angle0',None),('angle1',None)])
    
    num_grid=3
    num_iter=100
    BO=MultiObjectiveACBOGPUCB(reach)
    log_path='../'+BO.name()
    BO.run(num_grid=num_grid,num_iter=num_iter,log_path=log_path,log_interval=num_iter//10)
    reach.visualize(BO.get_best_on_metric(1)[0])