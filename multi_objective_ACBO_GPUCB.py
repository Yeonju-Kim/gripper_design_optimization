from multi_objective_BO_GPUCB import *
import scipy.optimize
import pdb
import matplotlib.pyplot as plt


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
            # print (-self.critic.predict(points))
            return -self.critic.predict(points).sum()
        if self.localOpt and len(self.policies)>0:
            x=[]
            bounds=[]
            for p in self.policies:
                x+=self.sigmoid_transform(p)
                bounds+=[(a,b) for a,b in zip(self.BO.problemBO.vmin[self.ndesign:],self.BO.problemBO.vmax[self.ndesign:])]
            x=scipy.optimize.minimize(obj_local,np.array(x),method='L-BFGS-B',bounds=bounds).x
            # pdb.set_trace()
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
    def __init__(self,problemBO, num_mc_samples, partition, d_sample_size, use_direct_for_design,
                 kappa=10.,nu=2.5,length_scale=1.,localOpt=True):
        if nu is not None:
            kernel=Matern(nu=nu,length_scale=length_scale)
        else: kernel=RBF(length_scale=length_scale)
        self.d_sample_size =d_sample_size
        self.use_direct_for_design = use_direct_for_design
        #create npolicy
        self.npolicy=0
        for pid in problemBO.vpolicyid:
            if pid>=0:
                self.npolicy+=1
        self.ndesign=len(problemBO.vpolicyid)-self.npolicy

        # num of metrics
        self.num_metric_OI = len([m for m in problemBO.metrics if not m.OBJECT_DEPENDENT])
        self.num_metric_OD = len([m for m in problemBO.metrics if m.OBJECT_DEPENDENT])

        #create GP
        self.gpOI=[] #gpOI[object-independent-metric-id]
        for i in range(self.num_metric_OI):
            self.gpOI.append(GaussianProcessScaled(kernel=kernel,n_restarts_optimizer=25,alpha=0.0001))
        self.gpOD=[] #gpOD[objectid][object-dependent-metric-id]
        for o in problemBO.objects:
            self.gpOD.append([ActorCritic(self,kernel,self.npolicy,self.ndesign,localOpt=localOpt) for m in range(self.num_metric_OD)])

        self.problemBO=problemBO
        self.kappa=kappa
        if len(self.problemBO.metrics)==1:
            raise RuntimeError('MultiObjectiveBO passed with single metric!')

        self.partition = partition
        self.metric_space_dim = self.num_metric_OI + self.num_metric_OD*len(partition)
        self.scores = np.empty((0, self.metric_space_dim))
        self.num_mc_samples = num_mc_samples

    def init(self,num_grid,log_path):
        coordinates=[np.linspace(vminVal,vmaxVal,num_grid) for vminVal,vmaxVal in zip(self.problemBO.vmin,self.problemBO.vmax)]
        if log_path is not None and os.path.exists(log_path+'/init.dat'):
            self.load(log_path+'/init.dat')
            pdb.set_trace()
        else:
            self.pointsOI=[]
            self.scoresOI=[]
            #scoresOI indexes: [pt_id][metric_id]
            #scoresOD indexes: [pt_id][object_id][metric_id]
            points=np.array([dimi.flatten() for dimi in np.meshgrid(*coordinates)]).T.tolist()
            scoresOI,scoresOD=self.problemBO.eval(points,mode='MAX_POLICY')
            self.update_gp(points,scoresOI,scoresOD)
            self.update_PF()
            if log_path is not None:
                self.save(log_path+'/init.dat')
            
    def iterate(self):
        self.update_PF()

        def obj(x,user_data=None):
            return -self.acquisition_MC_sampling(x),0

        if self.use_direct_for_design:
            design,acquisition_val,ierror=DIRECT.solve(obj,self.problemBO.vmin[:self.ndesign],
                                                       self.problemBO.vmax[:self.ndesign],
                                                       logfilename='../direct.txt', algmethod=1,
                                                       maxf=self.d_sample_size)
        else:
            design_samples = np.random.uniform(self.problemBO.vmin[:self.ndesign],
                                               self.problemBO.vmax[:self.ndesign],
                                               (self.d_sample_size, (self.ndesign)))
            obj_d = [obj(design_samples[d])[0] for d in range(len(design_samples))]
            design = design_samples[np.argmin(obj_d)]
        design=design.tolist()
        
        #recover solution point
        points=[]
        # offOD=0
        # for m in self.problemBO.metrics:
        #     if m.OBJECT_DEPENDENT:
        #         policies=[self.gpOD[io][offOD].estimate_best_policy(design) for io,o in enumerate(self.problemBO.objects)]
        #         points.append(design+np.array(policies).T.tolist())
        #         offOD+=1
        policies = [self.estimate_best_policy_per_env(io, design) for io, o in enumerate(self.problemBO.objects)]
        points = [design + np.array(policies).T.tolist()]
        print(points)
        scoresOI,scoresOD=self.problemBO.eval(points,mode='MAX_POLICY')
        self.update_gp(points,scoresOI,scoresOD)
        self.update_PF()

    def acquisition_MC_sampling(self, x):
        env_indep_metrics = np.empty((0, self.num_mc_samples))
        for m_oid in range(self.num_metric_OI):
            samples = self.gpOI[m_oid].sample_y([x], self.num_mc_samples)
            env_indep_metrics = np.vstack((env_indep_metrics, samples))

        env_dep_metrics = self.estimate_env_dep_metrics(point=x)
        costs = self.reconstruct_estimated_score(env_indep_metrics.T, env_dep_metrics)

        num_nondominated = self.is_pareto(costs)
        # print('non dominated', len(num_nondominated))
        return len(num_nondominated)

    def estimate_env_dep_metrics(self, point):
        num_samples = self.num_mc_samples
        policies = [self.estimate_best_policy_per_env(io, point) for io, o in enumerate(self.problemBO.objects)]
        f= []
        for metric_od_idx in range(self.num_metric_OD):
            f_per_metric= np.empty((0, num_samples))
            for oi in range(len(self.problemBO.objects)):
                f_per_metric = np.vstack((f_per_metric,
                                          self.gpOD[oi][metric_od_idx].critic.sample_y([np.hstack((point,policies[oi])).tolist()], num_samples)))
            f.append(f_per_metric)
        return f

    def estimate_best_policy_per_env(self, object_id, point):
        # get best policy from ACBO framework with considering multiple object-dependent metrics
        policy_candidates = []
        for m_id in range(self.num_metric_OD):
            policy_candidates.append(self.gpOD[object_id][m_id].estimate_best_policy(point))

        result= []
        for policy in policy_candidates:
            mul_mean = 1.
            sum_sigma = 0.
            for m_idx in range(self.num_metric_OD):
                m, sigma = self.gpOD[object_id][m_idx].critic.predict([np.hstack((point, policy)).tolist()], return_std=True)
                mul_mean *= m[0]
                sum_sigma += sigma[0]
            result.append(mul_mean+self.kappa*sum_sigma)

        return policy_candidates[np.argmax(result)]

    def reconstruct_estimated_score(self, scoresOI, scoresOD):
        #Same with reconstruct_score in MultiObjectiveBilevel class

        num_points = len(scoresOI)
        costs = np.empty((0, self.metric_space_dim))
        for pt_id in range(num_points):
            score = []
            for oimetric_id in range(self.num_metric_OI):
                score.append(scoresOI[pt_id][oimetric_id])
            for odmetric_id in range(self.num_metric_OD):
                for group in self.partition:
                    score_per_group = 0.
                    for obj_idx in group:
                        score_per_group += scoresOD[odmetric_id][obj_idx][pt_id]
                    score_per_group /= float(len(group))
                    score.append(score_per_group)
            costs = np.vstack((costs, score))
        return costs

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
                return -self.gpOI[offOI].predict([x])[0],0
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

    def is_pareto(self, points):
        pareto_set = self.currentPF
        num_nondomiated = []
        for i in range(len(points)):
            costs = np.vstack((points[i], pareto_set))
            is_efficient = np.ones(costs.shape[0], dtype=bool)
            for j, c in enumerate(costs):
                if is_efficient[j]:
                    is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
                    is_efficient[j] = True
            if is_efficient[0]:
                num_nondomiated.append(i)
        return num_nondomiated

    def update_PF(self):
        costs= np.array(self.scores)
        pareto_set = []
        non_pareto_set = []
        pareto_arg = []
        is_efficient = np.ones(len(costs), dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
                is_efficient[i] = True  # And keep self

        for i in range(costs.shape[0]):
            if is_efficient[i]:
                pareto_set.append(costs[i])
                pareto_arg.append(i)
            else:
                non_pareto_set.append(costs[i])
        self.currentPF = pareto_set

    def run(self, num_grid=5, num_iter=100, log_path=None, log_interval=100, keep_latest=5):
        self.num_grid=num_grid
        if log_path is not None and not os.path.exists(log_path):
            os.mkdir(log_path)
        i = self.load_log(log_path,log_interval,keep_latest)
        if i is 0 and num_grid>0:
            self.init(num_grid, log_path)
            i = 1
        while i<=num_iter:
            print("Multi-Objective ACBO Iter=%d!"%i)
            self.iterate()
            self.save_log(i,log_path,log_interval,keep_latest)
            i+=1

    def draw_plot(self, costs=None):
        plt.figure()

        if costs is not None:
            plt.scatter( costs[:, 0], costs[:, 1], c='b', s=5)
        c = np.array(self.scores)

        for i in range(c.shape[0]):
            plt.text(c[i, 0], c[i, 1], str(i), size=8)
        num_init_data = self.num_grid**len(self.problemBO.vmin)
        plt.scatter(c[:num_init_data,0],c[:num_init_data, 1], c ='cyan', s= 5)
        plt.scatter(c[num_init_data:, 0], c[num_init_data:, 1],c = 'r', s=5)
        plt.show()

    def reconstruct_scores(self):
        points = self.pointsOI
        for ip,p in enumerate(points):
            score=[]
            imOI=0
            imOD=0
            for m in self.problemBO.metrics:
                if not m.OBJECT_DEPENDENT:
                    score.append(self.scoresOI[ip][imOI])
                    imOI+=1
                else:
                    for group in self.partition:
                        meanScore = 0.
                        for io in group:
                            meanScore += self.gpOD[io][imOD].scores[ip]
                        score.append(meanScore/len(self.problemBO.objects))
                    imOD+=1
            self.scores = np.vstack((self.scores, score))
    
    def load(self,filename):
        data=pickle.load(open(filename,'rb'))
        self.pointsOI=data[0]
        self.scoresOI=data[1]

        for m_idx in range(self.num_metric_OI):
            self.gpOI[m_idx].fit(self.pointsOI,
                                 [self.scoresOI[pt_id][m_idx] for pt_id in range(len(self.scoresOI))])

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
        
    def update_gp(self, points, scoresOI, scoresOD):
        #scoresOI indexes: [pt_id][metric_id]
        self.pointsOI+=[pt[:self.ndesign] for pt in points]
        for score in scoresOI:
            self.scoresOI.append([score[im] for im,m in enumerate(self.problemBO.metrics) if not m.OBJECT_DEPENDENT])
        for m_idx in range(self.num_metric_OI):
            self.gpOI[m_idx].fit(self.pointsOI,
                                 [self.scoresOI[pt_id][m_idx] for pt_id in range(len(self.scoresOI))])
        
        #scoresOD indexes: [pt_id][object_id][metric_id]
        for io,o in enumerate(self.problemBO.objects):
            offOD=0
            for im,m in enumerate(self.problemBO.metrics):
                if m.OBJECT_DEPENDENT:
                    self.gpOD[io][offOD].add_points(self.points_object(points,io),
                                                    [scoresOD[ip][io][im] for ip,p in enumerate(points)])
                    offOD+=1

        #add scores to self.scores
        for ip,p in enumerate(points):
            score=[]
            imOI=0
            imOD=0
            for m in self.problemBO.metrics:
                if not m.OBJECT_DEPENDENT:
                    score.append(scoresOI[ip][imOI])
                    imOI+=1
                else:
                    for group in self.partition:
                        meanScore = 0.
                        for io in group:
                            meanScore += scoresOD[ip][io][imOD+self.num_metric_OI]
                        score.append(meanScore/len(self.problemBO.objects))
                    imOD+=1
            self.scores = np.vstack((self.scores, score))
                    
    def points_object(self,points,io):
        ret=[]
        for p in points:
            ret.append([pi[io] if isinstance(pi,list) else pi for pi in p])
        return ret
                    
    def name(self):
        if self.use_direct_for_design:
            return 'ACBO-DIRECT('+self.problemBO.name()+')'+'k='+str(self.kappa)+'d='+str(self.d_sample_size)
        else:
            return 'ACBO-uniform('+self.problemBO.name()+')'+'k='+str(self.kappa)+'d='+str(self.d_sample_size)


if __name__=='__main__':
    from reach_problem_BO import *
    objects=[(-0.5,1.0),(0.0,1.0),(0.5,1.0)]
    obstacles=[Circle((-0.35,0.5),0.1),Circle((0.35,0.5),0.1)]
    reach=ReachProblemBO(objects=objects, obstacles=obstacles,
                         policy_space=[('angle0',None),('angle1',None)])
    
    num_grid=3
    num_iter=100
    use_direct_for_design = False
    BO=MultiObjectiveACBOGPUCB(reach, num_mc_samples= 1000, kappa = 2.0,
                               partition = [[0,1,2]], d_sample_size = 100,
                               use_direct_for_design =use_direct_for_design)
    log_path='../ACBO_reach'
    BO.run(num_grid=num_grid,num_iter=num_iter,log_path=log_path,log_interval=num_iter//10)
    reach.visualize(BO.get_best_on_metric(1)[0])