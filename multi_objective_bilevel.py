from multi_objective_BO_GPUCB import *
import pdb
import matplotlib.pyplot as plt
import multiprocessing
import time
class Bilevel:
    def __init__(self, BO, kernel, npolicy, ndesign):
        self.BO=BO
        self.points=[]
        #the last npolicy variables of each point is policy
        self.npolicy = npolicy
        self.ndesign = ndesign
        #gp_critic maps from points to scores
        self.jointGP=GaussianProcessScaled(kernel=kernel,n_restarts_optimizer=25,alpha=0.0001)
        self.scores=[]

    def add_points(self, points, scores):
        self.points += points
        self.scores += scores
        self.jointGP.fit(self.points, self.scores)

    def predict(self, points, return_std):
        m, sigma = self.jointGP.predict(points,
                                        return_std=return_std)
        return m, sigma
    def sample_y(self, points, num_samples):
        return self.jointGP.sample_y(points, num_samples)

    # def scale_01_design_policy(self, points):
    #     return [[(d - a) / (b - a) for a, b, d in zip(self.BO.vmin, self.BO.vmax, pt)] for pt in points]
    def save(self):
        return [self.points, self.scores]

    def load(self, points, scores):
        self.add_points(points, scores)

    def estimate_best_policy(self, design, kappa =0.):
        def obj(x, user_data=None):
            mu, sigma = self.jointGP.predict([design.tolist()+x.tolist()],
                                             return_std =True)
            upper_bound = mu[0] + kappa * sigma[0]
            return -upper_bound,0

        policy, acquisition_val, ierror = DIRECT.solve(obj, self.BO.vmin[self.ndesign:],
                                                       self.BO.vmax[self.ndesign:],
                                                       logfilename='../direct.txt', algmethod=1)
        return policy

    def estimate_best_score(self, design, return_std= False):
        policy = self.estimate_best_policy(design).tolist()
        design_policy = design.tolist() + policy
        return self.jointGP.predict([design_policy], return_std = return_std)

class MultiObjectiveBOBilevel(MultiObjectiveBOGPUCB):
    def __init__(self, problemBO, d_sample_size, num_mc_samples, max_f_eval, use_direct, partition,parallel,
                 kappa=10., nu = None, length_scale = 1):
        if nu is not None:
            kernel = Matern(nu=nu, length_scale=length_scale)
        else:
            kernel = RBF(length_scale=length_scale)
        self.parallel = parallel
        # create npolicy
        self.npolicy = 0
        for pid in problemBO.vpolicyid:
            if pid >= 0:
                self.npolicy += 1
        self.ndesign = len(problemBO.vpolicyid) - self.npolicy

        # num of metrics
        self.num_metric_OI = len([m for m in problemBO.metrics if not m.OBJECT_DEPENDENT])
        self.num_metric_OD = len([m for m in problemBO.metrics if m.OBJECT_DEPENDENT])
        # create GP
        self.gpOI = []
        for i in range(self.num_metric_OI):
            self.gpOI.append(GaussianProcessScaled(kernel=kernel, n_restarts_optimizer=25, alpha=0.0001))
        self.gpOD = []
        for o in problemBO.objects:
            self.gpOD.append([Bilevel(problemBO, kernel, self.npolicy, self.ndesign) for i in range(self.num_metric_OD)])
            #gpOD[objectid][object_dependent_metric id]

        self.problemBO = problemBO
        self.kappa = kappa
        if len(self.problemBO.metrics) == 1:
            raise RuntimeError('MultiObjectiveBO passed with single metric!')

        self.d_sample_size =d_sample_size
        self.num_mc_samples = num_mc_samples
        self.max_f_eval = max_f_eval
        self.time_simulation = []
        self.time_config_opt = []
        self.time_fitting = []
        self.time_total_iter = []
        self.temp_fmax_policy = []
        self.use_direct= use_direct
        self.partition = partition #[[0,1,2],[3,4,5],[6,7]] for 7 objects
        self.metric_space_dim = self.num_metric_OI + self.num_metric_OD*len(partition)
        self.scores = np.empty((0, self.metric_space_dim))
        self.num_processes = max(1, multiprocessing.cpu_count()//2)

    def init(self, num_grid, log_path):
        self.num_grid = num_grid
        coordinates=[np.linspace(vminVal,vmaxVal,num_grid) for vminVal,vmaxVal in zip(self.problemBO.vmin,self.problemBO.vmax)]
        if log_path is not None and os.path.exists(log_path+'/init.dat'):
            self.load(log_path+'/init.dat')
            print('init!')
        else:
            self.pointsOI=[]
            self.scoresOI=[]
            #scoresOI indexes: [pt_id][metric_id]
            #scoresOD indexes: [pt_id][object_id][metric_id]
            points=np.array([dimi.flatten() for dimi in np.meshgrid(*coordinates)]).T.tolist()
            print(points)
            scoresOI,scoresOD=self.problemBO.eval(points,mode='MAX_POLICY')
            self.update_gp(points,scoresOI,scoresOD)
            self.update_PF()
            if log_path is not None:
                self.save(log_path+'/init.dat')


    def update_gp(self, points, scoresOI, scoresOD):
        #scoresOI indexes: [pt_id][metric_id]
        self.pointsOI += [pt[:self.ndesign] for pt in points]
        for score in scoresOI:
            self.scoresOI.append([score[im] for im, m in enumerate(self.problemBO.metrics) if not m.OBJECT_DEPENDENT])
        for m_idx in range(self.num_metric_OI):
            self.gpOI[m_idx].fit(self.pointsOI,#self.scale_01(self.pointsOI),
                                 [self.scoresOI[pt_id][m_idx] for pt_id in range(len(self.scoresOI))])

        # scoresOD indexes: [pt_id][object_id][metric_id]
        for io, o in enumerate(self.problemBO.objects):
            offOD = 0
            for im, m in enumerate(self.problemBO.metrics):
                if m.OBJECT_DEPENDENT:
                    self.gpOD[io][offOD].add_points(self.points_object(points, io),
                                                    [scoresOD[ip][io][im] for ip, p in enumerate(points)])
                    offOD += 1

        #reconstruct score
        for pt_id in range(len(points)):
            score = []
            for m_idx in range(self.num_metric_OI):
                score.append(scoresOI[pt_id][m_idx])
            for m_idx in range(self.num_metric_OD):
                for group in self.partition:
                    score_per_group = 0.
                    for obj_idx in group:
                        score_per_group += scoresOD[pt_id][obj_idx][m_idx+self.num_metric_OI]
                    score_per_group /= float(len(group))
                    score.append(score_per_group)
            self.scores = np.vstack((self.scores, score))

    def run(self, num_grid=5, num_iter=100, log_path = None, log_interval = 100, keep_latest= 5):
        self.num_grid=num_grid
        if log_path is not None and not os.path.exists(log_path):
            os.mkdir(log_path)

        i = self.load_log(log_path,log_interval,keep_latest)
        if i is 0 and num_grid>0:
            self.init(num_grid,log_path)
            i = 1
        while i <= num_iter:
            print("Multi-Objective Bilevel Iter=%d!" % i)
            self.iterate()
            self.save_log(i, log_path, log_interval, keep_latest)
            i += 1

    def points_object(self, points, io):
        ret = []
        for p in points:
            ret.append([pi[io] if isinstance(pi, list) else pi for pi in p])
        return ret

    def iterate(self):
        total_iter = time.time()
        design_samples = np.random.uniform(self.problemBO.vmin[:self.ndesign],
                                           self.problemBO.vmax[:self.ndesign],
                                           size=(self.d_sample_size, self.ndesign))
        acquisition_values = []
        self.update_PF()

        #compute acquisition function value for design samples
        time_list_compute_acq = []
        for i in range(self.d_sample_size):
            st_time = time.time()
            acquisition_values.append(self.acquisition_MC_sampling(design_samples[i]))
            time_list_compute_acq.append(time.time() - st_time)
        print('mean of computation acq', np.mean(time_list_compute_acq))
        print(acquisition_values)
        self.time_config_opt.append(np.mean(time_list_compute_acq))

        # Pick design+policy to be evaluated next
        idx_max_acq = np.argmax(acquisition_values)
        point = design_samples[idx_max_acq]
        design_policy = point.tolist() + np.array(self.temp_fmax_policy[idx_max_acq]).T.tolist()

        #Environment-independent score: mass
        time_eval_metric = time.time()
        scoresOI, scoresOD = self.problemBO.eval([design_policy], mode='MAX_POLICY', visualize=False)
        self.time_simulation.append(time.time() - time_eval_metric)
        print('simulation: ', time.time() - time_eval_metric)

        time_fit = time.time()
        self.update_gp([design_policy], scoresOI, scoresOD)
        print('fitting: ', time.time() - time_fit)
        self.time_fitting.append( time.time() - time_fit)
        self.time_total_iter.append(time.time()-total_iter)
        self.temp_fmax_policy.clear()

    def acquisition_MC_sampling(self, x):
        env_indep_metrics = np.empty((0,self.num_mc_samples ))

        for m_oid in range(self.num_metric_OI):
            samples = self.gpOI[m_oid].sample_y([x], self.num_mc_samples)
            env_indep_metrics = np.vstack((env_indep_metrics, samples))
        # env_idep_metrics = (num_metrics, sample_id)
        #env_dep_metrics = (object, num_samples,
        env_dep_metrics = self.estimate_env_dep_metrics(point=x)

        costs = self.reconstruct_score(env_indep_metrics.T, env_dep_metrics)

        # costs = np.vstack((env_indep_metrics, env_dep_metrics)).T

        # compare with current PF
        num_nondominated = self.is_pareto(costs)

        # self.draw_plot(costs)
        return len(num_nondominated)

    def estimate_env_dep_metrics(self, point):
        #mc samples for env-dependent metrics
        num_samples = self.num_mc_samples

        result = [None for m in self.problemBO.objects] # list of tuples: [(policy, gp value), .. ]
        policies = [None for i in self.problemBO.objects]

        if self.use_direct:
            config_opt_func = self.compute_max_gp_DIRECT
        else:
            config_opt_func = self.compute_max_gp_sampling

        parallel = self.parallel
        if parallel:
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
            pool = ProcessPoolExecutor(max_workers=self.num_processes)

        for i in range(len(self.problemBO.objects)):
            if parallel:
                result[i] = pool.submit(config_opt_func, i, point) # max_function_evaluation)
                print(result)
            else:
                temp_result = config_opt_func(i, point)
                policies[i] = temp_result

        if parallel:
            pool.shutdown(wait=True)

            for id in range(len(self.problemBO.objects)):
                policies[id] = result[id].result()
        print('policies= ', policies)


        #TODO: extend it to multiple env-dependent metrics
        # m_idx = 0
        f =[]
        for metric_od_idx in range(self.num_metric_OD):
            f_per_metric = np.empty((0, num_samples))
            for i in range(len(self.problemBO.objects)):
                f_per_metric = np.vstack((f_per_metric,
                                          self.gpOD[i][metric_od_idx].sample_y([np.hstack((point, policies[i])).tolist()], num_samples)))
            f.append(f_per_metric)
        self.temp_fmax_policy.append(policies)
        return f

    def compute_max_gp_sampling(self, object_id, point):
        config_candidates = np.random.uniform(self.problemBO.vmin[self.ndesign:],
                                              self.problemBO.vmax[self.ndesign:],
                                              (self.max_f_eval, self.npolicy))

        mul_mean = 1.
        sum_sigma = 0.
        for m_id in range(self.num_metric_OD):
            m, sigma = self.gpOD[object_id][m_id].predict(np.hstack((np.tile(point, (self.max_f_eval, 1)), config_candidates)),
                                                          return_std=True)
            mul_mean *= m
            sum_sigma += sigma

        predicted_val = mul_mean + np.expand_dims(sum_sigma, axis=1) * self.kappa
        arg = np.argmax(predicted_val, axis=0)
        print(config_candidates[arg][0])
        return config_candidates[arg][0]

    def compute_max_gp_DIRECT(self, object_id, point):
        def obj(x, user_data):
            mul_mean = 1.
            sum_sigma = 0.
            for m_id in range(self.num_metric_OD):
                m, sigma = self.gpOD[object_id][m_id].predict([point.tolist() + x.tolist()],
                                                              return_std= True)
                mul_mean *= m
                sum_sigma += sigma
            return -(mul_mean[0] + self.kappa * sum_sigma[0]),0
        policy, score, ierrer = DIRECT.solve(obj, self.problemBO.vmin[self.ndesign:],
                                              self.problemBO.vmax[self.ndesign:],
                                              logfilename='../direct.txt',
                                              maxf = self.max_f_eval, algmethod = 1)
        return policy


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

    def load(self, filename):
        data = pickle.load(open(filename, 'rb'))
        self.pointsOI = data[0]
        self.scoresOI = data[1]
        for m_idx in range(self.num_metric_OI):
            self.gpOI[m_idx].fit(self.pointsOI,
                                 [self.scoresOI[pt_id][m_idx] for pt_id in range(len(self.scoresOI))])
        data = data[2:]

        for io, o in enumerate(self.problemBO.objects):
            for offOD in range(self.num_metric_OD):
                self.gpOD[io][offOD].load(data[0], data[1])
                data = data[2:]

        #TODO: reconstruct score values
        for pt_id in range(len(self.pointsOI)):
            score = []
            for m_idx in range(self.num_metric_OI):
                score.append(self.scoresOI[pt_id][m_idx])
            for m_idx in range(self.num_metric_OD):
                for group in self.partition:
                    score_per_group = 0.
                    for obj_idx in group:
                        score_per_group += self.gpOD[obj_idx][m_idx].scores[pt_id]
                    score_per_group /= float(len(group))
                    score.append(score_per_group)
            self.scores = np.vstack((self.scores, score))
        # pdb.set_trace()

    def reconstruct_score(self, scoresOI, scoresOD):
        #scoresOD.shape
        # pdb.set_trace()
        num_points = len(scoresOI)
        costs = np.empty((0, self.metric_space_dim))
        for pt_id in range(num_points):
            score= []
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
        # pdb.set_trace()
        return costs


    def save(self, filename):
        data = [self.pointsOI, self.scoresOI]
        for io, o in enumerate(self.problemBO.objects):
            for offOD in range(self.num_metric_OD):
                data += self.gpOD[io][offOD].save()
        pickle.dump(data, open(filename, 'wb'))

    def get_best_on_metric(self, id, num_design_samples=20):
        num_design_samples = num_design_samples
        # find index
        offOD = 0
        offOI = 0
        for im, m in enumerate(self.problemBO.metrics):
            if m.OBJECT_DEPENDENT:
                if im == id:
                    break
                offOD += 1
            else:
                if im == id:
                    break
                offOI += 1

        #optimize
        def obj(x, user_data=None):
            print(x)
            if self.problemBO.metrics[id].OBJECT_DEPENDENT:
                muOIAvg = 1.
                for group in self.partition:
                    score_per_group = 0.
                    for obj_idx in group:
                        score_per_group += self.gpOD[obj_idx][offOD].estimate_best_score(x)
                    score_per_group /= len(group)
                    muOIAvg *= score_per_group
                return -muOIAvg, 0
            else: #object_independent
                return -self.gpOI[offOI].predict([x])[0],0
        if self.problemBO.metrics[id].OBJECT_DEPENDENT:
            design_samples = np.random.uniform(self.problemBO.vmin[:self.ndesign],
                                               self.problemBO.vmax[:self.ndesign],
                                               (num_design_samples, (self.ndesign)))
            obj_d = [obj(design_samples[d])[0] for d in range(len(design_samples))]
            design = design_samples[np.argmin(obj_d)]
            pdb.set_trace()
            print(obj_d, 'design', design)
        else:
            design, acquisition_val, ierror = DIRECT.solve(obj,self.problemBO.vmin[:self.ndesign],
                                                           self.problemBO.vmax[:self.ndesign],
                                                           logfilename='../direct.txt',algmethod=1)
        #recover solution point
        points = []
        offOD = 0
        for m in self.problemBO.metrics:
            if m.OBJECT_DEPENDENT:
                policies=[self.gpOD[io][offOD].estimate_best_policy(design) for io,o in enumerate(self.problemBO.objects)]
                points.append(design.tolist()+np.array(policies).T.tolist())
                offOD+=1

        return points

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

    def graph_gripper_plot(self):
        plt.figure()
        d= self.scores
        for i in range(d.shape[0]):
            plt.text(100*d[i, 0], d[i, 1], str(i), size=8)
        init_idx = 64
        plt.scatter(100*d[:init_idx, 0], d[:init_idx, 1], c='cyan', s=15, label='Initial designs ')
        plt.scatter(100*d[init_idx:, 0], d[init_idx:, 1], c='blue', s=15, label='New designs')
        # plt.scatter(-100 / p[:, 0], p[:, 1], c='r', s=5, label='Pareto fronts')

        plt.legend()
        plt.xlabel('100/Metric')
        plt.ylabel('Elapsed Time Metric')
        plt.show()

    def name(self):
        if self.use_direct:
            return 'BILEVEL-DIRECT('+self.problemBO.name()+')'+'k='+str(self.kappa)\
                   +'d='+str(self.d_sample_size) +'fmax='+str(self.max_f_eval)
        else:
            return 'BILEVEL-uniform('+self.problemBO.name()+')'+'k='+str(self.kappa)\
                   +'d='+str(self.d_sample_size) +'fmax='+str(self.max_f_eval)



    def save_log(self, i, log_path, log_interval, keep_latest):
        if log_path is not None and i > 0 and i % log_interval == 0:
            self.save(log_path + "/" + self.name() + "_" + str(i) + ".dat")

            # delete old
            i -= keep_latest * log_interval
            while i > 0:
                if os.path.exists(log_path + "/" + self.name() + "_" + str(i) + ".dat"):
                    os.remove(log_path + "/" + self.name() + "_" + str(i) + ".dat")
                    i -= log_interval
                else:
                    break

if __name__ == '__main__':
    from reach_problem_BO import *

    objects = [(-0.5, 1.0), (0.0, 1.0), (0.5, 1.0)]
    obstacles = [Circle((-0.35, 0.5), 0.1), Circle((0.35, 0.5), 0.1)]
    reach = ReachProblemBO(objects=objects, obstacles=obstacles, policy_space=[('angle0', None), ('angle1', None)])

    num_grid = 3
    num_iter = 10
    BO = MultiObjectiveBOBilevel(reach, d_sample_size=10,
                                 num_mc_samples = 1000, partition=[[0,1,2]],
                                 max_f_eval = 1000, kappa=2.0, nu=2.5, use_direct=True,
                                 parallel =False)
    log_path = '../bilevel'
    BO.run(num_grid=num_grid, num_iter=num_iter, log_path=log_path, log_interval=num_iter//10)
    BO.draw_plot()
    # design = BO.pointsOI[np.argmax(np.array(BO.scores)[:, 1])]
    # ar = np.argmax(np.array(BO.scores)[:, 1])
    # policy = np.array([BO.gpOD[0][0].points[ar][2:],BO.gpOD[1][0].points[ar][2:], BO.gpOD[2][0].points[ar][2:]]).T.tolist()
    # reach.visualize(design +policy )
    # pdb.set_trace()

    reach.visualize(BO.get_best_on_metric(1)[0])
    pdb.set_trace()