from codesign import *
import numpy as np
from gaussian_process_scaled import GaussianProcessScaled
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic)
import DIRECT
import pdb
import multiprocessing, pickle, os, time

class MOBBOSettings:
    """Stores settings for the optimizer"""
    def __init__(self, numIter, numD, numB, kappa, behOptimizationMethod,
                 initializeMethod='random', parallel = True, nu =2.5, mc = 1000,
                 numInitDes = 10, numInitBeh = 2):
        self.initializeMethod = initializeMethod #optimize/random
        self.numInitialDesignSamples = numInitDes
        self.numInitialBehaviorSamples = numInitBeh

        self.numIterations = numIter
        self.numDesignSamples = numD
        self.numBehaviorEvals = numB
        self.kappa = kappa
        self.behOptimizationMethod = behOptimizationMethod #random or DIRECT
        self.nu = nu
        self.numMCSamples = mc
        self.parallel = parallel
        self.maxTrial = 10000


class MOBBO:
    """
    Attributes:
        problem (CodesignProblem):
        samples (list of tuples):

    A sample is a tuple (design,fvec,behaviors,data,dMetricValues,bMetricValues)
    containing:

        - design: a DesignBase
        - fvec: a metric space vector.
        - behaviors: a matrix of behaviors, size len(environments) x
            len(behaviorMetrics).
        - data: a matrix of evaluationData() results, size len(environments) x
            len(behaviorMetrics).
        - dMetricValues: a list of design metric values
        - bMetricValues: a matrix of behavior metric values, size len(environments) x
            len(behaviorMetrics).
    """
    NUMBER_PROCESS = max(1, multiprocessing.cpu_count() // 2)

    def __init__(self,problem,settings=None):
        self.problem = problem
        self.settings = settings if settings is not None else MOBBOSettings()
        self.samples = []
        #more here
        if settings.nu is not None:
            kernel = Matern(nu=settings.nu)
        else:
            kernel = RBF()
        self.dSurrogate = []
        self.bSurrogate = [] #bsurrogate[envid][metricid]
        for i in range(len(self.problem.designMetrics)):
            self.dSurrogate.append(GaussianProcessScaled(kernel=kernel,n_restarts_optimizer=25,alpha=0.0001))

        for i in range(len(self.problem.environments)):
            self.bSurrogate.append([GaussianProcessScaled(kernel=kernel,n_restarts_optimizer=25,alpha=0.0001) for j in range(len(self.problem.behaviorMetrics))])

        self.elapsedTime = 0.

    def optimizeBehavior(self,design,environment,metric,settings=None):
        """Optimizes a behavior for a given design, env, and metric. Returns
        the tuple (b,d,g) where b is the behavior, d is the evaluation data,
        and g is the metric evaluated at that behavior.

        In rare cases, can return (None,-inf).
        """
        if settings is None:
            settings = self.settings
        assert settings.behOptimizationMethod == 'random',"DIRECT not implemented yet"
        bbest = None
        dbest = None
        gbest = -float('inf')
        for i in range(settings.numBehaviorEvals):
            b = self.problem.behaviorSpace.randomSample(design,environment)
            if not self.problem.behaviorSpace.isValid(design,environment,b):
                continue
            data = self.problem.behaviorSpace.evaluationData(design,environment,b)
            g = metric(design,environment,b,data)
            assert np.ndim(g)==0,"Can't optimize behavior for linked metrics"
            if g > gbest:
                bbest = b
        return bbest,dbest,gbest


    def optimizeBehaviorSurrogate(self,design,envid,metricid,settings=None):
        """Optimizes a behavior for a given design, env, and metric on the
        surrogate metric for the given environment/metric. Returns
        the pair (b,g) where b is the behavior and g is the surrogate evaluated at
        that behavior.

        In rare cases, can return (None,-inf).
        """
        if settings is None:
            settings = self.settings

        if settings.behOptimizationMethod == 'random':
            behCandidates = []
            for t in range(settings.maxTrial):
                newSample = self.problem.behaviorSpace.randomSample()
                if not self.problem.behaviorSpace.isValid(design, self.problem.environments[envid], newSample):
                    continue
                else:
                    behCandidates.append(newSample)
                    if len(behCandidates) == settings.numBehaviorEvals:
                        break

            # behCandidates = [self.problem.behaviorSpace.randomSample() for i in range(settings.numBehaviorEvals)]
            m, sigma = self.bSurrogate[envid][metricid].predict(np.hstack((np.tile(design, (settings.numBehaviorEvals, 1)),
                                                                           np.array(behCandidates))), return_std=True)
            predictedMetrics = m + sigma * settings.kappa
            arg = np.argmax(predictedMetrics)
            return behCandidates[arg], predictedMetrics[arg]
        elif settings.behOptimizationMethod == 'direct':
            def obj(x, user_data):
                if not self.problem.behaviorSpace.isValid(design, self.problem.environments[envid], behavior):
                    return float('inf'), 0
                m,sigma = self.bSurrogate[envid][metricid].predict([design.tolist()+x.tolist()],return_std=True)
                return -m[0]+settings.kappa*sigma[0], 0
            behavior, score, ierror = DIRECT.solve(obj, self.problem.behaviorSpace.xmin, self.problem.behaviorSpace.xmax,
                                                 logfilename='../direct.txt', algmethod=1, maxf=settings.numBehaviorEvals)
            return behavior, score
        elif settings.behOptimizationMethod == 'valid':
            robotDesign = self.problem.designSpace.makeDesign(design)
            behCandidates = self.problem.behaviorSpace.validSamples(robotDesign, self.problem.environments[envid],
                                                                    settings.numBehaviorEvals, settings.maxTrial)
            if len(behCandidates) == 0:
                behCandidates = [self.problem.behaviorSpace.randomSample() for i in range(settings.numBehaviorEvals)]
            m, sigma = self.bSurrogate[envid][metricid].predict(np.hstack((np.tile(design, (len(behCandidates), 1)),
                                                                           np.array(behCandidates))), return_std=True)
            predictedMetrics = m + sigma * settings.kappa
            arg = np.argmax(predictedMetrics)
            return behCandidates[arg], predictedMetrics[arg]


    def setInitialSamples(self, logPath):
        """Sets some initial design samples to seed the optimizer."""
        if logPath is not None and os.path.exists(logPath+'/init.dat'):
            self.loadState(logPath +'/init.dat')
        else:
            # initSettings = MOBBOSettings()
            # initSettings.numDesignSamples = self.settings.numInitialDesignSamples
            # initSettings.designOptimizationMethod = self.settings.designOptimizationMethod
            if self.settings.initializeMethod == 'optimize':
                dinit = [self.problem.designSpace.randomSample() for i in range(self.settings.numInitialDesignSamples)]
                for d in dinit:
                    self.samples.append(self.calcBilevelSample(d))
            elif self.settings.initializeMethod == 'random':
                dinit = [self.problem.designSpace.randomSample() for i in range(self.settings.numInitialDesignSamples)]
                for d in dinit:
                    self.samples += self.calcRandomGridSample(d)
            elif self.settings.initializeMethod == 'valid':
                self.samples += self.calcValidSample()
            if logPath is not None:
                self.saveState(logPath +'/init.dat')
                pdb.set_trace()

    def calcBilevelSample(self, design, settings=None):
        """Calculates a sample corresponding to a design, optimizing or random sampling the decisions
        for each item with the TRUE metrics.
        """
        if settings is None:
            settings = self.settings
        dMetricValues = [f(design) for f in self.problem.designMetrics]
        m = len(self.problem.environments)
        n = len(self.problem.behaviorMetrics)
        behaviors = np.empty((m,n),dtype=object)
        data = np.empty((m,n),dtype=object)
        bMetricValues = np.empty((m,n), dtype=float)
        fvec = [v for v in dMetricValues]
        for j, g in enumerate(self.problem.behaviorMetrics):
            for i, e in enumerate(self.problem.environments):
                b,d,m = self.optimizeBehavior(design,e,g,settings)
                behaviors[i,j] = b
                data[i,j] = d
                bMetricValues[i,j] = m
            for k in range(len(self.problem.metricWeights)):
                fvec.append(self.problem.metricWeights[k].dot(bMetricValues[:, j]))
        fvec = np.stack(fvec)
        return (design,fvec,behaviors,data,dMetricValues,bMetricValues)
    
    def calcRandomGridSample(self, design, settings=None):
        if settings is None:
            settings = self.settings

        samples = []
        m = len(self.problem.environments)
        n = len(self.problem.behaviorMetrics)
        behaviorSamples = self.problem.behaviorSpace.grid(settings.numInitialBehaviorSamples)

        for b in behaviorSamples:
            behaviors = np.empty((m,n), dtype=object)
            for j, g in enumerate(self.problem.behaviorMetrics):
                for i, e in enumerate(self.problem.environments):
                    behaviors[i,j] = b #same b for every metric& envs
            samples.append(self.evaluateSample(design, behaviors))

        return samples

    def calcValidSample(self, settings=None):
        if settings is None:
            settings = self.settings
        samples = []
        m = len(self.problem.environments)
        n = len(self.problem.behaviorMetrics)

        numDesTrial = 0
        while numDesTrial < self.settings.numInitialDesignSamples:
            design = self.problem.designSpace.randomSample()
            numDesTrial += 1
            numBehTrial = 0
            while numBehTrial < self.settings.numInitialBehaviorSamples:
                numBehTrial += 1
                behaviors = np.empty((m, n), dtype=object)
                fail = False
                for j, g in enumerate(self.problem.behaviorMetrics):
                    if fail:
                        break
                    for i, e in enumerate(self.problem.environments):
                        beh = self.problem.behaviorSpace.validSamples(self.problem.designSpace.makeDesign(design),
                                                                      e, 1, settings.maxTrial)
                        if len(beh) == 0:
                            fail = True
                            break
                        else:
                            assert len(beh) == 1
                            behaviors[i, j] = beh[0]
                if not fail:
                    samples.append(self.evaluateSample(design, behaviors))
        return samples

    def evaluateSample(self, design, behaviors):
        robotDesign = self.problem.designSpace.makeDesign(design)
        dMetricValues = [f(robotDesign) for f in self.problem.designMetrics]
        m = len(self.problem.environments)
        n = len(self.problem.behaviorMetrics)
        data = np.empty((m, n), dtype=object)
        bMetricValues = np.empty((m, n), dtype=float)
        fvec = [v for v in dMetricValues]

        if self.settings.parallel:
            from concurrent.futures import ProcessPoolExecutor
            pool = ProcessPoolExecutor(max_workers=MOBBO.NUMBER_PROCESS)

        for j, g in enumerate(self.problem.behaviorMetrics):
            for i, e in enumerate(self.problem.environments):
                if self.settings.parallel:
                    data[i, j] = pool.submit(self.problem.behaviorSpace.evaluationData, robotDesign, e, behaviors[i, j])
                else:
                    data[i, j]= self.problem.behaviorSpace.evaluationData(robotDesign, e, behaviors[i, j])
                    bMetricValues[i, j] = g(robotDesign, e, behaviors[i, j], data[i, j])

        if self.settings.parallel:
            pool.shutdown(wait=True)
            for j, g in enumerate(self.problem.behaviorMetrics):
                for i, e in enumerate(self.problem.environments):
                    data[i,j] = data[i,j].result()
                    bMetricValues[i, j] = g(robotDesign, e, behaviors[i, j], data[i,j])

        for j, g in enumerate(self.problem.behaviorMetrics):
            for k in range(len(self.problem.metricWeights)):
                fvec.append(self.problem.metricWeights[k].dot(bMetricValues[:, j]))
        fvec = np.stack(fvec)
        return (design,fvec,behaviors,data,dMetricValues,bMetricValues)

    def surrogateBilevelSample(self, design):
        """Caculates a sample corresponding to a design, optimizing the decisions
        for each item with the SURROGATE metrics.
        """
        #TODO
        # dMetricValues = [f(design) for f in self.problem.designMetrics]
        E = len(self.problem.environments)
        M = len(self.problem.behaviorMetrics)
        behaviors = np.empty((E, M), dtype=object)
        if self.settings.parallel:
            from concurrent.futures import ProcessPoolExecutor
            pool = ProcessPoolExecutor(max_workers=MOBBO.NUMBER_PROCESS)

        for i, e in enumerate(self.problem.environments):
            for j, g in enumerate(self.problem.behaviorMetrics):
                if self.settings.parallel:
                    behaviors[i, j] = pool.submit(self.optimizeBehaviorSurrogate, design, i, j)
                else:
                    b, _ = self.optimizeBehaviorSurrogate(design=design, envid=i, metricid=j)
                    behaviors[i, j] = b

        if self.settings.parallel:
            pool.shutdown(wait=True)
            for i, e in enumerate(self.problem.environments):
                for j, g in enumerate(self.problem.behaviorMetrics):
                    behaviors[i, j] = behaviors[i, j].result()[0]
        return design, behaviors

    def acquisitionFunction(self, design, behaviors):
        '''
        metricweights = [[1/3, 1/3, 1/3, 0, 0], [0, 0, 0, 1/2, 1/2]]
        metricWeights.shape = (P, |E|)
        '''
        M = len(self.problem.behaviorMetrics)
        P = self.problem.metricWeights.shape[0]
        numSamples = self.settings.numMCSamples
        dMetricSamples = np.empty((0, numSamples))
        bMetricSamples = np.empty((M*P, numSamples))

        for m in range(len(self.problem.designMetrics)):
            samples = self.dSurrogate[m].sample_y([design], numSamples)
            dMetricSamples = np.vstack((dMetricSamples, samples))

        for j in range(len(self.problem.behaviorMetrics)):
            fPerMetric = np.empty((0, self.settings.numMCSamples))
            for i in range(len(self.problem.environments)):
                fPerMetric = np.vstack((fPerMetric,
                                        self.bSurrogate[i][j].sample_y([np.hstack((design, behaviors[i, j]))], numSamples)))
            weightedScore = self.problem.metricWeights.dot(fPerMetric)
            assert weightedScore.shape == (P, numSamples)
            for p in range(P):
                bMetricSamples[M*p + j] = weightedScore[p]

        scores = np.vstack((dMetricSamples, bMetricSamples)).T
        numNonDominated = self.isPareto(scores)
        print(len(numNonDominated))

        return len(numNonDominated)

    def learnGaussianProcess(self):
        '''(design,fvec,behaviors,data,dMetricValues,bMetricValues) '''
        design = [s[0] for s in self.samples]
        for i in range(len(self.problem.designMetrics)):
            dMetricValues = [s[4][i] for s in self.samples]
            self.dSurrogate[i].fit(design, dMetricValues)

        for i in range(len(self.problem.environments)):
            for j in range(len(self.problem.behaviorMetrics)):
                jointInput = [list(s[0]) + list(s[2][i,j]) for s in self.samples]
                bMetricValues = [s[5][i,j] for s in self.samples]
                self.bSurrogate[i][j].fit(jointInput, bMetricValues)

    def isPareto(self, points):
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

    def updatePF(self):
        costs= np.array([s[1] for s in self.samples])
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
        self.currentPF_arg = pareto_arg

    def run(self, logPath, logInterval, keepLatest):
        """Runs the optimizer"""
        startTime = time.time()
        if logPath is not None and not os.path.exists(logPath):
            os.mkdir(logPath)

        iter = self.loadLog(logPath)
        if iter is 0 and len(self.samples) == 0:
            self.setInitialSamples(logPath)
            iter = 1

        self.updatePF()
        self.learnGaussianProcess()
        while iter <= self.settings.numIterations:
            print("Multi-Objective Bilevel Iter=%d" % iter)
            dtest = [self.problem.designSpace.randomSample() for i in range(self.settings.numDesignSamples)]
            sbest = None
            abest = 0
            for d in dtest:
                s = self.surrogateBilevelSample(d)
                a = self.acquisitionFunction(*s)
                if a > abest:
                    abest = a
                    sbest = s
            sactual = self.evaluateSample(*sbest)
            self.samples.append(sactual)
            self.updatePF()
            self.learnGaussianProcess()
            self.saveLog(iter, logPath, logInterval, keepLatest)
            iter += 1
        self.elapsedTime = time.time() - startTime
        return True

    def paretoFront(self):
        """Return the pareto front"""
        raise NotImplementedError()

    def optimalSample(self,metricWeights):
        """Returns the optimal sample in the metric space direction defined by metricWeights."""
        sbest = None
        fbest = -float('inf')
        for s in self.samples:
            fs = metricWeights.dot(s[1])
            if fs > fbest:
                sbest = s
                fbest = fs
        return sbest

    def loadState(self,fn):
        """Loads the state of the optimizer from disk"""
        # raise NotImplementedError()
        self.samples = pickle.load(open(fn, 'rb'))

    def saveState(self,fn):
        """Save the state of the optimizer from disk"""
        pickle.dump(self.samples, open(fn, 'wb'))

    def loadLog(self, logPath):
        i = 0
        if logPath is not None and os.path.exists(logPath):
            import glob
            for f in glob.glob(logPath+'/'+self.name()+'_*.dat'):
                fn = os.path.basename(f).split('_')[-1][:-4]
                try:
                    i = max(i, int(fn))
                except Exception as e:
                    continue
            if i > 0:
                self.loadState(logPath + "/" + self.name() + "_" + str(i) + ".dat")
                i += 1
        return i

    def saveLog(self, i, logPath, logInterval, keepLatest):
        if logPath is not None and i > 0 and i % logInterval == 0:
            self.saveState(logPath + "/" + self.name() + "_" + str(i) + ".dat")

            # delete old
            i -= keepLatest * logInterval
            while i > 0:
                if os.path.exists(logPath + "/" + self.name() + "_" + str(i) + ".dat"):
                    os.remove(logPath + "/" + self.name() + "_" + str(i) + ".dat")
                    i -= logInterval
                else:
                    break

    def name(self):
        return 'Bilevel-'+ self.settings.behOptimizationMethod + \
               '('+self.problem.name+')k=' + str(self.settings.kappa)+ \
               'd='+str(self.settings.numDesignSamples) + 'b=' + str(self.settings.numBehaviorEvals)

    def getStats(self):
        """Returns a dict giving the stats of the optimization progress"""
        res = dict()
        res['num samples'] = len(self.samples)
        res['time elapsed'] = self.elapsedTime #should be calculated in run()
        res['hypervolume'] = 0      #calculate here
        res['pareto spread'] = [0]  #calculate here
        return res

    def visualize(self, sampleIdx):
        sample = self.samples[sampleIdx]
        robotDesign = self.problem.designSpace.makeDesign(sample[0])
        self.problem.behaviorSpace.visualize(design = robotDesign, behavior = sample[2],
                                             envs=self.problem.environments)

    def plotSample(self,visualizer,sample,environment='median',metric='all'):
        #TODO: implement plotting samples
        """Given a DesignVisualizerBase, plots a sample in one or more
        environments.

        To avoid showing choices for all environments and metrics, can specify
        a single environment or auto-determine which environment is interesting
        for each metric.

        Args:
            visualizer (DesignVisualizerBase): a
            sample (tuple): a (design,behaviors,dmetrics,bmetrics) in the
                samples list.
            environment (int, str, or None): if int, an index of an environment
                to draw. If str, can be 'median', 'max', 'min', or 'all'.  If
                None, no environment is drawn.
            metric (int or str): If int, an index of a behavior metric for the
                selected behavior to draw.  If 'all', draws behaviors for
                all metrics.
        """
        (design,fvec,behaviors,data,dmetrics,bmetrics) = sample
        if environment is None:
            visualizer.showDesign(design)
            return
        if isinstance(environment,int) and isinstance(metric,int):
            visualizer.showDesignBehaviorMetric(design,
                                                self.problem.environments[environment],
                                                self.behaviors[environment,metric],
                                                self.bmetrics[environment,metric])
        elif isinstance(environment,int):
            assert metric=='all'
            envs = [self.problem.environments[environment]]*len(self.problem.behaviorMetrics)
            behs = self.behaviors[environment,:]
            metrics = self.bmetrics[environment,:]
            visualizer.showDesignBehaviorMetrics(design,envs,behs,metrics)
        else:
            metricsToDraw = [metric] if isinstance(metric,int) else list(range(len(self.problem.behaviorMetrics)))
            bmetrics_reduced = np.array([bmetrics[:,j] for j in metricsToDraw]).T
            envs = []
            behs = []
            metrics = []
            for j in range(bmetrics_reduced.shape[0]):
                data = bmetrics_reduced[j]
                if environment == 'median':
                    e = np.argsort(data)[len(data)//2]
                elif environment == 'min':
                    e = np.argmin(data)
                else:
                    e = np.argmax(data)
                envs.append(self.problem.environments[e])
                behs.append(self.behaviors[e,metricsToDraw[j]])
                metrics.append(self.bmetrics[e,metricsToDraw[j]])
            visualizer.showDesignBehaviorMetrics(design,envs,behs,metrics)


