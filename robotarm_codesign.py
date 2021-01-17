from klampt import *
from klampt import vis

from gl_vis import GLViewer
import pdb
import argparse
from codesign import DesignBase, DesignMetric, DesignSpace, BehaviorMetric, BehaviorSpace, CodesignProblem
from mobbo import MOBBOSettings, MOBBO
import lxml.etree as ET
from collections import OrderedDict
import os, math
import numpy as np


class RobotArmDesign(DesignBase):
    def __init__(self, parameters):
        DesignBase.__init__(self, parameters)
        self.worldfn = 'TRINA_world.xml'


class RobotArmDesignSpace(DesignSpace):
    def __init__(self):
        designSpace = OrderedDict([('distance', (0.05, 0.3)), ('pan', (0, 90)), ('tilt', (-45, 135))])
        xmin = []
        xmax = []
        self.xname = []
        for designParam, minmax in designSpace.items():
            self.xname.append(designParam)
            xmin.append(float(minmax[0]))
            xmax.append(float(minmax[1]))
        DesignSpace.__init__(self, xmax=xmax, xmin=xmin)

    def makeDesign(self, parameters):
        return RobotArmDesign(parameters)


class RobotArmBehaviorSpace(BehaviorSpace):
    def __init__(self, behaviorSpace):
        behaviorSpace = OrderedDict(behaviorSpace)
        self.behaviorName = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']
        xminTotal = [-2*math.pi] * 6
        xmaxTotal = [2*math.pi] * 6
        self.behaviorInit = [0.0322, -2.2055, -2.06, -1.1135, 4.3005, 0.7]

        xmin = []
        xmax = []
        self.xname = []
        for d in range(len(self.behaviorName)):
            if self.behaviorName[d] in behaviorSpace:
                var = behaviorSpace[self.behaviorName[d]]
                if isinstance(var, int) or isinstance(var, float):
                    if var <= xmaxTotal[d] and var >= xminTotal[d]:
                        self.behaviorInit[d] = var
                    else:
                        raise RuntimeError('Incorrect value for behavior')
                elif var is None:
                    self.xname.append(self.behaviorName[d])
                    xmin.append(xminTotal[d])
                    xmax.append(xmaxTotal[d])

        self.nominalConfig= [-5.514873331776197, 4.54462894654496, -5.916490220124983,
                             5.384545803127709, -4.695740116927796, 0.5840738805492091]

        BehaviorSpace.__init__(self, xmax=xmax, xmin=xmin)

    def evaluationData(self, design, environment, behavior, visualize=False):
        """
        parameters: visualize=True used for debugging.
        return trajectory score
        """
        print(design.parameters, environment.name, behavior)
        initConfig = []
        idx=0
        for c, var in enumerate(self.behaviorName):
            if idx < len(self.xname) and self.xname[idx] is var:
                initConfig.append(behavior[idx])
                idx += 1
            else:
                initConfig.append(self.behaviorInit[c])

        world = WorldModel()
        res = world.readFile(design.worldfn)
        if not res:
            raise RuntimeError('Unable to load model')

        robotArmEnv = GLViewer(world, visualization=visualize)
        robotArmEnv.get_robot(*design.parameters)
        robotArmEnv.create_rigidObject(environment.name)
        #data for reachability metric
        sc, vol = robotArmEnv.given_starting_config_score(init_config=initConfig,
                                                          pose=environment.pos,
                                                          vmax=environment.posMax,
                                                          vmin=environment.posMin,
                                                          is_vertical=environment.isVert,
                                                          is_left=True)
        print('score', sc)
        # sc = robotArmEnv.trajectory_score(init_config=initConfig, pose=environment.pos,
        #                                   vmax=environment.posMax, vmin=environment.posMin,
        #                                   is_vert=environment.isVert, is_vis=visualize,
        #                                   is_left=True)
        if visualize:
            vis.run(robotArmEnv)
        return sc

    def visualize(self, design, behavior, envs):
        recomputedScore= []
        world = WorldModel()
        res = world.readFile(design.worldfn)
        if not res:
            raise RuntimeError('Unable to load model')

        robotArmEnv = GLViewer(world, visualization=True)
        robotArmEnv.get_robot(*design.parameters)
        for id , environment in enumerate(envs):
            initConfig = []
            idx = 0
            for c, var in enumerate(self.behaviorName):
                if idx < len(self.xname) and self.xname[idx] is var:
                    initConfig.append(behavior[id][0][idx])
                    idx += 1
                else:
                    initConfig.append(self.behaviorInit[c])

            robotArmEnv.create_rigidObject(environment.name)
            # sc = robotArmEnv.trajectory_score(init_config=initConfig, pose=environment.pos,
            #                                   vmax=environment.posMax, vmin=environment.posMin,
            #                                   is_vert=environment.isVert, is_vis=True,
            #                                   is_left=True)
            sc, vol = robotArmEnv.given_starting_config_score(init_config=initConfig,
                                                              pose=environment.pos,
                                                              vmax=environment.posMax,
                                                              vmin=environment.posMin,
                                                              is_vertical=environment.isVert,
                                                              is_left=True)
            recomputedScore.append(sc)
            print(design.parameters, recomputedScore)
            vis.run(robotArmEnv)
            robotArmEnv.remove_rigidObject()
            robotArmEnv.vis_reset()


    def validSamples(self, design, environment, behSampleSize, maxTrial):
        world = WorldModel()
        res = world.readFile(design.worldfn)
        if not res:
            raise RuntimeError('Unable to load model')
        robotArmEnv = GLViewer(world, visualization=False)

        robotArmEnv.get_robot(*design.parameters)
        robotArmEnv.create_rigidObject(environment.name)
        configs = []
        trial = 0
        while len(configs) < behSampleSize and trial < maxTrial:
            initConfig = self.randomSample().tolist()
            position = (np.array(environment.pos) + np.random.uniform(environment.posMin, environment.posMax)).tolist()
            config = robotArmEnv.local_ik_solve(position, environment.isVert, initConfig)
            trial += 1
            if config is not None:
                configs.append(config)
        print('total num of config:', len(configs))
        return configs

class ReachabilityMetric(BehaviorMetric):
    def __init__(self):
        BehaviorMetric.__init__(self, None, 'ReachabilityMetric')

    def __call__(self, design, environment, behavior, data):
        return data

class TrajectoryMetric(BehaviorMetric):
    def __init__(self):
        BehaviorMetric.__init__(self, None, 'TrajectoryMetric')

    def __call__(self, design, environment, behavior, data):
        return data

class RobotArmEnv:
    def __init__(self, isVert, pos, posMax, posMin, name):
        self.isVert = isVert
        self.pos =pos
        self.posMax = posMax
        self.posMin = posMin
        self.name = name

    def __str__(self):
        return 'Environment vert={self.isVert} & name={self.name}'.format(self=self)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bilevel Gripper optimization')
    parser.add_argument('--numD', type=int, help='Number of design samples per iteration')
    parser.add_argument('--numB', type=int, help='Number of behavior samples for each design')
    parser.add_argument('--kappa', type=float, help='kappa needed for policy search')
    parser.add_argument('--logPath', type=str, default='../mobbo_TRINA', help='Directory path for logging')
    parser.add_argument('--numIter', type=int, default=10, help='The number of designs that will be generated.')
    parser.add_argument('--numMCSamples', type=int, default=1000,
                        help='Num of samples generated from gp posterior to compute acquisition funcion')
    parser.add_argument('--parallel', default=True, type=lambda x: (str(x).lower() == 'true'), help='To use multiprocessing')
    args = parser.parse_args()

    settings = MOBBOSettings(numIter=args.numIter, numD=args.numD, numB=args.numB, kappa=args.kappa,
                             initializeMethod='valid', mc=args.numMCSamples,
                             behOptimizationMethod='valid', parallel=args.parallel, numInitBeh=3, numInitDes=40)

    # Set Bounding boxes for each scenario
    pos = [[0.7, 0.3, 0.0], [0.9, 0.3, 0.2], [0.7, 0.3, 0.7], [0.7, 0.3, 0.7], [0.7, 0.3, 1.3]]
    posMax = [[0.2, 0.4, 0.5], [0.2, 0.4, 0.5], [0.3, 0.5, 0.5], [0.3, 0.5, 0.5], [0.2, 0.4, 0.5]]
    posMin = [[-0.2, -0.4, 0.0], [-0.2, -0.4, 0.0], [-0.3, -0.5, 0.0], [-0.3, -0.5, 0.0], [-0.2, -0.4, 0.0]]
    isVert = [True, False, True, False, False]
    name = [None, 'low_shelf', 'table', 'table', 'high_shelf']

    # Environments
    envs = [RobotArmEnv(isVert[i], pos[i], posMax[i], posMin[i], name[i]) for i in range(5)]

    # Behavior Space
    behaviorSpace = [('c0', None), ('c1', None), ('c2', None), ('c3', None), ('c4',None), ('c5', None)]
    robotArmBehSpace = RobotArmBehaviorSpace(behaviorSpace)

    # Design Space
    robotArmDesignSpace = RobotArmDesignSpace()

    robotArmProblem = CodesignProblem(designSpace=robotArmDesignSpace, behaviorSpace=robotArmBehSpace,
                                      environments=envs, behaviorMetrics=[TrajectoryMetric()],
                                      name='robotArmPlacement', mean=False)

    robotArmMOBBO = MOBBO(problem=robotArmProblem, settings=settings)
    robotArmMOBBO.run(logPath=args.logPath, logInterval=args.numIter // 10, keepLatest=5)