from codesign import DesignBase, DesignMetric, DesignSpace, BehaviorMetric, BehaviorSpace, CodesignProblem
from mobbo import MOBBOSettings, MOBBO

from compile_gripper import Gripper
from compile_world import World
from controller import Controller
import mujoco_py as mjc
import argparse

from compile_objects import set_simulator_option
import lxml.etree as ET
from collections import OrderedDict
import os, math, shutil

class GripperDesign(DesignBase):
    DEFAULT_PATH = 'gripper_data'

    def __init__(self, parameters):
        self.gripper = Gripper()
        self.fileName = GripperDesign.DEFAULT_PATH + '/gripper.xml'
        DesignBase.__init__(self, parameters)

    def getDesign(self):
        root = ET.Element('mujoco')
        set_simulator_option(root)
        asset = ET.SubElement(root, 'asset')
        body = ET.SubElement(root, 'worldbody')
        actuator = ET.SubElement(root, 'actuator')
        print(self.parameters)
        link = self.gripper.get_robot(**self.parameters)
        link.compile_gripper(body, asset, actuator)

        if not os.path.exists(GripperDesign.DEFAULT_PATH):
            os.mkdir(GripperDesign.DEFAULT_PATH)
        open(self.fileName, 'w').write(ET.tostring(root, pretty_print=True).decode())
        return link

    def getSim(self):
        model = mjc.load_model_from_path(self.fileName)
        return mjc.MjSim(model)

class GripperDesignSpace(DesignSpace):
    def __init__(self, designSpace):
        """designSpace = [('finger_length',(0.2,0.5)),('finger_curvature',(-2,2))] """
        designSpace = OrderedDict(designSpace)
        self.args0 = {}
        xmin = []
        xmax = []
        self.xname = []
        self.d = []
        for designParam, minmax in designSpace.items():
            if not isinstance(minmax, tuple) and not isinstance(minmax, float):
                raise RuntimeError('Incorrect format for design_space!')
            elif isinstance(minmax, tuple):
                xmin.append(float(minmax[0]))
                xmax.append(float(minmax[1]))
                self.xname.append(designParam)
            else:
                self.args0[designParam] = float(minmax)

        DesignSpace.__init__(self, xmax=xmax, xmin=xmin)

    def makeDesign(self, parameters):
        args = self.args0
        for a, b, n, d in zip(self.xmin, self.xmax, self.xname, parameters):
            assert d >= a and d <= b
            args[n] = d
        gripperDesign = GripperDesign(args)
        gripperDesign.getDesign()
        return gripperDesign

class GripperBehaviorSpace(BehaviorSpace):
    DEFAULT_PATH = 'gripper_world'

    def __init__(self, behaviorSpace):
        behaviorSpace = OrderedDict(behaviorSpace)
        self.behaviorName = ['theta', 'phi', 'beta', 'init_pose0', 'init_pose1', 'approach_coef0', 'approach_coef1',
                             'init_dist', 'grasp_dir']
        xminTotal = [0., math.pi / 4, 0., -math.pi / 2, -math.pi / 2, -1., -1., 2., -1.]
        xmaxTotal = [math.pi * 2, math.pi / 2 * 0.99, math.pi, math.pi / 2, math.pi / 2, 1., 1., 3.5, 1.]
        self.behaviorInit = [0., math.pi / 2 * 0.99, 0., math.pi / 2, 0., 1., 1., 3.5, None]
        # *0.99 to phi will avoid Gimbal lock of Euler angles

        self.xname = []
        xmin = []
        xmax = []

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

        BehaviorSpace.__init__(self, xmin=xmin, xmax=xmax)

    def evaluationData(self, design, environment, behavior, visualize=False):
        """
        return (elapsed timesteps, simulation timestep)
        """
        link = design.getDesign()
        control = []
        idx = 0
        print(behavior)

        for c, var in enumerate(self.behaviorName):
            if idx < len(self.xname) and self.xname[idx] is var:
                control.append(behavior[idx])
                idx += 1
            else:
                control.append(self.behaviorInit[c])
        print(control)
        assert len(control) == 9 and idx == len(self.xname)
        world = World()
        world.compile_simulator(path=GripperBehaviorSpace.DEFAULT_PATH, objects=[environment], link=link)
        ctrl = Controller(world)
        ctrl.reset(0, angle=control[0:3], init_pose=control[3:5], approach_coef=control[5:7],
                   init_dist=control[7], grasp_dir=control[8] if len(control) > 8 else None)
        viewer = mjc.MjViewer(world.sim) if visualize else None
        while not ctrl.step():
            if viewer is not None:
                viewer.render()
        #remove temporary file path
        os.remove(GripperBehaviorSpace.DEFAULT_PATH+'/world_PID='+str(os.getpid())+'.xml')

        return ctrl.elapsed, ctrl.sim.model.opt.timestep

class MassMetric(DesignMetric):
    def __init__(self):
        DesignMetric.__init__(self, None, 'MassMetric')

    def __call__(self, design):
        controller = design.getSim()
        if isinstance(controller, Controller):
            sim = controller.sim
        else:
            sim = controller
        model = sim.model
        bid = model.body_names.index('base')
        mass = model.body_subtreemass[bid]
        return 1. / mass


class ElapsedTimeMetric(BehaviorMetric):
    def __init__(self):
        BehaviorMetric.__init__(self, None, 'ElapsedMetric')

    def __call__(self, design, environment, behavior, data):
        # dt = data.sim.model.opt.timestep
        elapsed, dt = data
        return elapsed * dt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bilevel Gripper optimization')
    parser.add_argument('--numD', type=int, help='Number of design samples per iteration')
    parser.add_argument('--numB', type=int, help='Number of behavior samples for each design')
    parser.add_argument('--useDirect', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='To use DIRECT opt for policy search')
    parser.add_argument('--kappa', type=float, help='kappa needed for policy search')
    parser.add_argument('--logPath', type=str, default='../mobbo_gripper', help='Directory path for logging')
    parser.add_argument('--numIter', type=int, default=100, help='The number of designs that will be generated.')
    parser.add_argument('--numMCSamples', type=int, default=1000,
                        help='Num of samples generated from gp posterior to compute acquisition funcion')
    parser.add_argument('--parallel', default=True, type=lambda x: (str(x).lower() == 'true'), help='To use multiprocessing')
    args = parser.parse_args()
    behOptMethod = 'direct' if args.useDirect else 'random'
    settings = MOBBOSettings(numIter=args.numIter, numD= args.numD, numB= args.numB, kappa=args.kappa,
                             behOptimizationMethod=behOptMethod, parallel=args.parallel)

    from dataset_canonical import get_dataset_canonical
    envs = get_dataset_canonical()  # object list
    gripperDesignSpace = GripperDesignSpace([('hinge_rad', 0.04), ('finger_width', 0.3), ('finger_length', (0.1, 0.4)),
                                             ('num_finger', (2.1, 3.9)), ('base_off', 0.2), ('finger_curvature', (-4., 4.))])
    gripperBehaviorSpace = GripperBehaviorSpace([('beta', None), ('phi', None), ('theta', None)])
    gripperProblem = CodesignProblem(designSpace=gripperDesignSpace, behaviorSpace=gripperBehaviorSpace,
                                     environments=envs,
                                     designMetrics=[MassMetric()], behaviorMetrics=[ElapsedTimeMetric()],
                                     name='gripperDesign', mean=True)

    gripperMOBBO = MOBBO(problem=gripperProblem, settings=settings)
    gripperMOBBO.run(logPath=args.logPath, logInterval=args.numIter // 10, keepLatest=5)

    import matplotlib.pyplot as plt
    import pdb
    import numpy as np

    plt.figure()
    scores = np.array([s[1] for s in gripperMOBBO.samples ])
    for i in range(len(scores)):
        plt.text(scores[i,0], scores[i,1], str(i), size= 9)
    plt.scatter(scores[:,0], scores[:,1], s=6, c='b')
    plt.scatter(scores[:80, 0], scores[:80, 1], s=6, c='cyan')
    plt.scatter(scores[gripperMOBBO.currentPF_arg,0], scores[gripperMOBBO.currentPF_arg,1], c='r', s= 9)
    plt.show()
    pdb.set_trace()