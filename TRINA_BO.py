from problem_BO import ProblemBO
from metric import *
import numpy as np
import math,copy,os,shutil,time,multiprocessing
import pickle
from klampt import *
from klampt import vis

from gl_vis import GLViewer
from multi_objective_bilevel import *
import pdb
import argparse

class ReachabilityMetric(Metric):
    def __init__(self):
        Metric.__init__(self, OBJECT_DEPENDENT=True)
    def compute(self, robot_arm_env):
        return
class BoundingBoxMetric(Metric):
    def __init__(self):
        Metric.__init__(self, OBJECT_DEPENDENT=True)
    def compute(self, robot_arm_env):
        return


class RobotArmProblemBO(ProblemBO):
    def __init__(self, policy_space):
        self.pose, self.pos_max, self.pos_min, self.start_point, self.table, self.shelf = pickle.load(open('pos_and_dimension', 'rb'))
        self.objects = [None, 'low_shelf', 'table', 'table', 'high_shelf']
        self.policy_names = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']
        self.policy_vmin = [-2*math.pi] * 6
        self.policy_vmax = [2*math.pi] * 6
        self.policy_init = [0.0322, -2.2055, -2.06, -1.1135, 4.3005, 0.7]

        self.is_vert = [True, False, True, False, False]

        self.metrics= [ReachabilityMetric()]# BoundingBoxMetric()]
        self.design_space = [('distance', (0.05, 0.3)), ('pan', (0, 90)), ('tilt', (-45, 135))]
        ProblemBO.__init__(self, self.design_space, policy_space)

    def initialize(self, num_init_data):
        initial_pts= []

        while len(initial_pts) < num_init_data:
            pt = np.random.uniform(self.vmin, self.vmax, len(self.vmin)).tolist()
            design = pt[:len(self.design_space)]
            succeed_finding = True
            decision = []
            for obj_id in range(len(self.objects)):
                print(design)
                config = self.get_config_candidates(design, obj_id, 1, max_eval=1000)
                if len(config) is 0:
                    succeed_finding = False
                    break
                decision.append(config[0])

            if succeed_finding:
                # pdb.set_trace()
                assert len(decision) == 5
                initial_pts.append(design+ np.array(decision).T.tolist())
                # pdb.set_trace()
        return initial_pts

    # def validity_check(self, points, object_id=None):
    #     if self.world is None:
    #         print('no world')
    #         self.world = WorldModel()
    #         res = self.world.readFile('TRINA_world.xml')
    #         if not res:
    #             raise RuntimeError('Unable to load model')
    #         self.robot_arm_env = GLViewer(self.world, visualization=True, shelf=self.shelf, table=self.table)
    #
    #     if object_id is not None:
    #         # pdb.set_trace()
    #         min_xyz = np.array(self.pose[object_id]) + np.array(self.pos_min[object_id])
    #         max_xyz = np.array(self.pose[object_id]) + np.array(self.pos_max[object_id])
    #         min_xyz = min_xyz.tolist()
    #         max_xyz = max_xyz.tolist()
    #     else:
    #         min_xyz = [0.4, -0.2, 0.]
    #         max_xyz = [1.1, 0.8, 1.8]
    #     is_valid = [None for i in range(len(points))]
    #     #transform robot once
    #     design = points[0][:len(self.design_space)]
    #     self.robot_arm_env.get_robot(*design)
    #     for pt_id in range(len(points)):
    #         decision = points[pt_id][len(self.design_space):]
    #         self.robot_arm_env.set_partial_config(decision)
    #         _, t = self.robot_arm_env.gripper_transform(is_left=True)
    #         is_in_bb = np.all([min_xyz[i] < t[i] < max_xyz[i] for i in range(3)])
    #         if self.robot_arm_env.current_collide_check(is_left=True) and is_in_bb:
    #             is_valid[pt_id] = True
    #         else:
    #             is_valid[pt_id] = False
    #     self.robot_arm_env.get_robot_(*[-design[j] for j in range(3)])
    #
    #     return is_valid

    def get_config_candidates(self, design, object_id, max_samples, max_eval = 10000):
        assert len(design)==3
        world = WorldModel()
        res = world.readFile('TRINA_world.xml')
        if not res:
            raise RuntimeError('Unable to load model')

        robot_arm_env = GLViewer(world, visualization=True, shelf=self.shelf, table=self.table)

        num_local_ik = 0
        min_xyz = np.array(self.pose[object_id]) + np.array(self.pos_min[object_id])
        max_xyz = np.array(self.pose[object_id]) + np.array(self.pos_max[object_id])
        min_xyz = min_xyz.tolist()
        max_xyz = max_xyz.tolist()
        configs = []
        robot_arm_env.get_robot(*design)

        robot_arm_env.create_rigidObject(self.objects[object_id])
        while len(configs) < max_samples and num_local_ik < max_eval:
            if len(configs) is 0 or np.random.uniform() > 0.7:
                init_config = np.random.uniform(self.vmin[len(self.design_space):],
                                                self.vmax[len(self.design_space):]).tolist()
            else:
                init_config = configs[np.random.choice(len(configs), 1)[0]]
            position = np.random.uniform(min_xyz, max_xyz).tolist()
            print('design,', design)
            config = robot_arm_env.local_ik_solve(position, self.is_vert[object_id], init_config)
            num_local_ik += 1
            if config is not None:
                configs.append(config)

        print(len(configs), 'configs')
        print(max_eval, 'max eval')
        # vis.run(self.robot_arm_env)
        # robot_arm_env.remove_rigidObject()
        # robot_arm_env.vis_reset()
        # robot_arm_env.get_robot_(*[-design[j] for j in range(3)])
        return configs


    def compute_metrics(self, points, remove_tmp= True, parallel= True, visualize = False):
        indep_metrics = []
        all_metrics = []
        for pt_id in range(len(points)):
            world = WorldModel()
            res = world.readFile('TRINA_world.xml')
            if not res:
                raise RuntimeError('Unable to load model')

            # There's no Object-dependent metric in this problem
            robot_arm_env = GLViewer(world, visualization=False, shelf=self.shelf, table=self.table)
            robot_arm_env.get_robot(*points[pt_id][:len(self.design_space)])
            print(points[pt_id][:len(self.design_space)])
            policy_metrics= []
            for policy in self.policies:
                # init policy
                for id, v in zip(self.vpolicyid, points[pt_id]):
                    if id >= 0:
                        policy[id] = v
                for k, v in self.mimic:
                    policy[k[0]] = [p * v for p in points[pt_id][k[1]]] if isinstance(points[pt_id][k[1]], list) else points[pt_id][k[1]] * v

                object_metrics = []
                for i in range(len(self.objects)):
                    policyo = [p[i] if isinstance(p, list) else p for p in policy]
                    robot_arm_env.create_rigidObject(self.objects[i])
                    sc, vol= robot_arm_env.given_starting_config_score(init_config = policyo,
                                                                       pose=self.pose[i], vmax=self.pos_max[i],
                                                                       vmin=self.pos_min[i],
                                                                       is_vertical= self.is_vert[i], is_left= True)
                    if sc> 0 :
                        print(sc, vol)
                        # vis.run(robot_arm_env)
                        # pdb.set_trace()
                    robot_arm_env.remove_rigidObject()
                    robot_arm_env.vis_reset()
                    object_metrics.append([sc])
                policy_metrics.append(object_metrics)
            all_metrics.append(policy_metrics)
        return indep_metrics, all_metrics


    def plot_solution(self, point):
        world = WorldModel()
        res = world.readFile('TRINA_world.xml')
        if not res:
            raise RuntimeError('Unable to load model')

        #There's no Object-dependent metric in this problem
        all_metrics = []
        robot_arm_env = GLViewer(world, visualization=True, shelf=self.shelf, table = self.table)

        robot_arm_env.get_robot(*point[:len(self.design_space)])
        print(point[:len(self.design_space)])
        # vis.run(robot_arm_env)
        policy_metrics = []
        for policy in self.policies:
            # init policy
            for id, v in zip(self.vpolicyid, point):
                if id >= 0:
                    policy[id] = v
            for k, v in self.mimic:
                policy[k[0]] = [p * v for p in point[k[1]]] if isinstance(point[k[1]], list) else \
                point[k[1]] * v

            object_metrics = []
            for i in range(len(self.objects)):
                policyo = [p[i] if isinstance(p, list) else p for p in policy]
                robot_arm_env.create_rigidObject(self.objects[i])
                sc, vol = robot_arm_env.given_starting_config_score(init_config=policyo,
                                                                    pose=self.pose[i], vmax=self.pos_max[i],
                                                                    vmin=self.pos_min[i],
                                                                    is_vertical=self.is_vert[i], is_left=True)
                vis.run(robot_arm_env)
                robot_arm_env.remove_rigidObject()
                robot_arm_env.vis_reset()
                object_metrics.append([sc])
            policy_metrics.append(object_metrics)
        all_metrics.append(policy_metrics)
        print('metrics', all_metrics)
        return all_metrics

    def name(self):
        return 'TrinaRobotArm'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bilevel_gripper optimization')
    parser.add_argument('--num_design_samples', type=int, help='Number of design samples per iteration')
    parser.add_argument('--maxf', type=int, help='Number of design samples per iteration')
    parser.add_argument('--use_direct', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='To use DIRECT opt for policy search')
    parser.add_argument('--kappa', type=float, help='kappa needed for policy search')
    parser.add_argument('--logpath', type=str, default='../bilevel_trina', help='Directory path for logging')
    parser.add_argument('--num_iter', type=int, default=100, help='The number of designs that will be generated.')
    parser.add_argument('--num_grid', type=int, default=2, help='Gripper_problem = 2 6dim 64 initial points')
    parser.add_argument('--num_mc_samples', type=int, default=1000,
                        help='Num of samples generated from gp posterior to compute acquisition funcion')
    args = parser.parse_args()
    nu = 2.5
    policy_space = [('c0', None), ('c1', None), ('c2', None), ('c3', None), ('c4',None), ('c5', None)]
    # policy_space = [('x', None), ('y', None), ('z', None)]
    problembo = RobotArmProblemBO(policy_space)

    BO = MultiObjectiveBOBilevel(problemBO=problembo, d_sample_size=args.num_design_samples,
                                 num_mc_samples=args.num_mc_samples,
                                 partition=[[0],[1],[2],[3],[ 4]],
                                 max_f_eval=args.maxf, parallel=False,
                                 kappa=args.kappa, nu=nu, use_direct=args.use_direct)
    start_time = time.time()

    BO.run(num_grid=args.num_grid, num_iter=args.num_iter, log_path=args.logpath, log_interval=args.num_iter // 10)
    print('time ---- ', time.time() - start_time)
    # BO.graph_gripper_plot()
    pdb.set_trace()
    # import matplotlib.pyplot as plt
    # plt.figure()
    # m_r= np.mean(BO.scores[:,:5], axis= 1)
    # m_v = np.mean(BO.scores[:,5: ], axis=1)
    # for i in range(len(BO.scores)):
    #     plt.text(m_r[i], m_v[i], str(i), size=8)
    #
    # plt.scatter(m_r, m_v)
    # plt.xlabel('reachability')
    # plt.ylabel('0.5-volume')
    # plt.show()

    #TODO:
    from pareto import pareto_max
    from heuristics import farthest_first

    pf, npf, ar = pareto_max(BO.scores)
    arg_subset = farthest_first(pf, 5)
    arg = np.where(ar)[0][arg_subset]
    print(BO.scores[arg])
    pdb.set_trace()

    for idx in arg:
        print(arg >=100, 'arg > 100')
        # if idx>=100:
        BO.plot_solution(idx)


    pdb.set_trace()