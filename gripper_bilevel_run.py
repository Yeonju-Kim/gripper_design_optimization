from multi_objective_bilevel import *
import pdb
import argparse
from gripper_problem_BO import *


def create_gripper_problem_BO(high_dimensional_design_space=False, high_dimensional_policy_space=False):
    # load design space
    if high_dimensional_design_space:
        # 6D
        design_space = [('hinge_rad', (0.01, 0.08)), ('finger_width', (0.15, 0.4)), ('finger_length', (0.1, 0.4)),
                        ('num_finger', (2.1, 3.9)), ('base_off', (0.1, 0.4)), ('finger_curvature', (-4., 4.))]
    else:
        # 4D
        # design_space=[('hinge_rad',(0.01,0.08)),('finger_width',(0.15,0.4)),('finger_length',0.25),
        #               ('num_finger',(2.1,3.9)),('base_off',0.2),('finger_curvature',(-4.,4.))]
        # 3D
        design_space = [('hinge_rad', 0.04), ('finger_width', 0.3), ('finger_length', (0.1, 0.4)),
                        ('num_finger', (2.1, 3.9)), ('base_off', 0.2), ('finger_curvature', (-4., 4.))]
        # 0d
        # design_space = [('hinge_rad', 0.04), ('finger_width', 0.3), ('finger_length', 0.4),
        #                 ('num_finger', 2.1), ('base_off', 0.2), ('finger_curvature', 3.)]

    # load metrics
    metrics = [MassMetric(), ElapsedMetric()]

    # load canonical dataset
    from dataset_canonical import get_dataset_canonical
    objects = get_dataset_canonical()

    # load policy space
    if high_dimensional_policy_space:
        # 5D
        policy_space = [('beta', None), ('init_pose0', None), ('init_pose1', None), ('approach_coef0', None),
                        ('approach_coef1', None)]
    else:
        # 2D('beta',None),('grasp_dir',None)
        policy_space = [('beta', None), ('phi', None), ('theta', None)]

    # put everything together
    domain = GripperProblemBO(design_space=design_space,
                              metrics=metrics,
                              objects=objects,
                              policy_space=policy_space)
    print(domain)
    return domain


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bilevel_gripper optimization')
    parser.add_argument('--num_design_samples', type=int, help='Number of design samples per iteration')
    parser.add_argument('--maxf', type=int, help='Number of design samples per iteration')
    parser.add_argument('--use_direct', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='To use DIRECT opt for policy search')
    parser.add_argument('--kappa', type=float, help='kappa needed for policy search')
    parser.add_argument('--logpath', type=str, default='../bilevel_gripper', help='Directory path for logging')
    parser.add_argument('--num_iter', type=int, default=100, help='The number of designs that will be generated.')
    parser.add_argument('--num_grid', type=int, default=2, help='Gripper_problem = 2 6dim 64 initial points')
    parser.add_argument('--num_mc_samples', type=int, default=1000,
                        help='Num of samples generated from gp posterior to compute acquisition funcion')
    args = parser.parse_args()

    domain = create_gripper_problem_BO()
    BO = MultiObjectiveBOBilevel(problemBO=domain, d_sample_size=args.num_design_samples,
                                 num_mc_samples=args.num_mc_samples,
                                 partition=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
                                 max_f_eval=args.maxf, parallel=True,
                                 kappa=args.kappa, nu=2.5, use_direct=args.use_direct)
    start_time = time.time()
    BO.run(num_grid=args.num_grid, num_iter=args.num_iter, log_path=args.logpath, log_interval=args.num_iter // 10)
    print('time ---- ', time.time() - start_time)
    BO.graph_gripper_plot()
    pdb.set_trace()

    # if test:
    #     li =[64, 188, 151, 86, 165, 104, 67, 83, 75, 88, 3, 15, 78, 91]
    #     for i in li:
    #         print(i)
    #         sol = BO.points[i]
    #         domain.plot_solution(sol,i,view= True)
    #         ori = []
    #         for k in range(len(BO.problemBO.objects)):
    #             ori.append(BO.problemBO.train_label[k][i])
    #         print(sum(ori)/len(BO.problemBO.objects))
    #         print(BO.scores[i])