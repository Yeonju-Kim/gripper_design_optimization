from multi_objective_ACBO_GPUCB import *
import pdb
import argparse
from TRINA_BO import RobotArmProblemBO
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ACBO trina optimization')
    parser.add_argument('--num_design_samples', type=int, default=20, help='Number of design samples per iteration')
    parser.add_argument('--use_direct', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='To use DIRECT opt for design search')
    parser.add_argument('--kappa', type=float, default = 2.0 , help='kappa needed for policy search')
    parser.add_argument('--logpath', type=str, default='../acbo_trina', help='Directory path for logging')
    parser.add_argument('--num_iter', type=int, default=100, help='The number of designs that will be generated.')
    parser.add_argument('--num_grid', type=int, default=2, help='Gripper_problem = 2 6dim 64 initial points')
    parser.add_argument('--num_mc_samples', type=int, default=1000,
                        help='Num of samples generated from gp posterior to compute acquisition funcion')
    args = parser.parse_args()
    nu = 2.5
    policy_space = [('c0', None), ('c1', None), ('c2', None), ('c3', None), ('c4',None), ('c5', None)]
    # policy_space = [('x', None), ('y', None), ('z', None)]
    problembo = RobotArmProblemBO(policy_space)
    pdb.set_trace()
    BO = MultiObjectiveACBOGPUCB(problemBO=problembo,
                                 d_sample_size=args.num_design_samples,
                                 num_mc_samples=args.num_mc_samples,
                                 partition=[[0],[1],[2],[3],[4]],
                                 kappa=args.kappa, nu=nu, use_direct_for_design=args.use_direct)

    start_time = time.time()
    BO.run(num_grid=args.num_grid, num_iter=args.num_iter,
           log_path=args.logpath, log_interval=args.num_iter // 10)
    print('time ---- ', time.time() - start_time)
    BO.update_PF()
    BO.graph_gripper_plot()

    pdb.set_trace()

    from pareto import pareto_max
    from heuristics import farthest_first

    pf, npf, ar = pareto_max(BO.scores)
    arg_subset = farthest_first(pf, 6)
    arg = np.where(ar)[0][arg_subset]
    print(BO.scores[arg])
    pdb.set_trace()
    for i in arg:
        print(i)
        BO.plot_solution(i)

