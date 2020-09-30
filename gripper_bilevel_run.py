from multi_objective_bilevel import *
import pdb

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


plot = True
test = True
if __name__ == '__main__':
    Total = 100000
    num_design_samples = 5
    num_iter = 100
    num_grid = 2 #default gripper num_grid
    domain = create_gripper_problem_BO()
    BO = MultiObjectiveBOBilevel(problemBO=domain, d_sample_size=num_design_samples,
                                 num_mc_samples= 1000, partition=[[0,1,2,3,4,5,6,7,8,9,10,11,12]],
                                 max_f_eval=Total//num_design_samples, parallel=True,
                                 kappa=10.0, nu=2.5, use_direct=True)
    starttime = time.time()
    # main loop
    log_path = '../gripper_bilevel'
    BO.run(num_grid=num_grid, num_iter=num_iter, log_path=log_path, log_interval=num_iter//10)
    print('time ---- ', time.time() - starttime)
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