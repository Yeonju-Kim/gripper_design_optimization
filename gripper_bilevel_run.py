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
    from dataset_cup import get_dataset_cup

    Total = 2000
    num_design_samples = 3
    num_iter = Total // num_design_samples

    domain = create_gripper_problem_BO()
    BO = MultiObjectiveBOBilevel(problemBO=domain, d_sample_size=num_design_samples,
                                 num_mc_samples=  1000, partition=[[0,1,2,3,4,5,6,7,8,9,10,11,12]],
                                 max_f_eval=20000, kappa=10.0, nu=10.0, use_direct=False)
    starttime = time.time()
    # main loop
    save_path = BO.name() + '.dat'
    load_path = 'init.dat'
    if not os.path.exists(load_path):
        BO.run(num_grid=1, num_iter=10)
        np.save('scores', np.array(BO.scores))
        BO.save(save_path)
    else:
        BO.run_with_file(file_name=load_path, num_iter=1)
        np.save('scores', np.array(BO.scores))
        BO.save(save_path)
    print('time ---- ', time.time() - starttime)

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