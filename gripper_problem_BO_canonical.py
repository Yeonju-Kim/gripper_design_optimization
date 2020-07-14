from gripper_problem_BO import *

def create_gripper_problem_BO(high_dimensional_design_space=False,high_dimensional_policy_space=False):
    #load design space
    if high_dimensional_design_space:
        #6D
        design_space=   \
        'hinge_rad:0.01,0.08|finger_width:0.15,0.4|finger_length:0.1,0.4|num_finger:2.1,3.9|base_off:0.1,0.4|finger_curvature:-4.,4.'
    else:
        #4D
        design_space=   \
        'hinge_rad:0.01,0.08|finger_width:0.15,0.4|num_finger:2.1,3.9|finger_curvature:-4.,4.'
    
    #load metrics
    metrics='MassMetric|ElapsedMetric'
    
    #load canonical dataset
    from dataset_canonical import get_dataset_canonical
    objects=get_dataset_canonical()
    
    #load policy space
    if high_dimensional_policy_space:
        #5D
        policy_space={'beta':None,'init_pose0':None,'init_pose1':None,'approach_coef0':None,'approach_coef1':None}
    else: 
        #2D
        policy_space={'beta':None,'grasp_dir':None}
    
    #put everything together
    domain=GripperProblemBO(design_space=design_space,
                            metrics=metrics,
                            objects=objects,
                            policy_space=policy_space)
    print(domain)
    return domain

if __name__=='__main__':
    create_gripper_problem_BO(True,True)
    create_gripper_problem_BO(False,False)