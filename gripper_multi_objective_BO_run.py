from single_objective_BO_GPUCB import SingleObjectiveBOGPUCB
from gripper_problem_BO import GripperProblemBO
from problem_BO import HyperVolumeTransformedProblemBO
from compile_objects import auto_download
import os

plot=False
test=True
if __name__=='__main__':
    auto_download()
    
    domain=GripperProblemBO(design_space='finger_length:0.2,0.5|finger_curvature:-2,2',metrics='SizeMetric|Q1Metric',
                            object_file_name='data/ObjectNet3D/CAD/off/cup/05.off',
                            policy_space=[5,5,5,3.])
    domain=HyperVolumeTransformedProblemBO(domain,scale=100.)
    BO=SingleObjectiveBOGPUCB(domain,nu=10.)
    
    #main loop
    path='../'+BO.name()+'.dat'
    if not os.path.exists(path):
        BO.run(num_iter=10)
        BO.save(path)
    else:
        BO.load(path)
    
    #plot convergence history
    if plot:
        import matplotlib.pyplot as plt   
        BO.plot_iteration(plt,accumulate=True)
        BO.plot_iteration(plt,accumulate=False)
        plt.show()
        plt.close()
        
        #plot fitted GP at each iteration
        _,ani=BO.plot_func(plt)
        plt.show()
        plt.close()
    
    #plot best gripper design
    if test:
        sol=BO.get_best()
        domain.inner.plot_solution(sol,metric_id=0)