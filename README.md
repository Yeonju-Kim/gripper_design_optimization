
# Multi-Objective Bilevel Bayesian Optimization(MO-BBO)
MO-BBO is implemented in [mobbo.py](https://github.com/Yeonju-Kim/gripper_design_optimization/blob/mobbo/mobbo.py). The default implementation to define design problem are located in [codesign.py](https://github.com/Yeonju-Kim/gripper_design_optimization/blob/mobbo/codesign.py).


Step 1: Specify CodesignProblem by defining 

* Design Space
* Behavior Space
* Environments
* Design Metrics(dependent on design)
* Behavior Metrics(dependent on design and behavior)

Step 2: Set up the parameters used in MO-BBO. 
* num of design samples: smaller -> more exploration
* num of behavior samples
* kappa: smaller -> more exploitation than exploration 
* Behavior Optimization method: ['valid', 'random', 'direct']
* Initialization method: ['valid', 'random', 'optimize']
* num of design to be generated

# Example 1. gripper_design_optimization

## Mujoco Installation

Download ObjectNet3D.zip file from: https://cvgl.stanford.edu/projects/objectnet3d/
In Zherong's version, this download is automatic and defined in compile_objects.py::auto_download()

Simulator and Visualizer are both from Mujoco, to install MuJoCo, use the following steps:

Step 1: Acquire a licence from https://www.roboti.us/license.html

Step 2: Download MuJoCo 2.00 from https://www.roboti.us/index.html

Step 3: Extract mujoco200_linux, rename it to /home/${User}/.mujoco

Step 4: Put mjkey.txt in /home/${User}/.mujoco

Step 5: Add the following lines to /home/${User}/.bashrc

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zherong/.mujoco/mujoco200/bin:/usr/lib/nvidia-418
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
    
Step 6: Install MuJoCo's python interface by typing: sudo easy_install mujoco-py

Step 7: Run python3 compile_world.py

## C++ Grasp Metric Computer Installation

We use an advanced algorithm to compute Q_1 Q_{Inf} Q_{MSV} Q_{VEW} Q_{G11} metrics. 
These algorithms are implemented in C++, to install it, use the following commands:

    cd GraspMetric
    sudo python3 setup.py install
 
## Optimization 
    
    python3 gripper_codesign.py --numIter 100 --numD 10 --numB 20 --parallel True --kappa 2.0  

# Example 2. Robot Arm Placement Optimization
## Optimization 
    
    python3 robotarm_codesign.py --numIter 100 --numD 10 --numB 20 --parallel True --kappa 2.0  
