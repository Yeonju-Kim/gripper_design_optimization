# gripper_design_optimization


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