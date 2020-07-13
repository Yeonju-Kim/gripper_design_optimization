from controller import *
     
def test_dataset_canonical(design,policy,controller,ids):
    #create gripper
    gripper=Gripper()
    link=gripper.get_robot(**design)

    #create world    
    world=World()
    from dataset_canonical import get_dataset_canonical
    objects=get_dataset_canonical()
    if ids is not None:
        objects=[objects[id] for id in ids]
    world.compile_simulator(objects=objects,link=link)
    viewer=mjc.MjViewer(world.sim)
    
    #create controller
    controller['world']=world
    controller=Controller(**controller)
    
    id=0
    while True:
        policy['id']=id
        controller.reset(**policy)
        while not controller.step():
            viewer.render()
        id=(id+1)%len(controller.world.names)
    
if __name__=='__main__':
    #the need for positive curvature
    #design={'base_off':0.2,'finger_length':0.25,'finger_width':0.3,'finger_curvature':4.,'num_finger':3,'hinge_rad':0.01}
    #policy={'initial_pos':[0.1,0.,3.],'axial_rotation':math.pi/2}
    #controller={'lift_height':2.5}
    #test_dataset_canonical(design,policy,controller,ids=[1])
    
    #the need for negative curvature
    #design={'base_off':0.2,'finger_length':0.25,'finger_width':0.3,'finger_curvature':-4.,'num_finger':3,'hinge_rad':0.04}
    #policy={'initial_pos':[0.1,0.,3.],'axial_rotation':math.pi/2}
    #controller={'lift_height':2.5}
    #test_dataset_canonical(design,policy,controller,ids=[0])
    
    #negative curvature works for id=1 as well
    #design={'base_off':0.2,'finger_length':0.25,'finger_width':0.3,'finger_curvature':-4.,'num_finger':3,'hinge_rad':0.04}
    #policy={'initial_pos':[0.1,0.,3.],'axial_rotation':math.pi/2}
    #controller={'lift_height':2.5}
    #test_dataset_canonical(design,policy,controller,ids=[1])
    
    #hinge radius must be larger for id=2
    #design={'base_off':0.2,'finger_length':0.25,'finger_width':0.3,'finger_curvature':-4.,'num_finger':3,'hinge_rad':0.06}
    #policy={'initial_pos':[0.1,0.,3.],'axial_rotation':math.pi/2}
    #controller={'lift_height':2.5}
    #test_dataset_canonical(design,policy,controller,ids=[2])
    
    #surprisingly id=3 works with the same 
    #design={'base_off':0.2,'finger_length':0.25,'finger_width':0.3,'finger_curvature':4.,'num_finger':3,'hinge_rad':0.06}
    #policy={'initial_pos':[0.1,0.,3.],'axial_rotation':math.pi/2}
    #controller={'lift_height':2.5}
    #test_dataset_canonical(design,policy,controller,ids=[3])
    
    #the need of four fingers
    #design={'base_off':0.2,'finger_length':0.25,'finger_width':0.3,'finger_curvature':4.,'num_finger':4,'hinge_rad':0.06}
    #policy={'initial_pos':[0.1,0.,3.],'axial_rotation':math.pi/2}
    #controller={'lift_height':2.5}
    #test_dataset_canonical(design,policy,controller,ids=[4])
    
    #the need of zero curvature
    #design={'base_off':0.2,'finger_length':0.25,'finger_width':0.3,'finger_curvature':4.,'num_finger':3,'hinge_rad':0.06}
    #policy={'initial_pos':[0.1,0.,3.],'axial_rotation':math.pi/2}
    #controller={'lift_height':2.5}
    #test_dataset_canonical(design,policy,controller,ids=[4])
    
    #the four finger design is still nice
    #design={'base_off':0.2,'finger_length':0.25,'finger_width':0.3,'finger_curvature':4.,'num_finger':4,'hinge_rad':0.06}
    #policy={'initial_pos':[0.1,0.,3.],'axial_rotation':math.pi/2}
    #controller={'lift_height':2.5}
    #test_dataset_canonical(design,policy,controller,ids=[5])
    
    #the three finger is also fine
    #design={'base_off':0.2,'finger_length':0.25,'finger_width':0.3,'finger_curvature':4.,'num_finger':3,'hinge_rad':0.06}
    #policy={'initial_pos':[0.1,0.,3.],'axial_rotation':math.pi/2}
    #controller={'lift_height':2.5}
    #test_dataset_canonical(design,policy,controller,ids=[5])
    
    design={'base_off':0.2,'finger_length':0.25,'finger_width':0.3,'finger_curvature':4.,'num_finger':3,'hinge_rad':0.06}
    policy={'initial_pos':[0.1,0.,3.],'axial_rotation':math.pi/2}
    controller={'lift_height':2.5}
    test_dataset_canonical(design,policy,controller,ids=[6])