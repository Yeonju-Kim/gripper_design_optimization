from controller import *
     
def test_dataset_cup(design,policy,controller,ids=None,repeat=5):
    #create gripper
    gripper=Gripper()
    link=gripper.get_robot(**design)

    #create world    
    world=World()
    from dataset_cup import get_dataset_cup
    objects=get_dataset_cup(True)
    if ids is not None:
        objects=[objects[id] for id in ids]
    world.compile_simulator(objects=objects,link=link)
    viewer=mjc.MjViewer(world.sim)
    
    #create controller
    controller['world']=world
    controller=Controller(**controller)
    
    id=0
    numPass=0
    while True:
        policy['id']=id
        controller.reset(**policy)
        while not controller.step():
            viewer.render()
        id=(id+1)%len(controller.world.names)
        if id==0:
            numPass+=1
            if numPass==repeat:
                return

if __name__=='__main__':
    design={'base_off':0.2,'finger_length':0.15,'finger_width':0.3,'finger_curvature':4.,'num_finger':3,'hinge_rad':0.025}
    policy={'initial_pos':[0.1,0.,2.],'axial_rotation':math.pi/2}
    controller={}
    test_dataset_cup(design,policy,controller,ids=None)