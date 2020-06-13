from compile_world import World

class Controller:
    def __init__(self,world):
        self.world=world
    
    def set_approach_dir(self,initial_pos,initial_orient):
        pass
    
    def simulate(self,approach_vel,close_vel):
        pass
    
    def lift(self,height):
        pass
    
    def shake(self,frequency,dir=[3,0,0]):
        pass
    
if __name__=='__main__':
    auto_download()
    
    #create gripper
    gripper=Gripper()
    link=gripper.get_robot(base_off=0.3,finger_width=0.4,finger_curvature=2)

    #create world    
    world=World()
    world.compile_simulator(object_file_name='data/ObjectNet3D/CAD/off/cup/[0-9][0-9].off',link=link)
    
    #create controller
    controller=Controller(world)