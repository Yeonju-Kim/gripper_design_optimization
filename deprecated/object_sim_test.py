from Simulation.gl_vis import make_object_from_file
import glob
from klampt import *
from klampt.vis import GLRealtimeProgram
import numpy as np
from Simulation.moving_base import *
from klampt import vis


class Obj_simulation(GLRealtimeProgram):
    def __init__(self, world, sim, time_limit):
        GLRealtimeProgram.__init__(self, "GLTest")
        self.world = world
        self.sim = sim
        self.time_limit = time_limit
        self.sim.enableContactFeedbackAll()
        self.prev_obj_t = None
        self.stop_moving = False

    def is_object_stable(self):
        if self.prev_obj_t is not None:
            cur_obj_t = self.world.rigidObject(0).getTransform()
            dist = math.se3.distance(self.prev_obj_t, cur_obj_t)
            self.prev_obj_t = cur_obj_t
            threshold = 5e-6
            print(dist)
            return dist < threshold
        else:
            cur_obj_t = self.world.rigidObject(0).getTransform()
            self.prev_obj_t = cur_obj_t
            return False

    def turn_off_vis(self):
        vis.show(False)

    def display(self):
        self.sim.updateWorld()
        self.world.drawGL()

    def idle(self):
        if not self.stop_moving and 0.1 <= self.sim.getTime() < self.time_limit:
            if self.sim.getTime() < self.time_limit and self.is_object_stable():
                self.stop_moving = True
                self.turn_off_vis()

        if self.sim.getTime() > self.time_limit:
            self.turn_off_vis()
        self.sim.simulate(self.dt)
        return

    def get_result(self):
        return self.stop_moving


def get_object_files():
    list = glob.glob('../ObjectNet3D/CAD/off/chair/[0-9][0-9].off')
    print(list)
    return list


if __name__ == "__main__":
    object_files = get_object_files()# ["telephone/01.off", "telephone/02.off", "telephone/03.off"]
    world = WorldModel()
    res = world.readFile("Simulation/box_robot_floating.xml")
    if not res:
        raise RuntimeError("Unable to load world")
    robot = world.robot(0)
    for i in range(6):
        m = robot.link(i).getMass()
        m.setInertia([0.0001] * 3)
        robot.link(i).setMass(m)

    total_list = []
    init_config = robot.getConfig()
    for object_file_name in object_files:
        obj, object_r, object_T = make_object_from_file(world, object_file_name)
        set_moving_base_xform(robot, so3.identity(), [1, 1, 1])
        sim = Simulator(world)
        program = Obj_simulation(world, sim, 2.)
        program.run()
        if program.get_result():
            total_list += [object_file_name]
        world.remove(world.rigidObject(0))

    vis.show(False)
    vis.kill()
    print(len(object_files))
    print(len(total_list), total_list)
    list_2sec = np.asarray(total_list)

    np.save('chairs_2sec', list_2sec)