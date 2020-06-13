from .create_design import create_new_design
from .grasp_sim import *

from multiprocessing import Queue, Process
import queue

def make_object_from_file(world, file_name):
    object = Geometry3D()
    print(file_name)
    object.loadFile(file_name)

    obj = world.makeRigidObject("object")
    obj.geometry().set(object)
    obj.geometry().scale(0.35)
    contact_params = obj.getContactParameters()
    contact_params.kRestitution = 0.0
    contact_params.kFriction = 0.500000
    contact_params.kStiffness = 200000.0
    contact_params.kDamping = 1000.0
    obj.setContactParameters(contact_params)

    obj.setTransform(*math.se3.identity())
    bmin, bmax = obj.geometry().getBB()
    T = obj.getTransform()
    spacing = 0.006
    T = (T[0], math.vectorops.add(T[1], (-(bmin[0] + bmax[0]) * 0.5, -(bmin[1] + bmax[1]) * 0.5, -bmin[2] + spacing)))
    obj.setTransform(*T)
    obj.appearance().setColor(0.9, 0, 0.4, 1.0)

    r = np.max(np.abs([bmin[0], bmax[0], bmin[1], bmax[1], bmin[2], bmax[2]]))
    return object, np.sqrt(3) * r, T


def do_job(tasks_to_accomplish, result_queue):
    while True:
        try:
            world_file, object_file, length, width, link_angle, radius, link_tilted_angle, curvature, max_iter\
                = tasks_to_accomplish.get_nowait()
        except queue.Empty:
            break
        else:
            world = WorldModel()
            res = world.readFile(world_file)
            if not res:
                raise RuntimeError("Unable to load world")

            robot = world.robot(0)
            for i in range(6):
                m = robot.link(i).getMass()
                m.setInertia([0.0001] * 3)
                robot.link(i).setMass(m)

            s = create_new_design(robot)
            for i in range(9, 18):
                s.scale_link_length(i, scale_factor=length[i - 9])
                if "box" in world_file:
                    s.scale_link_width_box(i, scale_factor=width[i - 9])
                elif "robotiq" in world_file:
                    s.scale_link_width(i, scale_factor=width[i - 9])
            s.set_pos_on_palm(6, (radius[0], link_angle[0]), link_tilted_angle[0])
            s.set_pos_on_palm(7, (radius[1], link_angle[1] + 120), link_tilted_angle[1])
            s.set_pos_on_palm(8, (radius[2], link_angle[2] + 240), link_tilted_angle[2])
            if curvature is not None:
                s.change_curvature(curvature)

            init_config = robot.getConfig()
            obj, object_r, object_T = make_object_from_file(world, object_file)
            set_moving_base_xform(robot, so3.identity(), [1, 1, 1])
            robot.setConfig(init_config)
            grasp_test_module = GraspGL(world, object_r, object_T, max_iteration=max_iter)
            grasp_test_module.run_simulation()
            success, quality = grasp_test_module.get_result()
            result_queue.put((object_file, success, quality))
    return True


def grasp_test(world_file, length, width, link_angle, radius, link_tilted_angle,object_files, max_iter, curvature=None):
    """
    length : (9,)
    width : (9,)
    link_angle : (3,)
    radius : (3,)
    link_tilted_angle(3,)
    gamma : constant ( real value < 1)
    """
    world = WorldModel()
    res = world.readFile(world_file)
    if not res:
        raise RuntimeError("Unable to load world")

    robot = world.robot(0)
    for i in range(6):
        m = robot.link(i).getMass()
        m.setInertia([0.0001]*3)
        robot.link(i).setMass(m)

    original_mass = 0
    for i in range(5, robot.numLinks()):
        original_mass += robot.link(i).getMass().getMass()

    s = create_new_design(robot)
    for i in range(9, 18):
        s.scale_link_length(i, scale_factor=length[i - 9])
        if "box" in world_file:
            s.scale_link_width_box(i, scale_factor=width[i - 9])
        elif "robotiq" in world_file:
            s.scale_link_width(i, scale_factor=width[i - 9])
    s.set_pos_on_palm(6, (radius[0], link_angle[0]), link_tilted_angle[0])
    s.set_pos_on_palm(7, (radius[1], link_angle[1] + 120), link_tilted_angle[1])
    s.set_pos_on_palm(8, (radius[2], link_angle[2] + 240), link_tilted_angle[2])
    if curvature is not None:
        s.change_curvature(curvature)

    mass = 0
    for i in range(5, robot.numLinks()):
        mass += robot.link(i).getMass().getMass()

    grasp_result = []
    num_success =[]
    ''' 
    # original codes without multiprocessing module
    init_config = robot.getConfig()
    for object_file_name in object_files:
        obj, object_r, object_T = make_object_from_file(world, object_file_name)
        set_moving_base_xform(robot, so3.identity(), [1, 1, 1])
        robot.setConfig(init_config)
        grasp_test_module = GraspGL(world, object_r, object_T, max_iteration= max_iter)
        grasp_test_module.run_simulation()
        num_success_, grasp_quality = grasp_test_module.get_result()
        grasp_result += [grasp_quality]
        num_success += [num_success_]
    vis.show(False)
    vis.kill()
    '''

    # change the following codes into multi-processing codes.
    number_of_processes = 8
    object_name = []
    processes = []
    tasks_to_accomplish = Queue()
    for object in object_files:
        tasks_to_accomplish.put(
            [world_file, object, length, width, link_angle, radius, link_tilted_angle, curvature, max_iter])
    result_queue = Queue()
    for w in range(number_of_processes):
        p = Process(target=do_job, args=(tasks_to_accomplish, result_queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    prev_success = []
    prev_quality = []
    for i in range(len(object_files)):
        name, s, q = result_queue.get()
        prev_success += [s]
        prev_quality += [q]
        object_name += [name]

    assert len(object_files) == len(prev_success) and len(object_files) == len(prev_quality)

    arg_sort = np.argsort(object_name)
    for idx in range(len(object_files)):
        num_success += [prev_success[arg_sort[idx]]]
        grasp_result += [prev_quality[arg_sort[idx]]]

    return num_success, grasp_result, mass
