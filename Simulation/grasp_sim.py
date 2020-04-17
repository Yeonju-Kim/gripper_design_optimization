from klampt import *
from klampt.vis import GLRealtimeProgram
from klampt.model import contact
import numpy as np
from klampt import vis

from .moving_base import *
from . import utils
from .directions import Directions
from .computeQ1UpperBound import compute_Q1


class Grasp(GLRealtimeProgram):
    def __init__(self, world, sim):
        GLRealtimeProgram.__init__(self, "GLTest")
        self.world = world
        self.sim = sim
        self.sim.enableContactFeedbackAll()
        self.dt = 0.02
        self.angular_velocity = 0.5

        self.hand_state = 2
        self.joint_limits = self.world.robot(0).getJointLimits()

        self.is_forceClosure = False
        self.is_equilibrium = False
        self.object_id = self.world.rigidObject(0).getID()

        self.num_wrench_sampling = 100
        self.num_failures = 0
        self.num_forceClosure = 0

        self.result_success = None
        self.result_gws_vol = None
        self.result_gws_radius = None

        self.is_lifting = False
        self.time_start_lift = 0.
        self.lift_score = 0.

        self.is_shaking = False
        self.time_start_shake = 0.
        self.dir = None
        self.iter_shaking = 0
        self.dynamic_score = 0.0

    def check_contacts(self):
        num_contacts = 0
        contacted_links = []
        for idx in range(5, 18):
            link_id = self.world.robot(0).link(idx).getID()
            if self.sim.inContact(self.object_id, link_id):
                contact_points = self.sim.getContacts(self.object_id, link_id)
                num_contacts += len(contact_points)
                contacted_links += [idx]
        return num_contacts, contacted_links

    def check_contacts_terrain(self):
        num_contacts = 0
        terrain_id = self.world.terrain(0).getID()
        for idx in range(5, 18):
            link_id = self.world.robot(0).link(idx).getID()
            if self.sim.inContact(terrain_id, link_id):
                contact_points = self.sim.getContacts(terrain_id, link_id)
                num_contacts += len(contact_points)
        return num_contacts > 0

    def change_hand_state(self, is_success=None, gws_result=None, collide_terrain=None):
        if self.hand_state == 2:  # approaching -> grasping
            if collide_terrain:
                self.hand_state = 0
                # vis.show(False)
            else:
                self.hand_state = 3
                self.num_failures = 0
                self.num_forceClosure = 0
        elif self.hand_state == 3:  # grasping -> set_positon
            if is_success:
                self.result_success = is_success
                self.result_gws_vol = gws_result[0]
                self.result_gws_radius = gws_result[1]
                self.angular_velocity = 0.
                # self.hand_state = 4
                # self.time_start_lift = self.sim.getTime()
            else:
                self.result_success = is_success
                self.result_gws_vol = 0.
                self.result_gws_radius = 0.
                '''dynamic simulation
                self.angular_velocity = 0.
                self.hand_state = 1
                vis.show(False)'''
            #only grasping
            self.angular_velocity = 0.
            self.hand_state = 1
            # vis.show(False)
        # elif self.hand_state == 4:
        #     if self.lift_score == 0:
        #         self.hand_state = 1
        #         self.dynamic_score = 0.
        #         vis.show(False)
        #     else:
        #         self.dynamic_score = 0.5
        #         self.hand_state = 5
        #         self.iter_shaking = 0
        #         self.time_start_shake = self.sim.getTime()
        # elif self.hand_state == 5:
        #     self.hand_state = 1
        #     vis.show(False)

    def get_result(self):
        return self.result_success, self.result_gws_vol, self.result_gws_radius

    def get_result_shaking(self):
        return self.result_success, self.dynamic_score, self.result_gws_radius

    def get_hand_state(self):
        return self.hand_state

    def check_force_closure(self):
        clist = []
        for idx in range(self.world.robot(0).numLinks()):
            link_id = self.world.robot(0).link(idx).getID()
            if self.sim.inContact(self.object_id, link_id):
                contact_points = self.sim.getContacts(self.object_id, link_id)
                # forces = self.sim.getContactForces(self.object_id, link_id)
                for pt_idx in range(len(contact_points)):
                    x = np.asarray(contact_points[pt_idx][:3])
                    n = np.asarray(contact_points[pt_idx][3:6])
                    k = contact_points[pt_idx][6]
                    clist.append(contact.ContactPoint(x, n, k))
        return contact.forceClosure(clist)

    def construct_wrench_space(self):
        contact_list = []

        for idx in range(self.world.robot(0).numLinks()):
            link_id = self.world.robot(0).link(idx).getID()

            if self.sim.inContact(self.object_id, link_id):
                contact_list += self.sim.getContacts(self.object_id, link_id)

        contacts = np.asarray(contact_list)
        k_friction = self.world.rigidObject(0).getContactParameters().kFriction

        dirs = Directions(res=2)
        pss = []
        dss = []
        nss =[]

        num_contact = contacts[:, 3:6].shape[0]
        for idx in range(num_contact):
            p = np.asarray(contacts[idx, 0:3])
            c =np.asarray(self.sim.body(self.world.rigidObject(0)).getTransform()[1])
            n = np.asarray(contacts[idx, 3:6])
            p = p-c
            n = n/np.linalg.norm(n)
            dss.append(np.random.uniform(-1,1))
            nss.append(n)
            pss.append(p)
        q = compute_Q1(None, k_friction, 0., pss, dss, nss, dirs.dirs, analytic=True)
        return None, q

    # def display(self):
    #     self.sim.updateWorld()
    #     self.world.drawGL()

    def run(self):
        while self.hand_state > 1:
            self.idle()#self.sim.simulate(): Advance() & updateWorld()=sim.updateWorld()
            if self.sim.getStatus() > 0:
                self.hand_state = -1
                return

    def idle(self):
        controller = self.sim.controller(0)
        if self.sim.getTime() >= 1 and self.hand_state > 1:
            num_contacts, contacted_links = self.check_contacts()
            if self.hand_state == 2:
                if num_contacts > 0:
                    self.change_hand_state()
                else:
                    if self.check_contacts_terrain():
                        self.change_hand_state(collide_terrain=True)
                    T = self.world.robot(0).link(5).getTransform()
                    translation = math.vectorops.add(so3.apply(T[0], [0, 0, -0.01]), T[1])
                    send_moving_base_xform_linear(controller, T[0], translation, self.dt)
            elif self.hand_state == 3:
                q = controller.getCommandedConfig()
                if len(contacted_links) < 13:
                    for link_idx in range(9, 18):
                        new_q = q[link_idx] + self.angular_velocity * self.dt
                        q[link_idx] = np.clip(new_q, self.joint_limits[0][link_idx], self.joint_limits[1][link_idx])
                    controller.setPIDCommand(q, [0] * len(q))
                if len(contacted_links) < 4:
                    self.num_failures += 1
                    self.num_forceClosure = 0
                else:
                    self.is_forceClosure = self.check_force_closure()
                    if self.is_forceClosure:
                        self.num_forceClosure += 1
                        if self.num_forceClosure > 20:
                            avg, max_radius = self.construct_wrench_space()
                            self.change_hand_state(is_success=True, gws_result=(avg, max_radius))
                    else:
                        self.num_failures += 1
                        self.num_forceClosure = 0
                if self.num_failures > 80:
                    self.change_hand_state(is_success=False)
                    print('     fail!')
            # elif self.hand_state == 4:
            #     if len(contacted_links) == 0:
            #         self.change_hand_state()
            #     else:
            #         if self.sim.getTime() - self.time_start_lift < 1.:
            #             q = controller.getCommandedConfig()
            #             if len(contacted_links) < 13:
            #                 for link_idx in range(9, 18):
            #                     new_q = q[link_idx] + self.angular_velocity * self.dt
            #                     q[link_idx] = np.clip(new_q, self.joint_limits[0][link_idx], self.joint_limits[1][link_idx])
            #                 controller.setPIDCommand(q, [0] * len(q))
            #             T = self.world.robot(0).link(5).getTransform()
            #             translation = math.vectorops.add([0, 0, 0.05], T[1])
            #             send_moving_base_xform_linear(controller, T[0], translation, self.dt)
            #         elif self.sim.getTime() - self.time_start_lift >= 1.5:
            #             if len(contacted_links) >= 3:
            #                 self.lift_score = 1.
            #             elif len(contacted_links) == 2:
            #                 self.lift_score = 0.5
            #             else:
            #                 self.lift_score = 0.
            #             self.change_hand_state()
            # elif self.hand_state == 5:
            #     if len(contacted_links) == 0:
            #         self.change_hand_state()
            #     else:
            #         if self.sim.getTime() - self.time_start_shake < 3.:
            #             q = controller.getCommandedConfig()
            #             if len(contacted_links) < 13:
            #                 for link_idx in range(9, 18):
            #                     new_q = q[link_idx] + self.angular_velocity * self.dt
            #                     q[link_idx] = np.clip(new_q, self.joint_limits[0][link_idx], self.joint_limits[1][link_idx])
            #                 controller.setPIDCommand(q, [0] * len(q))
            #             if int((self.sim.getTime() - self.time_start_shake) // 0.3) >= self.iter_shaking:
            #                 self.iter_shaking += 1
            #                 self.dir = utils.sample_hemisphere(90)
            #             T = self.world.robot(0).link(5).getTransform()
            #             acc = 1.5
            #             translation = math.vectorops.add(0.5*self.dir*acc*(self.sim.getTime()-self.time_start_shake-self.iter_shaking + 1.)**2, T[1])
            #             send_moving_base_xform_linear(controller, T[0], translation, self.dt)
            #         elif self.sim.getTime() - self.time_start_shake >= 3.5:
            #             if len(contacted_links) >= 3:
            #                 print(len(contacted_links))
            #                 self.dynamic_score = 1.
            #             elif len(contacted_links) == 2:
            #                 print (len(contacted_links))
            #                 self.dynamic_score = 0.75
            #             self.change_hand_state()
        self.sim.simulate(self.dt)
        return


class GraspGL:
    def __init__(self, world, object_radius, object_T, max_iteration=10):
        self.world = world
        self.object_id = self.world.rigidObject(0).getID()
        self.object_r = object_radius + 0.4
        self.object_T = object_T
        self.hand_se3_goal = None

        self.joint_limits = self.world.robot(0).getJointLimits()
        self.init_q = self.world.robot(0).getConfig()

        self.iteration = 0
        self.max_iteration = max_iteration

        self.result_success_prob = []
        self.result_1 = [] #result_gws_volume or dynamic simulation result
        self.result_2 = [] #result gws max radius

    def _get_new_transform(self):
        object_origin = self.object_T[1]
        point_sphere = utils.sample_hemisphere(70)
        R = math.so3.canonical(point_sphere)
        new_R = np.zeros(9)
        new_R[6:9] = R[:3]
        new_R[:6] = R[3:9]
        t = math.vectorops.add(point_sphere * self.object_r, object_origin)
        return new_R, t

    def run_simulation(self):
        iteration = 0
        while iteration < self.max_iteration:
            # Set object position & set robot position
            self.hand_se3_goal = self._get_new_transform()
            set_moving_base_xform(self.world.robot(0), *self.hand_se3_goal)
            q = self.world.robot(0).getConfig()
            for i in range(9, 18):
                q[i] = self.init_q[i]
            self.world.robot(0).setConfig(q)
            self.world.rigidObject(0).setTransform(*self.object_T)
            is_simulation_success = self._simulation()
            if is_simulation_success:
                iteration += 1
        self.world.remove(self.world.rigidObject(0))
        return

    def _simulation(self):
        sim = Simulator(self.world)
        glRealProgram = Grasp(self.world, sim)
        glRealProgram.run()

        final_hand_state = glRealProgram.get_hand_state()
        if final_hand_state == 0:
            # Collide with Terrain
            return False
        if final_hand_state == -1:
            print("STATUS ERROR : status > normal")
            return False

        result = glRealProgram.get_result() #Original : glRealProgram.get_result()
        print("result : ", result)
        self.result_success_prob += [result[0]]
        self.result_1 += [result[1]]
        self.result_2 += [result[2]]
        return True

    def get_result(self):
        cond = np.asarray(self.result_success_prob, dtype=bool)
        if len(cond) != self.max_iteration:
            raise RuntimeError("Result size is not same with max iteration")
        num_success = cond.sum()
        result = np.empty((2, 0))
        if num_success > 0:
            result1 = np.extract(cond, np.asarray(self.result_1)) #vol or dynamic simulation result
            result2 = np.extract(cond, np.asarray(self.result_2)) #radius
            result = np.hstack((result, [result1, result2]))
        return result.shape[1], result
