from compile_gripper import Gripper
from compile_world import World
from compile_objects import *
import mujoco_py as mjc
import trimesh as tm
import numpy as np
import math,copy

class Controller:
    def __init__(self, world, *, approach_vel=0.5, thres_vel=5e-1, \
                 lift_vel=1.0, lift_height=3, \
                 shake_range=[0.4, 0, 0], shake_vel=15.0, shake_times=10, step_limit=1000):
        self.world = world
        self.sim = self.world.sim
        if self.world.link is None:
            raise RuntimeError('To initialize a controller, one must have link defined')
        else:
            self.link = self.world.link
        self.link.get_ctrl_address(self.sim)
        # param
        self.approach_vel = approach_vel
        self.thres_vel = thres_vel
        # approach
        self.approach_time_steps = step_limit
        # close
        self.close_steps = step_limit
        self.keep_closing = 100
        # lift
        self.lift_vel = lift_vel
        self.lift_height = lift_height
        # shake
        self.shake_range = shake_range
        self.shake_vel = shake_vel
        self.shake_times = shake_times
        # object id
        self.link_geom_ids, self.leaf_link_geom_ids = self.get_link_ids(self.link)
        self.floor_id = self.sim.model.geom_names.index('floor')

    def get_link_ids(self, link):
        model = self.sim.model
        bid = model.body_names.index(link.name)
        geom_adr = model.body_geomadr[bid]
        geom_num = model.body_geomnum[bid]

        link_geom_ids = [range(geom_adr, geom_adr + geom_num)]
        if len(link.children) == 0:
            leaf_link_geom_ids = link_geom_ids
        else:
            leaf_link_geom_ids = []

        for c in link.children:
            link, leaf = self.get_link_ids(c)
            link_geom_ids += link
            leaf_link_geom_ids += leaf
        return link_geom_ids, leaf_link_geom_ids

    def reset(self, id, angle=[0., math.pi * 0.95 / 2, 0.], init_pose=[math.pi / 2, 0.], approach_coef=[1., 1.],
              init_dist=3., grasp_dir=None):
        # we assume the gripper is always approaching from initial_pos to [0,0,0]
        self.world.test_object(id)
        state = self.sim.get_state()

        if grasp_dir is not None:
            assert isinstance(grasp_dir, float)
            if grasp_dir > 0.:
                init_pose[0] = math.pi / 2
            else:
                cos_val = self.link.base_off / (self.link.hinge_rad + self.link.finger_length) / self.link.num_segment
                if cos_val < 0.999:
                    init_pose[0] = -math.asin(cos_val)
                else:
                    init_pose[0] = -math.pi / 2
            init_pose = [init_pose[0] * abs(grasp_dir), 0.]
            approach_coef = [grasp_dir, grasp_dir]

        v0 = np.array([0, 0, 1], dtype=np.float64)
        v1 = np.array(
            [-math.cos(angle[0]) * math.cos(angle[1]), -math.sin(angle[0]) * math.cos(angle[1]), -math.sin(angle[1])],
            dtype=np.float64)
        R1 = tm.transformations.rotation_matrix(tm.transformations.angle_between_vectors(v0, v1),
                                                tm.transformations.vector_product(v0, v1))
        R0 = tm.transformations.rotation_matrix(angle=angle[2], direction=[0, 0, 1])
        R = np.matmul(R1, R0)[0:3, 0:3]
        x, y, z = tm.transformations.euler_from_matrix(R, 'rxyz')
        self.target_rot = [x, y, z]
        self.target_pos = (v1 * -init_dist).tolist()
        self.target_vel = [v * self.approach_vel for v in v1.tolist()]
        self.target_dir = v1

        self.init_pose = init_pose
        self.approach_coef = approach_coef
        self.link.set_PD_target(self.target_pos + self.target_rot + self.init_pose, state=state)
        self.sim.set_state(state)
        self.approached = False
        self.closed = False
        self.lifted = False
        self.shaked = False
        self.shake_count = 0
        self.approach_count = 0
        self.keep_closing_count = 0
        self.close_count = 0
        self.elapsed = 0.0
        self.contact_poses = []
        self.contact_normals = []
        self.init_state = copy.deepcopy(state)

    def record_contacts(self):
        # get object pos
        state = self.sim.get_state()
        addr = self.world.addrs[self.world.target_object_id]
        objpos = [state.qpos[addr[0]], state.qpos[addr[1]], state.qpos[addr[2]]]
        # get contacts
        self.contact_poses = []
        self.contact_normals = []
        for ic in range(self.sim.data.ncon):
            c = self.sim.data.contact[ic]
            g1clink = any([c.geom1 in rng for rng in self.link_geom_ids])
            g2clink = any([c.geom2 in rng for rng in self.link_geom_ids])
            if g1clink and not g2clink:
                if c.geom2 in self.world.target_geom_ids:
                    self.contact_poses.append([cp - op for cp, op in zip(c.pos, objpos)])
                    self.contact_normals.append(c.frame[0:3].tolist())
            elif not g1clink and g2clink:
                if c.geom1 in self.world.target_geom_ids:
                    self.contact_poses.append([cp - op for cp, op in zip(c.pos, objpos)])
                    # to compute Q_* metric, we assume normals are point inward i.e. n*p<0
                    self.contact_normals.append([-n for n in c.frame[0:3].tolist()])

    def link_contact_state(self):
        floor_contact = False
        obj_contact = False
        gcleaf = [False for l in self.leaf_link_geom_ids]
        contact_all_geoms = [False for l in self.link_geom_ids]
        for ic in range(self.sim.data.ncon):
            c = self.sim.data.contact[ic]
            g1clink = any([c.geom1 in rng for rng in self.link_geom_ids])
            g2clink = any([c.geom2 in rng for rng in self.link_geom_ids])
            if g1clink and not g2clink:
                if c.geom2 == self.floor_id:
                    floor_contact = True
                else:
                    obj_contact = True
            elif not g1clink and g2clink:
                if c.geom1 == self.floor_id:
                    floor_contact = True
                else:
                    obj_contact = True

            for idx in range(len(self.link_geom_ids)):
                contact_all_geoms[idx] = contact_all_geoms[idx] or (
                            c.geom1 in self.link_geom_ids[idx] and c.geom2 in self.world.target_geom_ids)
                contact_all_geoms[idx] = contact_all_geoms[idx] or (
                            c.geom2 in self.link_geom_ids[idx] and c.geom1 in self.world.target_geom_ids)

            for leafid in range(len(self.leaf_link_geom_ids)):
                gcleaf[leafid] = gcleaf[leafid] or (
                            c.geom1 in self.leaf_link_geom_ids[leafid] and c.geom2 in self.world.target_geom_ids)
                gcleaf[leafid] = gcleaf[leafid] or (
                            c.geom2 in self.leaf_link_geom_ids[leafid] and c.geom1 in self.world.target_geom_ids)
        return floor_contact, obj_contact, sum(contact_all_geoms) > 3  # all(gcleaf)

    def object_contact_state(self):
        for ic in range(self.sim.data.ncon):
            c = self.sim.data.contact[ic]
            g1cobject = c.geom1 in self.world.target_geom_ids
            if g1cobject and c.geom2 == self.floor_id:
                return True
            g2cobject = c.geom2 in self.world.target_geom_ids
            if g2cobject and c.geom1 == self.floor_id:
                return True
        return False

    def approach(self):
        state = self.sim.get_state()
        pos = np.array([state.qpos[self.link.joint_ids[d]] for d in range(3)])
        pos = (pos.dot(self.target_dir) * self.target_dir).tolist()
        x = [p + v for p, v in zip(pos, self.target_vel)] + self.target_rot + self.init_pose
        vx = self.target_vel + [0, 0, 0] + [0, 0]
        self.link.set_PD_target(x, vx)
        self.link.define_ctrl(self.sim, state.qpos, state.qvel)

        # return succeed or failed
        self.sim.step()
        fc, oc, _ = self.link_contact_state()
        if fc:  # if floor contact, immediately return false
            return False
        if oc:
            self.approached = True
            state = self.sim.get_state()
            self.x_approached = self.link.fetch_q(state.qpos)
            return True
        else:
            self.approach_count += 1
            if self.approach_count > self.approach_time_steps:
                print('time limit for approaching step : ', self.approach_count)
                return False
        return True

    def close(self):
        state = self.sim.get_state()
        qpos = self.link.fetch_q(state.qpos)
        x = self.x_approached[0:6] + [qpos[6] - self.approach_vel * self.approach_coef[0],
                                      qpos[7] - self.approach_vel * self.approach_coef[1]]
        vx = [0 for i in range(6)] + [-self.approach_vel * self.approach_coef[0],
                                      -self.approach_vel * self.approach_coef[1]]
        self.link.set_PD_target(x, vx)
        self.link.define_ctrl(self.sim, state.qpos, state.qvel)

        # return succeed or failed (actually close will always succeed)
        self.sim.step()
        # self.elapsed+=1
        fc, _, lc = self.link_contact_state()
        if fc:  # if floor contact, immediately return false
            return False
        state = self.sim.get_state()
        maxVel = max([abs(q) for q in self.link.fetch_q(state.qvel)])
        if lc:
            self.keep_closing_count += 1
        else:
            self.keep_closing_count = 0

        if maxVel < self.thres_vel or self.keep_closing_count > self.keep_closing:  # we assume closed when the velocity is small enough
            self.x_closed = self.link.fetch_q(state.qpos)
            self.record_contacts()  # this will be used to compute Q_* metric later
            self.closed = True
            self.keep_closing_count = 0
        else:
            self.close_count += 1
            if self.close_count > self.close_steps:
                return False
        return True

    def lift(self):
        state = self.sim.get_state()
        qpos = self.link.fetch_q(state.qpos)
        height = qpos[2]
        x = self.x_closed[0:6] + [qpos[6] - self.approach_vel * self.approach_coef[0],
                                  qpos[7] - self.approach_vel * self.approach_coef[1]]
        vx = [0 for i in range(6)] + [-self.approach_vel * self.approach_coef[0],
                                      -self.approach_vel * self.approach_coef[1]]
        vx[2] = self.lift_vel
        x[2] = height + vx[2]
        self.link.set_PD_target(x, vx)
        self.link.define_ctrl(self.sim, state.qpos, state.qvel)

        self.sim.step()
        self.elapsed += 1

        # return succeed or failed based on whether gripper is still in contact with object
        state = self.sim.get_state()
        if height > self.lift_height:
            if not self.object_contact_state():
                self.x_lifted = self.link.fetch_q(state.qpos)
                self.lifted = True
                return True
            else:
                return False
        return True

    def shake(self):
        sgn = 1 if self.shake_count % 2 == 0 else -1
        state = self.sim.get_state()
        pos = np.array([state.qpos[self.link.joint_ids[d]] for d in range(3)])
        srng = np.array(self.shake_range)
        len = np.linalg.norm(srng)
        srng *= 1 / len

        x = self.x_lifted[0:8]
        pos0 = np.array(x[0:3]) - np.array(x[0:3]).dot(srng) * srng
        pos0 += srng * (pos.dot(srng) + len * sgn)
        x[0:3] = pos0.tolist()

        qpos = self.link.fetch_q(state.qpos)
        x[6:8] = [qpos[6] - self.approach_vel * self.approach_coef[0],
                  qpos[7] - self.approach_vel * self.approach_coef[1]]
        vx = [0 for i in range(6)] + [-self.approach_vel * self.approach_coef[0],
                                      -self.approach_vel * self.approach_coef[1]]
        vx[0:3] = [v * sgn * self.shake_vel for v in srng]
        self.link.set_PD_target(x, vx)
        self.link.define_ctrl(self.sim, state.qpos, state.qvel)

        self.sim.step()
        self.elapsed += 1
        if self.object_contact_state():  # if floor contacts object
            return False

        # return succeed or failed based on whether gripper is still in contact with object
        state = self.sim.get_state()
        off = (np.array(pos[0:3]) - np.array(x[0:3])).dot(srng)
        if abs(off) > len:
            self.shake_count += 1
            if self.shake_count >= self.shake_times:
                self.x_shaked = self.link.fetch_q(state.qpos)
                self.shaked = True
                return True
        return True
    
    def step(self,will_close=True,will_lift=True,will_shake=True):
        #this function return whether an experiment has finished
        if not self.approached:
            if not self.approach():
                return True
            return False
        elif will_close and not self.closed:
            if not self.close():
                return True
            return False
        elif will_lift and not self.lifted:
            if not self.lift():
                return True
            return False
        elif will_shake and not self.shaked:
            if not self.shake():
                return True
            return False
        return True
