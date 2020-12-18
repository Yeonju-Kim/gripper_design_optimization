from klampt import *
from klampt.vis.visualization import VisualizationPlugin
from klampt import vis
import pdb
import math
import numpy as np
from klampt.math import vectorops,so3,se3
from klampt.model import ik, collide
import pickle
from klampt.model import trajectory


class GLViewer(VisualizationPlugin):
    def __init__(self, world, visualization, shelf=None, table = None):
        VisualizationPlugin.__init__(self)
        self.grid_size = 0.05 #5cm
        self.world = world
        self.is_vis = visualization
        self.shelf_dim, self.low_shelf_pos, self.high_shelf_pos =\
            ([0.8, 0.4, 0.5], [0.9, 0.3, 0.2], [0.7, 0.3, 1.3]) if shelf is None else shelf
        self.table_dim, self.table_pos = ([1.0, 0.6, 0.7], [0.7, 0.3, 0]) if table is None else table

        self.robot = world.robot(0)
        self.leftEELink = self.robot.link(14)
        self.rightEELink= self.robot.link(31) # 14+ 17
        self.left_active_dof = [8, 9, 10, 11, 12, 13]
        self.right_active_dof = [25, 26, 27, 28, 29, 30]
        self.left_arm_link_idx = range(8,23) # except for base link
        self.right_arm_link_idx = range(25, 40) # except for base link
        # self.robotWidget = RobotPoser(self.robot)
        # self.robotWidget.setActiveDofs(self.left_active_dof+self.right_active_dof)
        # self.addWidget(self.robotWidget)
        self.gripperDistance = 0.15
        self.init_orientation(6)
        self.add('world', world)
        self.collider = collide.WorldCollider(self.world)

    def init_orientation(self, sample_size_vert):
        assert sample_size_vert is 6
        self.hori_ori_table = []
        self.hori_ori_shelf = []
        for i in range(sample_size_vert):
            ori = so3.from_axis_angle(([0, 0, 1], i*2*math.pi/float(sample_size_vert)))
            if i is 0 or i is 1 or i is 5:
                self.hori_ori_shelf.append(ori)
            self.hori_ori_table.append(ori)

        self.hori_color_values_shelf = np.array([[202,0,32], [244, 165, 130],[146, 197, 222], [5, 113, 176]])/255.
        if sample_size_vert is 6:
            self.hori_color_values_table = np.array([[178, 24, 43], [239, 138, 98], [253, 219, 199], [247, 247, 247]
                                               ,[209, 229, 240], [103, 169, 207], [33, 102, 172]])/255.

        self.vert_orientation = []
        self.vert_orientation.append(so3.from_axis_angle(([0, 1, 0], math.pi / 2.)))
        self.vert_color_values =np.array([[202,0,32],[5,113,176]])/255.

    def get_robot(self, distance=0, pan=0., tilt = 0):
        baseQ = [0, 0.2, 0., 0., 0., 0., 0]
        leftArmQ = [0, -2.34, 4.84, -1.4, -0.12, -0.36, 0, 0]
        leftGripperQ = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        rightArmQ = [0, -2.34, 4.84, -1.4, -0.12, -0.36, 0, 0]
        rightGripperQ = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        q = baseQ + leftArmQ + leftGripperQ + rightArmQ + rightGripperQ
        self.robot.setConfig(q)
        # print(self.robot.getConfig())

        # For distance
        R_left, tr_left = self.robot.link(3).getParentTransform()
        tr_left[1] += distance / 2.
        self.robot.link(3).setParentTransform(R_left, tr_left)
        R_right, tr_right = self.robot.link(4).getParentTransform()
        tr_right[1] += distance / -2.
        self.robot.link(4).setParentTransform(R_right, tr_right)

        #  tilt [-30, 30] , pan [0, 90]
        tilt = (tilt) * math.pi / 180.
        # print('tilt : ', tilt)
        pan = pan * math.pi / 180.
        T_body_left = self.robot.link(3).getParentTransform()
        T_world_left = self.robot.link(3).getTransform()
        T_bs =(se3.inv(T_world_left)[0])
        body_tilt_axis_l = vectorops.unit(so3.apply(T_bs, [1, 0, 0]))
        new_body_left = so3.mul(so3.from_axis_angle((body_tilt_axis_l, tilt)), T_body_left[0])
        new_body_left = (new_body_left)
        self.robot.link(3).setParentTransform(new_body_left, T_body_left[1])

        T_body_left = self.robot.link(3).getParentTransform()
        T_world_left = self.robot.link(3).getTransform()
        T_bs = (se3.inv(T_world_left)[0])
        body_pan_axis_l = vectorops.unit(so3.apply(T_bs, [0, 0, 1]))
        new_body_left = so3.mul(so3.from_axis_angle((body_pan_axis_l, -1.* pan)), T_body_left[0])
        new_body_left = new_body_left
        self.robot.link(3).setParentTransform(new_body_left, T_body_left[1])


        T_world_right = self.robot.link(4).getTransform()
        T_body_right = self.robot.link(4).getParentTransform()
        T_bs_r =(se3.inv(T_world_right)[0])
        body_tilt_axis_r = vectorops.unit(so3.apply(T_bs_r, [1, 0 , 0]))
        new_body_right = so3.mul(so3.from_axis_angle((body_tilt_axis_r, -1.* tilt)), T_body_right[0])
        new_body_right= new_body_right
        self.robot.link(4).setParentTransform(new_body_right, T_body_right[1])


        T_world_right = self.robot.link(4).getTransform()
        T_body_right = self.robot.link(4).getParentTransform()
        T_bs_r =se3.inv(T_world_right)[0]
        body_pan_axis_r = vectorops.unit(so3.apply(T_bs_r, [0, 0, 1]))
        new_body_right = so3.mul(so3.from_axis_angle((body_pan_axis_r, pan)), T_body_right[0])
        new_body_right= new_body_right
        assert (so3.det(new_body_right) - 1 ) < 1e-5
        # print('det' , so3.det(new_body_right))

        self.robot.link(4).setParentTransform(new_body_right, T_body_right[1])


    def get_robot_(self, distance=0, pan=0., tilt = 0):

        # For distance
        R_left, tr_left = self.robot.link(3).getParentTransform()
        tr_left[1] += distance / 2.
        self.robot.link(3).setParentTransform(R_left, tr_left)
        R_right, tr_right = self.robot.link(4).getParentTransform()
        tr_right[1] += distance / -2.
        self.robot.link(4).setParentTransform(R_right, tr_right)

        #  tilt [-30, 30] , pan [0, 90]
        tilt = (tilt) * math.pi / 180.
        # print('tilt : ', tilt)
        pan = pan * math.pi / 180.

        T_body_left = self.robot.link(3).getParentTransform()
        T_world_left = self.robot.link(3).getTransform()
        T_bs = se3.inv(T_world_left)[0]
        body_pan_axis_l = vectorops.unit(so3.apply(T_bs, [0, 0, 1]))
        new_body_left = so3.mul(so3.from_axis_angle((body_pan_axis_l, -1. * pan)), T_body_left[0])
        new_body_left =(new_body_left)

        self.robot.link(3).setParentTransform(new_body_left, T_body_left[1])


        T_body_left = self.robot.link(3).getParentTransform()
        T_world_left = self.robot.link(3).getTransform()
        T_bs = se3.inv(T_world_left)[0]
        body_tilt_axis_l = vectorops.unit(so3.apply(T_bs, [1, 0, 0]))
        new_body_left = so3.mul(so3.from_axis_angle((body_tilt_axis_l, tilt)), T_body_left[0])
        new_body_left = (new_body_left)

        self.robot.link(3).setParentTransform(new_body_left, T_body_left[1])


        T_world_right = self.robot.link(4).getTransform()
        T_body_right = self.robot.link(4).getParentTransform()
        T_bs_r = (se3.inv(T_world_right)[0])
        body_pan_axis_r = vectorops.unit(so3.apply(T_bs_r, [0, 0, 1]))
        new_body_right = so3.mul(so3.from_axis_angle((body_pan_axis_r, pan)), T_body_right[0])
        new_body_right= (new_body_right)

        self.robot.link(4).setParentTransform(new_body_right, T_body_right[1])

        T_world_right = self.robot.link(4).getTransform()
        T_body_right = self.robot.link(4).getParentTransform()
        T_bs_r =(se3.inv(T_world_right)[0])
        body_tilt_axis_r = vectorops.unit(so3.apply(T_bs_r, [1, 0 ,0]))
        new_body_right = so3.mul(so3.from_axis_angle((body_tilt_axis_r, -1.* tilt)), T_body_right[0])
        new_body_right= (new_body_right)
        # print('det' , so3.det(new_body_right))

        self.robot.link(4).setParentTransform(new_body_right, T_body_right[1])


    def set_partial_config(self, arm_config, is_left=True, trans=0.2,rot = 0.,is_both = False):
        assert len(arm_config) == 6
        baseQ = [0,trans,rot,0,0,0,0]

        if is_both:
            leftArmQ = [0] + arm_config + [0]
            leftGripperQ = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            rightArmQ = [0] + [-arm_config[0], np.pi- arm_config[1], -arm_config[2] ,np.pi-arm_config[3], -arm_config[4], -arm_config[5]] + [0]
            rightGripperQ = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            q = baseQ + leftArmQ + leftGripperQ + rightArmQ + rightGripperQ
            self.robot.setConfig(q)
        else:
            if is_left:
                leftArmQ = [0] + arm_config + [0]
                leftGripperQ = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                left_arm_nominal =[-5.514873331776197, 4.54462894654496, -5.916490220124983,
                              5.384545803127709, -4.695740116927796, 0.5840738805492091]
                rightArmQ = [0] + [-left_arm_nominal[0], np.pi- left_arm_nominal[1],
                                    -left_arm_nominal[2] ,np.pi-left_arm_nominal[3], -left_arm_nominal[4],
                                    -left_arm_nominal[5]] +[0]
                rightGripperQ = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                q = baseQ + leftArmQ + leftGripperQ + rightArmQ + rightGripperQ
                self.robot.setConfig(q)
                # self.robotWidget.set(q)

            else:
                rightArmQ = [0] + arm_config + [0]
                rightGripperQ = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                leftArmQ = [0, -2.34, 4.84, -1.4, -0.12, -0.36, 0, 0]
                leftGripperQ = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                q = baseQ + leftArmQ + leftGripperQ + rightArmQ + rightGripperQ
                self.robot.setConfig(q)
                # self.robotWidget.set(q)


    def gripper_transform(self, is_left=True):
        R = [1, 0, 0, 0, 1, 0, 0, 0, 1]

        if is_left:
            pos = self.leftEELink.getTransform()
            R = so3.mul(pos[0], R)
            t = se3.apply(pos, [self.gripperDistance, 0, 0])
            return (R, t) # se3.mul(T1 = pose,T2= (R, [gripperDistance, 0, 0]))
        else:
            pos = self.rightEELink.getTransform()
            R = so3.mul(pos[0], R)
            t = se3.apply(pos, [self.gripperDistance, 0, 0])
            return (R, t)

    def grid_EE_position(self, pose, xyz_min, xyz_max):
        coordinates = []
        id_coord = []
        for i in range(3):
            current = xyz_min[i]
            grid = []
            while current < xyz_max[i]:
                # if i is 2 and (pose[i] + current) < 0:
                #     current += self.grid_size
                # else:
                grid.append(pose[i] + current)
                current += self.grid_size
            coordinates.append(grid)
            id_coord.append(np.arange(len(grid)))
        self.positions= np.array([dimi.flatten() for dimi in np.meshgrid(*coordinates)]).T.tolist()
        self.grid_id = np.array([dimi.flatten() for dimi in np.meshgrid(*id_coord)]).T.tolist()
        self.grid_id = [(self.grid_id[i][0], self.grid_id[i][1], self.grid_id[i][2])for i in range(len(self.grid_id))]

        self.grid_pos_dict = dict(zip(self.grid_id, self.positions))

        #TODO: milestone points in bounding box
        max_xyz = [pose[i] + xyz_max[i] for i in range(3)]
        min_xyz = [pose[i] + xyz_min[i] for i in range(3)]
        mid_xyz = [(max_xyz[i] + min_xyz[i])/2. for i in range(3)]

        self.trajectory_milestones = [[max_xyz, min_xyz],
                                      [[max_xyz[0], min_xyz[1], max_xyz[2]], [min_xyz[0], max_xyz[1], min_xyz[2]]],
                                      [[mid_xyz[0], max_xyz[1], mid_xyz[2]], [mid_xyz[0], min_xyz[1], mid_xyz[2]]],
                                      [[min_xyz[0], min_xyz[1], max_xyz[2]], [max_xyz[0], max_xyz[1], min_xyz[2]]],
                                      [[min_xyz[0], max_xyz[1], max_xyz[2]], [max_xyz[0], min_xyz[1], min_xyz[2]]],
                                      [[min_xyz[0], mid_xyz[1], mid_xyz[2]], [max_xyz[0], mid_xyz[1], mid_xyz[2]]]]

        # return positions

    def make_table(self, width, depth, height, name):
        desk_thickness = 0.07
        left_front = Geometry3D()
        right_front =Geometry3D()
        left_back = Geometry3D()
        right_back= Geometry3D()
        table_top = Geometry3D()

        left_front.loadFile("data_robot_arm/objects/cylinder.off")
        right_front.loadFile("data_robot_arm/objects/cylinder.off")
        left_back.loadFile("data_robot_arm/objects/cylinder.off")
        right_back.loadFile("data_robot_arm/objects/cylinder.off")
        table_top.loadFile("data_robot_arm/objects/cube.off")

        leg_radius_scale=0.03
        leg_x_pos = width*0.5-leg_radius_scale*1.1
        leg_y_pos = depth*0.5 - leg_radius_scale*1.1
        leg_height = height-desk_thickness

        left_front.transform([leg_radius_scale, 0,0 ,0, leg_radius_scale , 0 , 0 ,0, leg_height], [-leg_x_pos, -leg_y_pos, 0])
        right_front.transform([leg_radius_scale, 0,0 ,0, leg_radius_scale , 0 , 0 ,0, leg_height], [-leg_x_pos, leg_y_pos,0])
        left_back.transform([leg_radius_scale, 0,0 ,0, leg_radius_scale , 0 , 0 ,0, leg_height], [leg_x_pos, -leg_y_pos, 0])
        right_back.transform([leg_radius_scale, 0,0 ,0, leg_radius_scale , 0 , 0 ,0, leg_height], [leg_x_pos, leg_y_pos, 0])
        table_top.transform([width, 0, 0, 0,depth, 0,0,0,desk_thickness],
                            [-width * 0.5, -depth * 0.5, height-desk_thickness])
        tablegeom= Geometry3D()
        tablegeom.setGroup()
        for i, elem in enumerate([left_front, right_front, right_back,left_back, table_top]):
            g = Geometry3D(elem)
            tablegeom.setElement(i, g)
        table = self.world.makeRigidObject(name)
        table.geometry().set(tablegeom)
        table.appearance().setColor(160/256 ,180/256, 200/ 256, 1.0)
        # self.add('table_coord', table.getTransform())
        self.collider =  collide.WorldCollider(self.world)
        return table

    def make_shelf(self, width, depth, height, name, wall_thickness=0.005):
        """Makes a new axis-aligned "shelf" centered at the origin with
        dimensions width x depth x height. Walls have thickness wall_thickness.
        If mass=inf, then the box is a Terrain, otherwise it's a RigidObject
        with automatically determined inertia.
        """

        left = Geometry3D()
        right = Geometry3D()
        back = Geometry3D()
        bottom = Geometry3D()
        top = Geometry3D()
        left.loadFile("data_robot_arm/objects/cube.off")
        right.loadFile("data_robot_arm/objects/cube.off")
        back.loadFile("data_robot_arm/objects/cube.off")
        bottom.loadFile("data_robot_arm/objects/cube.off")
        top.loadFile("data_robot_arm/objects/cube.off")
        left.transform([wall_thickness, 0, 0, 0, depth, 0, 0, 0, height], [-width * 0.5, -depth * 0.5, 0])
        right.transform([wall_thickness, 0, 0, 0, depth, 0, 0, 0, height], [width * 0.5, -depth * 0.5, 0])
        back.transform([width, 0, 0, 0, wall_thickness, 0, 0, 0, height], [-width * 0.5, depth * 0.5, 0])
        bottom.transform([width, 0, 0, 0, depth, 0, 0, 0, wall_thickness], [-width * 0.5, -depth * 0.5, 0])
        top.transform([width, 0, 0, 0, depth, 0, 0, 0, wall_thickness],
                      [-width * 0.5, -depth * 0.5, height - wall_thickness])
        shelfgeom = Geometry3D()
        shelfgeom.setGroup()
        for i, elem in enumerate([left, right, back, bottom, top]):
            g = Geometry3D(elem)
            shelfgeom.setElement(i, g)
        shelf = self.world.makeRigidObject(name)
        shelf.geometry().set(shelfgeom)
        shelf.appearance().setColor(0.2, 0.6, 0.3, 1.0)
        self.collider = collide.WorldCollider(self.world)
        return shelf

    def solve_IK(self, init_config, is_left, is_vert):
        IKDeviation = 5

        if is_left:
            link= self.leftEELink
            activeDofs = self.left_active_dof
            idx_range = self.left_arm_link_idx
        else:
            link = self.rightEELink
            activeDofs = self.right_active_dof
            idx_range= self.right_arm_link_idx
        configurations = []
        counter = 0

        if is_vert:
            orientations = self.vert_orientation
            color_values = self.vert_color_values
        else:
            if 'shelf' in self.world.rigidObject(0).getName():
                orientations = self.hori_ori_shelf
                color_values = self.hori_color_values_shelf
            elif 'table' in self.world.rigidObject(0).getName():
                orientations = self.hori_ori_table
                color_values = self.hori_color_values_table

        # start_config = self.robot.getConfig()
        scores = []
        volumes =[]
        for pos in self.positions:
            per_pos_success = []
            for ori in orientations:
                # self.robot.setConfig(start_config)
                self.set_partial_config(init_config)

                # vis.run(self)
                local1 = [self.gripperDistance, 0,0]
                local2 = vectorops.add(local1, [1,0,0])
                local3 = vectorops.add(local1, [0,1,0])
                pt1 = pos
                pt2 = vectorops.add(so3.apply(ori, [1,0,0]), pt1)
                pt3 = vectorops.add(so3.apply(ori, [0,1,0]), pt1)
                # collider = collide.WorldCollider(self.world)
                goal = ik.objective(link, local=[local1, local2, local3], world = [pt1, pt2, pt3])
                if ik.solve(goal, activeDofs = activeDofs):
                    collisions = self.collider.robotSelfCollisions(self.robot)
                    # object_collisions = self.collider.robotObjectCollisions(self.robot)
                    # terrain_collisions = self.collider.robotTerrainCollisions(self.robot)

                    if len([c for c in collisions]) > 0: #or len([o for o in object_collisions]) > 0 or len([t for t in terrain_collisions]) > 0:
                        per_pos_success.append(0)
                    else:
                        configurations.append(self.robot.getConfig())
                        As = np.empty((0, 3))
                        Bs = np.empty((0,3))
                        for idx in idx_range:
                            a,b = self.robot.link(idx).geometry().getBBTight()
                            As = np.vstack((As,a))
                            Bs = np.vstack((Bs, b))
                        a = np.min(As, axis =0)
                        b = np.max(Bs, axis=0)
                        #final configuration's max bounding box
                        # if self.is_vis:
                        #     self.add(str(counter)+str(idx)+'a', a)
                        #     self.add(str(counter)+str(idx)+'b', b)
                        #     self.hideLabel(str(counter)+str(idx)+'a', hidden=True)
                        #     self.hideLabel(str(counter)+str(idx)+'b', hidden=True)

                        #TODO: volume computation  & standard deviation
                        volumes.append((b[0]-a[0])*(b[1]-a[1])*(b[2]-a[2]))
                        per_pos_success.append(1)
                else:
                     per_pos_success.append(0)

            if self.is_vis:
                self.add(str(counter), pos)
                ratio = sum(np.array(per_pos_success) > 0) / len(per_pos_success)
                self.setAttribute(str(counter), 'size', 8)
                self.setColor(str(counter), *color_values[sum(np.array(per_pos_success)>0)], a=ratio)
                self.hideLabel(str(counter), hidden=True)
            counter += 1
            scores.append(sum(np.array(per_pos_success)>0)/len(per_pos_success))
        score = sum(scores) / len(scores)
        print('score : ', score)
            # self.robot.setConfig(new_q, is_left)
        if len(volumes) is 0:
            volume_mean = 0.5
            volume_std = 0.
        else:
            volume_mean = np.mean(volumes)
            volume_std= np.std(volumes)
        self.set_partial_config(init_config)
        return configurations, score, (volume_mean, volume_std)

    def trajectory_score(self, is_vert, init_config, pose, vmax, vmin, is_vis=False, is_left=True):
        #TODO: pick two points in the self.positions
        # pos_arg = np.random.choice(len(self.positions), 3, replace=False)
        # milestones = [self.positions[i] for i in range(len(pos_arg))]
        self.grid_EE_position(pose, vmin, vmax)
        trajectory_visualize = is_vis
        score = []
        for traj_idx, milestone in enumerate(self.trajectory_milestones):
            traj = trajectory.Trajectory(milestones = milestone)
            #Discretize the trajectory
            discretized_pts = traj.discretize(0.01).milestones
            # if trajectory_visualize:
                # self.add("point", milestone[0])
                # self.add('traj' + str(traj_idx), traj)
                # self.hideLabel('traj' + str(traj_idx), traj)

            result = []
            new_init_config = None
            for idx, pos in enumerate(discretized_pts):
                if new_init_config is not None:
                    self.set_partial_config(new_init_config)
                    config = self.local_ik_solve(pos, is_vert, new_init_config)
                else:
                    self.set_partial_config(init_config)
                    config = self.local_ik_solve(pos, is_vert, init_config)

                if config is not None:
                    result.append(True)
                    new_init_config = config
                    if trajectory_visualize:
                        self.add(str(traj_idx)+str(idx), pos)
                        self.setColor(str(traj_idx) + str(idx), 0, 0, 1)
                        self.hideLabel(str(traj_idx) + str(idx), hidden=True)
                    # print(result)
                    # vis.run(self)
                else:
                    result.append(False)
                    # print(result)
                    if trajectory_visualize:
                        self.add(str(traj_idx)+str(idx), pos)
                        self.setColor(str(traj_idx)+str(idx), 1, 0, 0)
                        self.hideLabel(str(traj_idx)+str(idx), hidden=True)

                    # vis.run(self)
            score.append(np.array(result).sum()/len(discretized_pts))
        if trajectory_visualize:
            vis.run(self)
        return np.mean(score)


    def local_ik_solve(self, pos, is_vert, init_config, is_left=True):
        assert len(init_config) == 6
        success = False
        if is_vert:
            ori = self.vert_orientation[0]
        else:
            ori = self.hori_ori_table[0]

        if is_left:
            link = self.leftEELink
            activeDofs = self.left_active_dof
            activeDOF_range = slice(self.left_active_dof[0], self.left_active_dof[-1] + 1)
        else:
            link = self.rightEELink
            activeDofs = self.right_active_dof
            activeDOF_range = slice(self.right_active_dof[0], self.right_active_dof[-1] + 1)

        self.set_partial_config(init_config)
        # vis.run(self)
        local1 = [self.gripperDistance, 0, 0]
        local2 = vectorops.add(local1, [1, 0, 0])
        local3 = vectorops.add(local1, [0, 1, 0])
        pt1 = pos
        pt2 = vectorops.add(so3.apply(ori, [1, 0, 0]), pt1)
        pt3 = vectorops.add(so3.apply(ori, [0, 1, 0]), pt1)
        goal = ik.objective(link, local=[local1, local2, local3], world=[pt1, pt2, pt3])
        if ik.solve(goal, activeDofs=activeDofs):
            collisions = len([c for c in self.collider.robotSelfCollisions(self.robot)])
            object_collisions = len([o for o in self.collider.robotObjectCollisions(self.robot)])
            # terrain_collisions = len([t for t in self.collider.robotTerrainCollisions(self.robot)])

            if collisions > 0 or object_collisions > 0:
                    # or terrain_collisions > 0:
                print(collisions, object_collisions) #terrain_collisions)
                success = False
            else:
                success = True
                print('success')
                print(pos)
        if success:
            return self.robot.getConfig()[activeDOF_range]
        else:
            return None


    def given_starting_config_score(self, init_config, pose, vmin, vmax, is_vertical, is_left, nrTrial = 20):
        self.grid_EE_position(pose, vmin, vmax)
        configs, score, volume = self.solve_IK(init_config, is_left, is_vertical)
        return score, 0.5-volume[0]


    def IK_solve_starting_pose(self, starting_pose, is_vertical, is_left):
        orientation = so3.identity()
        if is_vertical:
            orientation = self.vert_orientation[0]

        if is_left:
            link = self.leftEELink
            activeDOF = self.left_active_dof
            activeDOF_range = slice(self.left_active_dof[0], self.left_active_dof[-1] + 1)
        else:
            link = self.rightEELink
            activeDOF = self.right_active_dof
            activeDOF_range = slice(self.right_active_dof[0], self.right_active_dof[-1] + 1)

        local1 = [self.gripperDistance, 0, 0]
        local2 = vectorops.add(local1, [1, 0, 0])
        local3 = vectorops.add(local1, [0, 1, 0])
        pt1 = starting_pose
        pt2 = vectorops.add(so3.apply(orientation, [1, 0, 0]), pt1)
        pt3 = vectorops.add(so3.apply(orientation, [0, 1, 0]), pt1)

        goal = ik.objective(link, local=[local1, local2, local3], world=[pt1, pt2, pt3])

        if ik.solve_global(goal, activeDofs=activeDOF, numRestarts=1000):
            collisions = self.collider.robotSelfCollisions(self.robot)
            object_collisions = self.collider.robotObjectCollisions(self.robot)
            terrain_collisions = self.collider.robotTerrainCollisions(self.robot)
            if len([c for c in collisions]) > 0 or len([o for o in object_collisions]) > 0 \
                    or len([t for t in terrain_collisions]) > 0:
                success = False
            else:
                success = True
        else:
            success = False
        return success, self.robot.getConfig()[activeDOF_range]


    def current_collide_check(self, is_left= True):
        collisions = self.collider.robotSelfCollisions(self.robot)
        object_collisions = self.collider.robotObjectCollisions(self.robot)
        terrain_collisions = self.collider.robotTerrainCollisions(self.robot)
        if len([c for c in collisions]) > 0 or len([o for o in object_collisions]) > 0 \
                or len([t for t in terrain_collisions]) > 0:
            success = False
        else:
            success = True
        return success

    def remove_rigidObject(self):
        if self.world.numRigidObjects() > 0:
            self.world.remove(self.world.rigidObject(0))

    def vis_reset(self):
        self.clear()
        self.add('world', self.world)

    def create_rigidObject(self, rigid_obj_name):
        print(rigid_obj_name)
        R = so3.from_axis_angle(([0, 0, 1], -0.5 * math.pi))
        if rigid_obj_name == 'low_shelf':
            shelf = self.make_shelf(* self.shelf_dim, rigid_obj_name)
            shelf.setTransform(R, self.low_shelf_pos)
        elif rigid_obj_name == 'high_shelf':
            shelf = self.make_shelf(* self.shelf_dim, rigid_obj_name)
            shelf.setTransform(R, self.high_shelf_pos)
        elif rigid_obj_name == 'table':
            table = self.make_table(* self.table_dim, rigid_obj_name)
            table.setTransform(R, self.table_pos)
        self.add('world', self.world)
        print(self.world.numRigidObjects())
        return

if __name__ == '__main__':
    world = WorldModel()
    res = world.readFile('TRINA_world.xml')
    if not res:
        raise RuntimeError("unable to load model")
    pose, vmax, vmin, start_point, table, shelf = pickle.load(open('pos_and_dimension', 'rb'))
    is_vert = [True, False, True, False, False]
    rigidObject = [None, 'low_shelf', 'table', 'table', 'high_shelf']
    c= np.array([[0.05, -90, -45],
           [4.34668917e-02, 3.69203388e+01, 4.43498990e+01]])

    result = []
    viewer = GLViewer(world, visualization = True, shelf = shelf, table = table)
    for j in range(len(c)):
        viewer.get_robot(*c[j])
        viewer.add('world', viewer.world)
        vis.run(viewer)
        result_per_robot = []
        for i in range(4,5):
            viewer.create_rigidObject(rigidObject[i])
            vis.run(viewer)
            result_per_robot.append(viewer.random_starting_config(pose[i], vmax[i], vmin[i], is_vert[i], is_left=True))
            vis.run(viewer)
            viewer.remove_rigidObject()
            viewer.vis_reset()
        result.append(result_per_robot)
        viewer.get_robot(-c[j][0], -c[j][1], -c[j][2])
    pdb.set_trace()
