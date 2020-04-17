from klampt import *
import numpy as np
from klampt.math import so3
import math


class create_new_design():
    def __init__(self, robot):
        self.robot = robot
        self.num_finger = 3
        self.num_links = 3

    def change_curvature(self, scale_factor):
        for link_idx in range(9, 18):
            g = self.robot.link(link_idx).geometry()
            mesh = g.getTriangleMesh()

            for idx in range(len(mesh.vertices) // 3):
                i = idx * 3
                vx, vy, vz = mesh.vertices[i], mesh.vertices[i + 1], mesh.vertices[i + 2]
                if vz > 0:
                    mesh.vertices[i+2] = vz * scale_factor
            g.setTriangleMesh(mesh)

            m = self.robot.link(link_idx).getMass()
            m.setMass(m.getMass()*(1.5166 + 0.5166*scale_factor))
            com = m.getCom()
            com[0] *= scale_factor
            m.setCom(com)
            self.robot.link(link_idx).setMass(m)

    def scale_link_length(self, link_idx, scale_factor):
        '''
        scaling link length along x_axis
        change translation of child link
        '''
        if link_idx < (9):
            print('cannot scale this link')
            return

        self.robot.link(link_idx).geometry().transform([scale_factor, 0., 0., 0., 1., 0., 0., 0., 1.], [0, 0, 0])

        if (link_idx + 1) % self.num_links != 0:
            T = self.robot.link(link_idx+1).getParentTransform()
            T[1][0] = T[1][0] * scale_factor
            self.robot.link(link_idx+1).setParentTransform(T[0], T[1])

        m = self.robot.link(link_idx).getMass()
        m.setMass(m.getMass()*scale_factor)
        com = m.getCom()
        com[0] *= scale_factor
        m.setCom(com)
        self.robot.link(link_idx).setMass(m)

    def scale_link_width(self, link_idx, scale_factor):
        '''
        scaling link length along z-axis
        save link_off_file
        :return:
        '''
        if link_idx < 9:
            print('cannot scale this link')
            return

        self.robot.link(link_idx).geometry().transform([1., 0., 0., 0., 1., 0., 0., 0., scale_factor], [0,0,0])
        m = self.robot.link(link_idx).getMass()
        m.setMass(m.getMass() * scale_factor)
        self.robot.link(link_idx).setMass(m)

    def scale_link_width_box(self, link_idx, scale_factor):
        '''
        scaling link length along z-axis
        save link_off_file
        :return:
        '''
        if link_idx < 9:
            print('cannot scale this link')
            return

        self.robot.link(link_idx).geometry().transform([1., 0., 0., 0., scale_factor, 0., 0., 0., 1.], [0, 0, 0])
        m = self.robot.link(link_idx).getMass()
        m.setMass(m.getMass() * scale_factor)
        self.robot.link(link_idx).setMass(m)

    def set_pos_on_palm(self, link_idx, pos, tilted_angle):
        '''
        change the position of link mounted on the palm to x, y
        :param pos: tuple value (r, theta) on palm coordinate
        :param link_idx: indicate link
        :return:
        '''

        if np.abs(tilted_angle) > 30:
            return
        print('pass')
        x = pos[0]*np.cos(pos[1]/180*np.pi)
        y = pos[0]*np.sin(pos[1]/180*np.pi)
        p_T = [x, y, 0]
        angle = np.arctan2(x, y)
        R_1 = so3.rotation([1, 0, 0], angle + math.radians(tilted_angle))
        R_2 = (0, 0, -1, 0, -1, 0, -1, 0, 0)
        p_R = so3.mul(R_2, R_1)
        self.robot.link(link_idx).setParentTransform(p_R, p_T)