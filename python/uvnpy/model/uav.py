#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 05 17:59:12 2020
@author: fran
"""
import numpy as np
import collections
import yaml

from gpsic.modelos.multicoptero import MulticopteroLTI
from gpsic.modelos.camara import cam
from gpsic.toolkit import linalg

from . import vehiculo, velocidad_rw
from uvnpy.filtering import kalman
from uvnpy.sensor.rango import Rango
from uvnpy.network.neighborhood import Neighborhood
from uvnpy.toolkit.ros import PositionAndRange

multi_dot = linalg.multi_dot


class uav(vehiculo):
    def __init__(
      self, id, cnfg_file='/tmp/uav.yaml',
      motion_kw={}, cam_kw={}):
        super(uav, self).__init__(id, type='uav')
        # read config file
        config = yaml.load(open(cnfg_file))
        motion_kw.update(config)
        # motion model
        self.motion = MulticopteroLTI(**motion_kw)
        # sensors
        self.range = Rango()
        self.gimbal = cam_kw.get('gimbal', linalg.rm.Ry(np.pi/2))
        self.cam = cam(
            pos=self.motion.p(),
            attitude=(self.motion.euler(), self.gimbal),
            sigma=4
        )
        # neighborhood
        self.N = Neighborhood(size=1, dof=3)
        self.vrw = velocidad_rw(dim=3, sigma=0.25)
        # filter
        xi = np.hstack(
          [np.random.normal(0, 20, self.N.dim), np.zeros(self.N.dim)]
        )
        dxi = np.hstack([20*np.ones(self.N.dim), np.ones(self.N.dim)])
        self.filter = kalman.EKF(xi, dxi)
        # information sharing
        self.msg = PositionAndRange(id=self.id, source='uav')
        self.set_msg(self.filter.x, self.filter.P)
        self.inbox = collections.deque(maxlen=10)

    def sim_step(self, u, t, fw=(0., 0., 0.)):
        self.motion.step(u, t, d_kw={'fw': fw})

    def ctrl_step(self, t, points):
        self.cam.update_pose(
          self.motion.p(), (self.motion.euler(), self.gimbal)
        )
        pose = self.noisy_pose()
        self.filter.prediction(self.f_vrw, t, None)
        if len(points) != 0:
            pixels = self.cam.view(*points)
            if len(pixels) != 0:
                self.filter.correction(self.h_cam, pixels.flatten(), *pose)
        for msg in self.inbox:
            if msg.source == 'Rover':
                self.N.update(msg.id)
                y = np.array([msg.range])
                self.filter.correction(self.h_range, y, msg.id, *pose[:2])
        self.inbox.clear()
        self.set_msg(self.filter.x, self.filter.P)

    def set_msg(self, x, P):
        self.msg.point.x = x[0]
        self.msg.point.y = x[1]
        self.msg.point.z = x[2]
        self.msg.covariance = P[0:3, 0:3].flatten()

    def noisy_pose(self):
        sigma_p = 0.3
        p = np.random.normal(self.motion.p(), sigma_p)
        cov_p = np.diag(np.square([sigma_p]*3))
        sigma_att = 0.15
        n, t = np.random.normal(0, 1, 3), np.random.normal(0, sigma_att)
        att = linalg.rm.from_any((n, t), self.motion.euler())
        cov_att = np.diag(np.square([sigma_att]*3))
        return p, cov_p, att, cov_att

    def f_vrw(self, x, t, u):
        return self.vrw.dot_x(x, u, t), self.vrw.F_x, self.vrw.F_e, self.vrw.Q

    def h_cam(self, x, p, cov_p, att, cov_att):
        K = self.cam.intrinsic[:3, :3]
        C = linalg.rm.from_any(att, self.gimbal)
        n = np.reshape(x[:3] - p, (-1, 1))
        P = np.matmul(K, C.T)
        zs_inv = np.matmul(P[2, :], n)**-1
        A = zs_inv * P[:2, :]
        m = np.matmul(A, n)
        Hn = A - zs_inv * np.matmul(m, P[[2], :])
        Ht = np.matmul(Hn, linalg.skew(n.flatten()))
        Rc = self.cam.sigma**2 * np.eye(2)
        R = multi_dot([Hn, cov_p, Hn.T]) + multi_dot([Ht, cov_att, Ht.T]) + Rc
        H = np.block([Hn, np.zeros([2, 3])])
        return m.flatten(), H, R

    def h_range(self, x, id, p, cov_p):
        """ This function takes two robot's estimated position and
        covariance to compute expected range measurement, and jacobians
        """
        #   Expected measurement
        pj = x[[0, 1, 2]]
        dist = linalg.distance(p, pj)
        hat_y = np.hstack([dist])
        #   Jacobian
        H = np.zeros([1, 2*self.N.dim])
        jac = np.subtract(p, pj)/dist
        H[0, 0:3] = -jac
        #   Noise
        Hi = np.array([jac])
        R = multi_dot([Hi, cov_p, Hi.T]) + self.range.R
        return hat_y, H, R

    # def h_cam_test_2(self, pi, pj, t):
    #     cam = cam(pos=pi, attitude=t)
    #     K = cam.intrinsic[:3,:3]
    #     C = cam.attitude
    #     n = (pj-cam.pos).reshape(-1,1)
    #     P = K @ C.T
    #     zs_inv = np.matmul(P[[2],:], n).item()**-1
    #     A = zs_inv * P[:2,:]
    #     m = np.matmul(A, n)
    #     Hn = A - zs_inv * np.matmul(m, P[[2],:])
    #     return m, Hn

    def p(self):
        return self.motion.p()

    def euler(self):
        return self.motion.euler()

    def hat_mp(self):
        return self.filter.x[:3]

    def cov_mp(self):
        return np.copy(self.filter.P[:3, :3])