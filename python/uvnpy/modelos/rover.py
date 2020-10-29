#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:41:07 2020
@author: fran
"""
import numpy as np
import collections
import yaml
import quaternion

from gpsic.toolkit import linalg

from . import vehiculo, control_velocidad, velocidad_rw
from uvnpy.filtering import kalman
from uvnpy.sensor.imu import Imu
from uvnpy.sensor.rango import Rango
from uvnpy.network.neighborhood import Neighborhood
from uvnpy.toolkit.ros import PositionAndRange


class rover(vehiculo):
    def __init__(
      self, id, cnfg_file='/tmp/rover.yaml',
      motion_kw={}, sensor_kw={}):
        super(rover, self).__init__(id, type='rover')
        # read config file
        config = yaml.load(open(cnfg_file))
        motion_kw.update(config)
        # motion model
        self.motion = control_velocidad(**motion_kw)
        # sensors
        self.imu = Imu()
        self.range = Rango()
        self.gps = sensor_kw.get('gps', False)
        # neighborhood
        self.N = Neighborhood(size=8, dof=3)
        self.nbh = velocidad_rw(dim=self.N.dim, sigma=0.25)
        # filter
        self.dim = 6 + 2*self.N.dim + 3
        self.Q = linalg.block_diag(
          self.imu.Q[:3, :3], self.nbh.Q, self.imu.Q[6:9, 6:9]
        )
        # parameters
        self.param = 3
        # filtro
        xi = np.hstack([self.p(),
                        self.v(),
                        np.random.normal(np.tile(self.p(), self.N.size), 20.),
                        np.zeros(self.N.dim),
                        np.zeros(self.param)])
        dxi = np.hstack([3*np.ones(3),
                         np.ones(3),
                         20*np.ones(self.N.dim),
                         np.ones(self.N.dim),
                         self.imu.accel.bias])
        self.filter = kalman.EKF(xi, dxi)
        # information sharing
        self.msg = PositionAndRange(id=self.id, source='rover')
        self.set_msg(self.filter.x, self.filter.P)
        self.inbox = collections.deque(maxlen=8)

    def __str__(self):
        return '{}({})'.format(self.type, self.id)

    def to6dof(self, u):
        return np.insert(u, [2, 2, 2], 0)

    def sim_step(self, u, t):
        self.motion.step(self.to6dof(u), t)

    def ctrl_step(self, t):
        accel, gyro, q = self.imu.measurement(
            self.linear_accel(),
            self.w(),
            self.euler()
        )
        self.filter.prediction(self.f_imu, t, accel, q)
        if self.gps:
            p, v = self.gps_measurement()
            y = np.hstack([p, v])
            self.filter.correction(self.h_gps, y)
        for msg in self.inbox:
            if msg.source == 'rover':
                self.N.update(msg.id)
                y = np.hstack([msg.point, msg.range])
                self.filter.correction(self.h_range, y, msg.id, msg.covariance)
            elif msg.source == 'Drone':
                y = np.hstack([msg.point])
                self.filter.correction(self.h_pos, y, msg.covariance)
        self.inbox.clear()
        self.set_msg(self.filter.x, self.filter.P)

    def p(self):
        return self.motion.x[[0, 1, 2]]

    def euler(self):
        return self.motion.x[[3, 4, 5]]

    def v(self):
        return self.motion.x[[6, 7, 8]]

    def w(self):
        return self.motion.x[[9, 10, 11]]

    def linear_accel(self):
        return self.motion.a[[0, 1, 2]]

    def hat_p(self):
        return self.filter.x[:3]

    def hat_v(self):
        return self.filter.x[3:6]

    def hat_mp(self):
        return self.filter.x[6:6 + self.N.dim]

    def hat_mv(self):
        return self.filter.x[6 + self.N.dim:6 + 2*self.N.dim]

    def hat_param(self):
        return self.filter.x[6 + 2*self.N.dim:]

    def cov_p(self):
        return np.copy(self.filter.P[:3, :3])

    def set_msg(self, x, P):
        self.msg.point.x = x[0]
        self.msg.point.y = x[1]
        self.msg.point.z = x[2]
        self.msg.covariance = P[0:3, 0:3].flatten()

    def gps_measurement(self):
        sigma_p = 0.3
        p = np.random.normal(self.motion.x[[0, 1, 2]], sigma_p)
        sigma_v = 0.15
        v = np.random.normal(self.motion.x[[6, 7, 8]], sigma_v)
        return p, v

    def f_imu(self, x, t, u, q):
        """ this functions represents the prediction model
        x = [p, v, Mp, Mv, p]^T
        u = [uf]
        """
        _, v = x[:3], x[3:6]
        _, mv = x[6:6 + self.N.dim], x[6+self.N.dim:6 + 2*self.N.dim]
        bf = x[6+2*self.N.dim:]
        R = quaternion.as_rotation_matrix(q)
        f = np.block([v,
                      np.matmul(R, (u-bf)),
                      mv,
                      np.zeros_like(mv),
                      np.zeros_like(bf)])

        B = np.zeros([self.dim, 6+self.N.dim])
        B[3:6, 0:3] = R
        Id = np.eye(self.N.dim)
        B[6 + self.N.dim:6 + 2*self.N.dim, 3:3 + self.N.dim] = Id
        B[6 + 2*self.N.dim:, 3 + self.N.dim:] = np.eye(3)

        sigma = np.hstack(
          [np.zeros(3), self.nbh.sigma, self.imu.accel.bias_drift]
        )
        e = np.random.normal(0, sigma)
        dot_x = f + np.matmul(B, e)

        F_x = np.zeros([self.dim, self.dim])
        F_x[0:3, 3:6] = np.eye(3)
        F_x[3:6, 6+2*self.N.dim:] = -R
        F_x[6:6 + self.N.dim, 6 + self.N.dim:6 + 2*self.N.dim] = Id

        return dot_x, F_x, B, self.Q

    def h_gps(self, x):
        """ This functions modelates a correction of the posterior belief
        based on gps measurements """
        #   Expected measurement
        hat_y = x[:6]
        #   Jacobian
        H = np.zeros([6, self.dim])
        H[:6, :6] = np.eye(6)
        #   Noise
        R = np.diag([0.3, 0.3, 0.3, 0.15, 0.15, 0.15])
        return hat_y, H, R

    def h_range(self, x, id, cov):
        """ This function takes two robot's estimated position and
        covariance to compute expected range measurement, and jacobians
        """
        slice_p, slice_v = self.N.index(id, start=self.motion.dof)
        #   Expected measurement
        pi, pj = x[[0, 1, 2]], x[slice_p]
        dist = linalg.distance(pi, pj)
        hat_y = np.hstack([pj, dist])
        #   Jacobian
        H = np.zeros([4, self.dim])
        jac = np.subtract(pi, pj)/dist
        H[0:3, slice_p] = np.eye(3)
        H[3, 0:3] = jac
        H[3, slice_p] = -jac
        #   Noise
        Cov = np.reshape(cov, (3, 3))
        R = linalg.block_diag(Cov, self.range.R)
        # if self.id == 1:
        #     print(id, hat_y)
        return hat_y, H, R

    def h_pos(self, x, cov_p):
        #   Expected measurement
        hat_y = x[[0, 1, 2]]
        #   Jacobian
        H = np.zeros([3, self.dim])
        H[:3, :3] = np.eye(3)
        #   Noise
        R = np.reshape(cov_p, (3, 3))
        return hat_y, H, R
