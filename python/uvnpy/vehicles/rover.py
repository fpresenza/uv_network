#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:41:07 2020
@author: fran
"""
import numpy as np
import collections
import yaml
from uvnpy.vehicles.unmanned_vehicle import UnmannedVehicle
from uvnpy.model.holonomic import VelocityModel
from uvnpy.navigation.kalman import Ekf
from uvnpy.sensor.imu import Imu
from uvnpy.sensor.range import RangeSensor
from uvnpy.toolkit.ros import PositionAndRange
from uvnpy.toolkit.linalg import block_diag, vector, rotation
from uvnpy.network.neighborhood import Neighborhood

class Rover(UnmannedVehicle):
    def __init__(self, id, *args, **kwargs):
        pi = kwargs.get('pi', )
        kwargs.update(type='Rover')
        super(Rover, self).__init__(id, *args, **kwargs)
        # read config file
        config = yaml.load(open('../config/rover.yaml'))
        config.update(kwargs)
        # motion model
        self.motion = VelocityModel(**config)
        # sensors
        self.imu = Imu('MPU9250')
        self.range = RangeSensor('xbee')
        # neighborhood
        self.N = Neighborhood(size=10, dof=3)
        self.vrw = 2.5*np.tile([*self.imu.accel.sigma], self.N.size)
        self.vrw_Q = np.diag(np.square(self.vrw))
        self.dim = 6 + 2*self.N.dim + 3
        # filter
        self.Q = block_diag(self.imu.Q[:3,:3], self.vrw_Q, self.imu.Q[6:9,6:9])
        self.p = 3 # parameters
        xi = np.hstack([self.xyz().flatten(),
                        self.vxyz().flatten(),
                        np.random.normal(np.tile(self.xyz().flatten(), self.N.size), 20.),
                        np.zeros(self.N.dim),
                        np.zeros(self.p)])
        dxi = np.hstack([np.ones(3),
                         0.2*np.ones(3),
                         10*np.ones(self.N.dim),
                         0.2*np.ones(self.N.dim),
                         self.imu.accel.bias()])
        self.filter = Ekf(xi, dxi)
        # information sharing
        self.msg = PositionAndRange(id=self.id)
        self.set_msg(self.filter.x, self.filter.P)
        self.inbox = collections.deque(maxlen=10)

    def __str__(self):
        return '{}({})'.format(self.type, self.id)

    def to6dof(self, u):
        return np.insert(u, [2,2,2], 0).reshape(-1,1)

    def step(self, u, t):
        self.motion.step(self.to6dof(u), t)
        accel, gyro, q = self.imu.measurement(
            self.linear_accel().flatten(),
            self.w().flatten(),
            self.euler().flatten()
        )
        self.filter.prediction(accel, self.f_imu, t, q)
        for msg in self.inbox:
            self.N.update(msg.id)
            y = np.vstack([*msg.point, msg.range])
            self.filter.correction(y, self.h_range, msg.id, msg.covariance)
        self.inbox.clear()

        self.set_msg(self.filter.x, self.filter.P)

    def xyz(self):
        return self.motion.x[[6,7,8]].copy()

    def vxyz(self):
        return self.motion.x[[0,1,2]].copy()

    def euler(self):
        return self.motion.x[[9,10,11]].copy()

    def pose(self):
        return self.motion.x[[6,7,11]].copy()

    def vel(self):
        return self.motion.x[[0,1,5]].copy()

    def w(self):
        return self.motion.x[[3,4,5]].copy()

    def accel(self):
        return self.motion.accel()[[0,1,5]].copy()

    def linear_accel(self):
        return self.motion.accel()[[0,1,2]].copy()

    def hat_xyz(self):
        return self.filter.x[:3].copy()

    def hat_vxyz(self):
        return self.filter.x[3:6].copy()

    def hat_mp(self):
        return self.filter.x[6:6+self.N.dim].copy()

    def hat_mv(self):
        return self.filter.x[6+self.N.dim:6+2*self.N.dim].copy()

    def hat_p(self):
        return self.filter.x[6+2*self.N.dim:].copy()

    def cov_xyz(self):
        return self.filter.P[:3,:3].copy()

    def set_msg(self, x, P):
        self.msg.point.x = x[0].item()
        self.msg.point.y = x[1].item()
        self.msg.point.z = x[2].item()
        self.msg.covariance = P[0:3,0:3].flatten()

    def f_imu(self, x, u, *q):
        """ this functions represents the prediction model
        x = [p, v, Mp, Mv, p]^T
        u = [uf]
        """
        p, v = x[:3], x[3:6]
        mp, mv = x[6:6+self.N.dim], x[6+self.N.dim:6+2*self.N.dim]
        bf = x[6+2*self.N.dim:]
        R = rotation.matrix.from_quat(*q)
        f = np.block([[v],
                      [R @ (u-bf)],
                      [mv],
                      [np.zeros_like(mv)],
                      [np.zeros_like(bf)]])

        B = np.zeros((self.dim, 6+self.N.dim))
        B[3:6, 0:3] = R
        B[6+self.N.dim:6+2*self.N.dim, 3:3+self.N.dim] = np.eye(self.N.dim)
        B[6+2*self.N.dim:, 3+self.N.dim:] = np.eye(3)  

        e = np.vstack([*np.zeros(3), *self.vrw, *self.imu.accel.bias_drift])

        F_x = np.zeros((self.dim, self.dim))
        F_x[0:3,3:6] = np.eye(3) 
        F_x[3:6,6+2*self.N.dim:] = -R
        F_x[6:6+self.N.dim,6+self.N.dim:6+2*self.N.dim] = np.eye(self.N.dim)

        return f+B@e, F_x, B, self.Q

    def h_range(self, x, id, cov):
        """ This function takes two robot's estimated position and
        covariance to compute expected range measurement, and jacobians
        """
        slice_p, slice_v = self.N.index(id, start=6)
        #   Expected measurement
        pi, pj = x[[0,1,2]], x[slice_p]
        dist = vector.distance(pi, pj)
        hat_y = np.vstack([pj, dist])
        #   Jacobian
        H = np.zeros((4, self.dim))
        j = np.subtract(pi,pj).flatten()/dist
        H[0:3,slice_p] = np.eye(3)
        H[3,0:3] = j
        H[3,slice_p] = -j
        #   Noise
        R = block_diag(cov.reshape(3,3), self.range.R)
        return hat_y, H, R