#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat June 20 14:44:12 2020
@author: fran
"""
import numpy as np
import yaml
from types import SimpleNamespace
from uvnpy.toolkit.linalg import vector
from uvnpy.toolkit.linalg import rotation
g = 9.81

class Accelerometer(object):
    """ This class is intended to save information 
    about a generic accelerometer """
    def __init__(self, **kwargs):
        self.config = SimpleNamespace(**kwargs)
        bw = 0.5 * self.config.rate
        accel_bias_xy_mks = self.config.bias_xy * g * 0.001
        accel_bias_z_mks = self.config.bias_z * g * 0.001
        sigma = self.config.nsd * np.sqrt(2 * bw) * g * 0.001 # m/s2
        self.sigma = vector.vec3(
            x=sigma,
            y=sigma,
            z=sigma
        ) 
        self.bias = vector.vec3(
            x=accel_bias_xy_mks,
            y=accel_bias_xy_mks,
            z=accel_bias_z_mks
        )
        self.bias_drift = vector.vec3(
            x=0.05 * accel_bias_xy_mks,
            y=0.05 * accel_bias_xy_mks,
            z=0.05 * accel_bias_z_mks
        )


class Gyroscope(object):
    """ This class is intended to save information 
    about a generic gyroscope """
    def __init__(self, **kwargs):
        self.config = SimpleNamespace(**kwargs)
        bw = 0.5 * self.config.rate
        gyro_bias_xyz = self.config.bias * (np.pi / 180)
        sigma = self.config.nsd * np.sqrt(2 * bw) * (np.pi / 180)   # rad/s
        self.sigma = vector.vec3(
            x=sigma,
            y=sigma,
            z=sigma
        ) 
        self.bias = vector.vec3(
            x=gyro_bias_xyz,
            y=gyro_bias_xyz,
            z=gyro_bias_xyz
        )
        self.bias_drift = vector.vec3(
            x=0.1* gyro_bias_xyz,
            y=0.1* gyro_bias_xyz,
            z=0.1* gyro_bias_xyz
        )


class Compass(object):
    """ This class is intended to save information 
    about a generic compass """
    def __init__(self, **kwargs):
        self.config = SimpleNamespace(**kwargs)
        bw = 0.5 * self.config.rate
        sigma = self.config.nsd * np.sqrt(2 * bw)  # microT
        self.sigma = vector.vec3(
            x=sigma,
            y=sigma,
            z=sigma
        ) 


class Imu(object):
    """ This class implements model of a Inertial
    Measurement Unit (IMU) """
    def __init__(self, name, **kwargs):
        # read config file
        config_dict = yaml.load(open('../config/{}.yaml'.format(name)))
        config = SimpleNamespace(**config_dict)
        # set components
        config.accel.update(kwargs)
        self.accel = Accelerometer(**config.accel)
        config.gyro.update(kwargs)
        self.gyro = Gyroscope(**config.gyro)
        config.mag.update(kwargs)
        self.mag = Compass(**config.mag)
        #  Noise covariance matrices
        self.Q = np.diag(np.square([*self.accel.sigma,
                                    *self.gyro.sigma,
                                    *self.accel.bias_drift,
                                    *self.gyro.bias_drift]))
        self.R = np.diag(np.square([*self.mag.sigma]))

    def measurement(self, a, w, e):
        """ This module takes as parameter accel, vel
        and pose of a vehicle and simulates an imu 
        biased and noisy measurement """
        accel = np.asarray(a)
        w = np.asarray(w)
        euler = np.asarray(e)
        #   accelerometer
        R = rotation.matrix.RZYX(*euler)
        accel_meas = R.T @ accel + (self.accel.bias + np.random.normal(0., self.accel.sigma(), 3))
        #   gyroscope
        gyro_meas = R.T @ w + (self.gyro.bias + np.random.normal(0., self.gyro.sigma(), 3))
        quaternion = rotation.quaternion.from_euler(e + np.random.normal(0, 0.05, 3))
        return (accel_meas.reshape(-1,1),
                gyro_meas.reshape(-1,1),
                quaternion)