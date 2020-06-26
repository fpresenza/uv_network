#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat June 20 14:44:12 2020
@author: fran
"""
import numpy as np
import quaternion
import yaml
from types import SimpleNamespace
import uvnpy.toolkit.linalg as linalg
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
        self.sigma = (sigma, sigma, sigma) 
        self.bias = (accel_bias_xy_mks, accel_bias_xy_mks, accel_bias_z_mks)
        self.bias_drift = ( 0.05 * accel_bias_xy_mks, 0.05 * accel_bias_xy_mks, 0.05 * accel_bias_z_mks)


class Gyroscope(object):
    """ This class is intended to save information 
    about a generic gyroscope """
    def __init__(self, **kwargs):
        self.config = SimpleNamespace(**kwargs)
        bw = 0.5 * self.config.rate
        gyro_bias_xyz = self.config.bias * (np.pi / 180)
        sigma = self.config.nsd * np.sqrt(2 * bw) * (np.pi / 180)   # rad/s
        self.sigma = (sigma, sigma, sigma)
        self.bias = (gyro_bias_xyz, gyro_bias_xyz, gyro_bias_xyz) 
        self.bias_drift = (0.1* gyro_bias_xyz, 0.1* gyro_bias_xyz, 0.1* gyro_bias_xyz)


class Compass(object):
    """ This class is intended to save information 
    about a generic compass """
    def __init__(self, **kwargs):
        self.config = SimpleNamespace(**kwargs)
        bw = 0.5 * self.config.rate
        sigma = self.config.nsd * np.sqrt(2 * bw)  # microT
        self.sigma = (sigma, sigma, sigma) 


class Imu(object):
    """ This class implements model of a Inertial
    Measurement Unit (IMU) """
    def __init__(self, cnfg_file='../config/MPU9250.yaml',
        accel_kw={}, gyro_kw={}, mag_kw={}):
        # read config file
        config_dict = yaml.load(open(cnfg_file))
        config = SimpleNamespace(**config_dict)
        # set components
        config.accel.update(accel_kw)
        self.accel = Accelerometer(**config.accel)
        config.gyro.update(gyro_kw)
        self.gyro = Gyroscope(**config.gyro)
        config.mag.update(mag_kw)
        self.mag = Compass(**config.mag)
        #  Noise covariance matrices
        self.Q = np.diag(np.square([*self.accel.sigma,
                                    *self.gyro.sigma,
                                    *self.accel.bias_drift,
                                    *self.gyro.bias_drift]))
        self.R = np.diag(np.square(self.mag.sigma))

    def measurement(self, a, w, e):
        """ This module takes as parameter accel, vel
        and pose of a vehicle and simulates an imu 
        biased and noisy measurement """
        accel = np.copy(a)
        w = np.copy(w)
        euler = np.copy(e)
        #   accelerometer
        q = linalg.quat.ZYX(euler)
        q_conj = q.conj()
        accel_meas = quaternion.rotate_vectors(q_conj, accel) + self.accel.bias + np.random.normal(0., self.accel.sigma, 3)
        #   gyroscope
        gyro_meas = quaternion.rotate_vectors(q_conj, w) + self.gyro.bias + np.random.normal(0., self.gyro.sigma, 3)
        return (accel_meas,
                gyro_meas,
                q)