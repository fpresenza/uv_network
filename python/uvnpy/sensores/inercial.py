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

g = 9.81

__all__ = [
  'acelerometro',
  'giroscopo',
  'magnetometro',
  'IMU']


class acelerometro(object):
    def __init__(self, **kwargs):
        """ Modelo de acelerómetro. """
        self.config = SimpleNamespace(**kwargs)
        bw = 0.5 * self.config.rate
        accel_bias_xy_mks = self.config.bias_xy * g * 0.001
        accel_bias_z_mks = self.config.bias_z * g * 0.001
        sigma = self.config.nsd * np.sqrt(2 * bw) * g * 0.001  # m/s2
        self.sigma = (sigma, sigma, sigma)
        self.bias = (accel_bias_xy_mks, accel_bias_xy_mks, accel_bias_z_mks)
        self.bias_drift = (
          0.05 * accel_bias_xy_mks,
          0.05 * accel_bias_xy_mks,
          0.05 * accel_bias_z_mks)


class giroscopo(object):
    def __init__(self, **kwargs):
        """ Modelo de giróscopo. """
        self.config = SimpleNamespace(**kwargs)
        bw = 0.5 * self.config.rate
        gyro_bias_xyz = self.config.bias * (np.pi / 180)
        sigma = self.config.nsd * np.sqrt(2 * bw) * (np.pi / 180)   # rad/s
        self.sigma = (sigma, sigma, sigma)
        self.bias = (gyro_bias_xyz, gyro_bias_xyz, gyro_bias_xyz)
        self.bias_drift = (
          0.1 * gyro_bias_xyz,
          0.1 * gyro_bias_xyz,
          0.1 * gyro_bias_xyz)


class magnetometro(object):
    def __init__(self, **kwargs):
        """ Modelo de magnetómetro. """
        self.config = SimpleNamespace(**kwargs)
        bw = 0.5 * self.config.rate
        sigma = self.config.nsd * np.sqrt(2 * bw)  # microT
        self.sigma = (sigma, sigma, sigma)


class IMU(object):
    """ Modelo de IMU = Accel + Giro + Mag. """
    def __init__(
      self, cnfg_file='/tmp/MPU9250.yaml', accel_kw={}, gyro_kw={}, mag_kw={}):
        # read config file
        config_dict = yaml.load(open(cnfg_file))
        config = SimpleNamespace(**config_dict)
        # set components
        config.accel.update(accel_kw)
        self.accel = acelerometro(**config.accel)
        config.gyro.update(gyro_kw)
        self.gyro = giroscopo(**config.gyro)
        config.mag.update(mag_kw)
        self.mag = magnetometro(**config.mag)
        #  Noise covariance matrices
        self.Q = np.diag(np.square(np.hstack([
            self.accel.sigma,
            self.gyro.sigma,
            self.accel.bias_drift,
            self.gyro.bias_drift])))
        self.R = np.diag(np.square(self.mag.sigma))

    def __call__(self, a, w, q):
        """ Simular medición de una IMU.

        Argumentos:

            a: aceleración en terna global
            w: velocidad angular en terna global
            q: actitud (cuaternion)
        """
        accel = np.copy(a)
        w = np.copy(w)
        #  accelerometro
        q_conj = q.conj()
        accel_body = quaternion.rotate_vectors(q_conj, accel)
        accel_noise = np.random.normal(self.accel.sigma)
        accel_meas = accel_body + self.accel.bias + accel_noise
        #  giróscopo
        gyro_body = quaternion.rotate_vectors(q_conj, w)
        gyro_noise = np.random.normal(self.gyro.sigma)
        gyro_meas = gyro_body + self.gyro.bias + gyro_noise
        return accel_meas, gyro_meas