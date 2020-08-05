#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:41:07 2020
@author: fran
"""
import numpy as np
import yaml
import gpsic.toolkit.linalg as linalg
from gpsic.modelos.discreto import SistemaDiscreto
from uvnpy.vehicles.uv import UnmannedVehicle
from uvnpy.model.holonomic import Integrador, ControlVelocidad
import uvnpy.navigation.ekf as ekf
from uvnpy.sensor.rango import Rango
from uvnpy.navigation.control import MPCInformativo
import uvnpy.navigation.metricas as metricas


class PointLocalization(ekf.EKF):
    """ Esta clase provee un filtro de localización de un modelo puntual de vehículo
    cuyo modelo de velocidad es un integrador puro:
        - Paso de predicción: 
            toma: acción de control
            retorna: estado actualizado a priori
        - Paso de corrección: 
            toma: medición de rango
            retorna: estado actualizado a posteriori
    """
    def __init__(self, x, dx, Q, R, ti=0.):
        super(PointLocalization, self).__init__(x, dx, ti=0.)
        I = np.identity(2)
        O = np.zeros_like(I)
        K = 3 * I 
        self.F_x = np.block([[O,  I],
                             [O, -K]])
        self.F_r = np.block([[O],
                             [K]])
        self.F_e = np.block([[O,  O],
                             [I, -K]])
        self.Q = Q
        self.R = R

    def f(self, t, r):
        x = self.x
        dot_x = np.matmul(self.F_x, x) + np.matmul(self.F_r, r)
        return dot_x, self.F_x, self.F_e, self.Q
    
    def h(self, landmarks):
        x = self.x
        #   Expected measurement
        p = x[:2]
        hat_y = [linalg.distance(p, l) for l in landmarks]
        #   Jacobian
        Hp = np.vstack([Rango.gradiente(p, l) for l in landmarks])
        H = np.hstack([Hp, np.zeros_like(Hp)])
        #   Noise
        R = self.R * np.identity( len(landmarks) )
        return hat_y, H, R

    def logs(self):
        """ Este modulo guarda la historia del filtro:
        t:  tiempo
        x:  estado
        P:  covarianza
        dy: innovación
        """
        return [self.time, self.x, metricas.sqrt_diagonal(self.P), metricas.eigvals(self.P)]


class Point(UnmannedVehicle):
    def __init__(self, name, config_dir='/home/fran/repo/uv_network/config/', config_file='point.yaml',
     motion_kw={}, sensor_kw={}):
        super(Point, self).__init__(name, type='Point')

        # read config file
        file_path = f'{config_dir}{config_file}'
        config = yaml.load(open(file_path))
        # dinamica del vehiculo
        freq = motion_kw.get('freq', 1)
        sigma = config.get('sigma')
        config.update(
            sigma=np.multiply(freq**0.5, sigma)
        )
        motion_kw.update(config)
        self.motion = ControlVelocidad(**motion_kw)

        # sensors
        range_file = sensor_kw.get('range_file', 'xbee.yaml')
        file_path = f'{config_dir}{range_file}'
        config = yaml.load(open(file_path))
        sensor_kw.update(config)
        self.rango = Rango(**sensor_kw)

        # filter
        x0 = self.kin
        dx0 = np.ones(4)
        self.filter = PointLocalization(
            x0, dx0,
            self.motion.Q,
            self.rango.R
        )

        # control
        self.control = MPCInformativo(
            (1,4.5,-2000), 
            (np.eye(2), np.eye(2), np.eye(2)),
            SistemaDiscreto,
            control_dim=2,
            horizonte=np.linspace(0.1, 1, 10)
        )

    @property
    def kin(self):
        return np.hstack([self.motion.p, self.motion.v])
    
    def control_step(self, t, landmarks=[]):
        self.filter.prediction(t, self.control.u)
        if len(landmarks) > 0: 
            range_meas = [self.rango.measurement(self.motion.p, l) for l in landmarks]
            self.filter.correction(range_meas, landmarks)
        self.control.update(self.filter.x[:2], t, (landmarks, self.rango.sigma))

    def motion_step(self, t):
        self.motion.step(t, self.control.u)