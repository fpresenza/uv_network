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
from uvnpy.model.holonomic import ControlVelocidad
import uvnpy.navigation.kalman as kalman
from uvnpy.sensor.rango import Rango
from uvnpy.navigation.control import MPCInformativo
import uvnpy.navigation.metricas as metricas


class PointLocalization(kalman.EKF):
    def __init__(self, x, dx, Q, R, ti=0.):
        """ Filtro de localización de un vehículo ControlVelocidad """
        super(PointLocalization, self).__init__(x, dx, ti=0.)
        self.dof = int(len(x) / 2)
        Id = np.identity(2)
        Z = np.zeros_like(Id)
        K = 3 * Id
        self.F_x = np.block([[Z, Id],
                             [Z, -K]])
        self.F_u = np.block([[Z],
                             [K]])
        self.F_z = np.block([[Z, Z],
                             [Id, -K]])
        self.Q = Q
        self.R = R

    @property
    def p(self):
        return self.x[:self.dof]

    @property
    def v(self):
        return self.x[self.dof:]

    def control(self, t, u):
        x = self.x
        dot_x = np.matmul(self.F_x, x) + np.matmul(self.F_u, u)
        return dot_x, self.F_x, self.F_z, self.Q

    def rango(self, landmarks):
        #   medición esperada
        p = self.p
        hat_y = [linalg.dist(p, lm) for lm in landmarks]
        #   Jacobiano
        Hp = np.vstack([Rango.gradiente(p, lm) for lm in landmarks])
        H = np.hstack([Hp, np.zeros_like(Hp)])
        #   Ruido
        R = self.R * np.identity(len(landmarks))
        return hat_y, H, R

    def logs(self):
        """ Historia del filtro. """
        return [self.time,
                self.x,
                metricas.sqrt_diagonal(self.P),
                metricas.eigvalsh(self.P)]


class Point(UnmannedVehicle):
    def __init__(
            self, name,
            config_dir='/tmp/', config_file='point.yaml',
            mov_kw={}, sensor_kw={}):
        super(Point, self).__init__(name, type='Point')

        # leer archivo de configuración
        file_path = '{}{}'.format(config_dir, config_file)
        config = yaml.load(open(file_path))

        # dinamica del vehiculo
        freq = mov_kw.get('freq', 1.)
        sigma = config.get('sigma')
        config.update(
            sigma=np.multiply(freq**0.5, sigma)
        )
        mov_kw.update(config)
        self.mov = ControlVelocidad(**mov_kw)

        # sensores
        range_file = sensor_kw.get('range_file', 'xbee.yaml')
        file_path = '{}{}'.format(config_dir, range_file)
        config = yaml.load(open(file_path))
        sensor_kw.update(config)
        self.rango = Rango(**sensor_kw)

        # filtro
        x0 = self.c
        dx0 = np.ones(4)
        self.filtro = PointLocalization(
            x0, dx0,
            self.mov.Q,
            self.rango.R
        )

        # control
        self.control = MPCInformativo(
            (1, 4.5, -2000),
            (np.eye(2), np.eye(2), np.eye(2)),
            SistemaDiscreto,
            control_dim=2,
            horizonte=np.linspace(0.1, 1, 10)
        )

    @property
    def c(self):
        return np.hstack([self.mov.p, self.mov.v])

    def control_step(self, t, landmarks=[]):
        self.filtro.prediccion(t, self.control.u, 'control')
        if len(landmarks) > 0:
            range_meas = [
              self.rango.measurement(self.mov.p, lm) for lm in landmarks]
            self.filtro.correccion(range_meas, 'rango', landmarks)
        self.control.update(self.filtro.p, t, (landmarks, self.rango.sigma))

    def mov_step(self, t):
        self.mov.step(t, self.control.u)
