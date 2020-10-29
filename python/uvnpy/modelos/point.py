#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:41:07 2020
@author: fran
"""
import numpy as np
import yaml
from types import SimpleNamespace

from gpsic.toolkit import linalg
from gpsic.modelos.discreto import SistemaDiscreto

from . import vehiculo, control_velocidad
from uvnpy.sensores import rango
from uvnpy.filtering import kalman, metricas
from uvnpy.control import mpc_informativo


__all__ = ['point']


class point_loc_ekf(kalman.EKF):
    def __init__(self, x, dx, Q, R, ti=0.):
        """ Filtro de localización de un vehículo control_velocidad """
        super(point_loc_ekf, self).__init__(x, dx, ti=0.)
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
        self.log_dict = SimpleNamespace(
          t=[self.time],
          x=[self.x],
          dvst=[metricas.sqrt_diagonal(self.P)],
          eigs=[metricas.eigvalsh(self.P)])

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
        Hp = np.vstack([rango.gradiente(p, lm) for lm in landmarks])
        H = np.hstack([Hp, np.zeros_like(Hp)])
        #   Ruido
        R = self.R * np.identity(len(landmarks))
        return hat_y, H, R

    def guardar(self):
        """Guarda los últimos datos. """
        self.log_dict.t.append(self.time)
        self.log_dict.x.append(self.x)
        self.log_dict.dvst.append(
          metricas.sqrt_diagonal(self.P))
        self.log_dict.eigs.append(
                metricas.eigvalsh(self.P))

    @property
    def logs(self):
        """Historia del filtro. """
        return self.log_dict


class point(vehiculo):
    def __init__(
            self, name,
            config_dir='/tmp/', config_file='point.yaml',
            kin_kw={}, sensor_kw={}):
        super(point, self).__init__(name, type='point')

        # leer archivo de configuración
        file_path = '{}{}'.format(config_dir, config_file)
        config = yaml.load(open(file_path))

        # dinamica del vehiculo
        freq = kin_kw.get('freq', 1.)
        sigma = config.get('sigma')
        config.update(
            sigma=np.multiply(freq**0.5, sigma)
        )
        kin_kw.update(config)
        self.kin = control_velocidad(**kin_kw)

        # sensores
        range_file = sensor_kw.get('range_file', 'xbee.yaml')
        file_path = '{}{}'.format(config_dir, range_file)
        config = yaml.load(open(file_path))
        sensor_kw.update(config)
        self.rango = rango.sensor(**sensor_kw)

        # filtro
        x0 = self.kin.x
        dx0 = np.ones(4)
        self.filtro = point_loc_ekf(
            x0, dx0,
            self.kin.Q,
            self.rango.R
        )

        # control
        self.control = mpc_informativo.controlador(
            mpc_informativo.det_delta_informacion,
            (1, 4.5, -2000),
            (np.eye(2), np.eye(2), np.eye(2)),
            SistemaDiscreto,
            control_dim=2,
            horizonte=np.linspace(0.1, 1, 10)
        )

    def control_step(self, t, landmarks=[]):
        self.filtro.prediccion(t, self.control.u, 'control')
        if len(landmarks) > 0:
            range_meas = [self.rango(self.kin.p, lm) for lm in landmarks]
            self.filtro.correccion(range_meas, 'rango', landmarks)
        self.control.update(self.filtro.p, t, (landmarks, self.rango.sigma))
