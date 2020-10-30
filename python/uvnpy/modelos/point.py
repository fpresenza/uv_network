#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:41:07 2020
@author: fran
"""
import numpy as np
import collections
from types import SimpleNamespace

from gpsic.analisis.core import cargar_yaml
from gpsic.toolkit import linalg
from gpsic.modelos.discreto import SistemaDiscreto

from . import vehiculo, control_velocidad
from uvnpy.sensores import rango
from uvnpy.filtering import consenso, kalman, metricas
from uvnpy.filtering import ajustar_sigma
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
          t=[self.t],
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
        self.log_dict.t.append(self.t)
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
      self, nombre,
      arxiv='/tmp/point.yaml',
      pi=np.zeros(2), vi=np.zeros(2)):
        super(point, self).__init__(nombre, tipo='point')

        # leer archivo de configuración
        config = cargar_yaml(arxiv)

        # dinamica del vehiculo
        din_kw = ajustar_sigma(config['din'])
        din_kw.update(pi=pi, vi=vi)
        self.din = control_velocidad(**din_kw)

        # sensores
        rango_kw = config['sensor']['rango']
        self.rango = rango.sensor(**rango_kw)

        # filtro
        x0 = self.din.x
        dx0 = np.ones(4)
        self.filtro = point_loc_ekf(
            x0, dx0,
            self.din.Q,
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

        # intercambio de información
        self.promedio = consenso.promedio()
        self.lpf = consenso.lpf()
        self.inbox = collections.deque(maxlen=10)
        self.outbox = {'id': self.id}

    def control_step(self, t, landmarks=[]):
        self.filtro.prediccion(t, self.control.u, 'control')
        if len(landmarks) > 0:
            range_meas = [self.rango(self.din.p, lm) for lm in landmarks]
            self.filtro.correccion(range_meas, 'rango', landmarks)
        self.control.update(self.filtro.p, t, (landmarks, self.rango.sigma))

    def iniciar_consenso(self, avg=None, lpf=None):
        if avg is not None:
            self.promedio.iniciar(avg)
            self.outbox.update(avg=avg)
        if lpf is not None:
            self.lpf.iniciar(lpf['x'])
            self.outbox.update(lpf=lpf)
        self.inbox.clear()

    def consenso_step(self, t, u=None):
        avg_x_j = [msg.get('avg') for msg in self.inbox]
        if not np.isin(None, avg_x_j):
            self.promedio(t, avg_x_j)
            self.outbox.update(avg=self.promedio.x)
        if u is not None:
            lpf_x_j = [msg['lpf']['x'] for msg in self.inbox]
            lpf_u_j = [msg['lpf']['u'] for msg in self.inbox]
            self.lpf(t, u, lpf_x_j, lpf_u_j)
            self.outbox.update(
                lpf={
                    'x': self.lpf.x,
                    'u': u})
        self.inbox.clear()
