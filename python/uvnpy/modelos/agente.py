#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date nov  4 12:21:01 -03 2020
"""
import numpy as np
from types import SimpleNamespace

from gpsic.analisis.core import cargar_yaml
from gpsic.toolkit import linalg  # noqa
from gpsic.modelos.discreto import SistemaDiscreto

from . import vehiculo, control_velocidad
from uvnpy.sensores import rango
from uvnpy.filtering import consenso, kalman, metricas  # noqa
from uvnpy.filtering import ajustar_sigma
from uvnpy.control import mpc_informativo
from uvnpy.redes import mensajeria


__all__ = ['agente']


class dkfce(kalman.DKF):
    def __init__(self, x, dx, Q, R, ti=0.):
        super(dkfce, self).__init__(x, dx, ti=0.)
        self.Id = np.identity(4)
        Id = self.Id[:2, :2]
        Z = np.zeros_like(Id)
        self.F_x = np.block([[Z, Id],
                             [Z, Z]])
        B = np.block([[Z],
                      [Id]])
        self.Q = np.linalg.multi_dot([B, Q, B.T])
        self.R = R
        self._logs = SimpleNamespace(
          t=[self.t],
          x=[self.x],
          dvst=[metricas.sqrt_diagonal(self.P)],
          eigs=[metricas.eigvalsh(self.P)])

    @property
    def p(self):
        return self.x[:2]

    @property
    def v(self):
        return self.x[2:]

    def f(self, dt, u):
        x = self._x
        P = self._P
        Phi = self.Id + dt * self.F_x
        x_prior = np.matmul(Phi, x)
        P_prior = np.matmul(Phi, np.matmul(P, Phi.T)) + self.Q * (dt**2)
        return x_prior, P_prior

    # def h(self, observaciones):
        """Modelo de observación.

        args:
            observaciones = [(j, rango_j, y_j, Y_j, x_j)]
        """
        # p = self.p
        # hat_z = [linalg.dist(p, pv) for pv in vecinos]
        # y_i =
        # Hp = np.vstack([rango.gradiente(p, pv) for pv in vecinos])
        # H = np.hstack([Hp, np.zeros_like(Hp)])
        # R = np.diag([self.R for _ in vecinos])
        # return hat_y, Y

    def guardar(self):
        """Guarda los últimos datos. """
        self._logs.t.append(self.t)
        self._logs.x.append(self.x)
        self._logs.dvst.append(
          metricas.sqrt_diagonal(self.P))
        self._logs.eigs.append(
                metricas.eigvalsh(self.P))

    @property
    def logs(self):
        """Historia del filtro. """
        return self._logs


class agente(vehiculo):
    def __init__(
      self, nombre, num_vec,
      arxiv='/tmp/agente.yaml',
      pi=np.zeros(2), vi=np.zeros(2)):
        super(agente, self).__init__(nombre, tipo='agente')

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
        # x0 = self.din.x
        x0 = np.zeros(4 * num_vec)
        dx0 = 20 * np.ones(4 * num_vec)
        # Q = np.diag([])
        self.filtro = dkfce(
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
        self.box = mensajeria.box(out={'id': self.id}, maxlen=30)
        self.promedio = []

    def control_step(self, t, landmarks=[]):
        self.filtro.prediccion(t, self.control.u)
        if len(landmarks) > 0:
            rango_med = [self.rango(self.din.p, lm) for lm in landmarks]
            h = self.filtro.rango
            self.filtro.actualizacion(h, rango_med, landmarks)
        self.control.update(self.filtro.p, t, (landmarks, self.rango.sigma))
