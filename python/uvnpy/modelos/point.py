#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:41:07 2020
@author: fran
"""
import numpy as np
from types import SimpleNamespace

from gpsic.analisis.core import cargar_yaml
from gpsic.toolkit import linalg
from gpsic.modelos.discreto import SistemaDiscreto

from . import vehiculo, control_velocidad
from uvnpy.sensores import rango
from uvnpy.filtering import consenso, kalman, metricas
from uvnpy.filtering import ajustar_sigma
from uvnpy.control import mpc_informativo
from uvnpy.redes import mensajeria


__all__ = ['point']


class point_loc_ekf(kalman.KF):
    def __init__(self, x, dx, Q, R, ti=0.):
        """ Filtro de localización de un vehículo control_velocidad """
        super(point_loc_ekf, self).__init__(x, dx, ti=0.)
        self.Id = np.identity(4)
        Id = self.Id[:2, :2]
        Z = np.zeros_like(Id)
        K = 3 * Id
        self.F_x = np.block([[Z, Id],
                             [Z, -K]])
        self.F_u = np.block([[Z],
                             [K]])
        B = np.block([[Z, Z],
                      [Id, -K]])
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
        dot_x = np.matmul(self.F_x, x) + np.matmul(self.F_u, u)
        Phi = self.Id + dt * self.F_x
        x_prior = x + dt * dot_x
        P_prior = np.matmul(Phi, np.matmul(P, Phi.T)) + self.Q * (dt**2)
        return x_prior, P_prior

    def rango(self, landmarks):
        #   medición esperada
        p = self.p
        hat_z = [linalg.dist(p, lm) for lm in landmarks]
        #   Jacobiano
        Hp = np.vstack([rango.gradiente(p, lm) for lm in landmarks])
        H = np.hstack([Hp, np.zeros_like(Hp)])
        #   Ruido
        R = np.diag([self.R for _ in landmarks])
        return hat_z, H, R

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
        self.box = mensajeria.box(out={'id': self.id}, maxlen=30)
        # self.promedio = consenso.promedio()
        self.promedio = []
        self.lpf = consenso.lpf()
        self.comparador = consenso.comparador()

    def control_step(self, t, landmarks=[]):
        self.filtro.prediccion(t, self.control.u)
        if len(landmarks) > 0:
            rango_med = [self.rango(self.din.p, lm) for lm in landmarks]
            h = self.filtro.rango
            self.filtro.correccion(h, rango_med, landmarks)
        self.control.update(self.filtro.p, t, (landmarks, self.rango.sigma))

    def consenso_promedio_step(self, num, t):
        promedio = self.promedio[num]
        x_j = self.box.extraer(('avg', num))
        promedio.step(t, ([x_j], ))
        self.box.actualizar_salida(('avg', num), promedio.x)

    def consenso_lpf_step(self, t, u):
        x_j = self.box.extraer('lpf', 'x')
        u_j = self.box.extraer('lpf', 'u')
        self.lpf.step(t, ([u, x_j, u_j], ))
        self.box.actualizar_salida(
            'lpf', {'x': self.lpf.x, 'u': u})

    def consenso_comparador_step(self):
        x_j = self.box.extraer('comparador', 'x')
        u_j = self.box.extraer('comparador', 'u')
        self.comparador.step(x_j, u_j)
        self.box.actualizar_salida(
            'comparador', {'x': self.comparador.x, 'u': self.comparador.u})
