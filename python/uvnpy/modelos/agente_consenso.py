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

from . import vehiculo, control_velocidad
from uvnpy.sensores import rango
from uvnpy.filtering import consenso, kalman, metricas
from uvnpy.redes import mensajeria


__all__ = ['agente_consenso']


class EKF(kalman.KF):
    def __init__(self, x, dx, Q, R, ti=0.):
        """ Filtro de localización de un vehículo control_velocidad """
        super(EKF, self).__init__(x, dx, ti=0.)
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

    def f(self, dt, x, u):
        Phi = self.Id + dt * self.F_x
        x_prior = np.matmul(Phi, x) + np.matmul(self.F_u, u) * dt
        Q = self.Q * (dt**2)
        return x_prior, Phi, Q

    def modelo_rango(self, landmarks):
        #   medición esperada
        p = self.x[:2]
        hat_z = [linalg.dist(p, lm) for lm in landmarks]
        #   Jacobiano
        Hp = np.vstack([rango.gradiente(p, lm) for lm in landmarks])
        H = np.hstack([Hp, np.zeros_like(Hp)])
        #   Ruido
        R = np.diag([self.R for _ in landmarks])
        return H, R, hat_z

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


class agente_consenso(vehiculo):
    def __init__(
      self, nombre,
      arxiv='/tmp/agente_consenso.yaml',
      pi=np.zeros(2), vi=np.zeros(2)):
        super(agente_consenso, self).__init__(nombre, tipo='agente_consenso')

        # leer archivo de configuración
        config = cargar_yaml(arxiv)

        # dinamica del vehiculo
        din_kw = config['din']
        din_kw.update(pi=pi, vi=vi)
        self.din = control_velocidad(**din_kw)

        # sensores
        rango_kw = config['sensor']['rango']
        self.rango = rango.sensor(**rango_kw)

        # filtro
        x0 = self.din.x
        dx0 = np.ones(4)
        self.filtro = EKF(
            x0, dx0,
            self.din.Q,
            self.rango.R
        )
        # intercambio de mensajes
        self.box = mensajeria.box(out={'id': self.id}, maxlen=30)
        self.promedio = []
        self.lpf = consenso.lpf()
        self.comparador = consenso.comparador()

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
