#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date lun nov  9 10:51:25 -03 2020
"""
import numpy as np
from types import SimpleNamespace

from gpsic.analisis.core import cargar_yaml
from gpsic.modelos.discreto import SistemaDiscreto

from . import vehiculo, integrador
from uvnpy.sensores import rango
from uvnpy.filtering import kalman, metricas
from uvnpy.control import mpc_informativo
from uvnpy.redes import mensajeria


__all__ = ['punto']


class ekf_autonomo(kalman.KF):
    def __init__(self, x, dx, Q, R, ti=0.):
        """Filtro de localización autónomo. """
        super(ekf_autonomo, self).__init__(x, dx, ti=0.)
        self.Id = np.identity(2)
        self.Q = Q
        self.R = R
        self._logs = SimpleNamespace(
            t=[self.t],
            x=[self.x],
            dvst=[metricas.sqrt_diagonal(self.P)],
            eigs=[metricas.eigvalsh(self.P)])

    @property
    def p(self):
        return self.x

    @property
    def logs(self):
        """Historia del filtro. """
        return self._logs

    def f(self, dt, x, u):
        Phi = self.Id
        x_prior = x + np.multiply(dt, u)
        Q = self.Q * (dt**2)
        return x_prior, Phi, Q

    def modelo_rango(self, landmarks):
        p = self.x
        hat_z = [rango.distancia(p, lm) for lm in landmarks]
        H = np.vstack([rango.gradiente(p, lm) for lm in landmarks])
        R = np.diag([self.R for _ in landmarks])
        return H, R, hat_z

    def update(self, t, u, rangos, landmarks):
        self.prediccion(t, u)
        H, R, hat_z = self.modelo_rango(landmarks)
        self.actualizacion(rangos, H, R, hat_z)

    def guardar(self):
        """Guarda los últimos datos. """
        self._logs.t.append(self.t)
        self._logs.x.append(self.x)
        self._logs.dvst.append(
            metricas.sqrt_diagonal(self.P))
        self._logs.eigs.append(
            metricas.eigvalsh(self.P))


class mpc_informativo_scipy(mpc_informativo.controlador):
    def __init__(self):
        super(mpc_informativo_scipy, self).__init__(
            mpc_informativo.det_matriz_innovacion,
            (1, 4.5, -2000),
            (np.eye(2), np.eye(2), np.eye(2)),
            SistemaDiscreto,
            control_dim=2,
            horizonte=np.linspace(0.1, 1, 10)
        )


class punto(vehiculo):
    def __init__(
            self, nombre,
            filtro=None, controlador=None,
            arxiv='/tmp/punto.yaml',
            pi=np.zeros(2)):
        super(punto, self).__init__(nombre, tipo='punto')

        # archivo de configuración
        config = cargar_yaml(arxiv)
        S = config['din']['sigma']      # dinamica del vehiculo
        self.din = integrador(pi, Q=np.matmul(S, S))
        S = config['rango']['sigma']    # rango
        self.rango = rango.sensor(sigma=S)

        # filtro
        x0 = self.din.x
        dx0 = np.ones(2)
        self.filtro = filtro(
            x0, dx0,
            self.din.Q,
            self.rango.R
        )

        # control
        self.control = controlador()

        # intercambio de mensajes
        self.box = mensajeria.box(out={'id': self.id}, maxlen=30)
