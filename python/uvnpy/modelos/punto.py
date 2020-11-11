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

from . import vehiculo, integrador
from uvnpy.sensores import rango
from uvnpy.filtering import kalman, metricas
from uvnpy.control.informativo import minimizar
from uvnpy.redes import mensajeria


__all__ = ['punto']


class integrador_ruidoso(integrador):
    def __init__(self, xi, Q, ti=0.):
        super(integrador_ruidoso, self).__init__(xi, ti)
        self.Q = np.asarray(Q)

    def dinamica(self, x, t, u):
        self._dx = np.random.multivariate_normal(u, self.Q)
        return self._dx


class ekf_autonomo(kalman.KF):
    def __init__(self, x, dx, Q, R, ti=0.):
        """Filtro de localización autónomo. """
        super(ekf_autonomo, self).__init__(x, dx, ti=0.)
        self.Id = np.identity(2)
        self.Q = Q
        self.R = R
        self.logs = SimpleNamespace(
            t=[self.t],
            x=[self.x],
            dvst=[metricas.sqrt_diagonal(self.P)],
            eigs=[metricas.eigvalsh(self.P)])

    @property
    def p(self):
        return self.x

    def f(self, dt, x, u):
        Phi = self.Id
        x_prior = x + np.multiply(dt, u)
        Q = self.Q * (dt**2)
        return x_prior, Phi, Q

    def modelo_rango(self, landmarks):
        p = self.x
        hat_z = rango.distancia(p, landmarks)
        H = rango.jacobiano(p, landmarks)
        R = self.R * np.eye(len(landmarks))
        return H, R, hat_z

    def update(self, t, u, rangos, landmarks):
        self.prediccion(t, u)
        H, R, hat_z = self.modelo_rango(landmarks)
        self.actualizacion(rangos, H, R, hat_z)

    def guardar(self):
        """Guarda los últimos datos. """
        self.logs.t.append(self.t)
        self.logs.x.append(self.x)
        self.logs.dvst.append(
            metricas.sqrt_diagonal(self.P))
        self.logs.eigs.append(
            metricas.eigvalsh(self.P))


det_innovacion = {
    'controlador': minimizar,
    'modelo': integrador,
    'metrica': metricas.det,
    'matriz': rango.matriz_innovacion,
    'Q': (np.eye(2), 4.5*np.eye(2), -100),
    'dim': 2,
    'horizonte': np.linspace(0.1, 1, 10)
}


class punto(vehiculo):
    def __init__(
            self, nombre,
            filtro=None, control=None,
            arxiv='/tmp/punto.yaml',
            pi=np.zeros(2)):
        super(punto, self).__init__(nombre, tipo='punto')

        # archivo de configuración
        config = cargar_yaml(arxiv)
        Q = config['din']['Q']      # dinamica del vehiculo
        self.din = integrador_ruidoso(pi, Q)
        sigma = config['rango']['sigma']    # rango
        self.rango = rango.sensor(sigma)

        if filtro is not None:
            xi = self.din.x
            dxi = np.ones(2)
            Q = self.din.Q
            R = self.rango.R
            self.filtro = filtro(xi, dxi, Q, R)

        if control is not None:
            kwargs = control.copy()
            controlador = kwargs.pop('controlador')
            self.control = controlador(**kwargs)

        # intercambio de mensajes
        self.box = mensajeria.box(out={'id': self.id}, maxlen=30)
