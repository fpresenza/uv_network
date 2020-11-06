#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:11:56 2020
@author: fran
"""
import numpy as np

from gpsic.integradores import EulerExplicito
from gpsic.modelos.discreto import SistemaDiscreto


class vehiculo(object):
    """ This class implements a unmanned vehicle instance
    to use as node in a graph. """
    def __init__(self, nombre, tipo='vehiculo', **kwargs):
        self.id = nombre
        self.tipo = tipo

    def __str__(self):
        return '{}({})'.format(self.tipo, self.id)


class integrador(EulerExplicito):
    def __init__(self, xi, ti=0., Q=None):
        """ Modelo de vehículo integrador.

        Argumentos:

            dof: grados de libertad
            xi: posición inicial
            Q: matriz de covarianza del ruido
                a la entrada.
        """
        super(integrador, self).__init__(xi=xi, ti=ti)
        self._v = np.zeros_like(xi)
        if Q is None:
            n = len(xi)
            Q = np.zeros((n, n))
        self.Q = Q

    @property
    def p(self):
        return self._x.copy()

    @property
    def v(self):
        return self._v.copy()

    def dinamica(self, x, t, u):
        self._v = np.random.multivariate_normal(u, self.Q)
        return self._v

    def step(self, t, u):
        super(integrador, self).step(t, ([u], ))


class control_velocidad(SistemaDiscreto):
    def __init__(
            self,
            pi=[0.], vi=None, ti=0.,
            K=[1.], sigma=[[0.], [0.]]):
        """ Modelo de cinemática con control de velocidad.

        Argumentos:

            dof: grados de libertad
            pi, vi:  posición, velocidad inicial
            ctrl_matrix:  controlador interno
            sigma: (std. dev. en la acción de los actuadores,
                    std. dev. en la medicion la velocidad)
        """
        self.iniciar(pi, vi=vi, ti=ti)
        self.dof = len(pi)
        self._a = np.zeros_like(pi)
        self.sigma = np.hstack(sigma)
        Id = np.identity(self.dof)
        Z = np.zeros_like(Id)
        K = np.asarray(K)
        #   matrices del sistema
        self.G_x = np.block([[Z, Id],
                             [Z, -K]])
        self.G_u = np.block([[Z],
                             [K]])
        self.G_z = np.block([[Z, Z],
                             [Id, -K]])
        #   matriz de covarianza del ruido
        self.Q = np.diag(np.square(self.sigma))

    def iniciar(self, pi, vi=None, ti=0.):
        if vi is None:
            vi = np.zeros_like(pi)
        xi = np.hstack([pi, vi])
        super(control_velocidad, self).iniciar(xi=xi, ti=ti)

    @property
    def p(self):
        return self._x[:self.dof].copy()

    @property
    def v(self):
        return self._x[self.dof:].copy()

    @property
    def a(self):
        return self._a.copy()

    @property
    def x(self):
        return self._x.copy()

    def dinamica(self, x, t, u):
        z = np.random.normal(0, self.sigma)
        dot_x = np.matmul(self.G_x, x) + \
            np.matmul(self.G_u, u) + np.matmul(self.G_z, z)
        self._a = dot_x[self.dof:]
        return dot_x


class velocidad_rw(SistemaDiscreto):
    def __init__(self, ti=0., dof=1, **kwargs):
        """ Modelo de proceso de caminata aleatoria o Wiener en velocidad.

        Argumentos:

            dof: <int> grados de libertad
            pi, vi: <list o tuple o array> posición, velocidad inicial
            sigma: <list o tuple o array> std. dev. en señal de entrada
        """
        pi = kwargs.get('pi', np.zeros(dof))
        vi = kwargs.get('vi', np.zeros(dof))
        self.dof = dof
        self._p = pi
        self._v = vi
        self.sigma = kwargs.get('sigma', np.ones(dof))
        Id = np.identity(dof)
        Z = np.zeros_like(Id)
        #   matrices del sistema
        self.G_x = np.block([[Z, Id],
                             [Z, Z]])
        self.G_z = np.block([[Z],
                             [Id]])
        #   matriz de covarianza del ruido de entrada
        self.Q = np.diag(np.square(self.sigma))

    @property
    def p(self):
        return self._x[:self.dof].copy()

    @property
    def v(self):
        return self._x[self.dof:].copy()

    @property
    def x(self):
        return self._x.copy()

    def dinamica(self, x, t, u):
        z = np.random.normal(0, self.sigma)
        return np.hstack([x[self.dof:], z])
