#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue nov 19 14:06:16 -03 2020
"""
import numpy as np

from gpsic.integradores import EulerExplicito


class integrador(EulerExplicito):
    def __init__(self, xi=[0.], ti=0.):
        """ Modelo de vehículo integrador. """
        super(integrador, self).__init__(xi, ti)
        self._dx = np.zeros_like(xi, dtype=float)

    @property
    def x(self):
        return self._x.copy()

    @property
    def dx(self):
        return self._dx.copy()

    def dinamica(self, x, t, u):
        self._dx = u
        return self._dx

    def step(self, t, u):
        x = super(integrador, self).step(t, ([u], ))
        return x


class integrador_ruidoso(integrador):
    def __init__(self, xi, Q, ti=0.):
        super(integrador_ruidoso, self).__init__(xi, ti)
        self.Q = np.asarray(Q)

    def dinamica(self, x, t, u):
        self._dx = np.random.multivariate_normal(u, self.Q)
        return self._dx


class doble_integrador(EulerExplicito):
    def __init__(self, xi=[0.], ti=0.):
        """ Modelo de vehículo doble integrador. """
        super(doble_integrador, self).__init__(xi, ti)
        self._dx = np.zeros_like(xi, dtype=float)

    @property
    def x(self):
        return self._x.copy()

    @property
    def dx(self):
        return self._dx.copy()

    def dinamica(self, x, t, u):
        n = len(u)
        self._dx[:n] = x[n:]
        self._dx[n:] = np.asarray(u)
        return self._dx

    def step(self, t, u):
        x = super(doble_integrador, self).step(t, ([u], ))
        return x
