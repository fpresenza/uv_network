#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue nov 19 14:06:16 -03 2020
"""
import numpy as np

# from gpsic.integradores import EulerExplicito


class integrator(object):
    def __init__(self, xi, ti=0.):
        """Modelo de vehiculo integrador."""
        self.init(xi, ti)

    def init(self, xi, ti=0.):
        self.t = ti
        self._x = xi.copy()

    @property
    def x(self):
        return self._x.copy()

    def step(self, t, u):
        dt = t - self.t
        self.t = t
        self._x += dt * u
        return self._x.copy()


class random_walk(integrator):
    def __init__(self, xi, Q, ti=0.):
        super(random_walk, self).__init__(xi, ti)
        self.Q = Q
        self._dot_x = np.zeros(xi.shape)

    @property
    def dot_x(self):
        return self._dot_x.copy()

    def step(self, t, u):
        self._dot_x = np.random.multivariate_normal(u.ravel(), self.Q)
        self._dot_x = self._dot_x.reshape(u.shape)
        x = super(random_walk, self).step(t, self._dot_x)
        return x


# class doble_integrador(EulerExplicito):
#     def __init__(self, xi=[0.], ti=0.):
#         """ Modelo de vehiculo doble integrador. """
#         super(doble_integrador, self).__init__(xi, ti)
#         self._dx = np.zeros_like(xi, dtype=float)

#     @property
#     def x(self):
#         return self._x.copy()

#     @property
#     def dx(self):
#         return self._dx.copy()

#     def dinamica(self, x, t, u):
#         n = len(u)
#         self._dx[:n] = x[n:]
#         self._dx[n:] = np.asarray(u)
#         return self._dx

#     def step(self, t, u):
#         x = super(doble_integrador, self).step(t, ([u], ))
#         return x
