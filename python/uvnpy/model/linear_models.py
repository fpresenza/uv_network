#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue nov 19 14:06:16 -03 2020
"""
import numpy as np
import collections


class integrator(object):
    def __init__(self, xi, ti=0., order=1):
        """Modelo de vehiculo integrador."""
        self.init(xi, ti)
        self.derivatives = collections.deque(maxlen=order)
        self.coefficients = {
            1: np.array([1]),             # euler
            2: np.array([1/2, 1/2])}      # heun

    def init(self, xi, ti=0.):
        self.t = ti
        self._x = xi.copy()

    @property
    def x(self):
        return self._x.copy()

    def step(self, t, u):
        dt = t - self.t
        self.t = t
        self.derivatives.append(u)
        k = self.coefficients[len(self.derivatives)]
        self._x += dt * k.dot(self.derivatives)
        return self._x.copy()


class random_walk(integrator):
    def __init__(self, xi, Q, ti=0., order=1):
        super(random_walk, self).__init__(xi, ti, order)
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


class double_integrator(object):
    def __init__(self, xi, vi, ti=0.):
        """ Modelo de vehiculo doble integrador. """
        self._dx = np.zeros_like(xi, dtype=float)
        self.init(xi, vi, ti)

    def init(self, xi, vi, ti=0.):
        self.t = ti
        self._x = xi.copy()
        self._v = vi.copy()

    @property
    def x(self):
        return self._x.copy()

    @property
    def v(self):
        return self._v.copy()

    def step(self, t, u):
        dt = t - self.t
        self.t = t
        self._x += dt * self._v
        self._v += dt * u
        return self.x, self.v
