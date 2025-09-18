#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
"""
import numpy as np


class EulerIntegrator(object):
    def __init__(self, x, t=0.0):
        self.initialize(x, t)

    def initialize(self, x, t=0.0, u=None):
        self.t = t
        self._x = x.copy()
        if u is None:
            self._u = np.zeros_like(x)

    def x(self):
        return self._x.copy()

    def u(self):
        return self._u.copy()

    def step(self, t, u):
        dt = t - self.t
        self._x += dt * self._u
        self.t = t
        self._u = u


class DoubleEulerIntegrator(object):
    def __init__(self, x, dotx, t=0.0):
        self.initialize(x, dotx, t)

    def initialize(self, x, dotx, t=0.0, u=None):
        self.t = t
        self._x = x.copy()
        self._dotx = dotx.copy()
        if u is None:
            self._u = np.zeros_like(x)

    def x(self):
        return self._x.copy()

    def dotx(self):
        return self._dotx.copy()

    def u(self):
        return self._u.copy()

    def step(self, t, u):
        dt = t - self.t
        self._x += dt * self._dotx
        self._dotx += dt * self._u
        self.t = t
        self._u = u


class HeunIntegrator(object):
    def __init__(self, x, t=0.0):
        self.initialize(x, t)

    def initialize(self, x, t=0.0, u=None):
        self.t = t
        self._x = x.copy()
        if u is None:
            self._u = np.zeros_like(x, shape=(2,) + x.shape)

    def x(self):
        return self._x.copy()

    def u(self):
        return self._u.copy()

    def step(self, t, u):
        dt = t - self.t
        self._x += dt * (self._u[0] + self._u[1]) * 0.5
        self.t = t
        self._u[0] = self._u[1]
        self._u[1] = u
