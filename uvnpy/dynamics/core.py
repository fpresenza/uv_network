#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
"""
import numpy as np


class EulerIntegrator(object):
    def __init__(self, x, t=0.0, u=np.zeros(1)):
        self.initialize(x, t, u)

    def initialize(self, x, t=0.0, u=np.zeros(1)):
        self.t = t
        self._x = x.copy()
        self._u = u.copy()

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
    def __init__(self, x, dotx, t=0.0, u=np.zeros(1)):
        self.initialize(x, dotx, t, u)

    def initialize(self, x, dotx, t=0.0, u=np.zeros(1)):
        self.t = t
        self._x = x.copy()
        self._dotx = dotx.copy()
        self._u = u.copy()

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
    def __init__(self, x, t=0.0, u=2*[np.zeros(1)]):
        self.initialize(x, t, u)

    def initialize(self, x, t=0.0, u=2*[np.zeros(1)]):
        self.t = t
        self._x = x.copy()
        self._u = u.copy()

    def x(self):
        return self._x.copy()

    def u(self):
        return self._u.copy()

    def step(self, t, u):
        dt = t - self.t
        self._x += dt * sum(self._u) * 0.5
        self.t = t
        self._u[0] = self._u[1]
        self._u[1] = u
