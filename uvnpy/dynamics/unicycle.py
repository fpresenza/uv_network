#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
"""
import numpy as np

from .core import EulerIntegrator


class Unicycle(object):
    def __init__(self, x, t=0.0):
        """
        args:
        -----
            x = (px, py, theta)
        """
        self.int = EulerIntegrator(x, t)

    def pose(self):
        return self.int.x()

    def vel(self):
        return self.int.u()

    def f(self, v, w):
        """
        args:
        -----
            v = heading velocity
            w = angular velocity
        """
        th = self.int._x[2]
        return np.array([v * np.cos(th), v * np.sin(th), w])

    def step(self, t, v, w):
        dotx = self.f(v, w)
        self.int.step(t, dotx)
