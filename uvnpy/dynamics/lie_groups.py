#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
"""
import numpy as np
import scipy.linalg

from uvnpy.dynamics.core import EulerIntegrator
from uvnpy.toolkit.geometry import rotation_matrix_from_vector


class EulerIntegratorLieGroup(EulerIntegrator):
    def exp_map(self, h):
        return scipy.linalg.expm(h)

    def step(self, t, u):
        dt = t - self.t
        self._x = self.exp_map(dt * self._u).dot(self._x)
        self.t = t
        self._u = u


class EulerIntegratorOrtogonalGroup(EulerIntegratorLieGroup):
    def exp_map(self, h):
        if np.allclose(h, 0.0):
            dx = np.eye(3)
        else:
            dx = rotation_matrix_from_vector(h)
        return dx
