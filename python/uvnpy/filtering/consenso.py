#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date vie oct 30 10:09:44 -03 2020
"""
import numpy as np

from gpsic.integradores import EulerExplicito


class promedio(EulerExplicito):
    def __init__(self, xi=[0.], ti=0.):
        super(promedio, self).__init__(xi, ti)

    @staticmethod
    def dinamica(x, t, x_j):
        d_i = len(x_j)
        return - d_i * x + np.sum(x_j, axis=0)


class lpf(EulerExplicito):
    def __init__(self, xi=[0.], ti=0.):
        super(lpf, self).__init__(xi, ti)

    @staticmethod
    def dinamica(x, t, u_i, x_j, u_j):
        d_i = len(x_j)
        sum_xj = np.sum(x_j, axis=0)
        sum_uj = np.sum(u_j, axis=0)
        return - (1 + 2 * d_i) * x + (u_i + sum_xj + sum_uj)


class comparador(object):
    def __init__(self, xi=[0.], ui=[0.], funcion=None):
        self.iniciar(xi, ui, funcion)

    @property
    def x(self):
        return self._x

    @property
    def u(self):
        return self._u

    def iniciar(self, xi, ui, funcion):
        self._x = np.copy(xi)
        self._u = np.copy(ui)
        self.f = funcion
        self.flag = True

    def step(self, x_j, u_j):
        x = self._x
        x_aug = [self._x] + x_j
        u_aug = [self._u] + u_j
        f_val = self.f(x_aug)
        f_idx = x_aug.index(f_val)
        self._x = np.copy(f_val)
        self._u = u_aug[f_idx]
        if not np.isclose(self._x, x):
            self.flag = False
        return self._x, self._u, self.flag
