#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date vie oct 30 10:09:44 -03 2020
"""
import numpy as np


class consenso(object):
    def __init__(self, xi=[0.], ti=0.):
        """ Protocolos de consenso distribuído """
        self.iniciar(xi, ti)

    def iniciar(self, xi, ti=0.):
        self.xi = np.copy(xi)
        self._x = self.xi.copy()
        self.t = ti

    def reiniciar(self):
        """Reinicia el protocolo"""
        self._x = np.copy(self.xi)
        self.t = np.copy(self.ti)

    @property
    def x(self):
        return self._x.copy()


class promedio(consenso):
    def __init__(self, xi=[0.], ti=0.):
        super(promedio, self).__init__(xi, ti)

    def __call__(self, t, x_j):
        Ts = t - self.t
        self.t = t
        d_i = len(x_j)
        self._x = (1 - Ts * d_i) * self._x + Ts * np.sum(x_j, axis=0)


class lpf(consenso):
    def __init__(self, xi=[0.], ti=0.):
        super(lpf, self).__init__(xi, ti)

    def __call__(self, t, u_i, x_j, u_j):
        Ts = t - self.t
        self.t = t
        d_i = len(x_j)
        sum_xj = np.sum(x_j, axis=0)
        sum_uj = np.sum(u_j, axis=0)
        A = (1 - (1 + 2 * d_i) * Ts)
        self._x = A * self._x + Ts * (u_i + sum_xj + sum_uj)
