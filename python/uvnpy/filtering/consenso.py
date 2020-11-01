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

    def dinamica(self, x, t, x_j):
        d_i = len(x_j)
        return - d_i * x + np.sum(x_j, axis=0)


class lpf(EulerExplicito):
    def __init__(self, xi=[0.], ti=0.):
        super(lpf, self).__init__(xi, ti)

    def dinamica(self, x, t, u_i, x_j, u_j):
        d_i = len(x_j)
        sum_xj = np.sum(x_j, axis=0)
        sum_uj = np.sum(u_j, axis=0)
        return - (1 + 2 * d_i) * x + (u_i + sum_xj + sum_uj)
