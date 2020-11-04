#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue Jan 14 16:15:16 2020
@author: fran
"""
import numpy as np
from numpy.linalg import multi_dot, inv


def fusionar(informacion):
    """ Fusion de Kalman.

    Fusión de una secuencia de distribuciones gaussianas
    parametrizadas en el espacio de información (fischer).

    Argumentos:

        informacion = ([y_1, I_1], ..., [y_n, I_n])
    """
    v_s, I_s = zip(*informacion)
    return sum(v_s), sum(I_s)


class kalman(object):
    def __init__(self, xi, dxi, ti=0.):
        """Filtros de Kalman. """
        self.t = ti
        self._x = np.copy(xi)
        self._P = np.diag(np.square(dxi))

    @property
    def x(self):
        return self._x.copy()

    @property
    def P(self):
        return self._P.copy()

    def prediccion(self, t, *args):
        """Paso de predicción

        Argumentos:

            t: tiempo
        """
        dt = t - self.t
        self.t = t
        self._x, self._P = self.f(dt, *args)


class KF(kalman):
    def __init__(self, xi, dxi, ti=0.):
        super(KF, self).__init__(xi, dxi, ti=0.)
        self._dz = None

    @property
    def dz(self):
        return self._dz

    def correccion(self, h, z, *args):
        """Paso de corrección

        Argumentos:

            h: modelo de medición exoceptiva
            z: medición
        """
        hat_z, H, R = h(*args)
        self._dz = np.subtract(z, hat_z)
        P_z = multi_dot([H, self._P, H.T]) + R
        K = multi_dot([self._P, H.T, inv(P_z)])
        self._x = self._x + np.matmul(K, self._dz)
        self._P = self._P - multi_dot([K, H, self._P])


class IF(kalman):
    def __init__(self, xi, dxi, ti=0.):
        super(IF, self).__init__(xi, dxi, ti=0.)
        self._dy = None

    @property
    def dy(self):
        return self._dy

    def correccion(self, h, z, *args):
        """Paso de corrección

        Argumentos:

            h: modelo de medición exoceptiva
            z: medición
        """
        dy, Y = h(*args)
        self._dy = dy
        I_prior = inv(self._P)
        self._P = inv(I_prior + Y)
        self._x = self._x + np.matmul(self._P, self._dy)
