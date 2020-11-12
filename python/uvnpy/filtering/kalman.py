#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue Jan 14 16:15:16 2020
@author: fran
"""
import numpy as np
from numpy.linalg import multi_dot, inv

from uvnpy.filtering import similaridad

matmul = np.matmul


def fusionar(v, F):
    """ Fusion de Kalman.

    Fusión de una secuencia de distribuciones gaussianas
    representadas en su forma canónica
    (espacio de información de fischer).

    args:
        v = (v_1, ..., v_n)
        F = (F_1, ..., F_n)
    """
    return np.sum(v, axis=0), np.sum(F, axis=0)


class kalman(object):
    def __init__(self, xi, dxi, ti=0.):
        """Filtros de Kalman. """
        self.iniciar(xi, dxi, ti)

    def iniciar(self, xi, dxi, ti=0., f=None):
        self.t = ti
        self._x = np.copy(xi)
        self._P = np.diag(np.square(dxi))
        if f is not None:
            self.f = f

    @property
    def x(self):
        return self._x.copy()

    @property
    def P(self):
        return self._P.copy()

    def prediccion(self, t, *args):
        """Paso de predicción

        args:

            t: tiempo
        """
        dt = t - self.t
        self.t = t
        x = self._x
        self._x, phi, Q = self.f(dt, x, *args)
        self._P = matmul(phi, matmul(self._P, phi.T)) + Q


class KF(kalman):
    def __init__(self, xi, dxi, ti=0.):
        """Filtro de Kalman en forma clásica. """
        super(KF, self).__init__(xi, dxi, ti=0.)
        self._dz = None

    @property
    def dz(self):
        return self._dz

    def actualizacion(self, z, H, R, hat_z=None):
        """Paso de corrección

        args:

            z: observación
            H: matriz de observación
            R: covarianza del sensor
            hat_z: predicción de la observación,
                usar solamente para EKF.
        """
        x, P = self._x, self._P
        if hat_z is None:
            hat_z = matmul(H, x)
        self._dz = dz = np.subtract(z, hat_z)
        P_z = multi_dot([H, P, H.T]) + R
        K = multi_dot([P, H.T, inv(P_z)])
        self._x = x + matmul(K, dz)
        self._P = P - multi_dot([K, H, P])


class KFi(kalman):
    def __init__(self, xi, dxi, ti=0.):
        """Filtro de Kalman en forma alternativa. """
        super(KFi, self).__init__(xi, dxi, ti=0.)

    def actualizacion(self, dy, Y):
        """Paso de corrección

        args:

            dy: innovacón en espacio de información
            Y: matriz de innovación
        """
        x, P = self._x, self._P
        F_prior = inv(P)
        self._P = inv(F_prior + Y)
        self._x = x + matmul(P, dy)


class KCF(kalman):
    def __init__(self, xi, dxi, ti=0.):
        """Filtro de Kalman por Consenso

        Ver:
            Olfati-Saber,
            ''Kalman-Consensus Filter: Optimality
              Stability and Performance'',
            IEEE Conference on Decision and Control (2009).
        """
        super(KCF, self).__init__(xi, dxi, ti=0.)
        self.t_a = ti

    def actualizacion(self, t, dy, Y, x_j):
        """Paso de corrección

        args:

            t: tiempo
            dy: innovacón en espacio de información
            Y: matriz de innovación
            x_j: tupla de estimados de los vecinos
        """
        dt = t - self.t_a
        self.t_a = t
        x, P = self._x, self._P
        F_prior = inv(P)
        P = inv(F_prior + Y)

        d_i = len(x_j)
        S = np.sum(x_j, axis=0) - d_i * x
        norm_P = np.linalg.norm(P, 'fro')
        c = dt / (norm_P + 1)

        self._x = x + matmul(P, dy) + c * np.matmul(P, S)
        self._P = P


class IF(object):
    def __init__(self, xi, dxi, ti=0.):
        """Filtros de Kalman. """
        self.iniciar(xi, dxi, ti)

    def iniciar(self, xi, dxi, ti=0., f=None):
        self.t = ti
        self._F = F = np.diag(1./np.square(dxi))
        self._v = matmul(F, xi)
        if f is not None:
            self.f = f

    @property
    def v(self):
        return self._v.copy()

    @property
    def F(self):
        return self._F.copy()

    @property
    def x(self):
        v, F = self._v, self._F
        return matmul(inv(F), v)

    @property
    def P(self):
        return inv(self._F)

    def prediccion(self, t, *args):
        """Paso de predicción

        args:

            t: tiempo
        """
        dt = t - self.t
        self.t = t
        x, P = similaridad(self._v, self._F)
        x, phi, Q = self.f(dt, x, *args)
        P = matmul(phi, matmul(P, phi.T)) + Q
        self._v, self._F = similaridad(x, P)

    def actualizacion(self, y, Y):
        """Paso de corrección

        args:

            y: contribuciones en espacio de información
            Y: matriz de contribución
        """
        self._v = self._v + y
        self._F = self._F + Y
