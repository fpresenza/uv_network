#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue Jan 14 16:15:16 2020
@author: fran
"""
import numpy as np
from numpy.linalg import multi_dot, inv

matmul = np.matmul


def fusionar(v, Fisher):
    """ Fusion de Kalman.

    Fusión de una secuencia de distribuciones gaussianas
    representadas en su forma canónica
    (espacio de información de fischer).

    args:
        v = (v_1, ..., v_n)
        Fisher = (F_1, ..., F_n)
    """
    return np.sum(v, axis=0), np.sum(Fisher, axis=0)


class kalman(object):
    def __init__(self, xi, dxi, ti=0.):
        """Filtros de Kalman. """
        self.iniciar(xi, dxi, ti)

    @property
    def x(self):
        return self._x.copy()

    @property
    def P(self):
        return self._P.copy()

    def iniciar(self, xi, dxi, ti=0., f=None):
        self.t = ti
        self._x = np.copy(xi)
        self._P = np.diag(np.square(dxi))

    def prediccion(self, t, *args):
        """Paso de predicción

        args:

            t: tiempo
        """
        dt = t - self.t
        self.t = t
        self.prior(dt, *args)


class KF(kalman):
    def __init__(self, xi, dxi, ti=0.):
        """Filtro de Kalman en forma clásica. """
        super(KF, self).__init__(xi, dxi, ti=0.)

    def actualizacion(self, dz, H, R):
        """Paso de corrección

        args:

            dz: innovación
            H: matriz de observación
            R: covarianza del sensor
        """
        x, P = self._x, self._P
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
        I_prior = inv(P)
        self._P = inv(I_prior + Y)
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

    def actualizacion(self, t, dy_i, Y_i, x_j, c=30.):
        """Paso de corrección

        args:

            t: tiempo
            dy_i: innovacones en espacio de información
            Y_i: matrices de innovación
            x_j: tupla de estimados
        """
        dt = t - self.t_a
        self.t_a = t
        x, P = self._x, self._P

        dy = np.sum(dy_i, axis=0)
        Y = np.sum(Y_i, axis=0)
        I_prior = inv(P)
        P = inv(I_prior + Y)

        d_i = len(x_j)
        suma = np.sum(x_j, axis=0) - d_i * x
        norm_P = np.linalg.norm(P, 'fro')
        c *= dt / (norm_P + 1)

        self._x = x + matmul(P, dy + c*suma)
        self._P = P


class IF(object):
    def __init__(self, xi, dxi, ti=0.):
        """Filtro de Informacion. """
        self.iniciar(xi, dxi, ti)

    @property
    def v(self):
        return self._v.copy()

    @property
    def Fisher(self):
        return self._I.copy()

    def iniciar(self, xi, dxi, ti=0., f=None):
        self.t = ti
        self._I = np.diag(1./np.square(dxi))
        self._v = matmul(self._I, xi)

    def transformar(self, u, M):
        Minv = inv(M)
        v = np.matmul(Minv, u)
        return v, Minv

    def prediccion(self, t, *args):
        """Paso de predicción

        args:

            t: tiempo
        """
        dt = t - self.t
        self.t = t
        x, P = self.transformar(self._v, self._I)
        x, P = self.prior(dt, x, P, *args)
        self._v, self._I = self.transformar(x, P)

    def actualizacion(self, y, Y):
        """Paso de corrección

        args:

            y: contribuciones en espacio de información
            Y: matriz de contribución
        """
        self._v = self._v + y
        self._I = self._I + Y
