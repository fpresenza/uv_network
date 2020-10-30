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
    y_s, F_s = zip(*informacion)
    return sum(y_s), sum(F_s)


class EKF(object):
    def __init__(self, xi, dxi, ti=0.):
        """ Filtro extendido de Kalman. """
        self.t = ti
        self._x = np.copy(xi)
        self._P = np.diag(np.square(dxi))
        self.Id = np.identity(self._x.size)

    def prediccion(self, t, u, sensor, *args):
        """ Paso de predicción:

        Argumentos:

            t: tiempo
            u: señal de excitación del modelo (lista o array)
            sensor: nombre de función de medición introceptiva (str)
        """
        Ts = t - self.t
        self.t = t
        f = eval('self.' + sensor)
        dot_x, F_x, F_e, Q = f(t, u, *args)
        self._x = self._x + Ts * dot_x
        Phi = self.Id + Ts * F_x
        Phi_P_Phi = [Phi, self._P, Phi.T]
        F_e_Q_F_e = [F_e, Q, F_e.T]
        self._P = multi_dot(Phi_P_Phi) + multi_dot(F_e_Q_F_e) * (Ts**2)

    def correccion(self, y, sensor, *args):
        """ Paso de corrección

        Argumentos:

            y: medición (lista o array)
            sensor: nombre de función de medición exoceptiva (str)
        """
        h = eval('self.' + sensor)
        hat_y, H, R = h(*args)
        self.dy = np.subtract(y, hat_y)
        P_dy = multi_dot([H, self._P, H.T]) + R
        K = multi_dot([self._P, H.T, inv(P_dy)])
        self._x = self._x + np.matmul(K, self.dy)
        self._P = self._P - multi_dot([K, H, self._P])

    @property
    def x(self):
        return self._x[:]

    @property
    def P(self):
        return self._P[:]
