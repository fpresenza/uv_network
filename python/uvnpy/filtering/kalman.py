#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue Jan 14 16:15:16 2020
@author: fran
"""
import numpy as np
from numpy.linalg import multi_dot, inv


def fusion(x, P):
    """ Fusion de Kalman de dos distribuiones gaussianas. """
    inv_sum_P = inv(sum(P))
    W = [
      np.matmul(P[1], inv_sum_P),
      np.matmul(P[0], inv_sum_P),
    ]
    media = np.matmul(W[0], x[0]) + np.matmul(W[1], x[1])
    covar = multi_dot([P[1], inv_sum_P, P[0]])
    return media, covar


def multifusion(observaciones):
    """ Fusion de Kalman de múltimples distribuciones gaussianas.

    A partir de una secuencia de muestras independientes,
    asumidas gaussianas con media y covarianza conocida,
    estimar el MLE junto con su covarianza

    Argumentos:
        observaciones = ([mean_1, covar_1], ..., [mean_n, covar_n])
    """
    x, P = observaciones[0]
    for x_new, P_new in observaciones[1:]:
        dx = np.subtract(x_new, x)
        inv_sum_P = inv(np.add(P, P_new))
        K = np.matmul(P, inv_sum_P)
        x = x + np.matmul(K, dx)
        P = P - np.matmul(K, P)
    return x, P


class EKF(object):
    def __init__(self, x, dx, ti=0.):
        """ Filtro extendido de Kalman. """
        self.time = ti
        self.x = np.copy(x)
        self.P = np.diag(np.square(dx))
        self.Id = np.identity(self.x.size)

    def prediccion(self, t, w, sensor, *args):
        """ Paso de predicción:

        Argumentos:

            t: tiempo
            w: señal de excitación del modelo
            sensor: nombre de función de medición introceptiva
        """
        Ts = t - self.time
        self.time = t
        f = eval('self.' + sensor)
        dot_x, F_x, F_e, Q = f(t, w, *args)
        self.x = self.x + Ts * dot_x
        Phi = self.Id + Ts * F_x
        Phi_P_Phi = [Phi, self.P, Phi.T]
        F_e_Q_F_e = [F_e, Q, F_e.T]
        self.P = multi_dot(Phi_P_Phi) + multi_dot(F_e_Q_F_e) * (Ts**2)

    def correccion(self, y, sensor, *args):
        """ Paso de corrección

        Argumentos:

            y: medición
            sensor: key de función de medición exoceptiva
        """
        h = eval('self.' + sensor)
        hat_y, H, R = h(*args)
        self.dy = np.subtract(y, hat_y)
        P_dy = multi_dot([H, self.P, H.T]) + R
        K = multi_dot([self.P, H.T, inv(P_dy)])
        self.x = self.x + np.matmul(K, self.dy)
        self.P = self.P - multi_dot([K, H, self.P])
