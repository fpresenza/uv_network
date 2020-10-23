#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue Jan 14 16:15:16 2020
@author: fran
"""
import numpy as np
from numpy.linalg import multi_dot, inv


def transformar(v, M):
    """ Transformación de similaridad.

    Transforma un vector y una matriz p.d. entre el espacio
    de los estados y el espacio de la información, en ambos
    sentidos.
    """
    C = inv(M)
    u = np.matmul(C, v)
    return u, C


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
            w: señal de excitación del modelo (lista o array)
            sensor: nombre de función de medición introceptiva (str)
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

            y: medición (lista o array)
            sensor: nombre de función de medición exoceptiva (str)
        """
        h = eval('self.' + sensor)
        hat_y, H, R = h(*args)
        self.dy = np.subtract(y, hat_y)
        P_dy = multi_dot([H, self.P, H.T]) + R
        K = multi_dot([self.P, H.T, inv(P_dy)])
        self.x = self.x + np.matmul(K, self.dy)
        self.P = self.P - multi_dot([K, H, self.P])
