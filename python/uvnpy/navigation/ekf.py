#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue Jan 14 16:15:16 2020
@author: fran
"""
import numpy as np
import types
import gpsic.toolkit.linalg as linalg

def fusion(realizaciones):
    """ Dada una lista de muestras de una variable aleatoria, utiliza el
    algoritmo de kalman para fusionar las mismas 
    Argumentos:
        realizaciones = ([mean_1, covar_1], ..., [mean_n, covar_n]) 
    """
    x, P = realizaciones[0]
    for y, Py in realizaciones[1:]:
        dy = np.subtract(y, x)
        P_dy = P + Py
        K = np.matmul(P, np.linalg.inv(P_dy))
        x = x + np.matmul(K, dy)
        P = P - np.matmul(K, P)
    return x, P


class EKF(object):
    """ Esta clase contiene pasos de predicción y de correción de un filtro extendido
    de Kalman """
    def __init__(self, x, dx, ti=0.):
        self.time = ti
        self.x = np.copy(x)
        self.P = np.diag(np.square(dx))
        self.eye = np.eye(self.x.size)

    def prediction(self, t, w, *args):
        """ This module makes a state prediction based on:
        time: t
        input: w
        sensor model: dot_x = f(x, t, w, e)
            F_x: jacobian of f w.r.t. x
            F_e: jacobian of f w.r.t. e
            Q: noise covariance matrix
        
        return
            predicted state: x
            predicted covariance: P
        """
        Ts = t - self.time
        self.time = t
        dot_x, F_x, F_e, Q = self.f(t, w, *args)
        self.x = self.x + Ts * dot_x
        Phi  = self.eye + Ts * F_x
        self.P = linalg.multi_matmul(Phi, self.P, Phi.T) + linalg.multi_matmul(F_e, Q, F_e.T) * (Ts**2)
    
    def correction(self, y, *args):
        """ This module makes a correction of the estimated state
        based on:
        measurement: y
        measurement model: y = h(x) + n
            h: h(x)
            H: jacobian of H w.r.t. x
            R: noise covariance matrix
        """
        hat_y, H, R = self.h(*args)
        self.dy = np.subtract(y, hat_y)
        P_dy = linalg.multi_matmul(H, self.P, H.T) + R
        K = linalg.multi_matmul(self.P, H.T, np.linalg.inv(P_dy))
        self.x = self.x + np.matmul(K, self.dy)
        self.P = self.P - linalg.multi_matmul(K, H, self.P)