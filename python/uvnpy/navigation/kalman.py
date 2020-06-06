#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue Jan 14 16:15:16 2020
@author: fran
"""
import numpy as np

class Ekf(object):
    """ This class contains info and modules to work with an EKF. """
    def __init__(self, x, dx, **kwargs):
        self.time = kwargs.get('t', 0.)
        self.x = np.asarray(x).reshape(-1,1)
        self.P = np.diag(np.square(np.asarray(dx).flat))
        self.eye = np.eye(self.x.size)

    def prediction(self, u, f, t, *args):
        """ This module makes a state prediction based on:
        time: t
        input: u
        sensor model: dot_x = f(x, u) + B*e
            f: f(x, u)
            F_x: jacobian of f w.r.t. x
            B: noise input matrix
            Q: noise covariance matrix
        
        return
            predicted state: x
            predicted covariance: P
        """
        Ts = t - self.time
        self.time = t
        dot_x, F_x, B, Q = f(self.x, u, *args)
        self.x += Ts * dot_x
        Phi  = self.eye + Ts * F_x
        self.P = Phi @ self.P @ Phi.T + B @ Q @ B.T * Ts
    
    def correction(self, y, h, *args):
        """ This module makes a correction of the estimated state
        based on:
        measurement: y
        measurement model: y = h(x) + n
            h: h(x)
            H: jacobian of H w.r.t. x
            R: noise covariance matrix
        """
        hat_y, H, R = h(self.x, *args)
        dy = y - hat_y
        P_dy = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(P_dy)
        self.x = self.x + np.dot(K, dy)
        self.P = self.P - K @ H @ self.P