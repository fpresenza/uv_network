#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue Jan 14 16:15:16 2020
@author: fran
"""
import numpy as np
import types
import uvnpy.toolkit.linalg as linalg
import uvnpy.navigation.metrics as metrics

class Ekf(object):
    """ This class contains info and modules to work with an EKF. """
    def __init__(self, x, dx, ti=0.):
        self.time = ti
        self.x = np.copy(x)
        self.P = np.diag(np.square(dx))
        self.eye = np.eye(self.x.size)
        self.log = types.SimpleNamespace(t=[], x=[], dx=[], dy=[])

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
        dot_x, F_x, B, Q = f(self.x, u, t, *args)
        self.x = self.x + Ts * dot_x
        Phi  = self.eye + Ts * F_x
        self.P = linalg.multi_matmul(Phi, self.P, Phi.T) + linalg.multi_matmul(B, Q, B.T) * Ts
    
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
        self.dy = y - hat_y
        P_dy = linalg.multi_matmul(H, self.P, H.T) + R
        K = linalg.multi_matmul(self.P, H.T, np.linalg.inv(P_dy))
        self.x = self.x + np.matmul(K, self.dy)
        self.P = self.P - linalg.multi_matmul(K, H, self.P)

    def save(self):
        """ This modules saves the history of the filter's variables:
        t:  time
        x:  state
        P:  covariance
        dy: innovation
        """
        self.log.t += [self.time]
        self.log.x += [self.x]
        self.log.dx += [metrics.metrics.sqrt_diagonal(self.P)]
        try:
            self.log.dy += [self.dy]
        except AttributeError:
            pass