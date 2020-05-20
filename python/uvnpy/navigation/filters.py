#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue Jan 14 16:15:16 2020
@author: fran
"""
import numpy as np

class Ekf(object):
    """ This class contains info and modules to work with an EKF. """
    def __init__(self):
        self.is_enable = False

    def begin(self, X, dX, **kwargs):
        """ Create all variables needed. """
        self.time = kwargs.get('t', 0.)
        self.X = X
        self.P = np.diag(np.square(dX.flat))
        self.eye = np.eye(self.X.size)
        self.is_enable = True
    
    def prediction(self, u, model, t, **kwargs):
        """ This module takes in input u, sensor model and absolute time 
        Ts to make a prediction. The sensor model must be a callable that
        returns dot_X, F_X, B and noise Q. """
        Ts = t - self.time
        self.time = t
        F_X, B, Q = model.fmat(self.X, u, **kwargs)
        Phi  = self.eye + Ts * F_X
        self.X = self.X + Ts * model.f(self.X, u)
        self.P = np.linalg.multi_dot([Phi, self.P, Phi.T]) + np.dot(B, np.dot(Q, B.T))*Ts
    
    def correction(self, y, model, **kwargs):
        """ This module makes a correction of the estimated state
        and covariance of the process based on the measurements and
        measurement model. The sensor model must be a callable and return
        expected measurement y, H, and noise R. """
        hat_y, H, R = model(self.X, **kwargs)
        dy = y - hat_y
        P_dy = np.linalg.multi_dot([H, self.P, H.T]) + R
        K = np.linalg.multi_dot([self.P, H.T, np.linalg.inv(P_dy)])
        self.X = self.X + np.dot(K, dy)
        self.P = self.P - np.linalg.multi_dot([K, H, self.P])