#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue Jan 14 16:15:16 2020
@author: fran
"""
import numpy as np

class EKFTools(object):
    """ This class contains info and modules to work with an EKF.
    """
    def __init__(self):
        """ Set flag to start propagating dynamics only when gps
        or range measurements are available.
        """
        self.is_enable = False

    def start(self, t, X, dX):
        """ Create all matrices needed to the filter.
        Create object of dynamic motion model of the robot.
        Starts running filter.
        """
        #   init time
        self.time = t
        #   init state and covariance
        self.X = X
        self.P = np.diag((dX**2).flatten())
        #   Update Flag
        self.is_enable = True
    
    def prediction(self, u, model, t):
        """ This module takes in input u, sensor model and step time 
        Ts to make a prediction. The sensor model must be a callable that
        returns dot_X, F_X, B and noise Q.
        """
        #   Time elapsed between successive iterations
        Ts = t - self.time
        self.time = t
        #   Dynamic Model
        dot_X, F_X, B, Q = model(self.X, u)
        Phi  = np.eye(self.X.size) + Ts*F_X
        #   integration
        self.X = self.X + Ts*dot_X
        self.P = np.linalg.multi_dot([Phi, self.P, Phi.T]) + np.linalg.multi_dot([B, Q, B.T])*Ts
    
    def correction(self, y, model, *argv):
        """ This module makes a correction of the estimated state
        and covariance of the process based on the measurements and
        measurement model. The sensor model must be a callable and return
        hay_y, H, and noise R.
        argv[0] is maximum number of partners.
        """
        #   Measurement model
        hat_y, H, R = model(self.X, *argv)
        dy = y - hat_y
        #   Kalman Gain
        P_dy = np.linalg.multi_dot([H, self.P, H.T]) + R
        K = np.linalg.multi_dot([self.P, H.T, np.linalg.inv(P_dy)])
        #   Correction
        self.X = self.X + np.dot(K, dy)
        self.P = self.P - np.linalg.multi_dot([K, H, self.P])