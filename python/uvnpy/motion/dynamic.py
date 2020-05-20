#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue May 19 20:47:34 2020
@author: fran
"""
import numpy as np
import recordclass

ss = recordclass.recordclass('ss', 'A B', defaults=(np.eye(1), np.eye(1)))
EqPoint = recordclass.recordclass('EqPoint', 'x u')

class DynamicModel(object):
    """ This class is to store models for all type of 
    vehicle, such as linear, nonlinear, holonomic and
    nonholonomic dyamics. All based in the model:
    dot_x = f(x, u) + noise """
    def __init__(self, *args, **kwargs):
        self.ti = kwargs.get('ti', 0.)
        self.t = self.ti
        self.xi = kwargs.get('xi', np.zeros((1, 1)))
        self.x = self.xi.copy()
        self.r = kwargs.get('r', np.zeros((1, 1)))
        self.u = kwargs.get('r', np.zeros((1, 1)))
        self.sigma = kwargs.get('sigma', np.zeros(1))
        self.e = np.zeros((self.sigma.shape[0], 1))

    def restart(self):
        """ Return all values to initial conditions """
        self.t = self.ti
        self.x = self.xi

    def step(self, u, t, **kwargs):
        """ Implements a dynamic system evolution """
        self.Ts = t - self.t
        self.t = t
        self.e = np.random.normal(0, self.sigma).reshape(-1,1)
        self.x = self.x + self.Ts * self.dot_x(self.x, u, **kwargs)        