#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue May 19 20:47:34 2020
@author: fran
"""
import numpy as np
import recordclass

class DiscreteModel(object):
    """ This class implements a generic descrete model of the 
    form dot_x = f(x, u) + B*e, where e equals noise. """
    def __init__(self, *args, **kwargs):
        self.ti = kwargs.get('ti', 0.)
        self.t = self.ti
        self.xi = kwargs.get('xi', np.zeros((1, 1)))
        self.x = self.xi.copy()
        self.r = kwargs.get('r', np.zeros((1, 1)))
        self.u = kwargs.get('u', np.zeros((1, 1)))
        self.sigma = np.asarray(kwargs.get('sigma', [[0.]]))
        self.e = np.zeros((self.sigma.shape[0], 1))

    def dot_x(self, x, u, **kwargs):
        """ Pure integrator model. It can be overwritten in a 
        child class to implement any dynamic """
        return u

    def restart(self):
        """ Return all values to initial conditions """
        self.t = self.ti
        self.x = self.xi.copy()

    def step(self, u, t, **kwargs):
        """ Implements a dynamic system evolution """
        self.Ts = t - self.t
        self.t = t
        self.e = np.random.normal(0, self.sigma).reshape(-1,1)
        self.x += self.Ts * self.dot_x(self.x, u, self.e, **kwargs)