#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue May 19 20:47:34 2020
@author: fran
"""
import numpy as np

class DiscreteModel(object):
    """ This class implements a generic descrete model dynamic """
    def __init__(self, ti=0., xi=np.zeros(1)):
        self.ti = ti
        self.xi = np.copy(xi)
        self.u = 0.
        self.set(ti, xi)

    def set(self, t, x):
        self.t = t
        self.x = np.copy(x)

    def restart(self):
        """ Return all values to initial conditions """
        self.set(self.ti, self.xi)

    def dot_x(self, x, u, t, *args, **kwargs):
        """ Pure integrator model. It can be overwritten
        to implement any dynamic """
        self.u = np.asarray(u)
        return self.u

    def imethod(self, dot_x, h, *args):
        """ Default integration method is Euler """
        self.x = self.x + dot_x * h

    def step(self, u, t, d_args=(), i_args=(), n_args=(), d_kw={}):
        """ Implemenh a step in the dynamic system evolution """
        h = t - self.t
        dot_x = self.dot_x(self.x, u, t, *d_args, **d_kw)
        self.imethod(dot_x, h, *i_args)
        try:
            self.normalize(*n_args)
        except (NameError, AttributeError):
            pass
        self.t = t
        return self.x.copy()
