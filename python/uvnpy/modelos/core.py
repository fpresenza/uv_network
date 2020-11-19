#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:11:56 2020
@author: fran
"""
import numpy as np

from gpsic.integradores import EulerExplicito


class vehiculo(object):
    """ This class implements a unmanned vehicle instance
    to use as node in a graph. """
    def __init__(self, nombre, tipo='vehiculo', **kwargs):
        self.id = nombre
        self.tipo = tipo

    def __str__(self):
        return '{}({})'.format(self.tipo, self.id)


class integrador(EulerExplicito):
    def __init__(self, xi=[0.], ti=0.):
        """ Modelo de vehículo integrador. """
        super(integrador, self).__init__(xi, ti)
        self._dx = np.zeros_like(xi, dtype=float)

    @property
    def x(self):
        return self._x.copy()

    @property
    def dx(self):
        return self._dx.copy()

    def dinamica(self, x, t, u):
        self._dx = np.asarray(u)
        return self._dx

    def step(self, t, u):
        x = super(integrador, self).step(t, ([u], ))
        return x


class doble_integrador(EulerExplicito):
    def __init__(self, xi=[0.], ti=0.):
        """ Modelo de vehículo doble integrador. """
        super(doble_integrador, self).__init__(xi, ti)
        self._dx = np.zeros_like(xi, dtype=float)

    @property
    def x(self):
        return self._x.copy()

    @property
    def dx(self):
        return self._dx.copy()

    def dinamica(self, x, t, u):
        n = len(u)
        self._dx[:n] = x[n:]
        self._dx[n:] = np.asarray(u)
        return self._dx

    def step(self, t, u):
        x = super(doble_integrador, self).step(t, ([u], ))
        return x
