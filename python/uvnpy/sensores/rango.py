#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue June 23 14:27:46 2020
@author: fran
"""
import numpy as np
from numpy.linalg import norm


__all__ = [
  'distancia',
  'gradiente',
  'delta_informacion',
  'delta_informacion_sum',
  'sensor'
]


def distancia(u, v):
    diff = np.subtract(u, v)
    return norm(diff)


def gradiente(u, v):
    diff = np.subtract(u, v)
    return diff / norm(diff)


def delta_informacion(u, v, sigma):
    h = gradiente(u, v)
    return sigma**(-2) * np.outer(h, h)


def delta_informacion_sum(u, vs, sigma):
    d_i = [delta_informacion(u, v, sigma) for v in vs]
    return sum(d_i)


class sensor(object):
    """Modelo de sensor de rango. """
    def __init__(self, sigma=1.):
        self.sigma = sigma
        self.R = np.diag([np.square(self.sigma)])

    def __call__(self, p, q):
        """Simula una medición ruidosa. """
        dist = distancia(p, q)
        return np.random.normal(dist, self.sigma)
