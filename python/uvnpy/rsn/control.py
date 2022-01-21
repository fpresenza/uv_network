#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue sep 23 17:04:15 -03 2021
"""
import numpy as np

from uvnpy.rsn import distances, rigidity
from uvnpy.toolkit import functions
from uvnpy.toolkit import calculus


class centralized_rigidity_maintenance(object):
    def __init__(self, dim, dmax, steepness, exponent, non_adjacent=False):
        """Control gradiente descendiente.

        args:
            dim: dimension del espacio (2 o 3)
            dmax: distancia maxima de conexion
            steepness: factor de decaimiento de intensidad de señal
            exponent: exponente (>0) al que se eleva el autovalor de rigidez
            non_adjacent: bool para considerar los enlaces no existentes
        """
        self.dim = dim
        self.midpoint = dmax
        self.steepness = steepness
        self.r = exponent
        self.dof = int(dim * (dim + 1)/2)
        self.non_adjacent = non_adjacent

    def gradient(self, x, eigenvalue, eigenvector):
        dS_dx = calculus.derivative_eval(self.weighted_rigidity_matrix, x)
        dlambda_dx = eigenvector.dot(dS_dx).dot(eigenvector)
        dlambda_dx = dlambda_dx.reshape(x.shape)
        u = -self.r * eigenvalue**(-self.r - 1) * dlambda_dx
        return u

    def weighted_rigidity_matrix(self, x):
        w = distances.matrix(x)
        w[w > self.midpoint] *= int(self.non_adjacent)
        w[w > 0] = functions.logistic(w[w > 0], self.midpoint, self.steepness)
        S = rigidity.symmetric_matrix(w, x)
        return S

    def update(self, x):
        S = self.weighted_rigidity_matrix(x)
        e, V = np.linalg.eigh(S)
        grad = self.gradient(x, e[self.dof], V[:, self.dof])
        return -grad


class communication_load(object):
    def __init__(self, dmax, steepness):
        self.dmax = dmax
        self.steepness = steepness

    def load(self, x, coeff):
        w = distances.matrix(x)
        w[w > 0] = functions.logistic(w[w > 0], self.dmax, self.steepness)
        deg = w.sum(-1)
        return (coeff * deg).sum(-1)

    def update(self, x, coeff):
        grad = calculus.gradient(self.load, x, coeff)
        return -grad


class collision_avoidance(object):
    def __init__(self, exponent=2, dmin=0):
        """Gradiente descendiente.

        args:
            exponent: potencia positiva a la que se eleva la distancia
            dmin: radio extra sobre el obstaculo
        """
        self.exponent = exponent
        self.dmin = dmin

    def update(self, x, o):
        """Calcular gradiente.

        args:
            x: posicion del agent
            o: posicion de los obstaculos
        """
        r = x - o
        d = np.sqrt(np.square(r).sum(axis=-1))
        d = d.reshape(-1, 1)
        e = self.exponent
        grad = - e * (d - self.dmin)**(-e - 1) * r / d
        return - grad.sum(axis=0)
