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
from uvnpy.toolkit.calculus import derivative_eval


class centralized_rigidity_maintenance(object):
    def __init__(self, dim, dmax, steepness, exponent):
        """Control gradiente descendiente.

        args:
            dim: dimension del espacio (2 o 3)
            dmax: distancia maxima de conexion
            steepness: factor de decaimiento de intensidad de señal
            exponent: exponente (>0) al que se eleva el autovalor de rigidez
        """
        self.dim = dim
        self.midpoint = dmax
        self.steepness = steepness
        self.r = exponent
        self.dof = int(dim * (dim + 1)/2)

    def gradient(self, x, eigenvalue, eigenvector):
        dS_dx = derivative_eval(self.weighted_rigidity_matrix, x)
        dlambda_dx = eigenvector.dot(dS_dx).dot(eigenvector)
        dlambda_dx = dlambda_dx.reshape(x.shape)
        u = -self.r * eigenvalue**(-self.r - 1) * dlambda_dx
        return u

    def weighted_rigidity_matrix(self, x):
        w = distances.matrix(x)
        w[w > 0] = functions.logistic(w[w > 0], self.midpoint, self.steepness)
        S = rigidity.symmetric_matrix(w, x)
        return S

    def update(self, x):
        S = self.weighted_rigidity_matrix(x)
        e, V = np.linalg.eigh(S)
        grad = self.gradient(x, e[self.dof], V[:, self.dof])
        return -grad
