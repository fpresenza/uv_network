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


class rigidity_maintenance(object):
    def __init__(self, dof, dmax, steepness, exponent):
        """Control gradiente descendiente.

        args:
        dof: degrees of freedom vehiculos
        dmax: distancia maxima de conexion
        steepness: factor de decaimiento de intensidad de señal
        exponent: exponente (>0) al que se eleva el autovalor de rigidez
        """
        self.dof = dof
        self.midpoint = dmax
        self.steepness = steepness
        self.r = exponent
        self.freedom = int(dof * (dof + 1)/2)

    def gradient(self, x, eigenvalue, eigenvector):
        dL_dx = derivative_eval(self.weighted_rigidity_matrix, x)
        eigengrad = eigenvector.dot(dL_dx).dot(eigenvector)
        eigengrad = eigengrad.reshape(x.shape)
        u = -self.r * eigenvalue**(-self.r - 1) * eigengrad
        return u

    def weighted_rigidity_matrix(self, x):
        w = distances.matrix(x)
        w[w > 0] = functions.logistic(w[w > 0], self.midpoint, self.steepness)
        S = rigidity.symmetric_matrix(w, x)
        return S

    def update(self, x):
        S = self.weighted_rigidity_matrix(x)
        e, V = np.linalg.eigh(S)
        grad = self.gradient(x, e[self.freedom], V[:, self.freedom])
        return -grad
