#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue sep 23 17:04:15 -03 2021
"""
import numpy as np

from uvnpy.distances import core
from uvnpy.toolkit import functions


class RigidityMaintenance(object):
    """
    Rigidity Maintenance Control based on the minimization
    of the rigidity matrix (nonzero) eigenvalues' inverses.

    args:
    -----
        dmax: maximum connectivity distance
        steepness: connectivity decrease factor
        power: positive number to exponentiate the eigenvalues
        adjacent_only: bool to consider non-adjacent vertices relations
        eigenvalues: which ones to consider (min of all)
        functional: wich function of the einengvalues (power or logarithmic)
    """
    def __init__(
            self,
            dmax,
            steepness,
            power=1.0,
            adjacent_only=False,
            eigenvalues='min',
            functional='pow'
            ):
        self.midpoint = dmax
        self.steepness = steepness
        self.r = abs(power)
        self.adjacent_only = adjacent_only

        if eigenvalues == 'min':
            self.eig_max = lambda x: self.dof(x) + 1
        elif eigenvalues == 'all':
            self.eig_max = lambda x: x.size
        else:
            ValueError('Invalid selection of eigenvalues.')

        if functional == 'pow':
            self.gradient = self.gradient_pow
        elif functional == 'log':
            self.gradient = self.gradient_log
        else:
            ValueError('Invalid selection of functional.')

    def dof(self, x):
        n, d = x.shape
        s = min(n - 1, d)
        return int((s + 1) * (2*d - s)/2)

    def gradient_pow(self, matrix_deriv, eigenvalue, eigenvector):
        eigenvalue_deriv = eigenvector.dot(matrix_deriv).dot(eigenvector)
        return - self.r * eigenvalue_deriv / eigenvalue**(self.r + 1)

    def gradient_log(self, matrix_deriv, eigenvalue, eigenvector):
        eigenvalue_deriv = eigenvector.dot(matrix_deriv).dot(eigenvector)
        return - eigenvalue_deriv / eigenvalue

    def weighted_rigidity_matrix(self, x):
        w = core.distance_matrix(x)
        if self.adjacent_only:
            w[w > self.midpoint] = 0.0
        w[w > 0] = functions.logistic(w[w > 0], self.midpoint, self.steepness)
        S = core.rigidity_laplacian_multiple_axes(w, x)
        return S

    def update(self, x):
        S = self.weighted_rigidity_matrix(x)
        e, V = np.linalg.eigh(S)
        dS_dx = functions.derivative_eval(self.weighted_rigidity_matrix, x)
        grad = sum([
            self.gradient(dS_dx, e[k], V[:, k])
            for k in range(self.dof(x), self.eig_max(x))
        ])
        return - grad.reshape(x.shape)
