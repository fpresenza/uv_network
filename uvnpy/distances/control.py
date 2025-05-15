#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue sep 23 17:04:15 -03 2021
"""
import numpy as np

from uvnpy.distances.core import (
    distance_matrix,
    rigidity_laplacian_multiple_axes
)
from uvnpy.toolkit import functions


class RigidityMaintenance(object):
    """
    Rigidity Maintenance Control based on the minimization
    of the rigidity matrix (nonzero) eigenvalues' inverses.

    args:
    -----
        dim: realization space dimension
        dmax: maximum connectivity distance
        steepness: connectivity decrease factor
        power: positive number to exponentiate the eigenvalues
        eigenvalues: which ones to consider (min of all)
        functional: wich function of the einengvalues (power or logarithmic)
    """
    def __init__(
            self,
            dim,
            dmax,
            steepness,
            power=1.0,
            threshold=1e-5,
            eigenvalues='min',
            functional='pow'
            ):
        self.dim = dim
        self.midpoint = dmax
        self.steepness = steepness
        self.r = abs(power)
        self.threshold = threshold

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
        return - eigenvalue_deriv / (eigenvalue - self.threshold)

    def weighted_rigidity_matrix(self, x):
        d = distance_matrix(x)
        w = 1.0 - functions.logistic_activation(
            x=d,
            midpoint=self.midpoint,
            steepness=self.steepness
        )
        w[..., np.eye(x.shape[-2], dtype=bool)] = 0.0

        S = rigidity_laplacian_multiple_axes(w, x)
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
