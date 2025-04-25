#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue sep 23 17:04:15 -03 2021
"""
import numpy as np

from uvnpy.distances.core import distance_matrix
from uvnpy.bearings.core import rigidity_laplacian_multiple_axes
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
        self.dof = dim + 1

        if eigenvalues == 'min':
            self.eig_max = lambda x: self.dof + 1
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

    def gradient_pow(self, matrix_deriv, eigenvalue, eigenvector):
        eigenvalue_deriv = eigenvector.dot(matrix_deriv).dot(eigenvector)
        return - self.r * eigenvalue_deriv / eigenvalue**(self.r + 1)

    def gradient_log(self, matrix_deriv, eigenvalue, eigenvector):
        eigenvalue_deriv = eigenvector.dot(matrix_deriv).dot(eigenvector)
        return - eigenvalue_deriv / (eigenvalue - self.threshold)

    def weighted_rigidity_matrix(self, x):
        w = distance_matrix(x)
        off_diag = np.logical_not(np.eye(x.shape[-2], dtype=bool))
        w[..., off_diag] = functions.logistic(
            x=w[..., off_diag],
            midpoint=self.midpoint,
            steepness=self.steepness
        )
        S = rigidity_laplacian_multiple_axes(w, x)
        return S

    def update(self, x):
        S = self.weighted_rigidity_matrix(x)
        e, V = np.linalg.eigh(S)
        dS_dx = functions.derivative_eval(self.weighted_rigidity_matrix, x)
        grad = sum([
            self.gradient(dS_dx, e[k], V[:, k])
            for k in range(self.dof, self.eig_max(x))
        ])
        return - grad.reshape(x.shape)
