#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
"""
import numpy as np

from uvnpy.bearings.real_d.core import bearing_rigidity_laplacian
from uvnpy.toolkit import functions


class RigidityMaintenance(object):
    """
    Rigidity Maintenance Control based on the minimization
    of the rigidity matrix (nonzero) eigenvalues' inverses.

    args:
    -----
        dim         : dimension of the realization space
        range_lims  : range low and high limits
        cos_lims    : cosine low and high limits
        power       : positive number to exponentiate the eigenvalues
        eigenvalues : which ones to consider (min of all)
        functional  : wich function of the einengvalues (power or logarithmic)
    """
    def __init__(
            self,
            dim,
            range_lims,
            cos_lims,
            power=1.0,
            threshold=1e-5,
            eigenvalues='min',
            functional='pow'
            ):
        self.dim = dim
        self.d_low, self.d_high = range_lims
        self.c_low, self.c_high = cos_lims
        self.r = power
        self.threshold = threshold
        self.dof = dim + 1

        if eigenvalues == 'min':
            self.eig_max = lambda x: self.dof + 1
        elif eigenvalues == 'all':
            self.eig_max = lambda x: len(x) * self.dim
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
        return - eigenvalue_deriv / np.abs(eigenvalue - self.threshold)

    def weighted_rigidity_matrix(self, x):
        p = x[..., :3]
        a = x[..., 3]
        axes = np.empty(a.shape + (3,), dtype=float)
        axes[..., 0] = np.cos(a)
        axes[..., 1] = np.sin(a)
        axes[..., 2] = 0.0

        r = p[..., np.newaxis, :, :] - p[..., np.newaxis, :]
        d = np.sqrt(np.square(r).sum(axis=-1))
        b = r / d[..., np.newaxis]
        c = np.matmul(b, axes[..., np.newaxis]).squeeze()

        wd = 1 - functions.cosine_activation(
            x=d,
            x_low=self.d_low,
            x_high=self.d_high,
        )
        wc = functions.cosine_activation(
            x=c,
            x_low=self.c_low,
            x_high=self.c_high,
        )

        w = wd * (wc + wc.swapaxes(-2, -1))
        w[..., np.eye(x.shape[-2], dtype=bool)] = 0.0

        S = bearing_rigidity_laplacian(w, p)
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
