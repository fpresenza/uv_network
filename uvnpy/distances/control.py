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
        dim: realization space dimension
        dmax: maximum connectivity distance
        steepness: connectivity decrease factor
        power: positive number to exponentiate the eigenvalues
        adjacent_only: bool to consider non-adjacent vertices relations
        eigenvalues: which ones to consider (min of all)
        functional: wich function of the einengvalues (power or logarithmic)
    """
    def __init__(
            self,
            dim,
            dmax,
            steepness,
            power=1.0,
            adjacent_only=False,
            eigenvalues='min',
            functional='pow'
            ):
        self.dim = dim
        self.midpoint = dmax
        self.steepness = steepness
        self.r = abs(power)
        self.dof = int(dim * (dim + 1)/2)
        self.adjacent_only = adjacent_only

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
            for k in range(self.dof, self.eig_max(x))
        ])
        return - grad.reshape(x.shape)


class CommunicationLoad(object):
    """
    Gradient based Communication Load minimization.

    args:
    -----
        dmax: maximum connectivity distance
        steepness: connectivity decrease factor
    """
    def __init__(self, dmax, steepness):
        self.dmax = dmax
        self.steepness = steepness

    def load(self, x, coeff):
        w = core.distance_matrix(x)
        w[w > 0] = functions.logistic(w[w > 0], self.dmax, self.steepness)
        deg = w.sum(-1)
        return (coeff * deg).sum(-1)

    def update(self, x, coeff):
        grad = functions.gradient(self.load, x, coeff)
        return -grad


class CollisionAvoidance(object):
    """
    Gradient based Collision Avoidance.

    args:
    -----
        power: positive number to exponentiate the distance
        dmin: minimum allowed distance
    """
    def __init__(self, power=2.0, dmin=0.0):
        self.power = power
        self.dmin = dmin

    def update(self, x, obstacles):
        r = x - obstacles
        d = np.sqrt(np.square(r).sum(axis=-1))
        d = d.reshape(-1, 1)
        e = self.power
        neg_grad = e * (d - self.dmin)**(-e - 1) * r / d
        return neg_grad.sum(axis=0)
