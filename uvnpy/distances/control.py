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


class CentralizedRigidityMaintenancePowEig(object):
    """
    Rigidity Maintenance Control based on the minimization of the
    inverse of the rigidity eigenvalue.

    args:
    -----
        dim: realization space dimension
        dmax: maximum connectivity distance
        steepness: connectivity decrease factor
        power: positive number to exponentiate the ridigity eigenvalue
        adjacent_only: bool to consider non-adjacent vertices relations
    """
    def __init__(self, dim, dmax, steepness, power, adjacent_only=False):
        self.dim = dim
        self.midpoint = dmax
        self.steepness = steepness
        self.r = power
        self.dof = int(dim * (dim + 1)/2)
        self.adjacent_only = adjacent_only

    def gradient(self, matrix_deriv, eigenvalue, eigenvector):
        dlambda_dx = eigenvector.dot(matrix_deriv).dot(eigenvector)
        return -self.r * eigenvalue**(-self.r - 1) * dlambda_dx

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
        grad = self.gradient(dS_dx, e[self.dof], V[:, self.dof])
        return -grad.reshape(x.shape)


class CentralizedRigidityMaintenanceLogDet(object):
    """
    Rigidity Maintenance Control based on the minimization of the
    logarithm of the product of all nonzero laplacian eigenvalues.

    args:
    -----
        dim: realization space dimension
        dmax: maximum connectivity distance
        steepness: connectivity decrease factor
        adjacent_only: bool to consider non-adjacent vertices relations
    """
    def __init__(self, dim, dmax, steepness, adjacent_only=False):
        self.dim = dim
        self.midpoint = dmax
        self.steepness = steepness
        self.dof = int(dim * (dim + 1)/2)
        self.adjacent_only = adjacent_only

    def gradient(self, matrix_deriv, eigenvalue, eigenvector):
        dlambda_dx = eigenvector.dot(matrix_deriv).dot(eigenvector)
        return - dlambda_dx / eigenvalue

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
        grad = [
            self.gradient(dS_dx, e[k], V[:, k])
            for k in range(self.dof, x.size)
        ]
        return - sum(grad).reshape(x.shape)


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
