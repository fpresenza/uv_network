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


class CentralizedRigidityMaintenance(object):
    """
    Gradient based Rigidity Maintenance Control.

    args:
    -----
        dim: realization space dimension
        dmax: maximum connectivity distance
        steepness: connectivity decrease factor
        power: positive number to exponentiate the ridigity eigenvalue
        non_adjacent: bool to consider non-adjacent vertices relations
    """
    def __init__(self, dim, dmax, steepness, power, non_adjacent=False):
        self.dim = dim
        self.midpoint = dmax
        self.steepness = steepness
        self.r = power
        self.dof = int(dim * (dim + 1)/2)
        self.non_adjacent = non_adjacent

    def gradient(self, x, eigenvalue, eigenvector):
        dS_dx = functions.derivative_eval(self.weighted_rigidity_matrix, x)
        dlambda_dx = eigenvector.dot(dS_dx).dot(eigenvector)
        dlambda_dx = dlambda_dx.reshape(x.shape)
        u = -self.r * eigenvalue**(-self.r - 1) * dlambda_dx
        return u

    def weighted_rigidity_matrix(self, x):
        w = core.distance_matrix(x)
        w[w > self.midpoint] *= int(self.non_adjacent)
        w[w > 0] = functions.logistic(w[w > 0], self.midpoint, self.steepness)
        S = core.rigidity_laplacian_multiple_axes(w, x)
        return S

    def update(self, x):
        S = self.weighted_rigidity_matrix(x)
        e, V = np.linalg.eigh(S)
        grad = self.gradient(x, e[self.dof], V[:, self.dof])
        return -grad


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
    def __init__(self, power=2, dmin=0):
        self.power = power
        self.dmin = dmin

    def update(self, x, obstacles):
        r = x - obstacles
        d = np.sqrt(np.square(r).sum(axis=-1))
        d = d.reshape(-1, 1)
        e = self.power
        grad = - e * (d - self.dmin)**(-e - 1) * r / d
        return - grad.sum(axis=0)
