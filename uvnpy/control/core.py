#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
"""
import numpy as np

from uvnpy.distances import core
from uvnpy.toolkit import functions


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


class CollisionAvoidanceVanishing(object):
    """
    Gradient based Collision Avoidance.

    args:
    -----
        power: positive number to exponentiate the distance
        dmin: minimum allowed distance
        dmax: distance at with the controla action vanish

    """
    def __init__(self, power=2.0, dmin=0.0, dmax=1.0):
        self.power = power
        self.dmin = dmin
        self.dmax = dmax

    def update(self, x, obstacles):
        r = x - obstacles
        d = np.sqrt(np.square(r).sum(axis=-1))
        d = d.reshape(-1, 1)
        e = self.power
        dm = d - self.dmin
        dM = d - self.dmax
        deriv = dM * (2*dm - e * dM) / dm**(e + 1)
        neg_grad = - deriv * (r / d)
        return neg_grad.sum(axis=0)


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
        w[w > 0] = 1.0 - functions.logistic_activation(
            w[w > 0], self.dmax, self.steepness
        )
        deg = w.sum(-1)
        return (coeff * deg).sum(-1)

    def update(self, x, coeff):
        grad = functions.gradient(self.load, x, coeff)
        return -grad
