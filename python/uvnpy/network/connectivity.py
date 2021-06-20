#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue mar 18 14:39:11 -03 2021
"""
import numpy as np


def power_strength(d, a):
    """Power function."""
    s = d**(-a)
    return s


def power_strength_derivative(d, a):
    D = -a * d**(-a - 1)
    return D


def logistic_strength(d, beta=1, e=0):
    """Logistic strength function.

    A logistic function or logistic curve is a common
    S-shaped curve (sigmoid curve) with equation

        1 / (1 + exp(-beta * (d - e)))

    d: distance between peers,
    beta: the logistic growth rate or steepness of the curve,
    e: the x value of the sigmoid's midpoint.
    """
    s = 1 / (1 + np.exp(beta * (d - e)))
    return s


def logistic_strength_derivative(d, beta=1, e=0):
    """Derivative  of the logistic function respect to distance."""
    D = - 0.5 * beta / (1 + np.cosh(beta * (d - e)))
    return D


def algebraic_connectivity(L):
    n = L.shape[-1]
    return np.linalg.matrix_rank(L) == n - 1
