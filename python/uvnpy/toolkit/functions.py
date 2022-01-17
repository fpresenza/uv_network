#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue mar 18 14:39:11 -03 2021
"""
import numpy as np
from numba import njit


@njit
def power(d, a):
    """Power function."""
    s = d**(-a)
    return s


@njit
def power_derivative(d, a):
    D = -a * d**(-a - 1)
    return D


@njit
def logistic(d, midpoint=0, steepness=1):
    """Logistic strength function.

    A logistic function or logistic curve is a common
    S-shaped curve (sigmoid curve) with equation

        1 / (1 + exp(-steepness * (d - midpoint)))

    d: distance between peers,
    steepness: the logistic growth rate or steepness of the curve,
    midpoint: the x value of the sigmoid's midpoint.
    """
    s = 1 / (1 + np.exp(steepness * (d - midpoint)))
    return s


@njit
def logistic_derivative(d, midpoint=0, steepness=1):
    """Derivative  of the logistic function respect to distance."""
    D = - 0.5 * steepness / (1 + np.cosh(steepness * (d - midpoint)))
    return D


@njit
def logistic_saturation(x, limit=1., slope=1.):
    """Funcion de saturacion logisitica"""
    K = 2 * limit
    steepness = 2 * slope / limit
    return K * (0.5 - logistic(x, steepness=steepness))


@njit
def ramp_saturation(x, limit=1., slope=1.):
    """Funcion de saturacion rampa"""
    return np.clip(slope * x, -limit, limit)
