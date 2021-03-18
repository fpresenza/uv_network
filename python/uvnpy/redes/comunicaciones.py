#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue mar 18 14:39:11 -03 2021
"""
import numpy as np


def logistic_strength(d, w=1, e=0):
    """Logistic strength function.

    A logistic function or logistic curve is a common
    S-shaped curve (sigmoid curve) with equation

        1 / (1 + exp(-w * (d - e)))

    d: distance between peers,
    w: the logistic growth rate or steepness of the curve,
    e: the x value of the sigmoid's midpoint.
    """
    s = 1 / (1 + np.exp(w * (d - e)))
    return s


def logistic_strength_derivative(d, w=1, e=0):
    """Derivative  of the logistic function respect to distance."""
    D = - 0.5 * w / (1 + np.cosh(w * (d - e)))
    return D
