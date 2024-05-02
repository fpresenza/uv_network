#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue mar 18 14:39:11 -03 2021
"""
import numpy as np
from numba import njit


def derivative_eval(f, x, *args, **kwargs):
    """ Partial numerical derivative of f.

    Compute numerically the partial derivatives
    f(x, *args): callable function
    x: np.ndarray

    If x has shape x.shape and y = f(x) with shape y.shape,
    then f must accept as parameter an np.ndarray of shape
    (2 * x.size, *x.shape) and return an np.ndarray of shape
    (2 * x.size, *y.shape) in order to perform the differences.
    """
    h = kwargs.pop('stepsize', 1e-6)
    size = x.size
    dx = np.empty((2 * size,) + x.shape, dtype=np.float64)
    p = np.diag(size * [h]).reshape(-1, *x.shape)
    dx[:size] = p
    dx[size:] = -p
    df = f(x + dx, *args, **kwargs)
    D = (df[:size] - df[size:]) / (2*h)
    return D


def gradient(f, x, *args, **kwargs):
    D = derivative_eval(f, x, *args, **kwargs)
    return D.reshape(x.shape)


def derivative(f):
    def df(x, *args, **kwargs):
        return derivative_eval(f, x, *args, **kwargs)
    return df


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
