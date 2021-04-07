#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date vie mar 26 20:26:33 -03 2021
"""
import numpy as np


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
    h = kwargs.get('h', 1e-3)
    size = x.size
    dx = np.empty((2 * size,) + x.shape)
    p = np.diag(size * [h]).reshape(-1, *x.shape)
    dx[:size] = p
    dx[size:] = -p
    df = f(x + dx, *args, **kwargs)
    D = (df[:size] - df[size:]) / (2*h)
    return D


def derivative(f):
    def df(x, *args, **kwargs):
        return derivative_eval(f, x, *args, **kwargs)
    return df


def circle2d(R=1., c=np.zeros(2), N=100):
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    gen = np.empty((N, 2))
    gen[:, 0] = np.cos(t)
    gen[:, 1] = np.sin(t)
    return R * gen + c
