#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date vie mar 26 20:26:33 -03 2021
"""
import numpy as np
import scipy.optimize


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


def circle2d(R=1., c=np.zeros(2), n=100):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    gen = np.empty((n, 2))
    gen[:, 0] = np.cos(t)
    gen[:, 1] = np.sin(t)
    return R * gen + c


def sphere(R=1, c=np.zeros(3), n2=100):
    u = np.linspace(0, 2 * np.pi, int(np.sqrt(n2)))
    v = np.linspace(0, np.pi,  int(np.sqrt(n2)))
    s = np.empty((n2, 3))
    s[:, 0] = np.outer(np.cos(u), np.sin(v)).ravel()
    s[:, 1] = np.outer(np.sin(u), np.sin(v)).ravel()
    s[:, 2] = np.outer(np.ones(u.shape), np.cos(v)).ravel()
    return c + R * s


def riesz_energy_sphere(n, s=1):
    """The Riesz s-energy (s>0) of a set of m points xj on the
    unit sphere is the sum over all pairs of distinct points of
    the terms 1/|xi−xj|^s. The standard Coulomb potential used
    to model electrons repelling each other corresponds to s=1.

    https://www.maths.unsw.edu.au/about/distributing-points-sphere
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.zeros((n, 3))
    x[:, 0] = 0.5 * np.cos(t)
    x[:, 1] = 0.5 * np.sin(t)

    def f(x):
        p = x.reshape(-1, 3)
        r = p[:, None] - p[None, :]
        d2 = np.square(r).sum(2)
        u = d2[d2 > 0]
        return (1/u**s).sum()

    def R(x):
        p = x.reshape(-1, 3)
        return 1 - np.square(p).sum(1)

    opt = scipy.optimize.minimize(
        f,
        x.ravel(),
        constraints={'type': 'ineq', 'fun': R},
        method='SLSQP',
    )
    print(opt)
    return opt.x.reshape(-1, 3)


def angular_energy_sphere(n):
    """work in progress"""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.zeros((n, 3))
    x[:, 0] = np.cos(t)
    x[:, 1] = np.sin(t)

    def f(x):
        p = x.reshape(-1, 3)
        n2 = np.square(p).sum(1)
        c = (p[:, None] * p).sum(2)
        r = np.triu(c, k=1)
        return r[r != 0].max() + (1/n2).sum()

    opt = scipy.optimize.minimize(
        f,
        x.ravel(),
        # constraints={'type': 'ineq', 'fun': R},
        method='SLSQP',
        # options={'eps': 0.03}
    )
    print(opt)
    x = opt.x.reshape(-1, 3)
    return x


def rayleigh_quotient(A, x):
    """Computa el cociente de Rayleigh

    A: matrix (n, n)
    x: m vectores stackeados por filas (m, n)
    """
    R = (x * x.dot(A.T)).sum(-1) / np.square(x).sum(-1)
    return R


def rayleigh_quotient_gradient(A, x):
    """Computa el gradiente del cociente de Rayleigh

    A: matrix (n, n)
    x: m vectores stackeados por filas (m, n)
    """
    n = np.square(x).sum(-1)
    R = (x * x.dot(A.T)).sum(-1) / n
    x_n = x / n.reshape(-1, 1)
    DR = 2 * (x_n.dot(A.T) - R.reshape(-1, 1) * x_n)
    return DR
