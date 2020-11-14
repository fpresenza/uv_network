#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue June 23 14:27:46 2020
@author: fran
"""
import numpy as np
import scipy.linalg


__all__ = [
    'distancia',
    'jacobiano',
    'matriz_innovacion',
    'sensor'
]


def norma(v):
    sqr_sum = np.multiply(v, v).sum(1)
    return np.sqrt(sqr_sum)


def distancia(p, qs):
    diff = np.subtract(p, qs)
    return norma(diff)


def distancias_grafo(p, D, n=2):
    Dt = np.kron(D.T, np.eye(n))
    diff = Dt.dot(p).reshape(-1, n)
    sqrdiff = diff * diff
    return np.sqrt(sqrdiff.sum(1))


def jacobiano(p, qs):
    diff = np.subtract(p, qs)
    dist = norma(diff).reshape(-1, 1)
    return diff / dist


def jacobiano_grafo(p, D, n=2):
    Dt = np.kron(D.T, np.eye(n))
    diff = Dt.dot(p).reshape(-1, n)
    sqrdiff = diff * diff
    dist = np.sqrt(sqrdiff.sum(1))
    w = diff / dist.reshape(-1, 1)
    W = scipy.linalg.block_diag(*w)
    return W.dot(Dt)


def matriz_innovacion(p, qs, sigma):
    diff = np.subtract(p, qs)
    dist = norma(diff).reshape(-1, 1)
    H = diff / dist
    return sigma**(-2) * np.matmul(H.T, H)


class sensor(object):
    """Modelo de sensor de rango. """
    def __init__(self, sigma=1.):
        self.sigma = sigma
        self.R = sigma**2

    def __call__(self, p, qs):
        """Simula una medición ruidosa. """
        d = distancia(p, qs)
        return np.random.normal(d, self.sigma)
