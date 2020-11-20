#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Wed Jun 34 13:28:15 2020
@author: fran
"""
import numpy as np
from numpy import trace  # noqa
from numpy.linalg import norm, det, eigvalsh  # noqa
from scipy.linalg import svdvals, sqrtm  # noqa

norma = norm


def traza2(M):
    return M[0, 0] + M[1, 1]


def traza_inv2(M):
    a, b, c, d = M.flat
    return (a + d)/(a*d - b*c)


def traza3(M):
    return M[0, 0] + M[1, 1] + M[2, 2]


def traza(M):
    return M.diagonal().sum()


def det2(M):
    return M[0, 0]*M[1, 1] - M[0, 1]*M[1, 0]


def inv2(M):
    a, b, c, d = M.flat
    return np.array([
        [d, -b],
        [-c, a]]) / (a*d - b*c)


def sqrt_diagonal(M):
    return np.sqrt(np.diagonal(M))
