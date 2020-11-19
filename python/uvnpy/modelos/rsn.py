#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mié nov 18 23:56:57 -03 2020
"""
import numpy as np
import scipy.linalg

__all__ = [
    'distancia_relativa',
    'distancia_relativa_jac',
]


def distancia_relativa(p, D, n=2):
    Dt = np.kron(D, np.eye(n)).T
    diff = Dt.dot(p).reshape(-1, n)
    sqrdiff = diff * diff
    return np.sqrt(sqrdiff.sum(1))


def distancia_relativa_jac(p, D, n=2):
    Dt = np.kron(D, np.eye(n)).T
    diff = Dt.dot(p).reshape(-1, n)
    sqrdiff = diff * diff
    dist = np.sqrt(sqrdiff.sum(1))
    h = diff / dist.reshape(-1, 1)
    M = scipy.linalg.block_diag(*h)
    return M.dot(Dt)
