#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date lun dic 14 15:37:42 -03 2020
"""
import numpy as np
import scipy.linalg
import itertools


def norma(v):
    sqr_sum = np.multiply(v, v).sum(1)
    return np.sqrt(sqr_sum)


def distancia(p, qs):
    diff = np.subtract(p, qs)
    return norma(diff)


def distancia_relativa(p, D):
    diff = D.T.dot(p)
    sqrdiff = diff * diff
    return np.sqrt(sqrdiff.sum(1))


def distancia_relativa_jac(p, D):
    diff = D.T.dot(p)
    sqrdiff = diff * diff
    dist = np.sqrt(sqrdiff.sum(1))
    r = diff / dist.reshape(-1, 1)
    M = scipy.linalg.block_diag(*r)
    Dn = np.kron(D, np.eye(len(p[0])))
    return M.dot(Dn.T)


def rp_jac(p, Dr, Dp):
    In = np.eye(len(p[0]))
    diff = Dr.T.dot(p)
    sqrdiff = diff * diff
    dist = np.sqrt(sqrdiff.sum(1))
    r = diff / dist.reshape(-1, 1)
    M = scipy.linalg.block_diag(*r)
    Dr = np.kron(Dr, In)
    Hr = M.dot(Dr.T)
    Hp = np.kron(Dp.T, In)
    return np.vstack([Hr, Hp])


def grafo_completo(V):
    return list(itertools.combinations(V, 2))


def disk_graph(p, d=1.):
    """ Devuelve lista de enlaces por proximidad. """
    d2 = d**2
    V = range(len(p))
    K = itertools.combinations(V, 2)
    E = [(i, j) for i, j in K if (p[i] - p[j]).dot(p[i] - p[j]) <= d2]
    return E


def matriz_incidencia(V, E):
    return [[1 - 2*e.index(v) if v in e else 0 for e in E] for v in V]


def matriz_adyacencia(V, E, w=None):
    if w is None:
        w = [1. for _ in E]
    n = len(V)
    A = np.zeros((n, n))
    for i, e in enumerate(E):
        A[e] = w[i]
    return A.T + A
