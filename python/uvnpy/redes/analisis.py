#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date lun dic 14 15:37:42 -03 2020
"""
import numpy as np
import scipy.linalg


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


def disk_graph(p, d=1.):
    """ Devuelve lista de enlaces por proximidad. """
    d2 = d**2
    V = range(len(p))
    E = []
    for i in V:
        for j in V:
            if j > i:
                diff = np.subtract(p[i], p[j])
                if diff.dot(diff) <= d2:
                    E.append((i, j))
    return E


def gen_mi(r, e):
    if r == e[0]:
        return 1.
    elif r == e[1]:
        return -1
    else:
        return 0


def matriz_incidencia(V, E):
    D = [[gen_mi(v, e) for v in V] for e in E]
    return np.transpose(D)


def matriz_adyacencia(V, E, w=None):
    if w is None:
        w = [1. for _ in E]
    n = len(V)
    A = np.zeros((n, n))
    for i, e in enumerate(E):
        A[e] = w[i]
    return A.T + A
