#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mié ago 11 21:15:33 -03 2021
"""
import numpy as np
import scipy.linalg
from transformations import unit_vector

from uvnpy.network import subsets


def classic_matrix(D, p):
    """Matriz de rigidez clásica"""
    Dt = D.T
    r = Dt.dot(p)
    R = Dt[:, :, None] * r[:, None]
    return R.reshape(-1, p.size)


def matrix(D, p):
    """Matriz de rigidez normalizada

    args:
        D: matriz de incidencia (n, m)
        p: array de posiciones (n, d)

    returns
        H: jacobiano (ne, n * d)
    """
    Dt = D.T
    r = unit_vector(Dt.dot(p), axis=-1)
    J = Dt[:, :, None] * r[:, None]
    return J.reshape(-1, p.size)


def matrix_from_adjacency(A, p):
    """Matriz de rigidez normalizada

    args:
        A: matriz de adyacencia (n, n)
        p: array de posiciones (n, d)

    returns
        H: (ne, n * d)
    """
    n, d = p.shape
    r = unit_vector(p[:, None] - p, axis=-1)
    ii = np.eye(n, dtype=bool)
    r[ii] = 0
    r *= A[..., None]               # aplicar pesos
    E = np.argwhere(np.triu(A) != 0)
    Ef = np.flip(E, axis=1)
    ne = len(E)
    J = np.zeros((ne, n, d))
    i = np.arange(ne).reshape(-1, 1)
    J[i, E] = r[E, Ef]
    J = J.reshape(ne, n * d)
    return J


def matrix_from_edges(E, p, w=np.array([1.])):
    """Matriz de rigidez normalizada

    args:
        E: enlaces (m, 2)
        p: array de posiciones (n, d)

    returns
        H: (ne, n * d)
    """
    n, d = p.shape
    r = unit_vector(p[E[:, 0]] - p[E[:, 1]], axis=-1)
    r *= w.reshape(-1, 1)           # aplicar pesos
    ne = len(E)
    J = np.zeros((ne, n, d))
    i = np.arange(ne)
    J[i, E[:, 0]] = r[i]
    J[i, E[:, 1]] = -r[i]
    J = J.reshape(ne, n * d)
    return J


def laplacian(A, p):
    """Laplaciano de rigidez.

        L =  H^T W H

    A[i, j] >= 0 respresenta el peso asociado a cada enlace.

    args:
        A: matriz de adyacencia (..., n, n)
        p: array de posiciones (..., n, d)

    returns
        L: laplaciano de rigidez (..., n * d, n * d)
    """
    n, d = p.shape[-2:]
    ii = np.eye(n, dtype=bool)
    r = unit_vector(p[..., None, :] - p[..., None, :, :], axis=-1)
    r[..., ii, :] = 0
    L = - r[..., None] * r[..., None, :]    # outer product
    L *= A[..., None, None]                 # aplicar pesos
    L[..., ii, :, :] -= L.sum(p.ndim - 1)
    L = L.swapaxes(-3, -2)
    s = list(L.shape)
    s[-4:] = n * d, n * d
    return L.reshape(s)


def complete_laplacian(p):
    n, d = p.shape[-2:]
    ii = np.eye(n, dtype=bool)
    r = unit_vector(p[..., None, :] - p[..., None, :, :], axis=-1)
    r[..., ii, :] = 0
    L = - r[..., None] * r[..., None, :]    # outer product
    L[..., ii, :, :] -= L.sum(p.ndim - 1)
    L = L.swapaxes(-3, -2)
    s = list(L.shape)
    s[-4:] = n * d, n * d
    return L.reshape(s)


def laplacian_diag(A, p):
    """Bloques diagonales del laplaciano de rigidez.

    args:
        A: matriz de adyacencia (..., n, n)
        p: array de posiciones (..., n, d)

    returns
        diag: bloques principales (..., n, d, d)
    """
    n, d = p.shape[-2:]
    ii = np.eye(n, dtype=bool)
    r = unit_vector(p[..., None, :] - p[..., None, :, :], axis=-1)
    r[..., ii, :] = 0
    L = r[..., None] * r[..., None, :]    # outer product
    L *= A[..., None, None]               # aplicar pesos
    diag = L.sum(p.ndim - 1)
    return diag


def local_laplacian(p, q, w=np.array(1.)):
    r = unit_vector(p[:, None] - q, axis=2)
    rw = r * w[..., None]
    L = r[..., None] * rw[..., None, :]
    Li = L.sum(1)
    return Li


def algebraic_condition(A, x, return_value=False):
    d = x.shape[-1]
    f = int(d * (d + 1)/2)
    L = laplacian(A, x)
    eig = np.linalg.eigvalsh(L)
    if return_value:
        return eig[f]
    return eig[f] > 1e-3


def trivial_motions(p):
    """Matriz cuyas columnas son una BON del espacio pose.

    args:
        p: array de posiciones (n, dof)

    returns
        M: matriz (n*dof, n*dof)
    """
    n = len(p)
    P = np.zeros((p.size, 3))
    r_cm = p - p.mean(0)

    P[::2, 0] = 1/np.sqrt(n)                    # dx
    P[1::2, 1] = 1/np.sqrt(n)                   # dy
    P[::2, 2] = -r_cm[:, 1]
    P[1::2, 2] = r_cm[:, 0]
    P[:, 2] /= np.sqrt(np.square(r_cm).sum())   # dt
    return P


def nontrivial_motions(p):
    T = trivial_motions(p)
    N = scipy.linalg.null_space(T.T)
    return N


def minimum_hops(A, x, threshold=1e-3):
    if not algebraic_condition(A, x):
        raise ValueError('Flexible Framework.')
    n = A.shape[0]
    d = x.shape[1]
    f = int(d * (d + 1)/2)
    hops = np.empty(n, dtype=int)
    for i in range(n):
        minimum_found = False
        h = 0
        while not minimum_found:
            h += 1
            if h > (n-1):
                break
            Ai, xi = subsets.multihop_subframework(A, x, i, h)
            Li = laplacian(Ai, xi)
            re = np.linalg.eigvalsh(Li)[f]
            if re > threshold:
                minimum_found = True
        hops[i] = h
    if minimum_found:
        return hops
    else:
        raise StopIteration('Minimum hops not found.')
