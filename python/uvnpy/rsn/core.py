#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar ene 26 18:33:51 -03 2021
"""
import numpy as np
import scipy.linalg
from transformations import unit_vector


def distances(p):
    """ Devuelve matriz de distancias.

    args:
        p: array de posiciones (n, dof)
    """
    r = p[:, np.newaxis] - p
    dist = np.sqrt(np.square(r).sum(axis=-1))
    return dist


def distances_from_edges(p, E):
    r = p[E[:, 0]] - p[E[:, 1]]
    dist = np.sqrt(np.square(r).sum(axis=-1))
    return dist


def distances_from_incidence(p, Dr):
    r = Dr.T.dot(p)
    dist = np.sqrt(np.square(r).sum(axis=-1))
    return dist


def distances_jac(p, Dr):
    nv, dof = p.shape
    ne = Dr.shape[1]
    J = np.zeros((ne, dof * nv))
    eye = np.eye(dof)
    r = unit_vector(Dr.T.dot(p), axis=-1)
    M = scipy.linalg.block_diag(*r)
    Dr = np.kron(Dr, eye)
    J = M.dot(Dr.T)
    return J


def positions_jac(nv, loops, dof):
    Dp = np.zeros((len(loops), nv))
    Dp[range(len(loops)), loops] = 1
    J = np.kron(Dp, np.eye(dof))
    return J


def distances_laplacian(p, Dr, w):
    """ Laplaciano de distancias.

    Devuelve el laplaciano pesado del modelo de distancias.
    El número de vehículos nv es tomado del vector
    de posiciones p.

    args:
        p: array de posiciones (nv, dof)
        Dr: matriz de incidencia (nv, ne)
        w: array de pesos (ne, )

    returns
        L: laplaciano (2 * nv, 2 * nv)
    """
    nv, dof = p.shape
    D = np.kron(Dr, np.eye(dof))
    rij = unit_vector(Dr.T.dot(p), axis=-1)
    r = rij.reshape(-1, dof, 1)
    r_T = rij.reshape(-1, 1, dof)
    Pij = np.matmul(r, r_T)
    Pij *= w.reshape(-1, 1, 1)
    P = scipy.linalg.block_diag(*Pij)
    return D.dot(P).dot(D.T)


def complete_distances_laplacian(p, w):
    """ Laplaciano de distancias completo.

    Devuelve el laplaciano pesado del modelo de distancias
    de un grafo completo. El número de vehículos nv
    es tomado del vector de posiciones p.

    args:
        p: array de posiciones (nv, dof)
        w: array de pesos (nv, nv)

    returns
        L: laplaciano (2 * nv, 2 * nv)
    """
    nv, dof = p.shape
    r = unit_vector(p[:, np.newaxis] - p, axis=-1)
    ii = np.diag([True] * nv)
    r[ii] = 0
    r = r.reshape(-1, nv,  dof, 1)
    r_T = r.reshape(-1, nv, 1, dof)
    L = - np.matmul(r, r_T)
    L *= w.reshape(nv, nv, 1, 1)
    L[ii] = - np.sum(L, axis=1)
    L = np.block(list(L)).reshape(p.size, p.size)
    return L


def positions_laplacian(nv, loops, dof, w=1):
    """Laplaciano de posiciones.
    """
    diag = np.zeros(nv)
    diag[loops] = w
    L = np.diag(np.repeat(diag, dof))
    return L
