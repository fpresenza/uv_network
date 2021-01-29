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


def distances_from_edges(E, p):
    r = p[E[:, 0]] - p[E[:, 1]]
    dist = np.sqrt(np.square(r).sum(axis=-1))
    return dist


def distances_from_incidence(Dr, p):
    r = Dr.T.dot(p)
    dist = np.sqrt(np.square(r).sum(axis=-1))
    return dist


def distances_jac(Dr, p):
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


def distances_innovation_laplacian(A, p):
    """ Laplaciano de innovación.

    Devuelve la matriz de innovación

            Y =  H^T R^{-1} H

    del modelo de distancias de un grafo de nv agentes
    determinado por la matriz de adyacencia A.

    Si A[i, i] > 0 el nodo i tiene medicion de posición
    Si A[i, j] > 0 los nodos i, j tienen medicion de distancia

    args:
        p: array de posiciones (nv, dof)
        A: matriz de adyacencia (nv, nv)

    returns
        L: laplaciano (2 * nv, 2 * nv)
    """
    nv, dof = p.shape
    r = unit_vector(p[:, np.newaxis] - p, axis=-1)
    ii = np.diag([True] * nv)
    r = r.reshape(-1, nv,  dof, 1)
    r_T = r.reshape(-1, nv, 1, dof)
    L = - np.matmul(r, r_T)
    L[ii] = np.eye(dof)
    L *= A.reshape(nv, nv, 1, 1)
    L[ii] -= np.sum(L[~ii].reshape(nv, -1, dof, dof), axis=1)
    L = np.block(list(L)).reshape(p.size, p.size)
    return L
