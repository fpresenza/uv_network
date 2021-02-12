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
    r = p[..., None, :] - p[..., None, :, :]
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
        p: array de posiciones (-1, nv, dof)
        A: matriz de adyacencia (-1, nv, nv)

    returns
        L: laplaciano (-1, nv * dof, nv * dof)
    """
    nv, dof = p.shape[-2:]
    r = unit_vector(p[..., None, :] - p[..., None, :, :], axis=-1)
    L = - r[..., None] * r[..., None, :]  # outer product
    ii = np.diag([True] * nv)
    L[:, ii] = np.eye(dof)
    L *= A[..., None, None]               # aplicar pesos
    L[:, ii] -= L[:, ~ii].reshape(-1, nv, nv - 1, dof, dof).sum(p.ndim - 1)
    L = L.swapaxes(-3, -2).reshape(-1, nv * dof, nv * dof)
    return L
