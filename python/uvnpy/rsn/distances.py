#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar ene 26 18:33:51 -03 2021
"""
import numpy as np
from transformations import unit_vector

import uvnpy.network.core as network


def all(p):
    """ Devuelve matriz de distancias.

    args:
        p: array de posiciones (n, d)
    """
    r = p[:, None] - p
    dist = np.sqrt(np.square(r).sum(axis=2))
    return dist


def all_aa(p):
    """ Devuelve matriz de distancias.

    args:
        p: array de posiciones (-1, n, d)
    """
    r = p[..., None, :] - p[..., None, :, :]
    dist = np.sqrt(np.square(r).sum(axis=-1))
    return dist


def from_edges(E, p):
    r = p[E[:, 0]] - p[E[:, 1]]
    dist = np.sqrt(np.square(r).sum(axis=-1))
    return dist


def from_incidence(D, p):
    r = D.T.dot(p)
    dist = np.sqrt(np.square(r).sum(axis=-1))
    return dist


def jacobian_from_adjacency(A, p):
    n, d = p.shape
    r = unit_vector(p[:, None] - p, axis=-1)
    ii = np.diag([True] * n)
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


# def distances_jac_aa(A, p):
#     N, n, d = p.shape[-3:]
#     r = unit_vector(p[..., None, :] - p[..., None, :, :], axis=-1)
#     ii = np.diag([True] * n)
#     r[:, ii] = 0
#     r *= A[..., None]               # aplicar pesos
#     E = np.argwhere(np.triu(A) != 0)
#     Ef = np.flip(E, axis=1)
#     ne = len(E)
#     J = np.empty((N, ne, n,  d))
#     i = np.arange(ne).reshape(-1, 1)
#     J[:, i, E] = r[:, E, Ef]
#     J = J.reshape(N, ne, n * d)
#     return J


def jacobian_from_incidence(D, p):
    """ Jacobiano del modelo de distancias.

    Devuelve el jacobiano de
        h(x) = [ ... ||x_i - x_j|| ...]^T

    a partir de la matriz de incidencia D. Cada columna de
    D representa un enlace D[e], y puede contener pesos
    D[e, i] = -D[e, j] = w_ij.

    args:
        D: matriz de incidencia (n, ne)
        p: array de posiciones (n, d)

    returns
        H: jacobiano (ne, n * d)

    """
    Dt = D.T
    r = unit_vector(Dt.dot(p), axis=-1)
    J = Dt[:, :, None] * r[:, None]
    return J.reshape(-1, p.size)


def innovation_matrix(A, p):
    """ Matriz de innovación.

    Devuelve la matriz de innovación

            Y =  H^T W H

    del modelo de distancias de un grafo de n agentes
    determinado por la matriz de adyacencia A.

    Si A[i, i] > 0 el nodo i tiene medicion de posición
    Si A[i, j] > 0 los nodos i, j tienen medicion de distancia,
    A[i, j] respresenta el peso asociado a cada enlace.

    args:
        A: matriz de adyacencia (n, n)
        p: array de posiciones (n, d)

    returns
        Y: matriz de innovacion (n * d, n * d)
    """
    n, d = p.shape
    r = unit_vector(p[:, None] - p, axis=2)
    Y = - r[..., None] * r[..., None, :]  # outer product
    ii = np.diag([True] * n)
    Y[ii] = np.eye(d)
    Y *= A[..., None, None]               # aplicar pesos
    Y[ii] -= Y[~ii].reshape(n, n - 1, d, d).sum(1)
    Y = Y.swapaxes(-3, -2).reshape(n * d, n * d)
    return Y


def innovation_matrix_aa(A, p):
    """ Matriz de innovación.

    Devuelve la matriz de innovación

            Y =  H^T W H

    del modelo de distancias de un grafo de n agentes
    determinado por la matriz de adyacencia A.

    Si A[i, i] > 0 el nodo i tiene medicion de posición
    Si A[i, j] > 0 los nodos i, j tienen medicion de distancia,
    A[i, j] respresenta el peso asociado a cada enlace.

    args:
        A: matriz de adyacencia (N, n, n)
        p: array de posiciones (N, n, d)

    returns
        Y: matriz de innovacion (N, n * d, n * d)
    """
    n, d = p.shape[-2:]
    r = unit_vector(p[..., None, :] - p[..., None, :, :], axis=-1)
    Y = - r[..., None] * r[..., None, :]  # outer product
    ii = np.diag([True] * n)
    Y[:, ii] = np.eye(d)
    Y *= A[..., None, None]               # aplicar pesos
    Y[:, ii] -= Y[:, ~ii].reshape(-1, n, n - 1, d, d).sum(p.ndim - 1)
    Y = Y.swapaxes(-3, -2).reshape(-1, n * d, n * d)
    return Y


def innovation_matrix_diag_aa(A, p):
    """ Diagonal por bloques de la Matriz de innovación.

    args:
        A: matriz de adyacencia (N, n, n)
        p: array de posiciones (N, n, d)

    returns
        diag: diagonales (-1, n, d, d)
    """
    n, d = p.shape[-2:]
    r = unit_vector(p[..., None, :] - p[..., None, :, :], axis=-1)
    ii = np.diag([True] * n)
    r[:, ii] = 0
    Y = r[..., None] * r[..., None, :]  # outer product
    Y *= A[..., None, None]
    diag = Y.sum(1)
    return diag


def rigidity(A, p):
    Y = innovation_matrix(A, p)
    d = p.shape[-1]
    rigid_rank = p.size - int(d * (d + 1)/2)
    return np.linalg.matrix_rank(Y) == rigid_rank


def rigidity_aa(A, p):
    Y = innovation_matrix_aa(A, p)
    n, d = p.shape[-2:]
    rigid_rank = d * n - int(d * (d + 1)/2)
    return np.linalg.matrix_rank(Y) == rigid_rank


def redundant_rigidity(A, p):
    Ae = network.remove_one_edge_adjacency(A)
    pe = np.tile(p, (len(Ae), 1, 1))
    rigid = rigidity_aa(Ae, pe)
    return np.all(rigid)


def edge_potencial_gradient(A, p):
    """Gradiente de un potencial función de la distancia de los enlaces.

    Devuelve el gradiente de

            V = sum_{ij in E} V_{ij},
            V_{ij} es función de d_{ij} = ||x_i - x_j||.

    A es la matriz de adyacencia donde cada componente
    A[i, j] = partial{V_{ij}} / partial{d_{ij}} es la derivada
    de V_{ij} respecto de la distancia. Si A[i, j] = 0, los nodos
    (i, j) no están conectados.

    args:
        A: matriz de adyacencia (n, n)
        p: array de posiciones (n, d)
    """
    n, d = p.shape
    r = unit_vector(p[:, None] - p, axis=-1)
    ii = np.diag([True] * n)
    r[ii] = 0
    r *= A[..., None]               # aplicar pesos
    grad = r.sum(1)
    return grad


def local_distances(p, q):
    r = p[:, None] - q
    dist = np.sqrt(np.square(r).sum(2))
    return dist


def local_innovation_matrix(p, q, w=np.array(1.)):
    r = unit_vector(p[:, None] - q, axis=2)
    rw = r * w[..., None]
    Y = r[..., None] * rw[..., None, :]
    Yi = Y.sum(1)
    return Yi


def local_edge_potencial_gradient(p, q, w):
    """Gradiente de un potencial función de la distancia de los enlaces.

    Devuelve el gradiente de

            V = sum_{j in N_i} V_{j},
            V_{j} es función de d_{ij} = ||x_i - x_j||.

    w es un array donde componente w[j] = partial{V_{j}} / partial{d_{ij}}
    es la derivada de V_{j} respecto de la distancia.
    """
    r = unit_vector(p[:, None] - q, axis=2)
    r *= w[..., None]             # aplicar pesos
    grad = r.sum(1)
    return grad
