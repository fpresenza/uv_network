#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar ene 26 18:33:51 -03 2021
"""
import numpy as np
from transformations import unit_vector


def matrix(p):
    """ Devuelve matriz de distancias.

    args:
        p: array de posiciones (..., n, d)
    """
    r = p[..., None, :] - p[..., None, :, :]
    dist = np.sqrt(np.square(r).sum(axis=-1))
    return dist


def from_edges(E, p):
    r = p[..., E[:, 0], :] - p[..., E[:, 1], :]
    dist = np.sqrt(np.square(r).sum(axis=-1))
    return dist


def from_adjacency(A, p):
    r = p[..., None, :] - p[..., None, :, :]
    dist = np.sqrt(np.square(r).sum(axis=-1)) * A
    return dist[dist > 0]


def from_incidence(D, p):
    r = D.T.dot(p)
    dist = np.sqrt(np.square(r).sum(axis=-1))
    return dist


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
    ii = np.eye(n, dtype=bool)
    r[ii] = 0
    r *= A[..., None]               # aplicar pesos
    grad = r.sum(1)
    return grad


def local_matrix(p, q):
    r = p[..., None, :] - q
    dist = np.sqrt(np.square(r).sum(-1))
    return dist


def local_edge_potencial_gradient(p, q, w):
    """Gradiente de un potencial función de la distancia de los enlaces.

    Devuelve el gradiente de

            V = sum_{j in N_i} V_{j},
            V_{j} es función de d_{ij} = ||x_i - x_j||.

    w es un array donde componente w[j] = partial{V_{j}} / partial{d_{ij}}
    es la derivada de V_{j} respecto de la distancia.
    """
    r = unit_vector(p[..., None, :] - q, axis=-1)
    r *= w[..., None]             # aplicar pesos
    grad = r.sum(-2)
    return grad
