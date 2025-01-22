#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mié ago 11 10:16:00 -03 2021
"""
import numpy as np
from numba import njit


@njit
def reach(A, hops):
    """La potencia k-esima de la matriz de adyacencia indica la
    cantidad de caminos de k-hops que hay entre dos nodos {i, j}."""
    Ak = [np.linalg.matrix_power(A, h) for h in hops]
    return Ak


@njit
def geodesics(A):
    G = A.copy()
    As = np.eye(len(A)) + A
    h = 2
    while not np.all(As):
        Ah = np.linalg.matrix_power(A, h)
        for i, g in enumerate(G):
            idx = np.logical_and(Ah[i] > 0, As[i] == 0)
            g[idx] = h
        As += Ah
        h += 1
    return G


def adjacency(A):
    A1 = np.eye(len(A)) + A
    return A1.astype(bool)


def multihop_adjacency(A, hops):
    """Los nodos que se pueden acceder a con k-hops o menos."""
    Ak = reach(A, np.arange(hops+1))
    return sum(Ak).astype(bool)


def neighborhood(A):
    N = A.astype(bool)
    return N


def neighbors_from_edges(E, i):
    e_in = E[E[:, 0] == i][:, 1]
    e_out = E[E[:, 1] == i][:, 0]
    return np.hstack([e_in, e_out])


def multihop_neighborhood(A, hops):
    Ak = reach(A, np.arange(hops+1))
    Nh = np.logical_not(sum(Ak[:-1]) + np.logical_not(Ak[-1]))
    return Nh


def subgraph(A, i):
    Ni = adjacency(A)[i]
    Ai = A[Ni][:, Ni]
    return Ai


def multihop_subgraph(A, i, hops=1):
    Ni = multihop_adjacency(A, hops)[i]
    Ai = A[Ni][:, Ni]
    return Ai


def multihop_edges(A, i, hops=1):
    Ai = multihop_subgraph(A, i, hops)
    return Ai.sum()/2


def subframework(A, x, i):
    Ni = adjacency(A)[i]
    Ai = A[Ni][:, Ni]
    xi = x[Ni]
    return Ai, xi


def multihop_subframework(A, x, i, hops=1):
    Ni = multihop_adjacency(A, hops)[i]
    Ai = A[Ni][:, Ni]
    xi = x[Ni]
    return Ai, xi


@njit
def multihop_subsets(A, centers, hops):
    """
    Requires:
        hops > 0
    """
    n = len(A)
    max_hop = np.max(hops)
    powers = np.arange(2, max_hop + 1)
    Ak = reach(A, powers)
    As = np.empty((max_hop, n, n), dtype=np.bool_)
    As[0] = np.eye(n) + A
    for k in powers:
        As[k - 1] = As[k - 2] + Ak[k - 2]
    return [As[h - 1, i] for i, h in zip(centers, hops)]


@njit
def degree_load(A, coeff):
    deg = A.sum(1)
    return coeff.dot(deg).sum()


def degree_load_std(A, hops):
    """Calcula una metrica de carga de la red en base al grado de los nodos.

        load = sum_{i in V} sum_{j in Vi} (hops_i - geo_ij) deg_j

    args:
        A: array(n x n)
        hops = array(n, )
    """
    geo = geodesics(A)
    coeff = (hops.reshape(-1, 1) - geo).clip(min=0)
    return degree_load(A, coeff)


@njit
def fast_degree_load_std(degree, hops, geodesics):
    coeff = np.reshape(hops, (-1, 1)) - geodesics
    coeff = np.clip(coeff, a_min=0, a_max=np.inf)
    return coeff.dot(degree).sum()


@njit
def fast_degree_load_flat(degree, hops, geodesics):
    coeff = np.reshape(hops, (-1, 1)) - geodesics
    coeff = np.clip(coeff, a_min=0, a_max=1)
    return coeff.dot(degree).sum()


def degree_load_flat(A, hops):
    """La funcion de peso es 1 para todo nodo en Vi excepto para
    aquellos en la frontera que tienen peso 0."""
    geo = geodesics(A)
    coeff = (hops.reshape(-1, 1) - geo).clip(min=0)
    return degree_load(A, np.sign(coeff))


def subgraph_union(A, hops):
    """Devuelve la union de los subgrafos dados por sus extensiones"""
    geo = geodesics(A)
    centers = np.nonzero(hops)[0]
    n = len(A)
    U = np.zeros((n, n))
    for i in centers:
        inside = np.where(geo[i] <= hops[i])[0]    # inside subgraph
        idx = np.ix_(inside, inside)
        U[idx] = A[idx].copy()

    return U
