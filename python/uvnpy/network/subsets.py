#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mié ago 11 10:16:00 -03 2021
"""
import numpy as np
import networkx as nx  # noqa

import uvnpy.network as network


def reach(A, hops):
    """La potencia k-esima de la matriz de adyacencia indica la
    cantidad de caminos de k-hops que hay entre dos nodos {i, j}."""
    Ak = [np.linalg.matrix_power(A, h) for h in hops]
    return np.array(Ak)


def adjacency(A):
    A1 = np.eye(len(A)) + A
    return A1.astype(bool)


def multihop_adjacency(A, hops):
    """Los nodos que se pueden acceder a con k-hops o menos."""
    Ak = reach(A, range(hops+1))
    return sum(Ak).astype(bool)


def neighborhood(A):
    N = A.astype(bool)
    return N


def neighbors_from_edges(E, i):
    e_in = E[E[:, 0] == i][:, 1]
    e_out = E[E[:, 1] == i][:, 0]
    return np.hstack([e_in, e_out])


def multihop_neighborhood(A, hops):
    Ak = reach(A, range(hops+1))
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
    geo = network.geodesics(A)
    coeff = (hops.reshape(-1, 1) - geo).clip(min=0)
    return degree_load(A, coeff)


def degree_load_flat(A, hops):
    """La funcion de peso es 1 para todo nodo en Vi excepto para
    aquellos en la frontera que tienen peso 0."""
    geo = network.geodesics(A)
    coeff = (hops.reshape(-1, 1) - geo).clip(min=0)
    return degree_load(A, np.sign(coeff))


def subgraph_union(A, hops):
    """Devuelve la union de los subgrafos dados por sus extensiones"""
    geo = network.geodesics(A)
    centers = np.nonzero(hops)[0]
    n = len(A)
    U = np.zeros((n, n))
    for i in centers:
        inside = np.where(geo[i] <= hops[i])[0]    # inside subgraph
        idx = np.ix_(inside, inside)
        U[idx] = A[idx].copy()

    return U


# def neighborhood_load(A, hops):
#     _h = hops - 1
#     n = len(_h)
#     N = [multihop_adjacency(A, _h[i])[i] for i in range(n)]
#     deg = A.sum(1)
#     return np.dot(N, deg).sum()/2


# def weighted_node_neighborhood_load(A, i, hop):
#     deg = A.sum(1)
#     G = nx.from_numpy_matrix(A)
#     lengths = nx.single_source_shortest_path_length(G, i, cutoff=hop)
#     k, v = lengths.keys(), lengths.values()
#     geo = np.tile(hop, len(deg))
#     geo[list(k)] = list(v)
#     return sum((hop - geo) * deg)
