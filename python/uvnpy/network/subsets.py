#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mié ago 11 10:16:00 -03 2021
"""
import numpy as np


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
