#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar abr  6 14:08:43 -03 2021
"""
from transformations import unit_vector


def matrix(p):
    """ Devuelve matriz de desplazamientos.

    args:
        p: array de posiciones (..., n, d)
    """
    r = p[..., None, :] - p[..., None, :, :]
    return r


def from_edges(E, p):
    r = p[E[:, 0]] - p[E[:, 1]]
    return r


def from_adjacency(A, p):
    r = p[:, None] - p
    r *= A[:, :, None]
    return r


def from_incidence(D, p):
    r = D.T.dot(p)
    return r


def rigidity_matrix(D, p):
    Dt = D.T
    r = Dt.dot(p)
    R = Dt[:, :, None] * r[:, None]
    return R.reshape(-1, p.size)


def edge_matrix(D, p):
    rt = p.T.dot(D)
    E = rt[None, :, :] * D[:, None, :]
    return E.reshape(p.size, -1)


def local_displacements(p, q):
    r = unit_vector(p[:, None] - q, axis=2)
    return r
