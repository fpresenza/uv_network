#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar abr  6 20:52:31 -03 2021
"""
import numpy as np


def edges(p, dmax=np.inf):
    """Devuelve array de enlaces por proximidad."""
    r = p[:, None] - p
    dist_2 = np.triu(np.square(r).sum(axis=-1))
    connected = (0 < dist_2) * (dist_2 < dmax**2)
    return np.argwhere(connected)


def inter_edges(p, q, dmax=np.inf):
    """Devuelve array de enlaces por proximidad entre dos grupos p y q."""
    r = p[:, None] - q
    dist_2 = np.square(r).sum(axis=-1)
    connected = dist_2 < dmax**2
    return np.argwhere(connected)


def adjacency(p, dmax=np.inf):
    """ Devuelve matriz de adyacencia por proximidad.

    args:
        p: array de posiciones (n, dof)
        dmax: distancia máxima de conexión

    returns:
        A: matriz adyacencia (n, n) donde el peso
        del enlace (i, j) es la distancia entre
        los vehículos i y j.
    """
    r = p[:, None] - p
    A = np.square(r).sum(axis=-1)
    A[A > dmax**2] = 0
    A[A != 0] = 1
    return A


def neighborhood(p, i, dmax=np.inf, inclusive=False):
    r = p[i] - p
    dist2 = np.square(r).sum(-1)
    idx = dist2 < dmax**2
    idx[i] = inclusive
    return idx


def neighborhood_histeresis(
        p, i, neighbors,
        dmin=1., dmax=np.inf, inclusive=False):
    r = p[i] - p
    dist2 = np.square(r).sum(-1)
    close = dist2 < dmin**2
    between = ~close * (dist2 < dmax**2)
    idx = close + neighbors * between
    idx[i] = inclusive
    return idx
