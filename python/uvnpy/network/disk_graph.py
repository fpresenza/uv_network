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
    dmax_2 = dmax**2 * (1 - np.eye(len(p)))
    r = p[:, None] - p
    dist_2 = np.square(r).sum(axis=-1)
    return np.argwhere(dist_2 < dmax_2)


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


def local_neighbors(p, q, dmax=np.inf):
    r = p[None] - q
    dist = np.sqrt(np.square(r).sum(-1))
    n = q[dist < dmax]
    return n
