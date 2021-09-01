#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar feb 16 11:17:50 -03 2021
"""
import numpy as np


def collision(p, axis=None):
    """Costo de repulsion entre agentes puntuales.

    El costo es igual a la suma de los cuadrados de la
    inversa de las distancias entre agentes.
    """
    r = p[..., None, :] - p[..., None, :, :]
    d2 = np.square(r).sum(axis=-1)
    d2[d2 != 0] **= -1
    return d2.sum(axis=axis)


def intercollision(p, q):
    """Costo de repulsion entre dos grupos de agentes puntuales.

    El costo es igual a la suma de los cuadrados de la
    inversa de las distancias entre agentes.
    """
    r = p[..., None, :] - q[..., None, :, :]
    d2 = np.square(r).sum(axis=-1)
    d2[d2 != 0] **= -1
    return d2.sum()


def maxdist(p):
    """Maxima distancia entre agentes puntuales."""
    r = p[..., None, :] - p[..., None, :, :]
    d2 = np.square(r).sum(axis=-1)
    dmax = np.sqrt(d2.max((-2, -1)))
    return dmax


def mindist(p):
    """Maxima distancia entre agentes puntuales."""
    r = p[..., None, :] - p[..., None, :, :]
    d2 = np.square(r).sum(axis=-1)
    ii = np.eye(d2.shape[-1], dtype=bool)
    dmin = np.sqrt(d2[..., ~ii].min(axis=-1))
    return dmin
