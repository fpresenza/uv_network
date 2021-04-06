#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar abr  6 14:08:43 -03 2021
"""
from transformations import unit_vector


def displacements(p):
    r = p[:, None] - p
    return r


def displacements_aa(p):
    """ Devuelve matriz de desplazamientos.

    args:
        p: array de posiciones (N, n, dof)
    """
    r = p[..., None, :] - p[..., None, :, :]
    return r


def from_adjacency(A, p):
    r = p[:, None] - p
    r *= A[:, :, None]
    return r


def local_displacements(p, q):
    r = unit_vector(p[:, None] - q, axis=2)
    return r
