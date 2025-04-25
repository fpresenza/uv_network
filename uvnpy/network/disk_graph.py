#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar abr  6 20:52:31 -03 2021
"""
import numpy as np


def edges_from_positions(p, dmax=np.inf):
    dmax = np.reshape(dmax, (-1, 1))
    r = p[:, np.newaxis] - p
    d2 = np.square(r).sum(axis=-1)
    A = d2 <= dmax**2
    A[np.eye(len(p), dtype=bool)] = 0.0
    return np.argwhere(A)


def adjacency_from_positions(p, dmax=np.inf):
    """
    Disk proximity matrix.

    args:
    -----
        p    : (n, d) state array
        dmax : maximum connectivity distance
    """
    r = p[:, None] - p
    A = np.square(r).sum(axis=-1)
    A[A > dmax**2] = 0
    A[A != 0] = 1
    return A
