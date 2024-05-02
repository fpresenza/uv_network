#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar abr  6 20:52:31 -03 2021
"""
import numpy as np


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


def adjacency_histeresis(connected, p, dmin, dmax):
    """
    Disk proximity matrix with histeresis.

    args:
    -----
        connected : (n, n) boolean array
        p         : (n, d) state array
        dmax      : maximum connectivity distance
    """
    r = p[:, None] - p
    d2 = np.square(r).sum(axis=-1)
    close = np.logical_and(1e-5 < d2, d2 < dmin**2)
    far = d2 > dmax**2
    between = np.logical_not(np.logical_or(close, far))
    now_connected = np.logical_or(close, np.logical_and(connected, between))
    return now_connected
