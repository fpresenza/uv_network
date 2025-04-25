#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar abr  6 20:52:31 -03 2021
"""
import numpy as np


def adjacency_matrix(positions, cones_axes, dmax=np.inf, fov=-1.0):
    """
    Cone graph adjacency matrix.

    args:
    -----
        positions  : (n, d) vector array
        cones_axes : (n, d) unit vector array
        dmax       : max conextion range
        fov        : cosine of cone's half angle
    """
    dmax = np.reshape(dmax, (-1, 1))
    r = positions - positions[:, np.newaxis]
    d = np.sqrt(np.square(r).sum(axis=-1))
    bearings = r / d[:, :, np.newaxis]
    cos = np.matmul(bearings, cones_axes[:, :, np.newaxis]).squeeze()
    A = np.logical_and(d <= dmax, cos >= fov)
    A[np.eye(len(positions), dtype=bool)] = 0.0
    return A
