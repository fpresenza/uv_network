#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar abr  6 20:52:31 -03 2021
"""
import numpy as np
import warnings


def adjacency_from_positions(p, dmax=np.inf):
    """
    Disk proximity matrix.

    args:
    -----
        p    : (n, d) state array
        dmax : maximum connectivity distance
    """
    warnings.warn(
        'This function is deprecated. Should use DiskGraph class instead.',
        DeprecationWarning
    )
    r = p[:, None] - p
    A = np.square(r).sum(axis=-1)
    A[A > dmax**2] = 0
    A[A != 0] = 1
    return A
