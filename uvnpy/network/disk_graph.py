#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar abr  6 20:52:31 -03 2021
"""
import numpy as np

from .graphs import DiskGraph


def adjacency_from_positions(p, dmax=np.inf):
    """
    Disk proximity matrix.

    args:
    -----
        p    : (n, d) state array
        dmax : maximum connectivity distance
    """
    raise DeprecationWarning(
        'This function is deprecated. Should use DiskGraph class instead.'
    )
    return DiskGraph(realization=p, dmax=dmax).adjacency_matrix().astype(float)
