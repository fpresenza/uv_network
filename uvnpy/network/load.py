#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from numba import njit


@njit
def one_token_for_all_sum(geodesics, extents):
    """
    Each node emits only one token that is broadcasted in order to reach
    all its subframework.
    """
    load = 0.0
    for i in np.nonzero(extents)[0]:
        n_broadcasters = np.sum(geodesics[i] < extents[i])
        load += n_broadcasters

    return load


@njit
def one_token_for_all_weighted_sum(geodesics, extents):
    """
    Each node emits only one token that is broadcasted in order to reach
    all its subframework.
    The load is weighted using the size of the token wich is modeled as the
    size of the subframework minus the center.
    """
    load = 0.0
    for i in np.nonzero(extents)[0]:
        token_size = np.sum(geodesics[i] <= extents[i]) - 1
        n_broadcasters = np.sum(geodesics[i] < extents[i])
        load += token_size * n_broadcasters

    return load


@njit
def one_token_for_each_sum(geodesics, extents):
    """
    Each node emits one token for each other node in its subframework.
    Tokens are broadcasted by layer: 1-hop, 2-hops and so on.
    """
    load = 0.0
    for i in np.nonzero(extents)[0]:
        for h in range(1, extents[i] + 1):
            n_tokens = np.sum(geodesics[i] == h)
            n_broadcasters = np.sum(geodesics[i] < h)
            load += n_tokens * n_broadcasters

    return load


@njit
def one_token_for_each(geodesics, extents):
    """
    Each node emits one token for each other node in its subframework.
    Tokens are broadcasted by layer: 1-hop, 2-hops and so on.
    """
    load = []
    for i in np.nonzero(extents)[0]:
        load.append(0.0)
        for h in range(1, extents[i] + 1):
            n_tokens = np.sum(geodesics[i] == h)
            n_broadcasters = np.sum(geodesics[i] < h)
            load[i] += n_tokens * n_broadcasters

    return load
