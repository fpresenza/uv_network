#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mié ago 11 10:16:00 -03 2021
"""
import numpy as np
from numba import njit
import itertools


from . import core


@njit
def subframeworks(geodesics, extents):
    return geodesics <= extents.reshape(-1, 1)


def superframework_extents(geodesics, extents):
    super_geodesics = geodesics * (geodesics <= extents)
    return np.max(super_geodesics, axis=1).astype(extents.dtype)


@njit
def complete_subframeworks(geodesics, extents):
    """
    Computes the framework obtained by completing all the remaining
    edges within subframeworks.
    """
    n = len(extents)
    K = np.zeros(geodesics.shape)

    for i in range(n):
        for j in range(i+1, n):
            in_subframework = np.any(
                (geodesics[i] <= extents) * (geodesics[j] <= extents)
            )
            K[i, j] = (geodesics[i, j] == 1) or in_subframework
            K[j, i] = K[i, j]

    return K


@njit
def isolated_edges(geodesics, extents):
    """
    Computes the set of isolated edges (those not in any subframework).
    """
    n = len(extents)
    edges = []

    for i in range(n):
        for j in range(i+1, n):
            if geodesics[i, j] == 1:
                if np.all(
                    (geodesics[i] > extents) + (geodesics[j] > extents)
                ):
                    edges.append((i, j))

    return edges


@njit
def isolated_nodes(geodesics, extents):
    """
    Computes the set of isolated nodes (those not in any subframework).
    """
    n = len(extents)
    nodes = []
    for i in range(n):
        if extents[i] == 0 and np.sum(geodesics[i] <= extents) == 1:
            nodes.append(i)

    return nodes


def sparse_subframeworks_full_search(
        A, p, extents, subframework_condition, max_extent, metric, **kwargs
        ):
    """
    Given an initial decomposition given by extents, compares each of the
    possible subframework decompositions that asserts the
    subframework_condition and doesn't exceed max_extent; and selects the one
    that minimizes the metric.

    Requires:
    ---------
        framework is rigid
    """
    n = len(p)
    geodesics = core.geodesics(A)
    search_space = itertools.product(
        *([0] + list(range(extents[i], max_extent + 1)) for i in range(n))
    )

    min_value = np.inf
    for h in list(search_space)[1:]:
        h = np.array(h)
        V = geodesics <= np.reshape(h, (-1, 1))
        C = V[h > 0]
        for i, c in enumerate(C):
            k = np.delete(C, i, axis=0)
            if np.all(~c + k, axis=1).any():
                continue
            Ai = A[c][:, c]
            pi = p[c]
            if not subframework_condition(Ai, pi, **kwargs):
                break
        new_value = metric(geodesics, h)
        if new_value < min_value:
            min_value = new_value
            h_opt = h

    return h_opt


def sparse_subframeworks_binary_search(geodesics, extents, metric):
    """
    Given an initial decomposition given by extents, compares each of the
    2^n possible subframework removals and selects the one that minimizes
    the metric.

    Requires:
    ---------
        framework is rigid
    """
    n = len(extents)
    search_factors = itertools.product((0, 1), repeat=n)

    min_value = np.inf
    for f in list(search_factors)[1:]:
        h = np.multiply(extents, f)
        new_value = metric(geodesics, h)
        if new_value < min_value:
            min_value = new_value
            h_opt = h

    return h_opt


def sparse_subframeworks_greedy_search(geodesics, extents, metric):
    """
    Given an initial decomposition given by extents, at each iteration
    removes the subframework with the greatest contribution to the metric.

    Requires:
    ---------
        framework is rigid
    """
    n = len(extents)
    hops = extents.copy()
    remain = np.arange(n)
    terminate = False

    min_value = np.inf
    while not terminate:
        remove = None
        for i in remain:
            sparsed = hops.copy()
            sparsed[i] = 0
            new_value = metric(geodesics, sparsed)
            if new_value < min_value:
                min_value = new_value
                remove = i

        if remove is not None:
            hops[remove] = 0
            remain = np.delete(remain, remain == remove)
        else:
            terminate = True
    return hops
