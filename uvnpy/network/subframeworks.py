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


def valid_extents(geodesics, condition, *args):
    extents = []
    for g in geodesics:
        extents.append([])
        ecc = int(g.max())
        for h in range(ecc + 1):
            subset = g <= h
            if condition(subset, *args):
                extents[-1].append(h)
    return extents


@njit
def subframework_vertices(geodesics, extents):
    return geodesics <= extents.reshape(-1, 1)


def subframework_adjacencies(geodesics, extents):
    S = geodesics <= extents.reshape(-1, 1)
    adjacency = geodesics.copy()
    adjacency[adjacency > 1] = 0
    return [adjacency[:, s][s] for s in S]


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
def isolated_links(geodesics, extents):
    """
    Computes the set of isolated links (edges not in any subframework).
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
        geodesics,
        valid_extents,
        metric,
        **kwargs
        ):
    """
    Given an initial decomposition given by valid_extents, compares each of the
    possible subframework decompositions that asserts the
    subframework_condition and doesn't exceed max_extent; and selects the one
    that minimizes the metric.

    Requires:
    ---------
        framework is rigid
    """
    search_space = itertools.product(*valid_extents)
    min_value = np.inf
    for h in search_space:
        h = np.array(h)
        new_value = metric(geodesics, h, **kwargs)
        if new_value < min_value:
            min_value = new_value
            h_opt = h

    return h_opt


# def sparse_subframeworks_greedy_unidirectional_search(
#         geodesics,
#         valid_extents,
#         metric
#         ):
#     """
#     Given an initial decomposition given by valid_extents, at each iteration
#     removes the subframework with the greatest contribution to the metric.

#     Requires:
#     ---------
#         framework is rigid
#     """
#     n = len(valid_extents)
#     hops = extents.copy()
#     remain = [i for i in range(n) if extents[i] > 0]
#     terminate = False

#     min_value = np.inf
#     while not terminate:
#         remove_sub = None
#         for i in remain:
#             sparsed = hops.copy()
#             sparsed[i] = 0
#             new_value = metric(geodesics, sparsed)
#             if new_value < min_value:
#                 min_value = new_value
#                 remove_sub = i

#         if remove_sub is not None:
#             hops[remove_sub] = 0
#             remain.remove(remove_sub)
#         else:
#             terminate = True
#     return hops


def sparse_subframeworks_greedy_search(
        geodesics,
        valid_extents,
        metric,
        initial_guess,
        **kwargs
        ):
    """
    Given the set of valid extents of each node, starts with all subframeworks
    with zero radiues and selects, at each iteration, the extent change the
    reduces the metric the most.

    Requires:
    ---------
        framework is rigid
        valid_extents is list of ordered lists in increasing order
    """
    n = len(geodesics)
    h_min = np.copy(initial_guess)
    nodes = np.arange(n)
    terminate = False

    min_value = metric(geodesics, h_min, **kwargs)
    while not terminate:
        i_min = None
        for i in nodes:
            if h_min[i] < valid_extents[i][-1]:
                curr_index = valid_extents[i].index(h_min[i])
                h_perturbed = h_min.copy()
                h_perturbed[i] = valid_extents[i][curr_index + 1]
                new_value = metric(geodesics, h_perturbed, **kwargs)
                if new_value < min_value:
                    min_value = new_value
                    i_min = i
                    index_min = curr_index + 1
            if h_min[i] > valid_extents[i][0]:
                curr_index = valid_extents[i].index(h_min[i])
                h_perturbed = h_min.copy()
                h_perturbed[i] = valid_extents[i][curr_index - 1]
                new_value = metric(geodesics, h_perturbed, **kwargs)
                if new_value < min_value:
                    min_value = new_value
                    i_min = i
                    index_min = curr_index - 1

        if i_min is None:
            terminate = True
        else:
            h_min[i_min] = valid_extents[i_min][index_min]

    return h_min
