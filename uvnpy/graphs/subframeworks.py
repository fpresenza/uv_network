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


from uvnpy.graphs.core import geodesics


def valid_extents(geodesics_matrix, condition, max_extent=None, args=()):
    extents = []
    for g in geodesics_matrix:
        extents.append([])
        if max_extent is None:
            max_extent = int(g.max())
        for h in range(max_extent + 1):
            subset = g <= h
            if condition(subset, *args):
                extents[-1].append(h)
    return extents


@njit
def subframework_vertices(geodesics_matrix, extents):
    return geodesics_matrix <= extents.reshape(-1, 1)


def subframework_adjacencies(geodesics_matrix, extents):
    S = geodesics_matrix <= extents.reshape(-1, 1)
    adjacency = geodesics_matrix.copy()
    adjacency[adjacency > 1] = 0
    return [adjacency[:, s][s] for s in S]


@njit
def subframework_diameters(geodesics_matrix, extents):
    n = len(extents)
    adjacency = geodesics_matrix.copy().ravel()
    adjacency[adjacency > 1] = 0
    adjacency = adjacency.reshape(n, n)
    diam = []
    for i in range(n):
        Vi = geodesics_matrix[i] <= extents[i]
        Ai = adjacency[Vi][:, Vi]
        diam.append(np.max(geodesics(Ai)))
    return diam


@njit
def superframework_extents(geodesics_matrix, extents):
    return np.array([np.max(extents[g <= extents]) for g in geodesics_matrix])


@njit
def superframework_geodesics(geodesics_matrix, extents):
    return np.array([np.max(g[g <= extents]) for g in geodesics_matrix])


@njit
def complete_subframeworks(geodesics_matrix, extents):
    """
    Computes the framework obtained by completing all the remaining
    edges within subframeworks.
    """
    n = len(extents)
    K = np.zeros(geodesics_matrix.shape)

    for i in range(n):
        for j in range(i+1, n):
            i_in_sub = (geodesics_matrix[i] <= extents)
            j_in_sub = (geodesics_matrix[j] <= extents)
            in_subframework = np.any(i_in_sub * j_in_sub)
            K[i, j] = (geodesics_matrix[i, j] == 1) or in_subframework
            K[j, i] = K[i, j]

    return K


@njit
def isolated_links(geodesics_matrix, extents):
    """
    Computes the set of isolated links (edges not in any subframework).
    """
    n = len(extents)
    edges = []

    for i in range(n):
        for j in range(i+1, n):
            if geodesics_matrix[i, j] == 1:
                i_not_in_sub = (geodesics_matrix[i] > extents)
                j_not_in_sub = (geodesics_matrix[j] > extents)
                if np.all(i_not_in_sub + j_not_in_sub):
                    edges.append((i, j))

    return edges


@njit
def isolated_nodes(geodesics_matrix, extents):
    """
    Computes the set of isolated nodes (those not in any subframework).
    """
    n = len(extents)
    nodes = []
    for i in range(n):
        if extents[i] == 0 and np.sum(geodesics_matrix[i] <= extents) == 1:
            nodes.append(i)

    return nodes


def sparse_subframeworks_full_search(
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
        new_value = metric(h, **kwargs)
        if new_value < min_value:
            min_value = new_value
            h_opt = h

    return h_opt


def sparse_subframeworks_greedy_search(
        valid_extents,
        metric,
        initial_guess,
        **kwargs
        ):
    """
    Given the set of valid extents of each node, starts with all subframeworks
    of an initial guess and then selects, at each iteration, the individual
    extent change that reduces the metric the most.

    Requires:
    ---------
        framework is rigid
        valid_extents is list of ordered lists in increasing order
    """
    n = len(valid_extents)
    h_min = np.copy(initial_guess)
    terminate = False

    min_value = metric(h_min, **kwargs)
    while not terminate:
        i_min = None
        for i in range(n):
            if h_min[i] < valid_extents[i][-1]:
                curr_index = valid_extents[i].index(h_min[i])
                h_perturbed = h_min.copy()
                h_perturbed[i] = valid_extents[i][curr_index + 1]
                new_value = metric(h_perturbed, **kwargs)
                if new_value < min_value:
                    min_value = new_value
                    i_min = i
                    index_min = curr_index + 1
            if h_min[i] > valid_extents[i][0]:
                curr_index = valid_extents[i].index(h_min[i])
                h_perturbed = h_min.copy()
                h_perturbed[i] = valid_extents[i][curr_index - 1]
                new_value = metric(h_perturbed, **kwargs)
                if new_value < min_value:
                    min_value = new_value
                    i_min = i
                    index_min = curr_index - 1

        if i_min is None:
            terminate = True
        else:
            h_min[i_min] = valid_extents[i_min][index_min]

    return h_min


def sparse_subframeworks_greedy_search_by_expansion(
        valid_extents,
        metric,
        **kwargs
        ):
    """
    Given the set of valid extents of each node, starts with all subframeworks
    at zero and then expands, at each iteration, the individual extent
    that reduces the metric the most.

    Requires:
    ---------
        framework is rigid
        valid_extents is a python list or tuple of ordered lists in increasing
            order
    """
    n = len(valid_extents)
    h_min = np.array([h[0] for h in valid_extents])
    terminate = False

    min_value = metric(h_min, **kwargs)
    while not terminate:
        i_min = None
        for i in range(n):
            if h_min[i] < valid_extents[i][-1]:
                curr_index = valid_extents[i].index(h_min[i])
                h_perturbed = h_min.copy()
                h_perturbed[i] = valid_extents[i][curr_index + 1]
                new_value = metric(h_perturbed, **kwargs)
                if new_value < min_value:
                    min_value = new_value
                    i_min = i
                    index_min = curr_index + 1

        if i_min is None:
            terminate = True
        else:
            h_min[i_min] = valid_extents[i_min][index_min]

    return h_min


def sparse_subframeworks_greedy_search_by_reduction(
        valid_extents,
        metric,
        **kwargs
        ):
    """
    Given the set of valid extents of each node, starts with all subframeworks
    at maximum valid extent and then reduces, at each iteration, the individual
    extent that reduces the metric the most.

    Requires:
    ---------
        framework is rigid
        valid_extents is list of ordered lists in increasing order
    """
    n = len(valid_extents)
    h_min = np.array([h[-1] for h in valid_extents])
    terminate = False

    min_value = metric(h_min, **kwargs)
    while not terminate:
        i_min = None
        for i in range(n):
            if h_min[i] > valid_extents[i][0]:
                curr_index = valid_extents[i].index(h_min[i])
                h_perturbed = h_min.copy()
                h_perturbed[i] = valid_extents[i][curr_index - 1]
                new_value = metric(h_perturbed, **kwargs)
                if new_value < min_value:
                    min_value = new_value
                    i_min = i
                    index_min = curr_index - 1

        if i_min is None:
            terminate = True
        else:
            h_min[i_min] = valid_extents[i_min][index_min]

    return h_min
