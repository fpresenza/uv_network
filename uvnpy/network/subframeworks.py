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
def links(geodesics, extents):
    """
    Computes the set of links (edges not in any subframework).
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
        geodesics, extents, metric,
        max_extent=None,
        subframework_condition=None
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
    if max_extent is None:
        max_extent = np.max(geodesics).astype(int)
    n = len(extents)
    adjacency = geodesics.copy()
    adjacency[adjacency > 1] = 0
    search_space = list(
        itertools.product(
            *([0] + list(range(extents[i], max_extent + 1)) for i in range(n))
        )
    )

    min_value = np.inf
    for h in search_space:
        h = np.array(h)
        V = geodesics <= np.reshape(h, (-1, 1))
        # ensures that no zero extent subframework is analyzed
        C = V[h > 0]
        for i, c in enumerate(C):
            k = np.delete(C, i, axis=0)
            if np.all(~c + k, axis=1).any():
                # checks if c is contained in another subframework
                continue
            if subframework_condition is not None:
                if not subframework_condition(c, adjacency[c][:, c]):
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
    for f in list(search_factors):
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
    remain = [i for i in range(n) if extents[i] > 0]
    terminate = False

    min_value = np.inf
    while not terminate:
        remove_sub = None
        for i in remain:
            sparsed = hops.copy()
            sparsed[i] = 0
            new_value = metric(geodesics, sparsed)
            if new_value < min_value:
                min_value = new_value
                remove_sub = i

        if remove_sub is not None:
            hops[remove_sub] = 0
            remain.remove(remove_sub)
        else:
            terminate = True
    return hops


def sparse_subframeworks_extended_greedy_search(
        geodesics,
        extents,
        metric,
        initial_guess,
        *args
        ):
    """
    Given the set of valid extents of each node, starts with all subframeworks
    with zero radiues and selects, at each iteration, the extent change the
    reduces the metric the most.

    Requires:
    ---------
        framework is rigid
        extents is list of ordered lists in increasing order
    """
    n = len(geodesics)
    h_subopt = np.copy(initial_guess)
    nodes = np.arange(n)
    terminate = False

    min_value = metric(geodesics, h_subopt, *args)
    while not terminate:
        update_sub = None
        for i in nodes:
            if h_subopt[i] < extents[i][-1]:
                curr_index = extents[i].index(h_subopt[i])
                perturbed_extents = h_subopt.copy()
                perturbed_extents[i] = extents[i][curr_index + 1]
                new_value = metric(geodesics, perturbed_extents, *args)
                if new_value < min_value:
                    min_value = new_value
                    update_sub = i
                    new_index = curr_index + 1
            if h_subopt[i] > extents[i][0]:
                curr_index = extents[i].index(h_subopt[i])
                perturbed_extents = h_subopt.copy()
                perturbed_extents[i] = extents[i][curr_index - 1]
                new_value = metric(geodesics, perturbed_extents, *args)
                if new_value < min_value:
                    min_value = new_value
                    update_sub = i
                    new_index = curr_index - 1

        if update_sub is None:
            terminate = True
        else:
            h_subopt[update_sub] = extents[update_sub][new_index]

    return h_subopt
