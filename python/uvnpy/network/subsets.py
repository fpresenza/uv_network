#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mié ago 11 10:16:00 -03 2021
"""


def neighborhood(A, i, inclusive=False):
    N = A.astype(bool)
    idx = N[i]
    idx[i] = inclusive
    return idx


def multihop_neighborhood(A, i, hops=1, inclusive=False):
    N = A.astype(bool)
    idx = N[i]
    for h in range(hops - 1):
        idx = N[idx].any(0)
    idx[i] = inclusive
    return idx


def subframework(A, x, i):
    Ni = neighborhood(A, i, True)
    Ai = A[Ni][:, Ni]
    xi = x[Ni]
    return Ai, xi


def multihop_subframework(A, x, i, hops=1):
    Ni = multihop_neighborhood(A, i, hops, True)
    Ai = A[Ni][:, Ni]
    xi = x[Ni]
    return Ai, xi
