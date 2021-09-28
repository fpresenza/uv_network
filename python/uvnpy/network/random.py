#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue sep 23 09:24:20 -03 2021
"""
import numpy as np


def erdos_renyi(n, prob):
    """La probabilidad de existencia de cada enlace es prob"""
    ij = np.triu((1 - np.eye(n))).astype(bool)
    a = np.random.choice([0, 1], size=int(n*(n-1)/2), p=(1-prob, prob))
    A = np.zeros((n, n))
    A[ij] = a
    A = A + A.T
    return A
