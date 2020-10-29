#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Wed Jun 34 13:28:15 2020
@author: fran
"""
import numpy as np
from numpy import trace
from numpy.linalg import norm, det, eigvalsh  # noqa
from scipy.linalg import svdvals, sqrtm  # noqa

traza = trace
norma = norm


def sqrt_diagonal(M):
    return np.sqrt(np.diagonal(M))
