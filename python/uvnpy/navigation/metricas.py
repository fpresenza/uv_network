#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Wed Jun 34 13:28:15 2020
@author: fran
"""
import numpy as np

def sqrt_diagonal(M):
    return np.sqrt(np.diagonal(M))

def sqrt_det(M):
    return np.sqrt(np.linalg.det(M))

def sqrt_trace(M):
    return np.sqrt(np.trace(M))

def sqrt_covar(M):
    eigval, V = np.linalg.eigh(M)
    sqrt_M = np.diag(np.sqrt(eigval))
    return np.matmul( V, np.matmul(sqrt_M, V.T) )

def eigvals(M):
    return np.linalg.eigvalsh(M)