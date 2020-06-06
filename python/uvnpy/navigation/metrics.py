#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Wed Jun 34 13:28:15 2020
@author: fran
"""
import numpy as np

class metrics(object):
    """ This class implements different types of metrics to evaluate 
    performance of navigation filters """
    @staticmethod
    def sqrt_diagonal(M):
        return np.sqrt(np.diagonal(M))

    @staticmethod
    def sqrt_det(M):
        return np.sqrt(np.linalg.det(M))

    @staticmethod
    def sqrt_trace(M):
        return np.sqrt(np.trace(M))

    @staticmethod
    def sqrt_covar(M):
        eigval, V = np.linalg.eigh(M)
        return V @ np.diag(np.sqrt(eigval)) @ V.T