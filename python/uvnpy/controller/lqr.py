#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Wed May 20 14:27:16 2020
@author: fran
"""
import numpy as np
import control.matlab as cm
cm.use_numpy_matrix(False)

class LQR(object):
    def __init__(self, A, B, *args, **kwargs):
        if not np.linalg.matrix_rank(cm.ctrb(A,B)) == A.shape[0]:
            raise ValueError('The system is not controllabe')
        Q = kwargs.get('Q', np.eye(*A.shape))
        R = kwargs.get('R', np.eye(B.shape[1]))
        self.K, S, eig = cm.lqr(A, B, Q, R)
        self.A_cl = A - np.dot(B, self.K)

    def update(self, x):
       return -np.dot(self.K, x) 