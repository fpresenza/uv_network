#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue June 23 14:27:46 2020
@author: fran
"""
import numpy as np
import gpsic.toolkit.linalg as linalg


class Rango(object):
    """ Esta clase implementa un modelo de sensor de rango """
    def __init__(self, sigma=1.):
        self.sigma = sigma
        #  Noise covariance matrices
        self.R = np.diag([np.square(self.sigma)])

    def measurement(self, p, q):
        """ Simula una medición ruidosa """
        return linalg.dist(p, q) + np.random.normal(0, self.sigma)

    @staticmethod
    def gradiente(p, q):
        return np.subtract(p, q)/linalg.dist(p, q)

    @classmethod
    def proyeccion(cls, p, q):
        r = cls.gradiente(p, q).reshape(-1, 1)
        return np.matmul(r, r.T)

    @classmethod
    def collection_matrix(cls, p, landmarks, sigma):
        return sum([sigma**(-2) * cls.proyeccion(p, q) for q in landmarks])

    @classmethod
    def collection_matrix_det(cls, u, y_p, Q, landmarks, sigma):
        cm_set = [cls.collection_matrix(y, landmarks, sigma) for y in y_p]
        return sum([np.linalg.det(cm) for cm in cm_set])
