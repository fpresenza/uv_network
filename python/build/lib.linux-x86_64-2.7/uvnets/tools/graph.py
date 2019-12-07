#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:41:07 2020
@author: fran
"""
import numpy as np

__all__ = ['Custom']

class Graph(object):
    def __init__(self, *args, **kwargs):
        self.V = kwargs.get('V', [1]) 
        self.E = kwargs.get('E', [(i, j) for i in self.V for j in self.V if i!=j and self.connect(i, j, *args)])
        self.W = np.array([[1 if (i,j) in self.E else 0 for j in self.V] for i in self.V])
        self.size = len(self.V)


    # def __call__(self):
    #     return self.neighbors

    # def __len__(self):
    #     return len(self.neighbors)

    # def incomplete(self):
    #     return len(self.neighbors) < self.size

    # def append(self, id):
    #     self.neighbors.append(id)


class Custom(Graph):
    def __init__(self, *args, **kwargs):
        self.connect = kwargs.get('connect', lambda i, j: True)
        super(Custom, self).__init__(*args, **kwargs)


class Random(Graph):
    def __init__(self, *args, **kwargs):
        self.connect = lambda i, j: np.random.choice([True, False])
        super(Random, self).__init__(*args, **kwargs)