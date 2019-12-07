#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:41:07 2020
@author: fran
"""
import numpy as np

class Neighborhood(object):
    def __init__(self, **kwargs):
        self.size = kwargs.get('size', 0)
        self.dim = kwargs.get('dim', 1)
        self.count = 0
        self.neighbors = []

    def __call__(self):
        return self.neighbors

    def __len__(self):
        return len(self.neighbors)

    def incomplete(self):
        return len(self.neighbors) < self.size

    def append(self, id):
        self.neighbors.append(id)
    
    def index(self, id, *start):
        order = self.neighbors.index(id)
        v_idx = sum(start) + self.dim*order
        p_idx = v_idx + self.dim*self.size
        return v_idx, p_idx