#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:41:07 2020
@author: fran
"""
import numpy as np
import collections

class Neighborhood(object):
    def __init__(self, **kwargs):
        self.size = kwargs.get('size', 5)
        self.dof = kwargs.get('dof', 3)
        self.dim = self.size * self.dof
        self.neighbors = collections.OrderedDict()

    def __call__(self):
        return list(self.neighbors.keys())

    def __getitem__(self, key): 
        return self.neighbors[key]

    def update(self, id, info=''):
        if id in self.neighbors.keys() or len(self.neighbors)<self.size:
            self.neighbors.update({id:info})
    
    def index(self, id, start=0):
        order = self().index(id)
        p_idx = start + self.dof*order
        v_idx = p_idx + self.dof*self.size
        return slice(p_idx, p_idx+self.dof), slice(v_idx, v_idx+self.dof)