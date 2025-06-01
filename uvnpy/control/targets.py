#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
"""
import numpy as np


class Targets(object):
    def __init__(self, n, dim, low_lim, up_lim, coverage):
        self.dim = dim
        self.data = np.empty((n, dim + 1), dtype=object)
        self.data[:, :dim] = np.random.uniform(low_lim, up_lim, (n, dim))
        self.data[:, dim] = True
        self.coverage = coverage

    def position(self):
        return self.data[:, :self.dim]

    def untracked(self):
        return self.data[:, self.dim]

    def allocation(self, p):
        alloc = {i: None for i in range(len(p))}
        untracked = self.data[:, self.dim].astype(bool)
        if untracked.any():
            targets = self.data[untracked, :self.dim].astype(float)
            r = p[:, None] - targets
            d2 = np.square(r).sum(axis=-1)
            for i in range(len(p)):
                j = d2[i].argmin()
                alloc[i] = targets[j]

        return alloc

    def update(self, p):
        r = p[..., None, :] - self.data[:, :self.dim]
        d2 = np.square(r).sum(axis=-1)
        c2 = (d2 < self.coverage**2).any(axis=0)
        self.data[c2, self.dim] = False

    def unfinished(self):
        return self.data[:, self.dim].any()


class TargetTracking(object):
    def __init__(self, tracking_radius, forget_radius, v_max):
        self.tracking_radius = tracking_radius
        self.forget_radius = forget_radius
        self.v_max = v_max

    def update(self, position, target):
        r = position - target
        d = np.sqrt(np.square(r).sum())
        if d < self.tracking_radius:
            v_tracking = self.v_max
        elif d < self.forget_radius:
            factor = (self.forget_radius - d) / \
                (self.forget_radius - self.tracking_radius)
            v_tracking = self.v_max * factor
        else:
            v_tracking = 0.0
        return - v_tracking * r / d
