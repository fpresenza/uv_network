#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
"""
import numpy as np


class Targets(object):
    def __init__(self, n, dim, low_lim, up_lim, collect_radius):
        self.positions = np.random.uniform(low_lim, up_lim, (n, dim))
        self.active = np.full(n, True)
        self.collect_radius = collect_radius

    def allocation(self, p):
        q = self.positions[self.active]
        r = p[:, np.newaxis] - q
        d2 = np.square(r).sum(axis=-1)
        return [q[d2[i].argmin()] for i in range(len(p))]

    def update(self, p):
        r = p[:, np.newaxis] - self.positions[self.active]
        d2 = np.square(r).sum(axis=-1)
        c2 = (d2 < self.collect_radius**2).any(axis=0)
        self.active[np.where(self.active)[0][c2]] = False


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
