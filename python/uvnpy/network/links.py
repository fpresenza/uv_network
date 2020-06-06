#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 08 19:00:32 2020
@author: fran
"""
import numpy as np
import collections
from uvnpy.tools.linalg import vector
from uvnpy.sensor.range import RangeSensor

Antenna = collections.namedtuple('Antenna', 'd0 P0 gain n')

class CommunicationLink(object):
    def __init__(self, source, target, *args, **kwargs):
        self.s = source
        self.t = target
        self.type = kwargs.get('type', 'CommunicationLink')
        self.rssi = 0. # dBm
        self.antenna = Antenna(1, -15, -4, 2)
        self.range = RangeSensor('xbee')

    def __str__(self):
        return '{}({},{})'.format(self.type, self.s, self.t)

    def dist(self):
        return vector.distance(self.s.xyz(), self.t.xyz())

    def range_measurement(self):
        return self.range.measurement(self.s.xyz(), self.t.xyz())

    def rss(self):
        d = self.dist()
        d0, P0, gain, n = self.antenna
        self.rssi = P0 - 10*n*np.log10(d/d0) + gain + np.random.normal(0,5)
        return self.rssi