#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 08 19:00:32 2020
@author: fran
"""
import numpy as np
import collections
from uvnpy.sensor.range import RangeSensor

Antenna = collections.namedtuple('Antenna', 'd0 P0 gain n')

def all_equal_protocol(link):
    r = link.range_measurement()
    msg = link.t.msg.copy()
    msg.range = r
    link.s.inbox.append(msg)
    msg = link.s.msg.copy()
    msg.range = r
    link.t.inbox.append(msg)

def diff_protocol(link):
    r = link.range_measurement()
    if link.s.type is 'Rover':
        if link.t.type is 'Rover':
            msg = link.t.msg.copy()
            msg.range = r
            link.s.inbox.append(msg)
            msg = link.s.msg.copy()
            msg.range = r
            link.t.inbox.append(msg)
        elif link.t.type is 'Drone':
            msg = link.t.msg.copy()
            link.s.inbox.append(msg)
            msg = link.s.msg.copy()
            msg.range = r
            link.t.inbox.append(msg)
    elif link.s.type is 'Drone':
        if link.t.type is 'Rover':
            msg = link.s.msg.copy()
            link.t.inbox.append(msg)
            msg = link.t.msg.copy()
            msg.range = r
            link.s.inbox.append(msg)
        if link.t.type is 'Drone':
            pass


class CommunicationLink(object):
    def __init__(self, source, target, *args, **kwargs):
        self.s = source
        self.t = target
        self.type = kwargs.get('type', 'CommunicationLink')
        self.rssi = 0. # dBm
        self.antenna = Antenna(1, -15, -4, 2)
        self.range = RangeSensor()

    def __str__(self):
        return '{}({},{})'.format(self.type, self.s, self.t)

    def range_measurement(self):
        return self.range.measurement(self.s.p(), self.t.p())

    def rss(self):
        d = self.dist()
        d0, P0, gain, n = self.antenna
        self.rssi = P0 - 10*n*np.log10(d/d0) + gain + np.random.normal(0,5)
        return self.rssi

    def exchange(self):
        all_equal_protocol(self)


