#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue May 06 14:15:45 2020
@author: fran
"""
import numpy as np
import copy


class Point(object):
    def __init__(self, *args, **kwargs):
        self.x = kwargs.get('x', args[0] if args else 0.)
        self.y = kwargs.get('y', args[1] if args else 0.)
        self.z = kwargs.get('z', args[2] if args else 0.)

    def __str__(self):
        return 'Point:\nx: {}\ny: {}\nz: {}'.format(self.x, self.y, self.z)


class Vector3(object):
    def __init__(self, *args, **kwargs):
        self.x = kwargs.get('x', args[0] if args else 0.)
        self.y = kwargs.get('y', args[1] if args else 0.)
        self.z = kwargs.get('z', args[2] if args else 0.)

    def __str__(self):
        return 'Vector3:\nx: {}\ny: {}\nz: {}'.format(self.x, self.y, self.z)


class Quaternion(object):
    def __init__(self, *args, **kwargs):
        self.x = kwargs.get('x', args[0] if args else 0.)
        self.y = kwargs.get('y', args[1] if args else 0.)
        self.z = kwargs.get('z', args[2] if args else 0.)
        self.w = kwargs.get('z', args[3] if args else 0.)

    def __str__(self):
        out = 'Quaternion:\nx: {}\ny: {}\nz: {}\nw: {}'
        return out.format(self.x, self.y, self.z, self.w)


class Pose(object):
    def __init__(self, **kwargs):
        self.pose = kwargs.get('pose', Point())
        self.orientation = kwargs.get('orientation', Quaternion())

    def __str__(self):
        pose_str = self.pose.__str__()
        orien_str = self.orientation.__str__()
        return 'Pose:\n{}\n{}'.format(pose_str, orien_str)


class PositionAndRange(object):
    def __init__(self, id, x=0., y=0., z=0., source='Unknown'):
        self.source = source
        self.id = id
        self.point = np.array([x, y, z])
        self.covariance = np.zeros(9)
        self.range = range

    def __str__(self):
        out = 'id: {}\nPoint:\nx: {}\ny: {}\nz: {}\ncovariance: {}\nrange: {}'
        return out.format(
            self.id,
            self.point[0],
            self.point[1],
            self.point[2],
            self.covariance,
            self.range)

    def copy(self):
        return copy.deepcopy(self)
