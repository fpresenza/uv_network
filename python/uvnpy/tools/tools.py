#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue May 06 14:15:45 2020
@author: fran
"""
import numpy as np
import collections
import recordclass

Vec3 = recordclass.recordclass('Vec3', 'x y z', defaults=(0,))

def from_arrays(arrays):
    as_list = map(lambda a: a.flat, arrays)
    return list(zip(*as_list))

def iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


class Rotation(object):
    def __init__(self, euler_seq):
        self.euler_seq = 'ZYX'

    @staticmethod
    def Rx(roll):
        cr = np.cos(roll)
        sr = np.sin(roll)
        return np.array([[1,  0,   0],
                         [0, cr, -sr],
                         [0, sr,  cr]])

    @staticmethod
    def Ry(pitch):
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        return np.array([[ cp,  0, sp],
                         [  0,  1,  0],
                         [-sp,  0, cp]])

    @staticmethod
    def Rz(yaw):
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        return np.array([[cy, -sy, 0],
                         [sy,  cy, 0],
                         [ 0,   0, 1]])

    @classmethod
    def Rzyx(cls, r, p, y):
        return cls.Rz(y) @ cls.Ry(p) @ cls.Rx(r)