#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue May 06 14:15:45 2020
@author: fran
"""
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