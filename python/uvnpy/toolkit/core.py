#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue May 06 14:15:45 2020
@author: fran
"""
from types import SimpleNamespace
import itertools
import numpy as np


def invertir_dic(dic):
    return {v: k for k, v in dic.items()}


class RecursiveNamespace(SimpleNamespace):
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)
        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))


def combinations(V, n):
    comb = itertools.combinations(V, n)
    return np.array(list(comb))


def permutations(V, n):
    perm = itertools.permutations(V, n)
    return np.array(list(perm))
