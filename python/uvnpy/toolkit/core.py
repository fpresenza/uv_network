#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue May 06 14:15:45 2020
@author: fran
"""
from types import SimpleNamespace

__all__ = [
  'invertir_dic',
  'iterable',
  'RecursiveNamespace'
]


def invertir_dic(dic):
    return {v: k for k, v in dic.items()}


def iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


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
