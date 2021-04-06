#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:11:56 2020
@author: fran
"""


class vehiculo(object):
    def __init__(self, nombre, **kwargs):
        self.id = nombre
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        try:
            return '{}({})'.format(self.tipo, self.id)
        except AttributeError:
            return 'vehiculo({})'.format(self.id)

    def __repr__(self):
        return self.__str__()
