#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:11:56 2020
@author: fran
"""


class UnmannedVehicle(object):
    """ This class implements a unmanned vehicle instance
    to use as node in a graph. """
    def __init__(self, name, type='UnmannedVehicle', **kwargs):
        self.id = name
        self.type = type

    def __str__(self):
        return '{}({})'.format(self.type, self.id)
