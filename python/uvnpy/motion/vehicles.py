#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:41:07 2020
@author: fran
"""
import numpy as np
from uvnpy.motion.holonomic import VelocityModel
from uvnpy.navigation.filters import Ekf

class UnmannedVehicle(object):
    """ This class implements a unmanned vehicle instance
    to use as node in a graph. """
    def __init__(self, id, *args, **kwargs):
        self.id = id
        self.type = kwargs.get('type', 'UnmannedVehicle')
        if not hasattr(self, 'motion'): self.motion = kwargs.get('motion', None)
        self.filter = kwargs.get('filter', Ekf())

    def __str__(self):
        return '{}({})'.format(self.type, self.id)


class Rover(UnmannedVehicle):
    def __init__(self, id, *args, **kwargs):
        kwargs['type'] = 'Rover'
        kwargs['dof'] =  3
        kwargs['ctrl_gain'] = (1., 1., 0.2)
        kwargs['sigma'] = np.array([0.2, 0.2, 0.1, 0.1, 0.1, 0.05])
        self.motion = VelocityModel(**kwargs)
        super(Rover, self).__init__(id, *args, **kwargs)

    def xy(self):
        return self.motion.x[3:5]

    def vxy(self):
        return self.motion.x[:2]