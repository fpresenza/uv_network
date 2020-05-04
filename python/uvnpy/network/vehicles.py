#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:41:07 2020
@author: fran
"""
import numpy as np
from uvnpy.motion.holonomic import VelocityModel

class UnmannedVehicle(object):
    """ This class implements a unmanned vehicle instance
    to use as node in a graph. """
    def __init__(self, id, *args, **kwargs):
        self.id = id
        self.type = kwargs.get('type', 'UnmannedVehicle')
        self.motion = kwargs.get('motion', VelocityModel())

    def __str__(self):
        return '{}({})'.format(self.type, self.id)

    def __repr__(self):
        return self.__str__()


class Rover(UnmannedVehicle):
    def __init__(self, id, *args, **kwargs):
        motion = {'dof': 3, 'ctrl_gain':(1.,1.,0.2), 'alphas':[[0.2,0.2,0.1],[0.1,0.1,0.05]]}
        kwargs = {'type': 'Rover', 'motion': VelocityModel(**motion)}
        super(Rover, self).__init__(id, *args, **kwargs)