#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue June 23 14:27:46 2020
@author: fran
"""
import numpy as np
import yaml
from types import SimpleNamespace
from uvnpy.toolkit.linalg import vector

class RangeSensor(object):
    """ This class implements model of a Range Sensor """
    def __init__(self, name, **kwargs):
        # read config file
        config_dict = yaml.load(open('../config/{}.yaml'.format(name)))
        config = SimpleNamespace(**config_dict)
        self.range = SimpleNamespace(**config.range)
        #  Noise covariance matrices
        self.R = np.diag(np.square([self.range.sigma]))

    def measurement(self, p, q):
        """ Calculates the diantance range based on two robot's position.
        """
        return (vector.distance(p, q) + np.random.normal(0, self.range.sigma)).item()