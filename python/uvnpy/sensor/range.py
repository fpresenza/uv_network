#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue June 23 14:27:46 2020
@author: fran
"""
import numpy as np
import yaml
from types import SimpleNamespace
import uvnpy.toolkit.linalg as linalg

class RangeSensor(object):
    """ This class implements model of a Range Sensor """
    def __init__(self, cnfg_file='../config/xbee.yaml'):
        # read config file
        config_dict = yaml.load(open(cnfg_file))
        config = SimpleNamespace(**config_dict)
        self.sigma = config.range['sigma']
        #  Noise covariance matrices
        self.R = np.diag([np.square(self.sigma)])

    def measurement(self, p, q):
        """ Calculates the diantance range based on two robot's position. """
        return linalg.distance(p, q) + np.random.normal(0, self.sigma)