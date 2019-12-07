#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 5 16:16:54 2020

@author: fran
"""

#   Import numpy to process data
import numpy as np
#   Import the messages we're interested in sending and receiving
from std_msgs.msg import Header
from uv_network.msg import FloatArrayStamped, NavFilter


class MissionTools(object):
    """
    This class contains info and modules operate vehicles
    based on mission planning and behavior such as 
    communication constraints or bounded motion, etc.
    """
    def __init__(self, vel_mean):
        self.random_vel = vel_mean
        self.random_vel_seq = 1
    

    # @staticmethod
    def limit_xy(self, pose, bound, gain=1.):
        """
        This function takes the pose of the vehicle
        and returns a velocity. If pose is inside
        operation area, then action is zero,
        otherwise generate velocity to constraint pose
        to operation area. 
        """
        action = np.array([0.,0.,0.])
        x = pose[0]
        y = pose[1]

        if np.abs(x) >= bound[0]:
            action[0] = - np.sign(x) * gain * (np.abs(x)-bound[0])
        
        if np.abs(y) >= bound[1]:
            action[1] = - np.sign(y) * gain * (np.abs(y)-bound[1])

        return action

    

    # @staticmethod
    def random_vel_generator(self, mean, covar, wait_seq=1):
        """
        This function generates a normal random signal
        with mean and covariance matrix. Updates vel
        at every entry if wait_seq is 1, otherwise wait n
        times to update.
        """
        if self.random_vel_seq > wait_seq:
            self.random_vel = np.random.multivariate_normal(mean, covar)
            self.random_vel_seq = 1

        self.random_vel_seq += 1

        return self.random_vel