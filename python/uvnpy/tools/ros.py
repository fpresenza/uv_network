#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue May 06 14:15:45 2020
@author: fran
"""
import numpy as np

# class ROSmsg(object):
# 	def __init__(self, *args, **kwargs):

# 	def __repr__(self):
# 		return self.str


class Point(object):
    def __init__(self, *args, **kwargs):
    	self.x = kwargs.get('x', args[0] if args else 0.)
    	self.y = kwargs.get('y', args[1] if args else 0.)
    	self.z = kwargs.get('z', args[2] if args else 0.)

    def __str__(self):
    	return 'Point:\nx: {}\ny: {}\nz: {}'.format(self.x, self.y, self.z)


class Vector3(object):
    def __init__(self, *args, **kwargs):
    	self.x = kwargs.get('x', args[0] if args else 0.)
    	self.y = kwargs.get('y', args[1] if args else 0.)
    	self.z = kwargs.get('z', args[2] if args else 0.)

    def __str__(self):
    	return 'Vector3:\nx: {}\ny: {}\nz: {}'.format(self.x, self.y, self.z)


class Quaternion(object):
    def __init__(self, *args, **kwargs):
    	self.x = kwargs.get('x', args[0] if args else 0.)
    	self.y = kwargs.get('y', args[1] if args else 0.)
    	self.z = kwargs.get('z', args[2] if args else 0.)
    	self.w = kwargs.get('z', args[3] if args else 0.)

    def __str__(self):
    	return 'Quaternion:\nx: {}\ny: {}\nz: {}\nw: {}'.format(self.x, self.y, self.z, self.w)


class Pose(object):
	def __init__(self, *args, **kwargs):
		self.pose = kwargs.get('pose', Point())
		self.orientation = kwargs.get('orientation', Quaternion())

	def __str__(self):
		return 'Pose:\n{}\n{}'.format(self.pose.__str__(), self.orientation.__str__())

