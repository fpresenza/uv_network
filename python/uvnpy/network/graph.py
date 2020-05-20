#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:41:07 2020
@author: fran
"""
import numpy as np
import graph_tool as gt
from uvnpy.motion.vehicles import UnmannedVehicle, Rover
from uvnpy.network.links import CommunicationLink
from uvnpy.tools.tools import iterable

class UnmannedVehicleGraph(gt.Graph):
	def __init__(self, *args, **kwargs):
		super(UnmannedVehicleGraph, self).__init__(*args, **kwargs)
		if not hasattr(self, 'vobj'): self.vobj = kwargs.get('vobj', UnmannedVehicle)
		if not hasattr(self, 'eobj'): self.eobj = kwargs.get('eobj', CommunicationLink)
		self.__ids = {}
		self.__idx = {}
		self.__set_property_maps()

	def __set_property_maps(self):
		self.vertex_properties['robots'] = self.new_vertex_property('object')
		self.edge_properties['links'] = self.new_edge_property('object')

	def __set_robot_as_vp(self, nv, **kwargs):
		obj = kwargs.get('object', self.vobj)
		pi = kwargs.get('pi', None)
		nv = nv if iterable(nv) else [nv]
		for v in nv:
			try:
				id = np.max([r.id for r in self.vp.robots if hasattr(r, 'id')]) + 1
			except ValueError:
				id = 1
			if pi is not None:
				self.vp.robots[v] = obj(id, pi=pi())
			else: 
				self.vp.robots[v] = obj(id)

	def __set_link_as_ep(self, ne, s, t,  obj):
		self.ep.links[ne] = obj(s, t)

	def __updt_hash(self):
		self.__ids.clear()
		self.__idx.clear()
		for idx in self.get_vertices():
			id = self.vp.robots[idx].id
			self.__ids[idx] = id
			self.__idx[id] = idx

	def idx(self, *ids):
		return [self.__idx[id] for id in ids]

	def ids(self, *indices):
		return [self.__ids[idx] for idx in indices]

	def add_robots(self, N, **kwargs):
		new_vertices = self.add_vertex(N)
		self.__set_robot_as_vp(new_vertices, **kwargs)
		self.__updt_hash()
		return new_vertices

	def add_link(self, s, t, **kwargs):
		obj = kwargs.get('object', self.eobj)
		try:
			i, j = self.__idx[s], self.__idx[t]
		except KeyError:
			raise KeyError('Invalid robot id')
		new_edge = self.add_edge(i, j, add_missing=False)
		self.__set_link_as_ep(new_edge, self.r(s), self.r(t), obj)
		return new_edge

	def remove_robot(self, *ids):
		try:
			idx = [self.__idx[id] for id in ids]
		except KeyError:
			raise KeyError('Invalid robot id')
		self.remove_vertex(idx)
		self.__updt_hash()

	def remove_link(self, s, t):
		try:
			i, j = self.__idx[s], self.__idx[t]
		except KeyError:
			raise KeyError('Invalid robot id')
		e = self.edge(i, j)
		if e:
			self.remove_edge(e)
		else:
			raise KeyError('No communication link')

	def robots(self):
		return self.vp.robots

	def links(self):
		return self.ep.links

	def get_robots(self):
		return np.array([self.__ids[idx] for idx in self.get_vertices()])

	def get_links(self):
		return np.array([self.ids(s, t) for (s, t) in self.get_edges()])

	def v(self, id):
		try:
			return self.vertex(self.__idx[id])
		except KeyError:
			raise KeyError('Invalid robot id')

	def r(self, id):
		try:
			return self.vp.robots[self.__idx[id]]
		except KeyError:
			raise KeyError('Invalid robot id')

	def e(self, s, t):
		try:
			e = self.edge(self.__idx[s], self.__idx[t])
		except KeyError:
			raise KeyError('Invalid robot id')
		if e:
			return e
		else:
			raise KeyError('No communication link')

	def l(self, s, t):
		try:
			e = self.edge(self.__idx[s], self.__idx[t])
		except KeyError:
			raise KeyError('Invalid robot id')
		if e:
			return self.ep.links[e]
		else:
			raise KeyError('No communication link')

	def screen(self): 
		print(self)
		for r in self.robots():
			print('{} at index {}'.format(r, self.__idx[r.id]))
		print('---')
		for l in self.links():
			print(l)

	def dist(self, i, j):
		return np.linalg.norm(self.r(i).xy() - self.r(j).xy())


class RoverGraph(UnmannedVehicleGraph):
	def __init__(self, *args, **kwargs):
		self.vobj = Rover
		super(RoverGraph, self).__init__(*args, **kwargs)