#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:41:07 2020
@author: fran
"""
import numpy as np
import graph_tool as gt
from uvnpy.vehicles.unmanned_vehicle import UnmannedVehicle
from uvnpy.vehicles.rover import Rover
from uvnpy.network.links import CommunicationLink
from uvnpy.toolkit.linalg import vector
from uvnpy.toolkit.pytools import iterable

def proximity(graph, i, range):
	all_links = graph.get_links().tolist()
	for j in graph.get_robots()[i:]:
		dist = graph.dist(i, j)
		if dist <= range and [i,j] not in all_links:
			graph.add_link(i,j)
		elif dist > range and [i,j] in all_links:
			graph.remove_link(i,j)


class UnmannedVehicleGraph(gt.Graph):
	def __init__(self, *args, **kwargs):
		self.wrapper = kwargs.pop('connect', None)
		super(UnmannedVehicleGraph, self).__init__(*args, **kwargs)
		if not hasattr(self, 'vertex_obj'): self.vertex_obj = kwargs.get('vertex_obj', UnmannedVehicle)
		if not hasattr(self, 'edge_obj'): self.edge_obj = kwargs.get('edge_obj', CommunicationLink)
		self.__ids = {}
		self.__idx = {}
		self.__set_property_maps()

	def __set_property_maps(self):
		""" set robot class and communication-link class as
		vertex and edge property respectively """
		self.vertex_properties['robots'] = self.new_vertex_property('object')
		self.edge_properties['links'] = self.new_edge_property('object')

	def __set_robot_as_vp(self, nv, **kwargs):
		""" Set robot class to new vertex added """
		obj = kwargs.get('object', self.vertex_obj)
		deploy = kwargs.get('deploy')
		nv = nv if iterable(nv) else [nv]
		for v in nv:
			try:
				id = max([r.id for r in self.vp.robots if hasattr(r, 'id')]) + 1
			except ValueError:
				id = 1
			try:
				kwargs.update({'pi':deploy(id)})
			except TypeError:
				raise TypeError('A function for pose deployment must be passed')
			self.vp.robots[v] = obj(id, **kwargs)

	def __set_link_as_ep(self, ne, s, t, obj):
		""" Set communication link class to new edge added """
		self.ep.links[ne] = obj(s, t)

	def __updt_hash(self):
		""" Updates the two dictonaries which points
		from vertex idx to ids and viceversa """
		indices = self.get_vertices()
		ids = [self.vp.robots[idx].id for idx in indices]
		self.__ids = dict(zip(indices, ids))
		self.__idx = dict(zip(ids, indices))

	def idx(self, *ids):
		""" given a list of robot ids, return a list of
		vertex indices """
		return [self.__idx[id] for id in ids]

	def ids(self, *indices):
		""" given a list of vertex indices, return a 
		list of robot ids """		
		return [self.__ids[idx] for idx in indices]

	def add_robots(self, N, **kwargs):
		""" Add N vertices to the graph with a robot as
		a vertex property """
		new_vertices = self.add_vertex(N)
		self.__set_robot_as_vp(new_vertices, **kwargs)
		self.__updt_hash()
		return new_vertices

	def add_link(self, s, t, **kwargs):
		""" Add one edge between robot s (source) and t (target) 
		and set communication link as edge property"""
		obj = kwargs.get('object', self.edge_obj)
		try:
			i, j = self.__idx[s], self.__idx[t]
		except KeyError:
			raise KeyError('Invalid robot id')
		new_edge = self.add_edge(i, j, add_missing=False)
		self.__set_link_as_ep(new_edge, self.r(s), self.r(t), obj)
		return new_edge

	def remove_robot(self, *ids):
		""" Remove a list of robots from graph """
		try:
			idx = [self.__idx[id] for id in ids]
		except KeyError:
			raise KeyError('Invalid robot id')
		self.remove_vertex(idx)
		self.__updt_hash()

	def remove_link(self, s, t):
		""" Remove a link from graph """
		try:
			i, j = self.__idx[s], self.__idx[t]
		except KeyError:
			raise KeyError('Invalid robot id')
		e = self.edge(i, j)
		if e:
			self.remove_edge(e)
		else:
			raise KeyError('No communication link to remove')

	def robots(self):
		""" returns an iterable of all robots objects """
		return self.vp.robots

	def links(self):
		""" returns an iterable of all communication links """
		return self.ep.links

	def get_robots(self):
		""" returns an array of all robots ids """
		return np.array([self.__ids[idx] for idx in self.get_vertices()])

	def get_links(self):
		""" returns an array of all communication links ids """
		return np.array([self.ids(s, t) for (s, t) in self.get_edges()])

	def v(self, id):
		""" return vertex where is stored the given robot """
		try:
			return self.vertex(self.__idx[id])
		except KeyError:
			raise KeyError('Invalid robot id')

	def r(self, id):
		""" return the object of given robot """
		try:
			return self.vp.robots[self.__idx[id]]
		except KeyError:
			raise KeyError('Invalid robot id')

	def e(self, s, t):
		""" return edge where is stored given communication link """
		try:
			e = self.edge(self.__idx[s], self.__idx[t])
		except KeyError:
			raise KeyError('Invalid robot id')
		if e:
			return e
		else:
			raise KeyError('No communication link')

	def l(self, s, t):
		""" returns the object of given communication link """
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
		print('---\n\nRobots:')
		for r in self.robots():
			print('{} at index {}'.format(r, self.__idx[r.id]))
		print('---\n\nCommunication Links:')
		for l in self.links():
			print(l)

	def dist(self, i, j):
		""" returns the euclidean distance between two robots """
		return vector.distance(self.r(i).xyz(), self.r(j).xyz())

	def connect(self, *args):
		return self.wrapper(self, *args)

	def share_msgs(self):
		for link in self.links():
			r = link.range_measurement()
			link.s.msg.range = r
			link.t.msg.range = r
			link.s.inbox.append(link.t.msg.copy())
			link.t.inbox.append(link.s.msg.copy())

	def share_between(self, i, j):
		if [i,j] in self.get_links().tolist():
			self.r(i).inbox.append(self.r(j).msg)
			self.r(j).inbox.append(self.r(i).msg)
		else:
			raise KeyError('No communication link')


class RoverGraph(UnmannedVehicleGraph):
	def __init__(self, *args, **kwargs):
		self.vertex_obj = Rover
		super(RoverGraph, self).__init__(*args, **kwargs)