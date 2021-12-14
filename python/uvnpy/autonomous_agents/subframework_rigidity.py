#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue 09 dic 2021 12:16:18 -03
"""
import numpy as np
import collections
from uvnpy.model.linear_models import integrator
# from uvnpy.rsn.control import decentralized_rigidity_maintenance
from uvnpy.rsn.localization import distances_to_neighbors_kalman


InterAgentMsg = collections.namedtuple(
    'InterAgentMsg',
    'id, \
    timestamp, \
    position, \
    covariance, \
    tokens')

NeigborhoodData = collections.namedtuple(
    'neighbors',
    'id, \
    timestamp, \
    range, \
    position, \
    covariance')

TokenData = collections.namedtuple(
    'TokenData',
    'center, \
    timestamp, \
    extent, \
    geodesic')

TokenData.__new__.__defaults__ = (
    None,
    None,
    None,
    np.inf)


class single_integrator(object):
    def __init__(self, id, pos, est_pos, cov, extent=None, t=0):
        self.id = id
        self.dim = len(pos)
        self.extent = extent
        self.current_time = t
        self.dm = integrator(pos)
        # self.ctrl = decentralized_rigidity_maintenance()
        ctrl_cov = 0.05**2 * np.eye(self.dim)
        range_cov = 1.
        self.gps_cov = gps_cov = 1. * np.eye(self.dim)
        self.loc = distances_to_neighbors_kalman(
            est_pos, cov, ctrl_cov, range_cov, gps_cov)
        self.neighbors = {}
        self.inclusion_group = {
            self.id: TokenData(
                center=self.id,
                timestamp=self.current_time,
                extent=self.extent,
                geodesic=0)
        }
        self.gps = {}

    def update_time(self, t):
        self.current_time = t

    def send_msg(self):
        IG = self.inclusion_group
        tokens = [
            token for token in IG.values() if token.geodesic < token.extent]
        msg = InterAgentMsg(
            id=self.id,
            timestamp=self.current_time,
            position=self.loc.position,
            covariance=self.loc.covariance,
            tokens=tokens)
        # self.inclusion_group = {self.id = self.inclusion_group[self.id]}
        return msg

    def receive_msg(self, msg, range_measurment):
        self.neighbors[msg.id] = NeigborhoodData(
            id=msg.id,
            timestamp=msg.timestamp,
            range=range_measurment,
            position=msg.position,
            covariance=msg.covariance)
        IG = self.inclusion_group
        for token in msg.tokens:
            tc = token.center
            IG[tc] = IG.get(tc, TokenData(center=tc))
            current_geodesic = IG[tc].geodesic
            IG[tc] = IG[tc]._replace(
                timestamp=token.timestamp,
                extent=token.extent,
                geodesic=min(current_geodesic, token.geodesic + 1))

    def control_step(self):
        u = np.zeros(self.dim)
        self.dm.step(self.current_time, u)
        self.loc.dynamic_step(self.current_time, u)

    def localization_step(self):
        if len(self.neighbors) > 0:
            neighbors_data = self.neighbors.values()
            z = np.array([
                neighbor.range for neighbor in neighbors_data])
            xj = np.array([
                neighbor.position for neighbor in neighbors_data])
            Pj = np.array([
                neighbor.covariance for neighbor in neighbors_data])
            self.loc.distances_step(z, xj, Pj)
            self.neighbors.clear()
        if len(self.gps) > 0:
            z = self.gps[max(self.gps.keys())]
            self.loc.gps_step(z)
            self.gps.clear()
