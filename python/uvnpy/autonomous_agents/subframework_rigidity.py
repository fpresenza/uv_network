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


interagent_msg = collections.namedtuple(
    'interagent_msg',
    'id, timestamp, position, covariance')

neighbor_data = collections.namedtuple(
    'neighbors',
    'id, timestamp, hops, range, position, covariance')


class single_integrator(object):
    def __init__(self, id, pos, est_pos, cov):
        ctrl_cov = 0.05**2 * np.eye(2)
        range_cov = 0.2
        self.id = id
        self.dim = len(pos)
        self.dm = integrator(pos)
        # self.ctrl = decentralized_rigidity_maintenance()
        self.loc = distances_to_neighbors_kalman(
            est_pos, cov, ctrl_cov, range_cov)
        self.neighbors = {}

    def send_msg(self, timestamp):
        msg = interagent_msg(
            id=self.id,
            timestamp=timestamp,
            position=self.loc.position,
            covariance=self.loc.covariance)
        return msg

    def receive_msg(self, msg, range_measurment):
        self.neighbors[msg.id] = neighbor_data(
            id=msg.id,
            timestamp=msg.timestamp,
            hops=1,
            range=range_measurment,
            position=msg.position,
            covariance=msg.covariance)

    def control_step(self, t):
        u = np.zeros(self.dim)
        self.dm.step(t, u)
        self.loc.prediction_step(t, u)

    def localization_step(self):
        if len(self.neighbors) > 0:
            neighbors_data = self.neighbors.values()
            xj = np.array([
                neighbor.position for neighbor in neighbors_data])
            Pj = np.array([
                neighbor.covariance for neighbor in neighbors_data])
            z = np.array([
                neighbor.range for neighbor in neighbors_data])
            self.loc.correction_step(z, xj, Pj)
            self.neighbors.clear()
