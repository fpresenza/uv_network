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
from uvnpy.autonomous_agents import routing_protocols


InterAgentMsg = collections.namedtuple(
    'InterAgentMsg',
    'id, \
    timestamp, \
    action_tokens, \
    state_tokens')


NeigborhoodData = collections.namedtuple(
    'NeigborhoodData',
    'id, \
    timestamp, \
    covariance, \
    position, \
    range')


class Neighborhood(dict):
    def update(self, id, timestamp, position, covariance, range_measurement):
        self[id] = NeigborhoodData(
            id=id,
            timestamp=timestamp,
            position=position,
            covariance=covariance,
            range=range_measurement)


class single_integrator(object):
    def __init__(self, id, pos, est_pos, cov, extent=None, t=0):
        self.id = id
        self.dim = len(pos)
        self.extent = extent
        self.current_time = t
        self.dm = integrator(pos)
        self.control_action = np.zeros(self.dim)
        # self.ctrl = decentralized_rigidity_maintenance()
        ctrl_cov = 0.05**2 * np.eye(self.dim)
        range_cov = 1.
        self.gps_cov = gps_cov = 1. * np.eye(self.dim)
        self.loc = distances_to_neighbors_kalman(
            est_pos, cov, ctrl_cov, range_cov, gps_cov)
        self.neighborhood = Neighborhood()
        self.routing = routing_protocols.subframework_rigidity(
            self.id, self.extent, self.current_time)
        self.gps = {}

    def update_time(self, t):
        self.current_time = t

    def send_msg(self):
        action_tokens, state_tokens = self.routing.broadcast(
            self.current_time,
            self.control_action,
            self.loc.position,
            self.loc.covariance)
        msg = InterAgentMsg(
            id=self.id,
            timestamp=self.current_time,
            action_tokens=action_tokens,
            state_tokens=state_tokens)
        return msg

    def receive_msg(self, msg, range_measurement):
        self.neighborhood.update(
            msg.id, msg.timestamp,
            msg.state_tokens[msg.id].data.position,
            msg.state_tokens[msg.id].data.covariance,
            range_measurement)
        routing = self.routing
        [routing.update_action(tkn) for tkn in msg.action_tokens.values()]
        [routing.update_state(tkn) for tkn in msg.state_tokens.values()]

    def control_step(self):
        self.control_action = np.zeros(self.dim)
        self.dm.step(self.current_time, self.control_action)

    def localization_step(self):
        self.loc.dynamic_step(self.current_time, self.control_action)
        if len(self.neighborhood) > 0:
            neighbors_data = self.neighborhood.values()
            z = np.array([
                neighbor.range for neighbor in neighbors_data])
            xj = np.array([
                neighbor.position for neighbor in neighbors_data])
            Pj = np.array([
                neighbor.covariance for neighbor in neighbors_data])
            self.loc.distances_step(z, xj, Pj)
            self.neighborhood.clear()
        if len(self.gps) > 0:
            z = self.gps[max(self.gps.keys())]
            self.loc.gps_step(z)
            self.gps.clear()
