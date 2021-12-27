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
from uvnpy.rsn.control import (
    centralized_rigidity_maintenance,
    communication_load)
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
    def __init__(
            self, id, pos, est_pos, cov,
            comm_range, extent=None, t=0):
        self.id = id
        self.dim = len(pos)
        self.dmin = np.min(comm_range)
        self.dmax = np.max(comm_range)
        self.extent = extent
        self.current_time = t
        self.dm = integrator(pos)
        self.maintenance = centralized_rigidity_maintenance(
            self.dim, self.dmin, 20/self.dmin, 1/3, non_adjacent=True)
        self.load = communication_load(self.dmax)
        self.control_action = np.zeros(self.dim)
        self.action = {}
        ctrl_cov = 0.05**2 * np.eye(self.dim)
        range_cov = 0.5
        self.gps_cov = gps_cov = 1. * np.eye(self.dim)
        self.loc = distances_to_neighbors_kalman(
            est_pos, cov, ctrl_cov, range_cov, gps_cov)
        self.neighborhood = Neighborhood()
        self.routing = routing_protocols.subgraph_protocol(
            self.id, self.extent, self.current_time)
        self.gps = {}
        self.state = {
            'position': self.loc.position,
            'covariance': self.loc.covariance
        }

    def update_time(self, t):
        self.current_time = t

    def send_msg(self):
        action_tokens, state_tokens = self.routing.broadcast(
            self.current_time,
            self.action,
            self.state)
        msg = InterAgentMsg(
            id=self.id,
            timestamp=self.current_time,
            action_tokens=action_tokens,
            state_tokens=state_tokens)
        return msg

    def receive_msg(self, msg, range_measurement):
        self.neighborhood.update(
            msg.id, msg.timestamp,
            msg.state_tokens[msg.id].data['position'],
            msg.state_tokens[msg.id].data['covariance'],
            range_measurement)
        routing = self.routing
        [routing.update_action(tkn) for tkn in msg.action_tokens.values()]
        [routing.update_state(tkn) for tkn in msg.state_tokens.values()]

    def control_step(self, cmd_ext=0):
        # obtengo posiciones del subframework
        position = self.routing.extract_state('position')
        p = np.empty((len(position) + 1, self.dim))
        p[0] = self.loc.position
        p[1:] = list(position.values())

        # obtengo la accion de control de rigidez
        u_r = self.maintenance.update(p)

        # obtengo la accion de control de carga
        geodesics = self.routing.geodesics()
        g = np.empty(len(geodesics) + 1)
        g[0] = 0
        g[1:] = list(geodesics.values())

        coeff = g < self.extent
        u_l = self.load.update(p, coeff)

        # sumo los objetivos del subframework
        u = 0.3 * u_r + 0.075 * u_l

        # genero la accion de control del centro
        cmd = self.routing.extract_action()
        u_center = u[0] + sum(cmd.values())

        # empaco las acciones de control del subframework
        self.action = {
            i: ui
            for i, ui in zip(position.keys(), u[1:])}

        # aplico acciones de control
        self.control_action = cmd_ext + u_center
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
        self.state['position'] = self.loc.position
        self.state['covariance'] = self.loc.covariance
