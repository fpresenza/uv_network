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
    communication_load,
    power_collision_avoidance)
from uvnpy.rsn.localization import distances_to_neighbors_kalman
from uvnpy.autonomous_agents import routing_protocols
from uvnpy.rsn import rigidity
from uvnpy.network import disk_graph
from uvnpy.toolkit.functions import logistic_saturation


InterAgentMsg = collections.namedtuple(
    'InterAgentMsg',
    'node_id, \
    timestamp, \
    action_tokens, \
    state_tokens')


NeigborhoodData = collections.namedtuple(
    'NeigborhoodData',
    'node_id, \
    timestamp, \
    covariance, \
    position, \
    range')


class Neighborhood(dict):
    def update(
            self, node_id, timestamp,
            position, covariance, range_measurement):
        self[node_id] = NeigborhoodData(
            node_id=node_id,
            timestamp=timestamp,
            position=position,
            covariance=covariance,
            range=range_measurement)


class single_integrator(object):
    def __init__(
            self, node_id, pos, est_pos, cov,
            comm_range, extent=None, queue=1, t=0):
        self.node_id = node_id
        self.dim = len(pos)
        self.dmin = np.min(comm_range)
        self.dmax = np.max(comm_range)
        self.extent = extent
        self.current_time = t
        self.dm = integrator(pos)
        self.maintenance = centralized_rigidity_maintenance(
            dim=self.dim, dmax=self.dmin,
            steepness=20/self.dmin, exponent=0.5, non_adjacent=True)
        self.load = communication_load(self.dmax, 3/self.dmax)
        self.collision = power_collision_avoidance(power=2)
        self.control_action_raw = np.zeros((1, self.dim))
        self.last_control_action = np.zeros(self.dim)
        self.action = {}
        ctrl_cov = 0.05**2 * np.eye(self.dim)
        range_cov = 0.5
        self.gps_cov = gps_cov = 1. * np.eye(self.dim)
        self.loc = distances_to_neighbors_kalman(
            est_pos, cov, ctrl_cov, range_cov, gps_cov)
        self.neighborhood = Neighborhood()
        self.routing = routing_protocols.subgraph_protocol(
            self.node_id, self.extent, queue)
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
            self.state,
            self.extent)
        msg = InterAgentMsg(
            node_id=self.node_id,
            timestamp=self.current_time,
            action_tokens=action_tokens,
            state_tokens=state_tokens)
        return msg

    def receive_msg(self, msg, range_measurement):
        self.neighborhood.update(
            msg.node_id, msg.timestamp,
            msg.state_tokens[msg.node_id].data['position'],
            msg.state_tokens[msg.node_id].data['covariance'],
            range_measurement)
        self.routing.update_action(msg.action_tokens.values())
        self.routing.update_state(msg.state_tokens.values())

    def rigidity_eigenvalue(self, hops):
        position = self.routing.extract_state('position', hops)
        p = np.empty((len(position) + 1, self.dim))
        p[0] = self.loc.position
        p[1:] = list(position.values())
        A = disk_graph.adjacency(p, self.dmin)
        re = rigidity.eigenvalue(A, p)
        return re

    def choose_extent(self):
        try:
            re = self.rigidity_eigenvalue(self.extent)
            new_re = self.rigidity_eigenvalue(self.extent - 1)
            if new_re > re:
                self.extent -= 1
        except ValueError:
            pass

    def steady(self):
        self.last_control_action = 0
        self.dm.step(self.current_time, 0)

    def control_step(self, u_ext=0):
        # obtengo posiciones del subframework
        position = self.routing.extract_state('position', self.extent)
        degree = len(position)
        if degree > 0:
            p = np.empty((degree + 1, self.dim))
            p[0] = self.loc.position
            p[1:] = list(position.values())

            # obtengo la accion de control de rigidez
            u = self.maintenance.update(p)
        else:
            u = np.zeros((1, self.dim))

        # genero la accion de control del centro
        cmd = self.routing.extract_action()
        u_r = u[0] + sum(cmd.values())

        # empaco las acciones de control del subframework
        self.action = {
            i: ui
            for i, ui in zip(position.keys(), u[1:])}

        # accion de evacion de colisiones
        obstacles = self.routing.extract_state('position', 1)
        if len(obstacles) > 0:
            obstacles_pos = list(obstacles.values())
            u_ca = self.collision.update(self.loc.position, obstacles_pos)
        else:
            u_ca = 0

        # aplico acciones de control
        control_action = logistic_saturation(
            5 * u_ext + 4 * u_r + 20 * u_ca, limit=2.5)
        self.control_action_raw[-1] = control_action
        self.last_control_action = self.control_action_raw[-1]
        self.dm.step(self.current_time, self.last_control_action)

    def localization_step(self):
        self.loc.dynamic_step(self.current_time, self.last_control_action)
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
