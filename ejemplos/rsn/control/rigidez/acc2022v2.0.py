#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on jue 09 dic 2021 20:11:05 -03
@author: fran
"""
import argparse
import collections
import time
import progressbar
import numpy as np
import matplotlib.pyplot as plt    # noqa

import uvnpy.network as network
from uvnpy.network import disk_graph
from uvnpy.rsn import distances, rigidity
from uvnpy.autonomous_agents import subframework_rigidity_agent


# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple(
    'Logs',
    'position, est_position, action, fre, re, adjacency')


class Formation(object):
    def __init__(
            self, node_ids, agent_model,
            pos, cov, comm_range, range_cov, gps_cov):
        """Clase para simular una red de agentes de forma centralizada"""
        self.n = len(node_ids)
        self.dim = pos.shape[1]
        self.timestamp = None
        self.dmin = np.min(comm_range)
        self.dmax = np.max(comm_range)
        self.range_cov = range_cov
        self.gps_cov = gps_cov
        self.vehicles = np.empty(self.n, dtype=object)
        self.position_array = np.empty(self.n, dtype=np.ndarray)
        self.est_position_array = np.empty(self.n, dtype=np.ndarray)
        self.action = np.zeros((self.n, self.dim))
        self.proximity_matrix = disk_graph.adjacency(
            pos, self.dmin).astype(bool)
        self.extents = rigidity.extents(self.proximity_matrix, pos)
        for i in node_ids:
            est_pos = np.random.multivariate_normal(pos[i], cov)
            self.vehicles[i] = agent_model(
                i, pos[i], est_pos, cov,
                self.dmin, self.extents[i])
            self.position_array[i] = self.vehicles[i].dm._x         # asigna el address # noqa
            self.est_position_array[i] = self.vehicles[i].loc._x    # asigna el address # noqa
        self.cloud = {v.node_id: [] for v in self.vehicles}

    @property
    def position(self):
        return np.vstack(self.position_array)

    @property
    def est_position(self):
        return np.vstack(self.est_position_array)

    def rigidity_eigenvalue(self):
        re = rigidity.eigenvalue(self.proximity_matrix, self.position)
        return re

    def rigidity_extents(self):
        h = rigidity.extents(self.proximity_matrix, self.position)
        return h

    def subframeworks(self):
        sf = [
            network.subsets.multihop_subframework(
                self.proximity_matrix, self.position,
                i, self.extents[i]) for i in range(self.n)]
        return sf

    def subframeworks_rigidity_eigenvalue(self):
        re = [
            rigidity.eigenvalue(A, x)
            for A, x in formation.subframeworks()]
        return re

    def update_time(self, t):
        self.timestamp = t
        [v.update_time(t) for v in self.vehicles]

    def update_proximity(self):
        # A = disk_graph.adjacency(self.position, self.dmin)
        A = disk_graph.adjacency_histeresis(
            self.proximity_matrix,
            self.position,
            self.dmin, self.dmax)
        self.proximity_matrix = A

    def get_gps(self, node_id):
        center = self.vehicles[node_id]
        gps_measurement = np.random.normal(center.dm.x, self.gps_cov)
        self.vehicles[node_id].gps = {self.timestamp: gps_measurement}

    def broadcast(self, node_id):
        center = self.vehicles[node_id]
        msg = center.send_msg()
        neighbors = self.vehicles[self.proximity_matrix[node_id]]
        for neighbor in neighbors:
            range_measurement = np.random.normal(
                distances.matrix_between(center.dm.x, neighbor.dm.x),
                self.range_cov)
            self.cloud[neighbor.node_id].append((msg, range_measurement))

    def receive(self, node_id):
        cloud = self.cloud.copy()
        for (msg, range_measurement) in cloud[node_id]:
            self.vehicles[node_id].receive_msg(msg, range_measurement)
        self.cloud[node_id].clear()

    def localization_step(self, i):
        self.vehicles[i].localization_step()

    def control_step(self, i):
        self.vehicles[i].control_step()
        self.action[i] = self.vehicles[i].control_action.copy()

# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(steps, formation, logs):
    # iteración
    bar = progressbar.ProgressBar(maxval=arg.tf).start()
    perf_time = []

    print(formation.extents)

    for i in node_ids:
        formation.broadcast(i)

    # for _ in range(10):
    #     for i in node_ids:
    #         formation.broadcast(i)
    #     for i in node_ids:
    #         formation.receive(i)
    #         formation.localization_step(i)

    for k, t in steps[1:]:
        t_a = time.perf_counter()

        formation.update_time(t)
        """parte de localizacion"""
        formation.get_gps(6)
        formation.get_gps(8)

        for i in node_ids:
            formation.receive(i)

        for i in node_ids:
            formation.localization_step(i)
            formation.control_step(i)

        for i in node_ids:
            formation.broadcast(i)

        t_b = time.perf_counter()

        formation.update_proximity()

        # log data
        logs.position[k] = formation.position.ravel()
        logs.est_position[k] = formation.est_position.ravel()
        logs.action[k] = formation.action.ravel()
        logs.fre[k] = formation.rigidity_eigenvalue()
        logs.re[k] = formation.subframeworks_rigidity_eigenvalue()
        logs.adjacency[k] = formation.proximity_matrix.ravel()

        perf_time.append((t_b - t_a)/n)
        bar.update(np.round(t, 3))

    bar.finish()

    st = arg.tf
    rt = sum(perf_time)
    prompt = 'RT={:.3f} secs, ST={:.3f} secs  ==>  RTF={:.3f}'
    print(prompt.format(rt, st, st / rt))

    return logs


# ------------------------------------------------------------------
# Parseo de argumentos
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-s', '--step',
    dest='h', default=100e-3, type=float, help='paso de simulación')
parser.add_argument(
    '-e', '--tf',
    default=1.0, type=float, help='time_interval final')

arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
time_interval = np.arange(0, arg.tf, arg.h)
steps = list(enumerate(time_interval))
n_steps = len(steps)

lim = 10
dim = 2
n = 10
# pos = np.random.uniform(-0.9*lim, 0.9*lim, (n, dim))
pos = np.array([
    [6.2343, 8.2097],
    [4.5759, 0.487],
    [-1.7355, -4.0296],
    [-2.1949, 5.8593],
    [-8.5903, 4.0411],
    [6.9784, -0.807],
    [-7.4382, -2.2692],
    [0.2778, 6.4405],
    [7.6862, 5.5808],
    [-1.0463, -0.8862]])

# pos = np.array([
#     [-7.4851,  7.0387],
#     [-2.793 ,  6.6516],
#     [-3.0055,  1.5672],
#     [ 3.4272,  4.1939],
#     [ 7.0671, -1.1412],
#     [ 4.6065,  5.3426],
#     [-6.767 ,  8.9778],
#     [ 4.0419, -1.5308],
#     [ 5.7891, -3.8129],
#     [ 6.4013, -4.1745]])


cov = 0.5**2 * np.eye(2)

node_ids = np.arange(n)

dmin = 0.85 * lim
dmax = 0.9 * lim

formation = Formation(
    node_ids,
    subframework_rigidity_agent.single_integrator,
    pos, cov,
    comm_range=(dmin, dmax),
    range_cov=0.5,
    gps_cov=1.)

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
# initialize()

logs = Logs(
    position=np.empty((n_steps, n*dim)),
    est_position=np.empty((n_steps, n*dim)),
    action=np.empty((n_steps, n*dim)),
    fre=np.zeros(n_steps),
    re=np.zeros((n_steps, n)),
    adjacency=np.empty((n_steps, n**2), dtype=int))
logs.position[0] = formation.position.ravel()
logs.est_position[0] = formation.est_position.ravel()
logs.action[0] = np.zeros(n*dim)
logs.fre[0] = formation.rigidity_eigenvalue()
logs.re[0] = formation.subframeworks_rigidity_eigenvalue()
logs.adjacency[0] = formation.proximity_matrix.ravel()

logs = run(steps, formation, logs)

np.savetxt('/tmp/t.csv', time_interval, delimiter=',')
np.savetxt('/tmp/position.csv', logs.position, delimiter=',')
np.savetxt('/tmp/est_position.csv', logs.est_position, delimiter=',')
np.savetxt('/tmp/action.csv', logs.action, delimiter=',')
np.savetxt('/tmp/fre.csv', logs.fre, delimiter=',')
np.savetxt('/tmp/re.csv', logs.re, delimiter=',')
np.savetxt('/tmp/adjacency.csv', logs.adjacency, delimiter=',')
np.savetxt('/tmp/hops.csv', formation.extents, delimiter=',')
