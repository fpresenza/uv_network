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
import matplotlib.pyplot as plt

import uvnpy.network as network
from uvnpy.network import disk_graph
from uvnpy.rsn import distances, rigidity
from uvnpy.autonomous_agents import subframework_rigidity


# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple(
    'Logs',
    'position, est_position, action, fre, re, adjacency')


class Formation(object):
    def __init__(self, node_ids, agent_model, pos, cov, range, sensor_noise):
        """Clase para simular una red de agentes de forma centralizada"""
        self.n = len(node_ids)
        self.dim = pos.shape[1]
        self.dmin = np.min(range)
        self.dmax = np.max(range)
        self.sensor_noise = sensor_noise
        self.vehicles = np.empty(n, dtype=np.ndarray)
        self.position_array = np.empty(n, dtype=np.ndarray)
        self.est_position_array = np.empty(n, dtype=np.ndarray)
        for i in node_ids:
            est_pos = np.random.multivariate_normal(pos[i], cov)
            self.vehicles[i] = agent_model(i, pos[i], est_pos, cov)
            self.position_array[i] = self.vehicles[i].dm._x         # asigna el address # noqa
            self.est_position_array[i] = self.vehicles[i].loc._x    # asigna el address # noqa
        self.update_proximity()

    def __getitem__(self, i):
        return self.vehicles[i]

    @property
    def position(self):
        return np.vstack(self.position_array)

    @property
    def est_position(self):
        return np.vstack(self.est_position_array)

    @property
    def rigidity_eigenvalue(self):
        A = disk_graph.adjacency(self.position, self.dmin)
        re = rigidity.eigenvalue(A, self.position)
        return re

    def update_proximity(self):
        A = disk_graph.adjacency(self.position, self.dmin)
        self.proximity_matrix = A.astype(bool)

    def spread(self, node_id, timestamp):
        center = self.vehicles[node_id]
        msg = center.send_msg(timestamp)
        neighbors = self.vehicles[self.proximity_matrix[node_id]]
        for neighbor in neighbors:
            range_measurement = distances.matrix_between(
                center.dm.x, neighbor.dm.x)
            range_measurement += np.random.normal(self.sensor_noise)
            neighbor.receive_msg(msg, range_measurement)


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(steps, formation, logs):
    # iteración
    bar = progressbar.ProgressBar(maxval=arg.tf).start()
    perf_time = []

    for k, t in steps[1:]:
        t_a = time.perf_counter()

        formation.update_proximity()

        """parte de localizacion"""
        for i in node_ids:
            formation[i].control_step(t)
            formation[i].localization_step()
            # if i in [15, 41]:
            #     pi = np.random.normal(x[i], pos_sd)
            #     Pi = localization[i].P
            #     Ki = Pi.dot(np.linalg.inv(Pi + pos_sd**2 * np.eye(2)))
            #     localization[i]._x += Ki.dot(pi - hatx[i])

            formation.spread(i, t)

        t_b = time.perf_counter()

        # log data
        logs.position[k] = formation.position.ravel()
        logs.est_position[k] = formation.est_position.ravel()
        # logs.action[k] = u.ravel()
        # logs.fre[k] = rigidity.eigenvalue(A, x)
        # logs.re[k] = re
        # logs.adjacency[k] = A.ravel()

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

cov = 1**2 * np.eye(2)

node_ids = np.arange(n)

dmin = 0.85 * lim
dmax = lim

formation = Formation(
    node_ids,
    subframework_rigidity.single_integrator,
    pos, cov,
    (dmin, dmax), 0.2)
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
logs.fre[0] = formation.rigidity_eigenvalue
# logs.re[0] = []
# logs.adjacency[0] = A.ravel()

logs = run(steps, formation, logs)

di = distances.matrix(logs.position[0].reshape(-1, 2))
est_di = distances.matrix(logs.est_position[0].reshape(-1, 2))
est_df = distances.matrix(logs.est_position[-1].reshape(-1, 2))

print(np.linalg.norm(di - est_di, 'fro'))
print(np.linalg.norm(di - est_df, 'fro'))


fig, ax = network.plot.figure()
network.plot.nodes(ax, logs.position[0].reshape(-1, 2), marker='o')
network.plot.edges(
    ax, logs.position[0].reshape(-1, 2), formation.proximity_matrix)
for est_pos in logs.est_position:
    network.plot.nodes(ax, est_pos.reshape(-1, 2), color='brown', marker='.')
    network.plot.nodes(
        ax, logs.est_position[-1].reshape(-1, 2), color='red', marker='x')

plt.show()

# np.savetxt('/tmp/t.csv', time_interval, delimiter=',')
# np.savetxt('/tmp/x.csv', logs.x, delimiter=',')
# np.savetxt('/tmp/hatx.csv', logs.hatx, delimiter=',')
# np.savetxt('/tmp/u.csv', logs.u, delimiter=',')
# np.savetxt('/tmp/fre.csv', logs.fre, delimiter=',')
# np.savetxt('/tmp/re.csv', logs.re, delimiter=',')
# np.savetxt('/tmp/adjacency.csv', logs.adjacency, delimiter=',')
# np.savetxt('/tmp/hops.csv', hops, delimiter=',')
