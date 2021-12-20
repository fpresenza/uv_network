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
        self.vehicles = np.empty(n, dtype=object)
        self.position_array = np.empty(n, dtype=np.ndarray)
        self.est_position_array = np.empty(n, dtype=np.ndarray)
        A = disk_graph.adjacency(pos, self.dmin)
        extents = rigidity.extents(A, pos)
        for i in node_ids:
            est_pos = np.random.multivariate_normal(pos[i], cov)
            self.vehicles[i] = agent_model(i, pos[i], est_pos, cov, extents[i])
            self.position_array[i] = self.vehicles[i].dm._x         # asigna el address # noqa
            self.est_position_array[i] = self.vehicles[i].loc._x    # asigna el address # noqa
        self.update_proximity()
        self.cloud = {v.id: [] for v in self.vehicles}

    def __getitem__(self, i):
        return self.vehicles[i]

    @property
    def position(self):
        return np.vstack(self.position_array)

    @property
    def est_position(self):
        return np.vstack(self.est_position_array)

    def rigidity_eigenvalue(self):
        A = disk_graph.adjacency(self.position, self.dmin)
        re = rigidity.eigenvalue(A, self.position)
        return re

    def extents(self):
        h = rigidity.extents(self.proximity_matrix, self.position)
        return h

    def update_time(self, t):
        self.timestamp = t
        [v.update_time(t) for v in self.vehicles]

    def update_proximity(self):
        A = disk_graph.adjacency(self.position, self.dmin)
        self.proximity_matrix = A.astype(bool)

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
            self.cloud[neighbor.id].append((msg, range_measurement))

    def receive(self, node_id):
        cloud = self.cloud.copy()
        for (msg, range_measurement) in cloud[node_id]:
            self.vehicles[node_id].receive_msg(msg, range_measurement)
        self.cloud[node_id].clear()


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(steps, formation, logs):
    # iteración
    bar = progressbar.ProgressBar(maxval=arg.tf).start()
    perf_time = []

    geodesics = np.empty((n_steps, n, n))
    geodesics[0] = network.geodesics(formation.proximity_matrix)
    est_geodesics = np.full((n_steps, n, n), n)
    extents = np.empty((n_steps, n))
    extents[0] = formation.extents()

    print(formation.extents())

    for i in node_ids:
        formation.broadcast(i)

    for k, t in steps[1:]:
        t_a = time.perf_counter()

        formation.update_time(t)
        geodesics[k] = network.geodesics(formation.proximity_matrix)
        extents[k] = extents[k - 1]

        """parte de localizacion"""
        # formation.get_gps(6)
        # formation.get_gps(8)

        print('---', k, '---')
        for i in node_ids:
            formation.receive(i)
            # print(k, i, 4 in formation[i].routing.action)
            # state_tokens = formation[i].routing.state.values()
            # print([tkn.center for tkn in state_tokens])

            # formation[i].localization_step()
            action_tokens = formation[i].routing.action.values()
            # print([tkn.center for tkn in action_tokens])
            j = [token.center for token in action_tokens]
            gij = [token.hops_travelled for token in action_tokens]
            est_geodesics[k, j, i] = gij
            # for vj in formation[i].routing.action_tokens():
            #     j = vj.center
            #     est_geodesics[k, j, i] = vj.hops_travelled

        # print(formation[7].routing.state_tokens())
        for i in node_ids:
            formation.broadcast(i)
        # print([tkn.path for tkn in formation[7].routing.state_tokens()])
        # print(formation[0].routing.state[0].hops_to_target)

        # for i in node_ids:
        #     formation[i].control_step()

        t_b = time.perf_counter()

        formation.update_proximity()
        # if 10 < k < 15:
        #     formation.proximity_matrix[0, 1] = formation.proximity_matrix[1, 0] = 0   # noqa
        #     formation.proximity_matrix[1, 7] = formation.proximity_matrix[7, 1] = 0   # noqa
            # formation.proximity_matrix[4, 9] = formation.proximity_matrix[9, 4] = 1   # noqa

        # log data
        logs.position[k] = formation.position.ravel()
        logs.est_position[k] = formation.est_position.ravel()
        # logs.action[k] = u.ravel()
        # logs.fre[k] = rigidity.eigenvalue(A, x)
        # logs.re[k] = re
        # logs.adjacency[k] = A.ravel()

        perf_time.append((t_b - t_a)/n)
        # bar.update(np.round(t, 3))

    bar.finish()

    # fig, ax = plt.subplots(10, 10)
    # # ax = ax.ravel()
    # for i in node_ids:
    #     for j in node_ids:
    #         ax[i, j].plot(extents[:, i], ds='steps-post', color='C2')
    #         ax[i, j].plot(geodesics[:, i, j], ds='steps-post', color='C0')
    #         ax[i, j].plot(
    #             est_geodesics[:, i, j], ds='steps-post', ls='--', color='k')
    #         ax[i, j].set_xticks(range(n_steps))
    #         ax[i, j].set_xticklabels([])
    #         ax[i, j].set_ylim(-0.25, 3.25)
    #         ax[i, j].set_yticks([0, 1, 2, 3])
    #         ax[i, j].set_yticklabels([])
    #         if np.all(geodesics[:, i, j] > extents[:, i]):
    #             ax[i, j].set_facecolor('0.8')
    #         # if np.any(geodesics[:, i, j] > extents[:, i]):
    #         #     ax[i, j].set_facecolor('0.8')
    # plt.show()

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


cov = 1**2 * np.eye(2)

node_ids = np.arange(n)

dmin = 0.85 * lim
dmax = lim

formation = Formation(
    node_ids,
    subframework_rigidity_agent.single_integrator,
    pos, cov,
    comm_range=(dmin, dmax),
    range_cov=1.,
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
# logs.re[0] = []
# logs.adjacency[0] = A.ravel()

logs = run(steps, formation, logs)

xi = logs.position[0]
est_xi = logs.est_position[0]
est_xf = logs.est_position[-1]

# print(np.linalg.norm(xi - est_xi))
# print(np.linalg.norm(xi - est_xf))

# fig, ax = network.plot.figure()
# network.plot.nodes(ax, logs.position[0].reshape(-1, 2), marker='o')
# network.plot.edges(
#     ax, logs.position[0].reshape(-1, 2), formation.proximity_matrix)

# for est_pos in logs.est_position:
#     network.plot.nodes(
#         ax, est_pos.reshape(-1, 2), color='gray', marker='.', s=10)

# network.plot.nodes(
#     ax, logs.est_position[-1].reshape(-1, 2), color='red', marker='x')
# network.plot.nodes(
#     ax, logs.est_position[0].reshape(-1, 2), color='blue', marker='o', s=10)

# plt.show()

# np.savetxt('/tmp/t.csv', time_interval, delimiter=',')
# np.savetxt('/tmp/x.csv', logs.x, delimiter=',')
# np.savetxt('/tmp/hatx.csv', logs.hatx, delimiter=',')
# np.savetxt('/tmp/u.csv', logs.u, delimiter=',')
# np.savetxt('/tmp/fre.csv', logs.fre, delimiter=',')
# np.savetxt('/tmp/re.csv', logs.re, delimiter=',')
# np.savetxt('/tmp/adjacency.csv', logs.adjacency, delimiter=',')
# np.savetxt('/tmp/hops.csv', hops, delimiter=',')
