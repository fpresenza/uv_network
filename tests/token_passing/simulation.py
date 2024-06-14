#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: fran
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
"""
import argparse
import collections
import time
import progressbar
import numpy as np

from uvnpy.network import core
from uvnpy.routing.token_passing import TokenPassing
from uvnpy.dynamics.linear_models import DiscreteIntegrator
from uvnpy.network.disk_graph import (
    adjacency_from_positions,
    adjacency_histeresis
)

# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
np.set_printoptions(
    suppress=True,
    precision=6,
    linewidth=200
)

Logs = collections.namedtuple(
    'Logs',
    'position, state, control, adjacency, extents'
)


class TokenPassingAgent(object):
    def __init__(self, node_id, position, state):
        self.node_id = node_id
        self.dim = len(position)
        self.position_dynamics = DiscreteIntegrator(position)
        self.state_dynamics = DiscreteIntegrator(state)
        self.token_passing = TokenPassing(self.node_id)
        self.last_control_action = np.zeros(self.dim)
        self.last_token_extent = None

    def get_tokens_to_transmit(self):
        # ########    TO DO     #########
        # ## enviar a los vecinos lo que sea necesario

        # en este ejemplo se envia el estado a 1 hop
        data = self.state_dynamics.x
        self.last_token_extent = 1

        # ###############################

        return self.token_passing.tokens_to_transmit(
            data, self.last_token_extent
        )

    def position_update(self, extra_inputs=None):
        # ########    TO DO     #########
        # ## calcular la accion de control de posicion

        self.last_control_action = np.zeros(self.dim)

        # ###############################

        self.position_dynamics.step(self.last_control_action)

    def state_update(self, extra_inputs=None):
        # ########    TO DO     #########
        # ## calcular la actualizacion del estado

        # en este ejemplo se hace la diferencia
        # con los estados de los vecinos
        # esto deberia converger al promedio de todos
        # los estados si el grafo es conexo y la ganancia
        # epsilon es lo suficientemente pequeña
        # v = 0.0
        data = self.token_passing.extract_data(hops=1)
        x = self.state_dynamics.x
        epsilon = 0.1
        v = -epsilon * sum([x - y for y in data.values()])

        # ###############################
        self.state_dynamics.step(v)


class Network(object):
    def __init__(
        self,
        adjacency_matrix,
        agents,
        comm_range,
        delay=1
    ):
        """Clase para simular una red de agentes"""
        self.adjacency_matrix = adjacency_matrix.astype(bool)
        self.agents = agents
        self.dmin = np.min(comm_range)
        self.dmax = np.max(comm_range)
        # una lista fifo para simular el retardo en las comunicaciones
        self.cloud = collections.deque(maxlen=delay)

    def neighbors(self, node_id):
        return np.where(self.adjacency_matrix[node_id])[0]

    def update_adjacency(self):
        self.adjacency_matrix = adjacency_histeresis(
            self.adjacency_matrix,
            self.collect_positions(),
            self.dmin, self.dmax
        )

    def cloud_append(self):
        self.cloud.append({agent.node_id: [] for agent in self.agents})

    def upload_to_cloud(self, node_id):
        tokens = self.agents[node_id].get_tokens_to_transmit()
        for neighbor_id in self.neighbors(node_id):
            self.cloud[-1][neighbor_id].append(tokens)

    def dowload_from_cloud(self, node_id):
        for tokens in self.cloud[0][node_id]:
            self.agents[node_id].token_passing.update_record(tokens)

    def collect_positions(self):
        return np.array([agent.position_dynamics.x for agent in self.agents])

    def collect_states(self):
        return np.array([agent.state_dynamics.x for agent in self.agents])

    def collect_control_actions(self):
        return np.array([agent.last_control_action for agent in self.agents])

    def collect_extents(self):
        return np.array([agent.last_token_extent for agent in self.agents])


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(steps, network, logs):
    # iteración
    bar = progressbar.ProgressBar(maxval=arg.tf).start()
    perf_time = []
    k_init = 20

    for k, t in steps[1:]:
        t_a = time.perf_counter()

        # communication step
        network.cloud_append()      # create new space in cloud
        # first upload from all
        for agent in network.agents:
            network.upload_to_cloud(agent.node_id)
        # second download for all
        for agent in network.agents:
            network.dowload_from_cloud(agent.node_id)

        for agent in network.agents:
            # update states
            agent.state_update()
            # update positions after a initial prudential time
            if k > k_init:
                agent.position_update()
            else:
                pass

        t_b = time.perf_counter()

        network.update_adjacency()

        # log data
        logs.position[k] = network.collect_positions().ravel()
        logs.state[k] = network.collect_states()
        logs.control[k] = network.collect_control_actions().ravel()
        logs.adjacency[k] = network.adjacency_matrix.ravel()
        logs.extents[k] = network.collect_extents()

        perf_time.append((t_b - t_a)/n)
        bar.update(np.round(t, 3))

    bar.finish()

    rt = arg.tf
    st = sum(perf_time)
    prompt = 'ST={:.3f} secs, RT={:.3f} secs  ==>  RTF={:.3f}'
    print(prompt.format(st, rt, rt / st))

    return logs


# ------------------------------------------------------------------
# Parseo de argumentos
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-s', '--step',
    default=25e-3, type=float, help='paso de simulación'
)
parser.add_argument(
    '-e', '--tf',
    default=1.0, type=float, help='tiempo total de simulación'
)
parser.add_argument(
    '-d', '--delay',
    default=1, type=int, help='largo de la cola del cloud'
)

arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
np.random.seed(1)
timestamps = np.arange(0, arg.tf + arg.step, arg.step)
steps = list(enumerate(timestamps))
n_steps = len(steps)

n = 10                # number of agents
dim = 2               # space dimension
lim = 20              # spatial domain length
dmin = 0.42 * lim     # connection range
dmax = 0.45 * lim     # disconnection range

position = np.random.uniform(-lim / 2, lim / 2, (n, dim))
adjacency_matrix = adjacency_from_positions(position, dmin)
state = np.random.uniform(0, 1, n)

network = Network(
    adjacency_matrix=adjacency_matrix,
    agents=[TokenPassingAgent(i, position[i], state[i]) for i in range(n)],
    comm_range=(dmin, dmax),
    delay=arg.delay
)

print("\n")
print("Adjacency matrix: \n", adjacency_matrix)
print(
    "Network is connected: ",
    core.algebraic_connectivity(adjacency_matrix) > 1e-8
)
print("\n")

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
# initialize()

logs = Logs(
    position=np.empty((n_steps, n*dim)),
    state=np.empty((n_steps, n)),
    control=np.empty((n_steps, n*dim)),
    adjacency=np.empty((n_steps, n**2), dtype=int),
    extents=np.zeros((n_steps, n))
)
logs.position[0] = network.collect_positions().ravel()
logs.state[0] = network.collect_states()
logs.control[0] = np.zeros(n*dim)
logs.adjacency[0] = network.adjacency_matrix.ravel()
logs.extents[0] = network.collect_extents()

logs = run(steps, network, logs)

print("Initial state: ", logs.state[0])
print("Final state: ", logs.state[-1])

np.savetxt('/tmp/timestamps.csv', timestamps, delimiter=',')
np.savetxt('/tmp/position.csv', logs.position, delimiter=',')
np.savetxt('/tmp/control.csv', logs.control, delimiter=',')
np.savetxt('/tmp/adjacency.csv', logs.adjacency, delimiter=',')
np.savetxt('/tmp/extents.csv', logs.extents, delimiter=',')
