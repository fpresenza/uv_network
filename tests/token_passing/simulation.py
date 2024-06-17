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
from uvnpy.distances.control import CollisionAvoidance

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
        self.last_control_action = np.zeros(self.dim)
        self.last_token_extent = None
        self.position_dynamics = DiscreteIntegrator(position)
        self.state_dynamics = DiscreteIntegrator(state)
        self.token_passing = TokenPassing(self.node_id)
        self.collision_avoidance = CollisionAvoidance(power=2)

    def get_tokens_to_transmit(self):
        # ########    TO DO     #########
        # ## enviar a los vecinos lo que sea necesario

        # en este ejemplo se envia el estado a 1 hop
        # data = self.state_dynamics.x
        # self.last_token_extent = 1

        # en este ejemplo se envia la posicion a 1 hop
        data = self.position_dynamics.x
        self.last_token_extent = 1

        # ###############################

        return self.token_passing.tokens_to_transmit(
            data, self.last_token_extent
        )

    def position_update(self, extra_inputs=None):
        # ########    TO DO     #########
        # ## calcular la accion de control de posicion

        # en este ejemplo no se hace nada
        # self.last_control_action = np.zeros(self.dim)

        # en est ejemplo se extrae la posicion a 1 hop
        setpoint = np.array([[5, 0], [0, 5]], dtype=float)
        position = self.position_dynamics.x
        u_setpoint = (setpoint[self.node_id] - position)

        neighbors = self.token_passing.extract_data(hops=1)
        if len(neighbors) > 0:
            obstacles = np.array(list(neighbors.values()), dtype=float)
            u_collision = self.collision_avoidance.update(position, obstacles)
        else:
            u_collision = np.zeros(self.dim)

        stepsize = 0.01
        # diferentes pesos para evitar minimos locales
        weights = [[0.6, 0.0], [0.5, 1.0]]
        self.last_control_action = stepsize * (
            weights[self.node_id][0] * u_setpoint
            + weights[self.node_id][1] * u_collision
        )
        # ###############################

        self.position_dynamics.step(self.last_control_action)

    def state_update(self, extra_inputs=None):
        # ########    TO DO     #########
        # ## calcular la actualizacion del estado

        # en este ejemplo no se hace nada
        v = 0.0

        # en este ejemplo se hace la diferencia
        # con los estados de los vecinos
        # esto deberia converger al promedio de todos
        # los estados si el grafo es conexo y la ganancia
        # epsilon es lo suficientemente pequeña
        # data = self.token_passing.extract_data(hops=1)
        # x = self.state_dynamics.x
        # epsilon = 0.1
        # v = -epsilon * sum([x - y for y in data.values()])

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

    def download_from_cloud(self, node_id):
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
            network.download_from_cloud(agent.node_id)

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
        logs.position[k] = network.collect_positions().ravel().copy()
        logs.state[k] = network.collect_states().copy()
        logs.control[k] = network.collect_control_actions().ravel().copy()
        logs.adjacency[k] = network.adjacency_matrix.ravel().copy()
        logs.extents[k] = network.collect_extents().copy()

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

n = 2                 # number of agents
dim = 2               # space dimension
lim = 10              # spatial domain length
dmin = 0.4 * lim      # connection range
dmax = 0.5 * lim      # disconnection range

# position = np.random.uniform(-lim / 2, lim / 2, (n, dim))
position = np.array([[-5, 0], [0, -5]], dtype=float)
adjacency_matrix = adjacency_from_positions(position, dmin)
state = np.random.uniform(0, 1, n)

network = Network(
    adjacency_matrix=adjacency_matrix,
    agents=[TokenPassingAgent(i, position[i], state[i]) for i in range(n)],
    comm_range=(dmin, dmax),
    delay=arg.delay
)

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
logs.position[0] = network.collect_positions().ravel().copy()
logs.state[0] = network.collect_states().copy()
logs.control[0] = np.zeros(n*dim)
logs.adjacency[0] = network.adjacency_matrix.ravel().copy()
logs.extents[0] = network.collect_extents().copy()

adjacency_matrix = logs.adjacency[0].reshape(n, n)
print("\nInitial:")
print("Position: ", logs.position[0])
print("State: ", logs.state[0])
print("Adjacency matrix: \n", adjacency_matrix)
print(
    "Network is connected: ",
    core.algebraic_connectivity(adjacency_matrix) > 1e-8
)

logs = run(steps, network, logs)

adjacency_matrix = logs.adjacency[-1].reshape(n, n)
print("\nFinal")
print("Position: ", logs.position[-1])
print("State: ", logs.state[-1])
print("Adjacency matrix: \n", adjacency_matrix)
print(
    "Network is connected: ",
    core.algebraic_connectivity(adjacency_matrix) > 1e-8
)

np.savetxt('/tmp/timestamps.csv', timestamps, delimiter=',')
np.savetxt('/tmp/position.csv', logs.position, delimiter=',')
np.savetxt('/tmp/control.csv', logs.control, delimiter=',')
np.savetxt('/tmp/adjacency.csv', logs.adjacency, delimiter=',')
np.savetxt('/tmp/extents.csv', logs.extents, delimiter=',')
