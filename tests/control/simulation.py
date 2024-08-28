#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mié 29 dic 2021 16:41:13 -03
@author: fran
"""
import argparse
import collections
import time
import progressbar
import numpy as np

from uvnpy.network import core
from uvnpy.distances.localization import KalmanBasedFilter
from uvnpy.routing.token_passing import TokenPassing
from uvnpy.dynamics.linear_models import Integrator
from uvnpy.toolkit.functions import logistic_saturation
from uvnpy.network.subframeworks import superframework_extents
from uvnpy.network.disk_graph import (
    adjacency_from_positions,
    adjacency_histeresis
)
from uvnpy.distances.core import (
    is_inf_rigid,
    rigidity_eigenvalue,
    minimum_rigidity_extents
)
from uvnpy.distances.control import (
    CentralizedRigidityMaintenance,
    CollisionAvoidance
)


# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
np.set_printoptions(
    suppress=True,
    precision=6
)

Logs = collections.namedtuple(
    'Logs',
    'position, \
    estimated_position,  \
    action, \
    fre, \
    re, \
    adjacency, \
    action_extents, \
    targets')

Robot = collections.namedtuple(
    'Robot',
    'dynamics, \
    computer')

InterRobotMsg = collections.namedtuple(
    'InterRobotMsg',
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
            position, covariance, range_measurement
            ):
        self[node_id] = NeigborhoodData(
            node_id=node_id,
            timestamp=timestamp,
            position=position,
            covariance=covariance,
            range=range_measurement
        )


class SubframeworkRigidityRobot(object):
    def __init__(
            self, node_id, est_pos,
            comm_range, action_extent=None, state_extent=None, t=0
            ):
        self.node_id = node_id
        self.dim = len(est_pos)
        self.dmin = np.min(comm_range)
        self.dmax = np.max(comm_range)
        self.action_extent = action_extent
        self.state_extent = state_extent
        self.current_time = t
        # self.dm = Integrator(pos)
        self.maintenance = CentralizedRigidityMaintenance(
            dim=self.dim, dmax=self.dmin,
            steepness=20/self.dmin, power=0.5, non_adjacent=True
        )
        self.collision = CollisionAvoidance(power=2)
        self.last_control_action = np.zeros(self.dim)
        self.action = {}
        ctrl_cov = 0.05**2 * np.eye(self.dim)
        range_cov = 0.5
        # range_cov = 0.
        gps_cov = 1. * np.eye(self.dim)
        # gps_cov = 0. * np.eye(self.dim)
        cov = 0.5**2 * np.eye(self.dim)
        self.loc = KalmanBasedFilter(
            est_pos, cov, ctrl_cov, range_cov, gps_cov
        )
        self.neighborhood = Neighborhood()
        self.routing = TokenPassing(self.node_id)
        self.gps = {}
        self.state = {
            'position': self.loc.position,
            'covariance': self.loc.covariance
        }

    def update_clock(self, t):
        self.current_time = t

    def create_msg(self):
        action_tokens, state_tokens = self.routing.broadcast(
            self.current_time,
            self.action,
            self.state,
            self.action_extent,
            self.state_extent
        )
        msg = InterRobotMsg(
            node_id=self.node_id,
            timestamp=self.current_time,
            action_tokens=action_tokens,
            state_tokens=state_tokens
        )
        return msg

    def handle_received_msg(self, msg, range_measurement):
        self.neighborhood.update(
            msg.node_id, msg.timestamp,
            msg.state_tokens[msg.node_id].data['position'],
            msg.state_tokens[msg.node_id].data['covariance'],
            range_measurement
        )
        self.routing.update_action(msg.action_tokens.values())
        self.routing.update_state(msg.state_tokens.values())

    def rigidity_eigenvalue(self, hops):
        position = self.routing.extract_state('position', hops)
        p = np.empty((len(position) + 1, self.dim))
        p[0] = self.loc.position
        p[1:] = list(position.values())
        A = adjacency_from_positions(p, self.dmin)
        return rigidity_eigenvalue(A, p)

    def set_control_action(self, u):
        self.last_control_action = u

    def compute_control_action(self, u_ext=0):
        # obtengo posiciones del subframework
        position = self.routing.extract_state('position', self.action_extent)
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
            for i, ui in zip(position.keys(), u[1:])
        }

        # accion de evacion de colisiones
        obstacles = self.routing.extract_state('position', 1)
        if len(obstacles) > 0:
            obstacles_pos = list(obstacles.values())
            u_ca = self.collision.update(self.loc.position, obstacles_pos)
        else:
            u_ca = 0

        # aplico acciones de control
        self.last_control_action = 0.5 * logistic_saturation(
            5 * u_ext + 4 * u_r + 20 * u_ca, limit=2.5
        )
        return self.last_control_action

    def localization_step(self):
        self.loc.dynamic_step(self.current_time, self.last_control_action)
        if len(self.neighborhood) > 0:
            neighbors_data = self.neighborhood.values()
            z = np.array([
                neighbor.range for neighbor in neighbors_data
            ])
            xj = np.array([
                neighbor.position for neighbor in neighbors_data
            ])
            Pj = np.array([
                neighbor.covariance for neighbor in neighbors_data
            ])
            self.loc.distances_step(z, xj, Pj)
            self.neighborhood.clear()
        if len(self.gps) > 0:
            z = self.gps[max(self.gps.keys())]
            self.loc.gps_step(z)
            self.gps.clear()
        self.state['position'] = self.loc.position
        self.state['covariance'] = self.loc.covariance


class Network(object):
    def __init__(
            self,
            adjacency_matrix,
            robots,
            comm_range,
            range_cov,
            gps_cov,
            queue=1
            ):
        """Clase para simular una red de robotes de forma centralizada"""
        self.ids = np.arange(len(adjacency_matrix))
        self.adjacency_matrix = adjacency_matrix.astype(bool)
        self.robots = robots

        self.dim = 2
        self.timestamp = None

        self.dmin = np.min(comm_range)
        self.dmax = np.max(comm_range)

        self.range_cov = range_cov
        self.gps_cov = gps_cov

        self.cloud = collections.deque(maxlen=queue)

    def collect_positions(self):
        return np.array([
            robot.dynamics.x for robot in self.robots
        ])

    def collect_estimated_positions(self):
        return np.array([
            robot.computer.loc.position for robot in self.robots
        ])

    def collect_action_extents(self):
        return np.array([
            robot.computer.action_extent for robot in self.robots
        ])

    def collect_control_actions(self):
        return np.array([
            robot.computer.last_control_action for robot in self.robots
        ])

    def rigidity_eigenvalue(self):
        return rigidity_eigenvalue(
            self.adjacency_matrix, self.collect_positions()
        )

    def subframeworks_rigidity_eigenvalue(self):
        geodesics = core.geodesics(self.adjacency_matrix.astype(float))
        eigs = []
        for robot in self.robots:
            subset = geodesics[robot.computer.node_id] <= \
                robot.computer.action_extent
            A = self.adjacency_matrix[np.ix_(subset, subset)]
            p = self.collect_positions()[subset]   # TODO: IMPROVE SLICE
            eigs.append(rigidity_eigenvalue(A, p))

        return eigs

    def update_adjacency(self):
        self.adjacency_matrix = adjacency_histeresis(
            self.adjacency_matrix,
            self.collect_positions(),
            self.dmin, self.dmax
        )

    def get_gps(self, node_id):
        gps_measurement = np.random.normal(
            self.robots[node_id].dynamics.x, self.gps_cov
        )
        self.robots[node_id].computer.gps = {self.timestamp: gps_measurement}

    def upload_to_cloud(self, node_id):
        msg = self.robots[node_id].computer.create_msg()
        for neighbor_id in np.where(self.adjacency_matrix[node_id] == 1)[0]:
            distance = np.sqrt(np.square(
                (self.robots[node_id].dynamics.x -
                 self.robots[neighbor_id].dynamics.x)
            ).sum())
            range_measurement = np.random.normal(
                distance,
                self.range_cov
            )
            self.cloud[-1][neighbor_id].append((msg, range_measurement))

    def download_from_cloud(self, node_id):
        cloud = self.cloud[0].copy()
        for (msg, range_measurement) in cloud[node_id]:
            self.robots[node_id].computer.handle_received_msg(
                msg, range_measurement
            )


class Targets(object):
    def __init__(self, n, xlim, ylim, range=3.0):
        self.set(n, xlim, ylim)
        self.range = range

    def set(self, n, xlim, ylim):
        lower, upper = zip(xlim, ylim)
        self.data = np.empty((n, 3), dtype=object)
        # self.data[:, :2] = np.random.uniform(lower, upper, (n, 2))
        self.data[:, :2] = np.array([
            [24.84, -3.8821],
            [-35.0464, -23.2889],
            [-5.6765, 6.6739],
            [1.7441, -6.9632],
            [-13.5915, -20.8073],
            [-33.5566, 8.4019],
            [2.5689, 6.999],
            [34.7281, 4.8964],
            [28.1627, 35.4352],
            [24.4927, 37.6256],
            [38.7665, -25.9539],
            [-4.573, 8.6123],
            [-31.323, 7.2874],
            [-27.2087, 22.2981],
            [-26.9443, 9.2272],
            [31.5733, -9.4084],
            [-15.8237, -5.8978],
            [-30.5408, 10.6855],
            [-24.998, 33.2679],
            [-7.549, -1.3672],
            [17.5176, -23.2711],
            [2.432, 36.942],
            [8.7371, -38.0221],
            [32.9201, 9.4081],
            [3.7534, 13.4704],
            [-18.5513, 25.1056],
            [5.0705, 16.6772],
            [-21.334, -19.3797],
            [-36.8785, 1.1317],
            [9.3937, -10.0719]
        ])

        self.data[:, 2] = True

    def position(self):
        return self.data[:, :2]

    def untracked(self):
        return self.data[:, 2]

    def allocation(self, p):
        untracked = self.data[:, 2].astype(bool)
        targets = self.data[untracked, :2].astype(float)
        r = p[:, None] - targets
        d2 = np.square(r).sum(axis=-1)
        a = {}
        for i in range(len(p)):
            try:
                j = d2[i].argmin()
                a[i] = targets[j]
            except ValueError:
                a[i] = p[i]
        return a

    def update(self, p):
        r = p[..., None, :] - self.data[:, :2]
        d2 = np.square(r).sum(axis=-1)
        c2 = (d2 < self.range**2).any(axis=0)
        self.data[c2, 2] = False

    def unfinished(self):
        return self.data[:, 2].any()


def tracking(position, target, R):
    r = position - target
    d = np.sqrt(np.square(r).sum())
    x = max(d, 10.)
    K = np.exp((R - x)/10)
    return - K * r / d


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(steps, network, logs):
    # iteración
    bar = progressbar.ProgressBar(maxval=arg.tf).start()
    perf_time = []
    t_init = 5.

    for k, t in steps[1:]:
        t_a = time.perf_counter()

        # update clocks
        network.timestamp = t
        for robot in robots:
            robot.computer.update_clock(t)

        # communication step
        network.cloud.append({robot.computer.node_id: [] for robot in robots})
        for i in network.ids:
            network.upload_to_cloud(i)
        for i in network.ids:
            network.download_from_cloud(i)

        # localization and control step
        network.get_gps(6)
        network.get_gps(8)

        p = network.collect_positions()
        alloc = targets.allocation(p)

        for robot in robots:
            robot.computer.localization_step()
            if t >= t_init and targets.unfinished():
                # might be est position
                u_track = tracking(
                    p[robot.computer.node_id],
                    alloc[robot.computer.node_id],
                    targets.range
                )
                u = robot.computer.compute_control_action(u_track)
                robot.dynamics.step(t, u)
            else:
                u = np.zeros(2)
                robot.computer.set_control_action(u)
                robot.dynamics.step(t, u)

        targets.update(network.collect_positions())

        t_b = time.perf_counter()

        network.update_adjacency()

        # log data
        logs.position[k] = network.collect_positions().ravel()
        logs.estimated_position[k] = network \
            .collect_estimated_positions().ravel()
        logs.action[k] = network.collect_control_actions().ravel()
        logs.fre[k] = network.rigidity_eigenvalue()
        sfre = network.subframeworks_rigidity_eigenvalue()
        logs.re[k] = sfre
        logs.adjacency[k] = network.adjacency_matrix.ravel()
        logs.action_extents[k] = network.collect_action_extents()
        logs.targets[k] = targets.data.ravel()

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
    default=1, type=float, help='paso de simulación en miliseg'
)
parser.add_argument(
    '-e', '--tf',
    default=10.0, type=float, help='tiempo total de simulación en seg'
)
parser.add_argument(
    '-q', '--queue',
    default=1, type=int, help='largo de la cola del cloud'
)
arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
step_milli = arg.step / 1000.0
time_interval = np.arange(0, arg.tf, step_milli)
steps = list(enumerate(time_interval))
n_steps = len(steps)

lim = 10
dim = 2
n = 10
# position = np.random.uniform(-0.9*lim, 0.9*lim, (n, dim))
position = np.array([
    [6.2343, 8.2097],
    [4.5759, 0.487],
    [-1.7355, -4.0296],
    [-2.1949, 5.8593],
    [-8.5903, 4.0411],
    [6.9784, -0.807],
    [-7.4382, -2.2692],
    [0.2778, 6.4405],
    [7.6862, 5.5808],
    [-1.0463, -0.8862]
])


dmin = 0.85 * lim
dmax = 0.90 * lim

adjacency_matrix = adjacency_from_positions(position, dmin)
if not is_inf_rigid(adjacency_matrix, position):
    raise ValueError('Framework should be infinitesimally rigid.')

geodesics = core.geodesics(adjacency_matrix)
action_extents = minimum_rigidity_extents(geodesics, position)
state_extents = superframework_extents(
    geodesics, action_extents
)

robots = [
    Robot(
        dynamics=Integrator(position[i]),
        computer=SubframeworkRigidityRobot(
            i,
            np.random.normal(position[i],  0.5**2),
            (dmin, dmax),
            action_extents[i],
            state_extents[i]
        )
    )
    for i in range(n)
]

network = Network(
    adjacency_matrix,
    robots,
    comm_range=(dmin, dmax),
    range_cov=0.5,
    gps_cov=1.,
    queue=arg.queue
)

print(network.collect_action_extents())
print(network.collect_positions())
print(network.collect_estimated_positions())

n_targets = 30
targets = Targets(n_targets, (-40, 40), (-40, 40))

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
# initialize()

logs = Logs(
    position=np.empty((n_steps, n*dim)),
    estimated_position=np.empty((n_steps, n*dim)),
    action=np.empty((n_steps, n*dim)),
    fre=np.zeros(n_steps),
    re=np.zeros((n_steps, n)),
    adjacency=np.empty((n_steps, n**2), dtype=int),
    action_extents=np.zeros((n_steps, n)),
    targets=np.empty((n_steps, 3*n_targets))
)
logs.position[0] = network.collect_positions().ravel()
logs.estimated_position[0] = network.collect_estimated_positions().ravel()
logs.action[0] = np.zeros(n*dim)
logs.fre[0] = network.rigidity_eigenvalue()
logs.re[0] = network.subframeworks_rigidity_eigenvalue()
logs.adjacency[0] = network.adjacency_matrix.ravel()
logs.action_extents[0] = network.collect_action_extents()
logs.targets[0] = targets.data.ravel()

logs = run(steps, network, logs)

print(network.collect_positions())
print(network.collect_estimated_positions())

np.savetxt('/tmp/t.csv', time_interval, delimiter=',')
np.savetxt('/tmp/position.csv', logs.position, delimiter=',')
np.savetxt('/tmp/est_position.csv', logs.estimated_position, delimiter=',')
np.savetxt('/tmp/action.csv', logs.action, delimiter=',')
np.savetxt('/tmp/fre.csv', logs.fre, delimiter=',')
np.savetxt('/tmp/re.csv', logs.re, delimiter=',')
np.savetxt('/tmp/adjacency.csv', logs.adjacency, delimiter=',')
np.savetxt('/tmp/extents.csv', logs.action_extents, delimiter=',')
np.savetxt('/tmp/targets.csv', logs.targets, delimiter=',')
