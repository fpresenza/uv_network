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
from uvnpy.distances.localization import FirstOrderKalmanFilter
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
    minimum_rigidity_extents,
    distance_matrix
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
    covariance, \
    action, \
    vel_meas_err, \
    gps_meas_err, \
    range_meas_err, \
    fre, \
    re, \
    adjacency, \
    action_extents, \
    targets')

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


class Robot(object):
    def __init__(
            self,
            node_id,
            pos,
            comm_range,
            action_extent=None,
            state_extent=None,
            t=0
            ):
        self.node_id = node_id
        self.dim = len(pos)
        self.dmin = np.min(comm_range)
        self.dmax = np.max(comm_range)
        self.action_extent = action_extent
        self.state_extent = state_extent
        self.current_time = t
        self.maintenance = CentralizedRigidityMaintenance(
            dim=self.dim, dmax=self.dmin,
            steepness=20.0/self.dmin, power=0.5, non_adjacent=True
        )
        self.collision = CollisionAvoidance(power=2.0)
        self.last_control_action = np.zeros(self.dim)
        self.action = {}
        vel_meas_cov = 0.0225 * np.eye(self.dim)
        range_meas_cov = 100.0
        gps_meas_cov = 100.0 * np.eye(self.dim)
        pos_cov = 25.0 * np.eye(self.dim)
        self.loc = FirstOrderKalmanFilter(
            pos, pos_cov, vel_meas_cov, range_meas_cov, gps_meas_cov
        )
        self.neighborhood = Neighborhood()
        self.routing = TokenPassing(self.node_id)
        self.state = {
            'position': self.loc.position(),
            'covariance': self.loc.covariance()
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

    def handle_received_msgs(self, msgs):
        for (msg, range_measurement) in msgs:
            self.neighborhood.update(
                msg.node_id, msg.timestamp,
                msg.state_tokens[msg.node_id].data['position'],
                msg.state_tokens[msg.node_id].data['covariance'],
                range_measurement
            )
            self.routing.update_action(msg.action_tokens.values())
            self.routing.update_state(msg.state_tokens.values())

    def set_control_action(self, u):
        self.last_control_action = u

    def compute_control_action(self, target):
        # go to allocated target
        r = self.loc.position() - target
        d = np.sqrt(np.square(r).sum())
        # v_collect = 1.0 if d < 100.0 else np.exp(1.0 - d/100.0)
        tracking_radius = 100.0    # radius
        forget_radius = 400.0      # radius
        v_collect_max = 2.5
        if d < 100.0:
            v_collect = v_collect_max
        elif d < 500.0:
            v_collect = v_collect_max * \
                (1.0 - (d - tracking_radius) / forget_radius)
        else:
            v_collect = 0.0
        u_collect = - v_collect * r / d

        # obtengo posiciones del subframework
        position = self.routing.extract_state('position', self.action_extent)
        degree = len(position)
        if degree > 0:
            p = np.empty((degree + 1, self.dim))
            p[0] = self.loc.position()
            p[1:] = list(position.values())

            # obtengo la accion de control de rigidez
            u_sub = self.maintenance.update(p)
            # u_sub = np.zeros((degree + 1, self.dim), dtype=float)
        else:
            u_sub = np.zeros((1, self.dim), dtype=float)

        # genero la accion de control del centro
        cmd = self.routing.extract_action()
        u_rigid = u_sub[0] + sum(cmd.values())

        # empaco las acciones de control del subframework
        self.action = {
            i: ui
            for i, ui in zip(position.keys(), u_sub[1:])
        }

        # accion de evacion de colisiones
        obstacles = self.routing.extract_state('position', 1)
        if len(obstacles) > 0:
            obstacles_pos = list(obstacles.values())
            u_evasion = self.collision.update(
                self.loc.position(), obstacles_pos
            )
        else:
            u_evasion = np.zeros(self.dim, dtype=float)

        # aplico acciones de control
        self.last_control_action = logistic_saturation(
            u_collect + 40.0 * u_rigid + 20000.0 * u_evasion, limit=2.5
        )

    def localization_step(self, vel_meas, gps_meas):
        self.loc.dynamic_step(self.current_time, vel_meas)
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
            self.loc.range_step(z, xj, Pj)
            self.neighborhood.clear()
        if gps_meas is not None:
            self.loc.gps_step(gps_meas)
        self.state['position'] = self.loc.position()
        self.state['covariance'] = self.loc.covariance()


class Robots(list):
    def collect_estimated_positions(self):
        return np.array([robot.loc.position() for robot in self])

    def collect_covariances(self):
        return np.array([
            np.linalg.eigvalsh(robot.loc.covariance()) for robot in self])

    def collect_action_extents(self):
        return np.array([robot.action_extent for robot in self])

    def collect_control_actions(self):
        return np.array([robot.last_control_action for robot in self])


class World(object):
    def __init__(
            self,
            robot_dynamics,
            network,
            comm_range,
            vel_meas_stdev,
            range_meas_stdev,
            gps_meas_stdev,
            queue=1
            ):
        """Clase para simular una red de robots"""
        self.robot_dynamics = robot_dynamics
        self.n = len(robot_dynamics)
        self.adjacency_matrix = network.astype(bool)
        self.dmin = np.min(comm_range)
        self.dmax = np.max(comm_range)

        self.vel_meas_stdev = vel_meas_stdev
        self.range_meas_stdev = range_meas_stdev
        self.gps_meas_stdev = gps_meas_stdev

        self.vel_meas = np.full((self.n, 2), np.nan)
        self.gps_meas = np.full((self.n, 2), np.nan)
        self.range_meas = np.full((self.n, self.n), np.nan)

        self.cloud = collections.deque(maxlen=queue)

    def collect_positions(self):
        return np.array([robot.x for robot in self.robot_dynamics])

    def collect_vel_meas_err(self):
        vel = np.array([robot.derivatives[0] for robot in self.robot_dynamics])
        return vel - self.vel_meas

    def collect_gps_meas_err(self):
        pos = np.array([robot.x for robot in self.robot_dynamics])
        return pos - self.gps_meas

    def collect_range_meas_err(self):
        dist = distance_matrix(self.collect_positions())
        dist[np.logical_not(self.adjacency_matrix)] = np.nan
        return dist - self.range_meas

    def update_adjacency(self, positions):
        self.adjacency_matrix = adjacency_histeresis(
            self.adjacency_matrix,
            positions,
            self.dmin, self.dmax
        )

    def velocity_measurement(self, node_index):
        self.vel_meas[node_index] = np.random.normal(
            self.robot_dynamics[node_index].derivatives[0], self.vel_meas_stdev
        )
        return self.vel_meas[node_index]

    def gps_measurement(self, node_index):
        self.gps_meas[node_index] = np.random.normal(
            self.robot_dynamics[node_index].x, self.gps_meas_stdev
        )
        return self.gps_meas[node_index]

    def upload_to_cloud(self, msg, node_index):
        neighbors = np.where(self.adjacency_matrix[node_index])[0]
        for neighbor_index in neighbors:
            distance = np.sqrt(np.square(
                (self.robot_dynamics[node_index].x -
                 self.robot_dynamics[neighbor_index].x)
            ).sum())
            self.range_meas[node_index, neighbor_index] = np.random.normal(
                distance,
                self.range_meas_stdev
            )
            self.cloud[-1][neighbor_index].append(
                (msg, self.range_meas[node_index, neighbor_index])
            )

    def download_from_cloud(self, node_index):
        return self.cloud[0][node_index].copy()


class Targets(object):
    def __init__(self, n, coverage):
        self.set(n)
        self.coverage = coverage

    def set(self, n):
        self.data = np.empty((n, 3), dtype=object)
        self.data[:, :2] = np.array([
            [748.400, 461.179],
            [149.536, 267.111],
            [443.235, 566.739],
            [517.441, 430.368],
            [364.085, 291.927],
            [164.434, 584.019],
            [525.689, 569.990],
            [847.281, 548.964],
            [781.627, 854.352],
            [744.927, 876.256],
            [887.665, 240.461],
            [454.270, 586.123],
            [186.770, 572.874],
            [227.913, 722.981],
            [230.557, 592.272],
            [815.733, 405.916],
            [341.763, 441.022],
            [194.592, 606.855],
            [250.020, 832.679],
            [424.510, 486.328],
            [675.176, 267.289],
            [524.320, 869.420],
            [587.371, 119.779],
            [829.201, 594.081],
            [537.534, 634.704],
            [314.487, 751.056],
            [550.705, 666.772],
            [286.660, 306.203],
            [131.215, 511.317],
            [593.937, 399.281]
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
        a = []
        for i in range(len(p)):
            try:
                j = d2[i].argmin()
                a.append(targets[j])
            except ValueError:
                a.append(p[i])
        return a

    def update(self, p):
        r = p[..., None, :] - self.data[:, :2]
        d2 = np.square(r).sum(axis=-1)
        c2 = (d2 < self.coverage**2).any(axis=0)
        self.data[c2, 2] = False

    def unfinished(self):
        return self.data[:, 2].any()


def framework_rigidity_eigenvalue(world):
    return rigidity_eigenvalue(
        world.adjacency_matrix, world.collect_positions()
    )


def subframeworks_rigidity_eigenvalue(robots, world):
    geodesics = core.geodesics(world.adjacency_matrix.astype(float))
    eigs = []
    for robot in robots:
        subset = geodesics[robot.node_id] <= \
            robot.action_extent
        A = world.adjacency_matrix[np.ix_(subset, subset)]
        p = world.collect_positions()[subset]   # TODO: IMPROVE SLICE
        eigs.append(rigidity_eigenvalue(A, p))

    return eigs


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(steps, world, logs):
    # iteración
    bar = progressbar.ProgressBar(maxval=arg.simu_time).start()
    perf_time = []

    for k, t in steps[1:]:
        t_a = time.perf_counter()

        # update clocks
        for robot in robots:
            robot.update_clock(t)

        # communication step
        world.cloud.append([[] for _ in robots])
        for robot in robots:
            msg = robot.create_msg()
            index = index_map[robot.node_id]
            world.upload_to_cloud(msg, index)
        for robot in robots:
            index = index_map[robot.node_id]
            msgs = world.download_from_cloud(index)
            robot.handle_received_msgs(msgs)

        # localization and control step
        # TODO: should be est position
        p = world.collect_positions()
        alloc = targets.allocation(p)

        for robot in robots:
            node_index = index_map[robot.node_id]

            if t >= t_init and targets.unfinished():
                robot.compute_control_action(alloc[node_index])
            else:
                robot.set_control_action(np.zeros(2, dtype=float))
            world.robot_dynamics[node_index].step(t, robot.last_control_action)

            vel_meas = world.velocity_measurement(node_index)
            if robot.node_id in [6, 8]:
                gps_meas = world.gps_measurement(node_index)
            else:
                gps_meas = None
            robot.localization_step(vel_meas, gps_meas)

        world.update_adjacency(world.collect_positions())
        targets.update(world.collect_positions())

        t_b = time.perf_counter()

        # log data
        logs.position[k] = world.collect_positions().ravel()
        logs.estimated_position[k] = robots \
            .collect_estimated_positions().ravel()
        logs.covariance[k] = robots \
            .collect_covariances().ravel()
        logs.action[k] = robots.collect_control_actions().ravel()

        logs.vel_meas_err[k] = world.collect_vel_meas_err().ravel()
        logs.gps_meas_err[k] = world.collect_gps_meas_err().ravel()
        logs.range_meas_err[k] = world.collect_range_meas_err().ravel()
        logs.fre[k] = framework_rigidity_eigenvalue(world)
        logs.re[k] = subframeworks_rigidity_eigenvalue(robots, world)
        logs.adjacency[k] = world.adjacency_matrix.ravel()
        logs.action_extents[k] = robots.collect_action_extents()
        logs.targets[k] = targets.data.ravel()

        perf_time.append((t_b - t_a)/n)
        bar.update(np.round(t, 3))

    bar.finish()

    rt = arg.simu_time
    st = sum(perf_time)
    prompt = 'ST={:.3f} secs, RT={:.3f} secs  ==>  RTF={:.3f}'
    print(prompt.format(st, rt, rt / st))

    return logs


# ------------------------------------------------------------------
# Parseo de argumentos
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-s', '--simu_step',
    default=1, type=int, help='simulation step in milli seconds'
)
parser.add_argument(
    '-t', '--simu_time',
    default=10.0, type=float, help='total simulation time in seconds'
)
parser.add_argument(
    '-q', '--queue',
    default=1, type=int, help='largo de la cola del cloud'
)
arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
step_sec = arg.simu_step / 1000.0
time_interval = np.arange(0.0, arg.simu_time, step_sec)
steps = list(enumerate(time_interval))
n_steps = len(steps)

position = np.array([
    [562.343, 582.097],
    [545.759, 504.870],
    [482.645, 459.704],
    [478.051, 558.593],
    [414.097, 540.411],
    [569.784, 491.930],
    [425.618, 477.308],
    [502.778, 564.405],
    [576.862, 555.808],
    [489.537, 491.138]
])
n, dim = position.shape

coverage = 30.0
dmin = 85.0
dmax = 90.0

adjacency_matrix = adjacency_from_positions(position, dmin)
if not is_inf_rigid(adjacency_matrix, position):
    raise ValueError('Framework should be infinitesimally rigid.')

world = World(
    robot_dynamics=[Integrator(position[i]) for i in range(n)],
    network=adjacency_matrix,
    comm_range=(dmin, dmax),
    vel_meas_stdev=0.15,
    range_meas_stdev=10.0,
    gps_meas_stdev=10.0,
    queue=arg.queue
)

geodesics = core.geodesics(adjacency_matrix)
action_extents = minimum_rigidity_extents(geodesics, position)
state_extents = superframework_extents(geodesics, action_extents)

robots = Robots([
    Robot(
        i,
        np.random.normal(position[i],  5.0),
        (dmin, dmax),
        action_extents[i],
        state_extents[i]
    )
    for i in range(n)
])

index_map = {robots[i].node_id: i for i in range(n)}
print('Index map: {}'.format(index_map))
# print(robots.collect_action_extents())
# print(world.collect_positions())
# print(robots.collect_estimated_positions())

n_targets = 30
targets = Targets(n_targets, coverage)

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
# initialize()

t_init = np.ceil(np.max(geodesics) * step_sec)

logs = Logs(
    position=np.empty((n_steps, n*dim)),
    estimated_position=np.empty((n_steps, n*dim)),
    covariance=np.empty((n_steps, n*dim)),
    action=np.empty((n_steps, n*dim)),
    vel_meas_err=np.empty((n_steps, n*dim)),
    gps_meas_err=np.empty((n_steps, n*dim)),
    range_meas_err=np.empty((n_steps, n*n)),
    fre=np.zeros(n_steps),
    re=np.zeros((n_steps, n)),
    adjacency=np.empty((n_steps, n**2), dtype=int),
    action_extents=np.zeros((n_steps, n)),
    targets=np.empty((n_steps, 3*n_targets))
)
logs.position[0] = world.collect_positions().ravel()
logs.estimated_position[0] = robots.collect_estimated_positions().ravel()
logs.covariance[0] = robots.collect_covariances().ravel()
logs.action[0] = np.zeros(n*dim)
logs.vel_meas_err[0] = np.zeros(n*dim)
logs.gps_meas_err[0] = np.zeros(n*dim)
logs.range_meas_err[0] = np.zeros(n*n)
world.adjacency_matrix
logs.fre[0] = framework_rigidity_eigenvalue(world)
logs.re[0] = subframeworks_rigidity_eigenvalue(robots, world)
logs.adjacency[0] = world.adjacency_matrix.ravel()
logs.action_extents[0] = robots.collect_action_extents()
logs.targets[0] = targets.data.ravel()

logs = run(steps, world, logs)

# print(world.collect_positions())
# print(robots.collect_estimated_positions())

np.savetxt('data/t.csv', time_interval, delimiter=',')
np.savetxt('data/position.csv', logs.position, delimiter=',')
np.savetxt('data/est_position.csv', logs.estimated_position, delimiter=',')
np.savetxt('data/covariance.csv', logs.covariance, delimiter=',')
np.savetxt('data/action.csv', logs.action, delimiter=',')
np.savetxt('data/vel_meas_err.csv', logs.vel_meas_err, delimiter=',')
np.savetxt('data/gps_meas_err.csv', logs.gps_meas_err, delimiter=',')
np.savetxt('data/range_meas_err.csv', logs.range_meas_err, delimiter=',')
np.savetxt('data/fre.csv', logs.fre, delimiter=',')
np.savetxt('data/re.csv', logs.re, delimiter=',')
np.savetxt('data/adjacency.csv', logs.adjacency, delimiter=',')
np.savetxt('data/extents.csv', logs.action_extents, delimiter=',')
np.savetxt('data/targets.csv', logs.targets, delimiter=',')
