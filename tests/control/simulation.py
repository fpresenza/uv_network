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
import copy

from uvnpy.network import core
from uvnpy.distances.localization import FirstOrderKalmanFilter
from uvnpy.routing.token_passing import TokenPassing
from uvnpy.dynamics.linear_models import Integrator
from uvnpy.toolkit.functions import logistic_saturation
from uvnpy.network.disk_graph import adjacency_from_positions
from uvnpy.distances.core import is_inf_rigid, rigidity_eigenvalue
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
    'time, \
    position, \
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
    state_extents, \
    targets')

InterRobotMsg = collections.namedtuple(
    'InterRobotMsg',
    'node_id, \
    timestamp, \
    in_balls, \
    action_tokens, \
    state_tokens')

NeigborhoodData = collections.namedtuple(
    'NeigborhoodData',
    'node_id, \
    timestamp, \
    position, \
    covariance, \
    range, \
    isolated_edge')


class Neighborhood(dict):
    def update(
            self,
            node_id,
            timestamp,
            position,
            covariance,
            range_measurement,
            isolated_edge
            ):
        self[node_id] = NeigborhoodData(
            node_id=node_id,
            timestamp=timestamp,
            position=position,
            covariance=covariance,
            range=range_measurement,
            isolated_edge=isolated_edge
        )


class Robot(object):
    def __init__(
            self,
            node_id,
            pos,
            comm_range,
            action_extent=0,
            state_extent=1,
            t=0.0
            ):
        self.node_id = node_id
        self.dim = len(pos)
        self.comm_range = comm_range
        self.safe_comm_range = 0.94 * comm_range
        self.action_extent = action_extent
        self.state_extent = state_extent
        self.current_time = t
        self.centered_ball = {node_id} if (action_extent > 0) else set()
        self.in_balls = self.centered_ball
        self.maintenance = CentralizedRigidityMaintenance(
            dim=self.dim, dmax=self.safe_comm_range,
            steepness=20.0/self.safe_comm_range, power=0.5, non_adjacent=True
        )
        self.collision = CollisionAvoidance(power=2.0)
        self.u_target = np.zeros(self.dim, dtype=float)
        self.u_collision = np.zeros(self.dim, dtype=float)
        self.u_rigidity = np.zeros(self.dim, dtype=float)
        self.last_control_action = np.zeros(self.dim, dtype=float)
        self.action = {}
        self.loc = FirstOrderKalmanFilter(
            pos,
            pos_cov=25.0 * np.eye(self.dim),
            vel_meas_cov=0.0225 * np.eye(self.dim),
            range_meas_cov=100.0,
            gps_meas_cov=100.0 * np.eye(self.dim)
        )
        self.state = {'position': self.loc.x, 'covariance': self.loc.P}
        self.neighborhood = Neighborhood()
        self.routing = TokenPassing(self.node_id)

    def update_clock(self, t):
        self.current_time = t

    def create_msg(self):
        action_tokens, state_tokens = self.routing.broadcast(
            self.current_time,
            copy.deepcopy(self.action),
            copy.deepcopy(self.state),
            self.action_extent,
            self.state_extent
        )
        msg = InterRobotMsg(
            node_id=self.node_id,
            timestamp=self.current_time,
            in_balls=self.in_balls.copy(),
            action_tokens=action_tokens,
            state_tokens=state_tokens
        )
        return msg

    def handle_received_msgs(self, msgs):
        self.neighborhood.clear()
        for (msg, range_measurement) in msgs:
            self.neighborhood.update(
                node_id=msg.node_id,
                timestamp=msg.timestamp,
                position=msg.state_tokens[msg.node_id].data['position'],
                covariance=msg.state_tokens[msg.node_id].data['covariance'],
                range_measurement=range_measurement,
                isolated_edge=self.in_balls.isdisjoint(msg.in_balls)
            )
            self.routing.update_action(msg.action_tokens.values())
            self.routing.update_state(msg.state_tokens.values())

        self.in_balls = self.centered_ball.union(self.routing.action_centers())
        # self.state_extent = max(1, self.max_action_extent())

    def set_control_action(self, u):
        self.last_control_action = u

    def target_collection_control_action(self, target):
        if (target is not None):
            # go to allocated target
            r = self.loc.position() - target
            d = np.sqrt(np.square(r).sum())
            # v_collect = 1.0 if d < 100.0 else np.exp(1.0 - d/100.0)
            tracking_radius = 100.0    # radius
            forget_radius = 400.0      # radius
            v_collect_max = 2.5
            if d < tracking_radius:
                v_collect = v_collect_max
            elif d < (tracking_radius + forget_radius):
                v_collect = v_collect_max * \
                    (1.0 - (d - tracking_radius) / forget_radius)
            else:
                v_collect = 0.0
            self.u_target = - v_collect * r / d
        else:
            self.u_target = np.zeros(self.dim, dtype=float)

    def collision_avoidance_control_action(self):
        # get obstacles (other robots positions)
        obstacles = self.routing.extract_state('position', 1)
        if len(obstacles) > 0:
            obstacles_pos = list(obstacles.values())
            self.u_collision = self.collision.update(
                self.loc.position(), obstacles_pos
            )
        else:
            self.u_collision = np.zeros(self.dim, dtype=float)

    def rigidity_maintenance_control_action(self):
        # get subframework positions
        position = self.routing.extract_state('position', self.action_extent)
        n_sub = len(position)
        if n_sub > 0:
            p = np.empty((n_sub + 1, self.dim), dtype=float)
            p[0] = self.loc.position()
            p[1:] = list(position.values())
            # get rigidity maintenance control action
            u_sub = self.maintenance.update(p)
            # u_sub = np.zeros((n_sub + 1, self.dim), dtype=float)
        else:
            u_sub = np.zeros((1, self.dim), dtype=float)

        # compose all containing subframeworks control actions
        cmd = self.routing.extract_action()
        self.u_rigidity = u_sub[0] + sum(cmd.values())

        # empaco las acciones de control del subframework
        self.action = {
            i: ui
            for i, ui in zip(position.keys(), u_sub[1:])
        }

    def compose_control_actions(self):
        # aplico acciones de control
        self.last_control_action = logistic_saturation(
            self.u_target +
            20000.0 * self.u_collision +
            40.0 * self.u_rigidity,
            limit=2.5
        )

    def velocity_measurement_step(self, vel_meas):
        self.loc.dynamic_step(self.current_time, vel_meas)

    def range_measurement_step(self):
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

    def gps_measurement_step(self, gps_meas):
        self.loc.gps_step(gps_meas)


class Robots(list):
    def collect_estimated_positions(self):
        return np.array([robot.loc.position() for robot in self])

    def collect_covariances(self):
        return np.array([
            np.linalg.eigvalsh(robot.loc.covariance()) for robot in self])

    def collect_action_extents(self):
        return np.array([robot.action_extent for robot in self])

    def collect_state_extents(self):
        return np.array([robot.state_extent for robot in self])

    def collect_control_actions(self):
        return np.array([robot.last_control_action for robot in self])


class World(object):
    def __init__(
            self,
            robot_dynamics,
            network,
            comm_range,
            gps_available,
            vel_meas_stdev,
            range_meas_stdev,
            gps_meas_stdev,
            queue=1
            ):
        """Clase para simular una red de robots"""
        self.robot_dynamics = robot_dynamics
        self.n = len(robot_dynamics)
        self.adjacency_matrix = network.astype(bool)
        self.comm_range = comm_range
        self.gps_available = gps_available

        self.vel_meas_stdev = vel_meas_stdev
        self.range_meas_stdev = range_meas_stdev
        self.gps_meas_stdev = gps_meas_stdev

        self.vel_meas_err = np.full((self.n, 2), np.nan)
        self.gps_meas_err = np.full((self.n, 2), np.nan)
        self.range_meas_err = np.full((self.n, self.n), np.nan)

        self.cloud = collections.deque(maxlen=queue)

    def collect_positions(self):
        return np.array([robot.x for robot in self.robot_dynamics])

    def collect_vel_meas_err(self):
        return self.vel_meas_err.copy()

    def collect_gps_meas_err(self):
        return self.gps_meas_err.copy()

    def collect_range_meas_err(self):
        return self.range_meas_err.copy()

    def update_adjacency(self):
        self.adjacency_matrix = adjacency_from_positions(
            self.collect_positions(), self.comm_range
        ).astype(bool)

    def framework_rigidity_eigenvalue(self):
        return rigidity_eigenvalue(
            self.adjacency_matrix, self.collect_positions()
        )

    def subframeworks_rigidity_eigenvalue(self, robots):
        geodesics = core.geodesics_dict(self.adjacency_matrix)
        eigs = []
        for robot in robots:
            if robot.action_extent > 0:
                g_i = geodesics[robot.node_id]
                h_i = robot.action_extent
                subset = [j for j, g_ij in g_i.items() if g_ij <= h_i]
                A = self.adjacency_matrix[np.ix_(subset, subset)]
                p = self.collect_positions()[subset]   # TODO: IMPROVE SLICE
                eigs.append(rigidity_eigenvalue(A, p))

        return eigs

    def velocity_measurement(self, node_index):
        if len(self.robot_dynamics[node_index].derivatives) > 0:
            vel = self.robot_dynamics[node_index].derivatives[0]
            self.vel_meas_err[node_index] = np.random.normal(
                scale=self.vel_meas_stdev, size=2
            )
            return vel + self.vel_meas_err[node_index]
        else:
            self.vel_meas_err[node_index] = np.nan
            return None

    def gps_measurement(self, node_index):
        if node_index in self.gps_available:
            pos = self.robot_dynamics[node_index].x
            self.gps_meas_err[node_index] = np.random.normal(
                scale=self.gps_meas_stdev, size=2
            )
            return pos + self.gps_meas_err[node_index]
        else:
            self.gps_meas_err[node_index] = np.nan
            return None

    def upload_to_cloud(self, msg, node_index):
        for neighbor_index in range(self.n):
            if self.adjacency_matrix[node_index, neighbor_index]:
                distance = np.sqrt(np.square(
                    (self.robot_dynamics[node_index].x -
                     self.robot_dynamics[neighbor_index].x)
                ).sum())
                self.range_meas_err[node_index, neighbor_index] = \
                    np.random.normal(
                        scale=self.range_meas_stdev, size=1
                    )
                self.cloud[-1][neighbor_index].append(
                    (
                        msg,
                        distance +
                        self.range_meas_err[node_index, neighbor_index]
                    )
                )
            else:
                self.range_meas_err[node_index, neighbor_index] = np.nan

    def download_from_cloud(self, node_index):
        return self.cloud[0][node_index].copy()


class Targets(object):
    def __init__(self, n, coverage):
        self.set(n)
        self.coverage = coverage

    def set(self, n):
        self.data = np.empty((n, 3), dtype=object)
        self.data[:, :2] = np.random.uniform(0.0, 1000.0, (n, 2))

        self.data[:, 2] = True

    def position(self):
        return self.data[:, :2]

    def untracked(self):
        return self.data[:, 2]

    def allocation(self, p):
        alloc = {i: None for i in range(len(p))}
        untracked = self.data[:, 2].astype(bool)
        if untracked.any():
            targets = self.data[untracked, :2].astype(float)
            r = p[:, None] - targets
            d2 = np.square(r).sum(axis=-1)
            for i in range(len(p)):
                j = d2[i].argmin()
                alloc[i] = targets[j]

        return alloc

    def update(self, p):
        r = p[..., None, :] - self.data[:, :2]
        d2 = np.square(r).sum(axis=-1)
        c2 = (d2 < self.coverage**2).any(axis=0)
        self.data[c2, 2] = False

    def unfinished(self):
        return self.data[:, 2].any()


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------
def initialize_robots():
    k = 0
    comm_events = 0
    while comm_events < 2 * np.max(action_extents):
        # update clocks
        t = time_steps.pop(0)
        for robot in robots:
            robot.update_clock(t)

        # communication step
        if (t % comm_step) < simu_step:
            world.cloud.append([[] for _ in robots])
            for robot in robots:
                msg = robot.create_msg()
                index = index_map[robot.node_id]
                world.upload_to_cloud(msg, index)
            for robot in robots:
                index = index_map[robot.node_id]
                msgs = world.download_from_cloud(index)
                robot.handle_received_msgs(msgs)
                robot.range_measurement_step()
            comm_events += 1
            print('Communication event {} finished'.format(comm_events))

        # localization step
        for robot in robots:
            node_index = index_map[robot.node_id]

            gps_meas = world.gps_measurement(node_index)
            if (gps_meas is not None):
                robot.gps_measurement_step(gps_meas)

            vel_meas = world.velocity_measurement(node_index)
            if (vel_meas is not None):
                robot.velocity_measurement_step(vel_meas)

        for robot in robots:
            node_index = index_map[robot.node_id]
            world.robot_dynamics[node_index].step(t, robot.last_control_action)
        world.update_adjacency()
        targets.update(world.collect_positions())

        # log data
        logs.time.append(t)
        logs.position.append(world.collect_positions().ravel())
        logs.estimated_position.append(
            robots.collect_estimated_positions().ravel()
        )
        logs.covariance.append(robots.collect_covariances().ravel())
        logs.action.append(robots.collect_control_actions().ravel())
        logs.vel_meas_err.append(world.collect_vel_meas_err().ravel())
        logs.gps_meas_err.append(world.collect_gps_meas_err().ravel())
        logs.range_meas_err.append(world.collect_range_meas_err().ravel())
        logs.fre.append(world.framework_rigidity_eigenvalue())
        logs.re.append(world.subframeworks_rigidity_eigenvalue(robots))
        logs.adjacency.append(world.adjacency_matrix.ravel())
        logs.action_extents.append(robots.collect_action_extents())
        logs.state_extents.append(robots.collect_state_extents())
        logs.targets.append(targets.data.ravel().copy())

        k += 1

    print(
        'Initialization completed after {} communication events.'
        .format(comm_events)
    )


def run_mission():
    bar = progressbar.ProgressBar(maxval=arg.simu_time).start()
    perf_time = []

    k = 0
    while len(time_steps) > 0:
        t_a = time.perf_counter()

        # update clocks
        t = time_steps.pop(0)
        for robot in robots:
            robot.update_clock(t)

        # communication step
        if (t % comm_step) < simu_step:
            world.cloud.append([[] for _ in robots])
            for robot in robots:
                msg = robot.create_msg()
                index = index_map[robot.node_id]
                world.upload_to_cloud(msg, index)
            for robot in robots:
                index = index_map[robot.node_id]
                msgs = world.download_from_cloud(index)
                robot.handle_received_msgs(msgs)
                robot.range_measurement_step()
                robot.rigidity_maintenance_control_action()
        # localization and control step
        # TODO: should be est position
        p = world.collect_positions()
        alloc = targets.allocation(p)

        for robot in robots:
            node_index = index_map[robot.node_id]

            robot.target_collection_control_action(alloc[node_index])
            robot.collision_avoidance_control_action()
            robot.compose_control_actions()

            gps_meas = world.gps_measurement(node_index)
            if (gps_meas is not None):
                robot.gps_measurement_step(gps_meas)

            vel_meas = world.velocity_measurement(node_index)
            if (vel_meas is not None):
                robot.velocity_measurement_step(vel_meas)

        for robot in robots:
            node_index = index_map[robot.node_id]
            world.robot_dynamics[node_index].step(t, robot.last_control_action)
        world.update_adjacency()
        targets.update(world.collect_positions())

        # log data
        logs.time.append(t)
        logs.position.append(world.collect_positions().ravel())
        logs.estimated_position.append(
            robots.collect_estimated_positions().ravel()
        )
        logs.covariance.append(robots.collect_covariances().ravel())
        logs.action.append(robots.collect_control_actions().ravel())
        logs.vel_meas_err.append(world.collect_vel_meas_err().ravel())
        logs.gps_meas_err.append(world.collect_gps_meas_err().ravel())
        logs.range_meas_err.append(world.collect_range_meas_err().ravel())
        logs.fre.append(world.framework_rigidity_eigenvalue())
        logs.re.append(world.subframeworks_rigidity_eigenvalue(robots))
        logs.adjacency.append(world.adjacency_matrix.ravel())
        logs.action_extents.append(robots.collect_action_extents())
        logs.state_extents.append(robots.collect_state_extents())
        logs.targets.append(targets.data.ravel().copy())

        k += 1

        t_b = time.perf_counter()
        perf_time.append((t_b - t_a)/n)
        bar.update(np.round(t, 3))

    bar.finish()

    rt = arg.simu_time
    st = sum(perf_time)
    prompt = 'ST={:.3f} secs (per robot), RT={:.3f} secs  ==>  RTF={:.3f}'
    print(prompt.format(st, rt, rt / st))


# ------------------------------------------------------------------
# Parseo de argumentos
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-s', '--simu_step',
    default=1, type=float, help='simulation step in milli seconds'
)
parser.add_argument(
    '-c', '--comm_step',
    default=1, type=float, help='communication step in milli seconds'
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
np.random.seed(0)
simu_step = arg.simu_step / 1000.0
comm_step = arg.comm_step / 1000.0
simu_time = arg.simu_time
time_steps = [simu_step * k for k in range(int(simu_time / simu_step))]
n_steps = len(time_steps)

print(
    'Simulation Time: begin = {}, end = {}, step = {}'
    .format(0.0, simu_time, simu_step)
)
print(
    'Communication Time: begin = {}, end = {}, step = {}'
    .format(0.0, simu_time, comm_step)
)

position = np.array([
    [162.343, 182.097],
    [145.759, 104.870],
    [82.645,  59.704],
    [78.051, 158.593],
    [14.097, 140.411],
    [169.784,  91.930],
    [25.618,  77.308],
    [102.778, 164.405],
    [176.862, 155.808],
    [89.537,  91.138]
])

n, dim = position.shape

comm_range = 90.0

adjacency_matrix = adjacency_from_positions(position, comm_range)
print(
    'Adjacency list: \n' +
    '\n'.join(
        '\t {}: {}'.format(key, val)
        for key, val in core.adjacency_dict(adjacency_matrix).items()
    )
)
if not is_inf_rigid(adjacency_matrix, position):
    raise ValueError('Framework should be infinitesimally rigid.')

action_extents = np.array([0, 2, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int)
print(
    'Action extents: \n' +
    '\n'.join(
        '\t node = {}, extent = {}'.format(i, r)
        for i, r in enumerate(action_extents) if r > 0
    )
)

world = World(
    robot_dynamics=[Integrator(position[i]) for i in range(n)],
    network=adjacency_matrix,
    comm_range=comm_range,
    gps_available=[6, 8],
    vel_meas_stdev=0.15,
    range_meas_stdev=10.0,
    gps_meas_stdev=10.0,
    queue=arg.queue
)

robots = Robots([
    Robot(
        node_id=i,
        pos=np.random.normal(position[i],  5.0),
        comm_range=comm_range,
        action_extent=int(action_extents[i]),
        state_extent=2
    )
    for i in range(n)
])

index_map = {robots[i].node_id: i for i in range(n)}
# print('Index map: {}'.format(index_map))

n_targets = 80
coverage = 30.0
targets = Targets(n_targets, coverage)

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
logs = Logs(
    time=[],
    position=[],
    estimated_position=[],
    covariance=[],
    action=[],
    vel_meas_err=[],
    gps_meas_err=[],
    range_meas_err=[],
    fre=[],
    re=[],
    adjacency=[],
    action_extents=[],
    state_extents=[],
    targets=[]
)

initialize_robots()
print(
    'State extents: \n' +
    '\n'.join(
        '\t node = {}, extent = {}'
        .format(robot.node_id, robot.state_extent)
        for robot in robots
    )
)
run_mission()

np.savetxt('data/t.csv', logs.time, delimiter=',')
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
np.savetxt('data/action_extents.csv', logs.action_extents, delimiter=',')
np.savetxt('data/state_extents.csv', logs.state_extents, delimiter=',')
np.savetxt('data/targets.csv', logs.targets, delimiter=',')
