#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import collections
import time
import progressbar
import numpy as np
import copy
import transformations

from uvnpy.network import core
from uvnpy.bearings.localization import FirstOrderKalmanFilter
from uvnpy.routing.token_passing import TokenPassing
from uvnpy.dynamics.linear_models import Integrator
# from uvnpy.toolkit.functions import logistic_saturation
from uvnpy.network.disk_graph import adjacency_from_positions
from uvnpy.bearings.control import RigidityMaintenance
from uvnpy.control.core import Targets, CollisionAvoidanceVanishing
from uvnpy.bearings.core import (
    is_inf_rigid,
    minimum_rigidity_extents,
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
    time_comm, \
    position, \
    estimated_position,  \
    covariance, \
    target_action, \
    collision_action, \
    rigidity_action, \
    vel_meas_err, \
    gps_meas_err, \
    bearing_meas_err, \
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
    bearing, \
    is_isolated_edge')


class Neighborhood(dict):
    def update(
            self,
            node_id,
            timestamp,
            position,
            covariance,
            bearing_measurement,
            is_isolated_edge
            ):
        self[node_id] = NeigborhoodData(
            node_id=node_id,
            timestamp=timestamp,
            position=position,
            covariance=covariance,
            bearing=bearing_measurement,
            is_isolated_edge=is_isolated_edge
        )


class Robot(object):
    def __init__(
            self,
            node_id,
            position,
            comm_range,
            action_extent=0,
            state_extent=1,
            t=0.0
            ):
        self.node_id = node_id
        self.dim = len(position)
        self.action_extent = action_extent
        self.state_extent = state_extent
        self.current_time = t
        self.self_centered_ball = {node_id} if (action_extent > 0) else set()
        self.in_balls = self.self_centered_ball
        self.maintenance = RigidityMaintenance(
            dim=3,
            dmax=0.92 * comm_range,
            steepness=10.0,
            threshold=1e-4,
            eigenvalues='all',
            functional='log'
        )
        self.collision = CollisionAvoidanceVanishing(
            power=2.0,
            dmin=1.0,
            dmax=comm_range
        )
        self.u_target = np.zeros(self.dim, dtype=float)
        self.u_collision = np.zeros(self.dim, dtype=float)
        self.u_rigidity = np.zeros(self.dim, dtype=float)
        self.last_control_action = np.zeros(self.dim, dtype=float)
        self.action = {}
        self.loc = FirstOrderKalmanFilter(
            position,
            position_cov=0.0 * np.eye(self.dim),
            vel_meas_cov=0.0 * np.eye(self.dim),
            bearing_meas_cov=0.0 * np.eye(self.dim),
            gps_meas_cov=0.0 * np.eye(self.dim)
        )
        self.neighborhood = Neighborhood()
        self.routing = TokenPassing(self.node_id)

    def update_clock(self, t):
        self.current_time = t

    def create_msg(self):
        action_tokens, state_tokens = self.routing.broadcast(
            timestamp=self.current_time,
            action=copy.deepcopy(self.action),
            state={
                'position': self.loc.state(),
                'covariance': self.loc.covariance()
            },
            action_extent=self.action_extent,
            state_extent=self.state_extent
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
        for (msg, bearing_measurement) in msgs:
            self.neighborhood.update(
                node_id=msg.node_id,
                timestamp=msg.timestamp,
                position=msg.state_tokens[msg.node_id].data['position'],
                covariance=msg.state_tokens[msg.node_id].data['covariance'],
                bearing_measurement=bearing_measurement,
                is_isolated_edge=self.in_balls.isdisjoint(msg.in_balls)
            )
            self.routing.update_action(msg.action_tokens.values())
            self.routing.update_state(msg.state_tokens.values())

        self.in_balls = self.self_centered_ball.union(
            self.routing.action_centers()
        )

    def update_state_extent(self):
        self.state_extent = max(1, self.routing.max_action_extent())

    def set_control_action(self, u):
        self.last_control_action = u

    def target_collection_control_action(self, target):
        if (target is not None):
            # go to allocated target
            r = self.loc.position() - target
            d = np.sqrt(np.square(r).sum())
            tracking_radius = 20.0    # radius
            forget_radius = 100.0     # radius
            v_collect_max = 2.5
            if d < tracking_radius:
                v_collect = v_collect_max
            elif d < forget_radius:
                factor = (forget_radius - d)/(forget_radius - tracking_radius)
                v_collect = v_collect_max * factor
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
            # collision control gain
            self.u_collision *= 0.75
        else:
            self.u_collision = np.zeros(self.dim, dtype=float)

    def rigidity_maintenance_control_action(self):
        # get actions for ball subframework
        position = self.routing.extract_state('position', self.action_extent)
        n_sub = len(position)
        if n_sub > 0:
            p = np.empty((n_sub + 1, self.dim), dtype=float)
            p[0] = self.loc.position()
            p[1:] = list(position.values())
            # get rigidity maintenance control action
            u_sub = self.maintenance.update(p)
        else:
            u_sub = np.zeros((1, self.dim), dtype=float)

        # pack control action for other robots within ball
        self.action = {
            i: ui
            for i, ui in zip(position.keys(), u_sub[1:])
        }

        # compose all control actions from containing balls
        cmd = self.routing.extract_action()
        self.u_rigidity = u_sub[0] + sum(cmd.values())

        # add action for isolated edges
        for neighbor in self.neighborhood.values():
            if neighbor.is_isolated_edge:
                p = np.vstack([self.loc.position(), neighbor.position])
                self.u_rigidity += self.maintenance.update(p)[0]

        # rigidity control gain
        self.u_rigidity *= 0.15

    def compose_actions(self):
        # compose control actions from different objectives and
        # apply logistic saturation
        # self.last_control_action = logistic_saturation(
        #     (self.u_target + self.u_collision + self.u_rigidity) * 0.85,
        #     limit=3.0
        # )
        self.last_control_action = \
            self.u_target + self.u_collision + self.u_rigidity

    def velocity_measurement_step(self, vel_meas):
        self.loc.dynamic_step(self.current_time, vel_meas)

    def bearing_measurement_step(self):
        if len(self.neighborhood) > 0:
            neighbors_data = self.neighborhood.values()
            z = np.array([
                neighbor.bearing for neighbor in neighbors_data
            ])
            xj = np.array([
                neighbor.position for neighbor in neighbors_data
            ])
            Pj = np.array([
                neighbor.covariance for neighbor in neighbors_data
            ])
            self.loc.bearing_step(z, xj, Pj)

    def gps_measurement_step(self, gps_meas):
        self.loc.gps_step(gps_meas)


class Robots(list):
    def collect_estimated_positions(self):
        return np.hstack([robot.loc.position() for robot in self])

    def collect_covariances(self):
        return np.hstack([
            np.linalg.eigvalsh(robot.loc.covariance()) for robot in self])

    def collect_action_extents(self):
        return np.hstack([robot.action_extent for robot in self])

    def collect_state_extents(self):
        return np.hstack([robot.state_extent for robot in self])

    def collect_target_actions(self):
        return np.hstack([robot.u_target for robot in self])

    def collect_collision_actions(self):
        return np.hstack([robot.u_collision for robot in self])

    def collect_rigidity_actions(self):
        return np.hstack([robot.u_rigidity for robot in self])


class World(object):
    def __init__(
            self,
            dim,
            robot_dynamics,
            network,
            comm_range,
            gps_available,
            vel_meas_stdev,
            bearing_meas_stdev,
            gps_meas_stdev,
            queue=1
            ):
        """Clase para simular una red de robots"""
        self.dim = dim
        self.robot_dynamics = robot_dynamics
        self.n = len(robot_dynamics)
        self.adjacency_matrix = network.astype(bool)
        self.comm_range = comm_range
        self.gps_available = gps_available

        self.vel_meas_stdev = vel_meas_stdev
        self.bearing_meas_stdev = bearing_meas_stdev
        self.gps_meas_stdev = gps_meas_stdev

        self.vel_meas_err = np.full((self.n, self.dim), np.nan)
        self.gps_meas_err = np.full((self.n, self.dim), np.nan)
        self.bearing_meas_err = np.full((self.n, self.n), np.nan)

        self.cloud = collections.deque(maxlen=queue)

    def positions(self, subset=None):
        if subset is None:
            return np.array([robot.x for robot in self.robot_dynamics])
        else:
            return np.array([self.robot_dynamics[i].x for i in subset])

    def collect_positions(self):
        return np.hstack([robot.x for robot in self.robot_dynamics])

    def collect_vel_meas_err(self):
        return self.vel_meas_err.copy().ravel()

    def collect_gps_meas_err(self):
        return self.gps_meas_err.copy().ravel()

    def collect_bearing_meas_err(self):
        return self.bearing_meas_err.copy().ravel()

    def update_adjacency(self):
        self.adjacency_matrix = adjacency_from_positions(
            self.positions(), self.comm_range
        ).astype(bool)

    def velocity_measurement(self, node_index):
        if len(self.robot_dynamics[node_index].derivatives) > 0:
            vel = self.robot_dynamics[node_index].derivatives[0]
            self.vel_meas_err[node_index] = np.random.normal(
                scale=self.vel_meas_stdev, size=self.dim
            )
            return vel + self.vel_meas_err[node_index]
        else:
            self.vel_meas_err[node_index] = np.nan
            return None

    def gps_measurement(self, node_index):
        if node_index in self.gps_available:
            position = self.robot_dynamics[node_index].x
            self.gps_meas_err[node_index] = np.random.normal(
                scale=self.gps_meas_stdev, size=self.dim
            )
            return position + self.gps_meas_err[node_index]
        else:
            self.gps_meas_err[node_index] = np.nan
            return None

    def upload_to_cloud(self, msg, node_index):
        for neighbor_index in range(self.n):
            if self.adjacency_matrix[node_index, neighbor_index]:
                noise = np.random.normal(
                        scale=self.bearing_meas_stdev, size=self.dim
                )
                noisy_bearing = transformations.unit_vector(
                    self.robot_dynamics[node_index].x -
                    self.robot_dynamics[neighbor_index].x +
                    noise
                )
                self.bearing_meas_err[node_index, neighbor_index] = \
                    np.dot(noise, noise)
                self.cloud[-1][neighbor_index].append((msg, noisy_bearing))
            else:
                self.bearing_meas_err[node_index, neighbor_index] = np.nan

    def download_from_cloud(self, node_index):
        return self.cloud[0][node_index].copy()


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def initialize_robots(simu_counter):
    comm_events = 0
    while comm_events < 2 * np.max(action_extents):
        # update clocks
        t = time_steps[simu_counter]
        for robot in robots:
            robot.update_clock(t)

        # communication step
        if (simu_counter % comm_skip == 0):
            comm_events += 1
            logs.time_comm.append(t)
            world.cloud.append([[] for _ in robots])
            for robot in robots:
                msg = robot.create_msg()
                node_index = index_map[robot.node_id]
                world.upload_to_cloud(msg, node_index)
            for robot in robots:
                node_index = index_map[robot.node_id]
                msgs = world.download_from_cloud(node_index)
                robot.handle_received_msgs(msgs)
                robot.update_state_extent()
                robot.bearing_measurement_step()
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

        # log data
        logs.time.append(t)
        logs.position.append(world.collect_positions())
        logs.estimated_position.append(robots.collect_estimated_positions())
        logs.covariance.append(robots.collect_covariances())
        logs.target_action.append(robots.collect_target_actions())
        logs.collision_action.append(robots.collect_collision_actions())
        logs.rigidity_action.append(robots.collect_rigidity_actions())
        logs.vel_meas_err.append(world.collect_vel_meas_err())
        logs.gps_meas_err.append(world.collect_gps_meas_err())
        logs.bearing_meas_err.append(world.collect_bearing_meas_err())
        logs.adjacency.append(world.adjacency_matrix.ravel())
        logs.action_extents.append(robots.collect_action_extents())
        logs.state_extents.append(robots.collect_state_extents())
        logs.targets.append(targets.data.ravel().copy())

        for robot in robots:
            node_index = index_map[robot.node_id]
            world.robot_dynamics[node_index].step(t, robot.last_control_action)
        world.update_adjacency()
        targets.update(world.positions())

        simu_counter += 1

    print(
        'Initialization completed after {} communication events.'
        .format(comm_events)
    )

    return simu_counter


def run_mission(simu_counter, end_counter):
    bar = progressbar.ProgressBar(maxval=arg.simu_time).start()
    perf_time = []
    while simu_counter < end_counter:
        t_a = time.perf_counter()

        # update clocks
        t = time_steps[simu_counter]
        for robot in robots:
            robot.update_clock(t)

        # communication step
        if (simu_counter % comm_skip == 0):
            logs.time_comm.append(t)
            world.cloud.append([[] for _ in robots])
            for robot in robots:
                msg = robot.create_msg()
                node_index = index_map[robot.node_id]
                world.upload_to_cloud(msg, node_index)
            for robot in robots:
                node_index = index_map[robot.node_id]
                msgs = world.download_from_cloud(node_index)
                robot.handle_received_msgs(msgs)
                robot.bearing_measurement_step()
                robot.rigidity_maintenance_control_action()

        # localization and control step
        # TODO: should be est position
        alloc = targets.allocation(world.positions())

        for robot in robots:
            node_index = index_map[robot.node_id]

            robot.target_collection_control_action(alloc[node_index])
            robot.collision_avoidance_control_action()
            robot.compose_actions()

            gps_meas = world.gps_measurement(node_index)
            if (gps_meas is not None):
                robot.gps_measurement_step(gps_meas)

            vel_meas = world.velocity_measurement(node_index)
            if (vel_meas is not None):
                robot.velocity_measurement_step(vel_meas)

        # log data
        logs.time.append(t)
        logs.position.append(world.collect_positions())
        logs.estimated_position.append(robots.collect_estimated_positions())
        logs.covariance.append(robots.collect_covariances())
        logs.target_action.append(robots.collect_target_actions())
        logs.collision_action.append(robots.collect_collision_actions())
        logs.rigidity_action.append(robots.collect_rigidity_actions())
        logs.vel_meas_err.append(world.collect_vel_meas_err())
        logs.gps_meas_err.append(world.collect_gps_meas_err())
        logs.bearing_meas_err.append(world.collect_bearing_meas_err())
        logs.adjacency.append(world.adjacency_matrix.ravel())
        logs.action_extents.append(robots.collect_action_extents())
        logs.state_extents.append(robots.collect_state_extents())
        logs.targets.append(targets.data.ravel().copy())

        for robot in robots:
            node_index = index_map[robot.node_id]
            world.robot_dynamics[node_index].step(t, robot.last_control_action)
        world.update_adjacency()
        targets.update(world.positions())

        simu_counter += 1

        t_b = time.perf_counter()
        perf_time.append((t_b - t_a)/n)
        bar.update(np.round(t, 3))

    bar.finish()

    rt = arg.simu_time
    st = sum(perf_time)
    prompt = 'ST={:.3f} secs (per robot), RT={:.3f} secs  ==>  RTF={:.3f}'
    print(prompt.format(st, rt, rt / st))

    return simu_counter


# ------------------------------------------------------------------
# Parseo de argumentos
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-s', '--simu_step',
    default=1, type=float, help='simulation step in milli seconds'
)
parser.add_argument(
    '-t', '--simu_time',
    default=10.0, type=float, help='total simulation time in seconds'
)
parser.add_argument(
    '-c', '--comm_skip',
    default=1, type=int, help='communication skip in number of simu_step'
)
parser.add_argument(
    '-q', '--queue',
    default=1, type=int, help='communication cloud queue length'
)
arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------


def index_of(t): return int(t / simu_step)


# Simulation parameters

np.random.seed(0)
simu_time = arg.simu_time
simu_step = arg.simu_step / 1000.0
time_steps = [simu_step * k for k in range(int(simu_time / simu_step))]
n_steps = int(simu_time / simu_step)
comm_skip = arg.comm_skip

print(
    'Simulation Time: begin = {}, end = {}, step = {}'
    .format(0.0, simu_time, simu_step)
)
print(
    'Communication Time: begin = {}, end = {}, step = {}'
    .format(0.0, simu_time, comm_skip * simu_step)
)

# world parameters

n = 20
position = np.array([
    [8.8073523063, 6.3042370627, 0.],
    [17.1029732669, 11.0046986205, 0.],
    [31.186538034, 8.9747394789, 0.],
    [18.7144623157, 35.2741964133, 0.],
    [24.0062670026, 32.5964895071, 0.],
    [35.0546596323, 28.2255372366, 0.],
    [37.480498427, 16.9108985815, 0.],
    [2.5700695761, 10.8970998756, 0.],
    [5.5603965902, 15.6577290526, 0.],
    [4.7723726409, 22.9948840075, 0.],
    [10.2448254248, 22.3585615506, 0.],
    [11.6161346706, 13.955768194, 0.],
    [19.3209996703, 28.0969737243, 0.],
    [11.8278550027, 34.442105737, 0.],
    [32.3836034169, 1.741932494, 0.],
    [27.763696182, 25.6107016014, 0.],
    [1.0308446707, 27.2387302286, 0.],
    [23.4271927906, 10.6842377268, 0.],
    [22.6177090084, 22.1869906253, 0.],
    [16.5353740973, 19.6605780599, 0.]
])
print(position)

comm_range = 15.0
print('Communication range: {}'.format(comm_range))
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

world = World(
    dim=3,
    robot_dynamics=[Integrator(position[i]) for i in range(n)],
    network=adjacency_matrix,
    comm_range=comm_range,
    gps_available=range(n),
    vel_meas_stdev=0.0,
    bearing_meas_stdev=0.0,
    gps_meas_stdev=0.0,
    queue=arg.queue
)

robots = Robots([
    Robot(
        node_id=i,
        position=np.random.normal(position[i],  0.0),
        comm_range=comm_range,
        action_extent=1,
        # state_extent=2
    )
    for i in range(n)
])

index_map = {robots[i].node_id: i for i in range(n)}
# print('Index map: {}'.format(index_map))

targets = Targets(
    n=100,
    dim=3,
    low_lim=(0.0, 0.0, 10.0),
    up_lim=(100.0, 100.0, 50.0),
    coverage=5.0
)

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
logs = Logs(
    time=[],
    time_comm=[],
    position=[],
    estimated_position=[],
    covariance=[],
    target_action=[],
    collision_action=[],
    rigidity_action=[],
    vel_meas_err=[],
    gps_meas_err=[],
    bearing_meas_err=[],
    adjacency=[],
    action_extents=[],
    state_extents=[],
    targets=[]
)

simu_counter = 0
for t_break in [simu_time]:
    position = world.positions()
    adjacency_matrix = world.adjacency_matrix.copy().astype(float)
    geodesics_matrix = core.geodesics(adjacency_matrix)
    action_extents = minimum_rigidity_extents(geodesics_matrix, position)
    print(
        'Action extents: \n' +
        '\n'.join(
            '\t node = {}, extent = {}'.format(i, r)
            for i, r in enumerate(action_extents) if r > 0
        )
    )
    for robot in robots:
        node_index = index_map[robot.node_id]
        robot.action_extent = action_extents[node_index]
        robot.last_control_action = np.zeros(3, dtype=float)

    simu_counter = initialize_robots(simu_counter)
    print(
        'State extents: \n' +
        '\n'.join(
            '\t node = {}, extent = {}'
            .format(robot.node_id, robot.state_extent)
            for robot in robots
        )
    )
    simu_counter = run_mission(simu_counter, end_counter=index_of(t_break))

np.savetxt('data/t.csv', logs.time, delimiter=',')
np.savetxt('data/tc.csv', logs.time_comm, delimiter=',')
np.savetxt('data/position.csv', logs.position, delimiter=',')
np.savetxt('data/est_position.csv', logs.estimated_position, delimiter=',')
np.savetxt('data/covariance.csv', logs.covariance, delimiter=',')
np.savetxt('data/target_action.csv', logs.target_action, delimiter=',')
np.savetxt('data/collision_action.csv', logs.collision_action, delimiter=',')
np.savetxt('data/rigidity_action.csv', logs.rigidity_action, delimiter=',')
np.savetxt('data/vel_meas_err.csv', logs.vel_meas_err, delimiter=',')
np.savetxt('data/gps_meas_err.csv', logs.gps_meas_err, delimiter=',')
np.savetxt('data/bearing_meas_err.csv', logs.bearing_meas_err, delimiter=',')
np.savetxt('data/adjacency.csv', logs.adjacency, delimiter=',')
np.savetxt('data/action_extents.csv', logs.action_extents, delimiter=',')
np.savetxt('data/state_extents.csv', logs.state_extents, delimiter=',')
np.savetxt('data/targets.csv', logs.targets, delimiter=',')
