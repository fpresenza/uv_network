#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import collections
import time
import progressbar
import numpy as np
import copy

from uvnpy.graphs import core
from uvnpy.graphs.models import DiskGraph
from uvnpy.graphs.subframeworks import superframework_extents
from uvnpy.distances.localization import DistanceBasedKalmanFilter
from uvnpy.network.token_passing import TokenPassing
from uvnpy.dynamics.core import EulerIntegrator
from uvnpy.distances.control import RigidityMaintenance
from uvnpy.distances.core import (
    is_inf_distance_rigid, minimum_distance_rigidity_extents
)
from uvnpy.control.core import CollisionAvoidanceVanishing
from uvnpy.control.targets import Targets, TargetTracking


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
    est_position,  \
    target_action, \
    collision_action, \
    rigidity_action, \
    adjacency, \
    action_extents, \
    state_extents, \
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
    position, \
    range')


class Neighborhood(dict):
    def update(
            self,
            node_id,
            timestamp,
            position,
            range_measurement
            ):
        self[node_id] = NeigborhoodData(
            node_id=node_id,
            timestamp=timestamp,
            position=position,
            range=range_measurement,
        )


class Robot(object):
    def __init__(
            self,
            node_id,
            position,
            comm_range,
            action_extent=None,
            state_extent=None,
            t=0.0
            ):
        self.node_id = node_id
        self.dim = len(position)
        self.action_extent = action_extent
        self.state_extent = state_extent
        self.current_time = t
        self.tracking = TargetTracking(
            tracking_radius=20.0,
            forget_radius=100.0,
            v_max=1.25
        )
        self.collision = CollisionAvoidanceVanishing(
            power=2.0,
            dmin=1.0,
            dmax=comm_range
        )
        self.maintenance = RigidityMaintenance(
            dmax=0.85 * comm_range,
            steepness=2.0,
            threshold=1e-4,
            eigenvalues='all',
            functional='log'
        )
        self.u_target = np.zeros(self.dim, dtype=float)
        self.u_collision = np.zeros(self.dim, dtype=float)
        self.u_rigidity = np.zeros(self.dim, dtype=float)
        self.last_control_action = np.zeros(self.dim, dtype=float)
        self.action = {}
        self.loc = DistanceBasedKalmanFilter(
            position,
            position_cov=1.0 * np.eye(self.dim),
            vel_meas_cov=0.01 * np.eye(self.dim),
            range_meas_cov=1.0,
            gps_meas_cov=1.0 * np.eye(self.dim)
        )
        self.neighborhood = Neighborhood()
        self.routing = TokenPassing(self.node_id)

    def update_clock(self, t):
        self.current_time = t

    def create_msg(self):
        action_tokens, state_tokens = self.routing.broadcast(
            timestamp=self.current_time,
            action=copy.deepcopy(self.action),
            state={'position': self.loc.position()},
            action_extent=self.action_extent,
            state_extent=self.state_extent
        )
        msg = InterRobotMsg(
            node_id=self.node_id,
            timestamp=self.current_time,
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
                range_measurement=range_measurement
            )
            self.routing.update_action(msg.action_tokens.values())
            self.routing.update_state(msg.state_tokens.values())

    def update_state_extent(self):
        self.state_extent = max(1, self.routing.max_action_extent())

    def target_tracking_control_action(self, target):
        if (target is not None):
            # go to allocated target
            self.u_target = self.tracking.update(
                self.loc.position(), target
            )
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
            self.u_collision *= 0.25
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

        # rigidity control gain
        self.u_rigidity *= 0.375

    def compose_actions(self):
        self.last_control_action = \
            self.u_target + self.u_collision + self.u_rigidity

    def control_action_step(self):
        self.loc.dynamic_step(self.current_time, self.last_control_action)

    def range_measurement_step(self):
        for data in self.neighborhood.values():
            self.loc.range_step(
                data.range, data.position, np.zeros((2, 2), dtype=float)
            )

    def gps_measurement_step(self, gps_meas):
        self.loc.gps_step(gps_meas)


class Robots(list):
    def collect_est_positions(self):
        return np.hstack([robot.loc.position() for robot in self])

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
            graph,
            gps_available,
            ctrl_action_stdev,
            range_meas_stdev,
            gps_meas_stdev
            ):
        """Clase para simular una red de robots"""
        self.dim = dim
        self.robot_dynamics = robot_dynamics
        self.n = len(robot_dynamics)
        self.graph = graph
        self.gps_available = gps_available

        self.ctrl_action_stdev = ctrl_action_stdev
        self.range_meas_stdev = range_meas_stdev
        self.gps_meas_stdev = gps_meas_stdev

        self.cloud = [[] for _ in range(self.n)]

    def positions(self, subset=None):
        if subset is None:
            return np.array([robot.x() for robot in self.robot_dynamics])
        else:
            return np.array([self.robot_dynamics[i].x() for i in subset])

    def collect_positions(self):
        return np.hstack([robot.x() for robot in self.robot_dynamics])

    def update_graph(self):
        self.graph.update(self.positions())

    def apply_control_action(self, t, node_index, control_action):
        ctrl_action_err = np.random.normal(
            scale=self.ctrl_action_stdev, size=self.dim
        )
        self.robot_dynamics[node_index].step(
            t, control_action + ctrl_action_err
        )

    def gps_measurement(self, node_index):
        if node_index in self.gps_available:
            position = self.robot_dynamics[node_index].x()
            gps_meas_err = np.random.normal(
                scale=self.gps_meas_stdev, size=self.dim
            )
            return position + gps_meas_err
        else:
            return None

    def upload_to_cloud(self, msg, node_index):
        for neighbor_index in self.graph.out_neighbors(node_index):
            dist = np.sqrt(np.square(
                self.robot_dynamics[node_index].x() -
                self.robot_dynamics[neighbor_index].x()
            ).sum())
            range_meas_err = np.random.normal(
                    scale=self.range_meas_stdev, size=1
            )
            self.cloud[neighbor_index].append((msg, dist + range_meas_err))

    def download_from_cloud(self, node_index):
        msgs = self.cloud[node_index].copy()
        self.cloud[node_index].clear()
        return msgs


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------

def run_mission(simu_counter, end_counter):
    bar = progressbar.ProgressBar(maxval=arg.simu_time).start()
    perf_time = []
    while simu_counter < end_counter:
        t = time_steps[simu_counter]
        t_a = time.perf_counter()

        # -- World update -- #
        for i, robot in enumerate(robots):
            world.apply_control_action(t, i, robot.last_control_action)
        world.update_graph()
        targets.update(world.positions())

        # -- Robot logic -- #
        # update clocks
        for robot in robots:
            robot.update_clock(t)

        # control step
        if (simu_counter > 0) and (simu_counter % ctrl_skip == 0):
            alloc = targets.allocation(world.positions())
            for i, robot in enumerate(robots):
                robot.control_action_step()
                robot.target_tracking_control_action(alloc[i])
                robot.collision_avoidance_control_action()
                robot.rigidity_maintenance_control_action()
                robot.compose_actions()
                msg = robot.create_msg()
                world.upload_to_cloud(msg, i)
            for i, robot in enumerate(robots):
                msgs = world.download_from_cloud(i)
                robot.handle_received_msgs(msgs)
                # robot.range_measurement_step()

        # # gps step
        # if (simu_counter > 0) and (simu_counter % gps_skip == 0):
        #     for i, robot in enumerate(robots):
        #         gps_meas = world.gps_measurement(i)
        #         if (gps_meas is not None):
        #             robot.gps_measurement_step(gps_meas)

        # -- Data log -- #
        if (simu_counter % log_skip == 0):
            logs.time.append(t)
            logs.position.append(world.collect_positions())
            logs.adjacency.append(world.graph.adjacency_matrix().ravel())
            logs.targets.append(targets.data.ravel().copy())
            logs.est_position.append(robots.collect_est_positions())
            logs.target_action.append(robots.collect_target_actions())
            logs.collision_action.append(robots.collect_collision_actions())
            logs.rigidity_action.append(robots.collect_rigidity_actions())
            logs.action_extents.append(robots.collect_action_extents())
            logs.state_extents.append(robots.collect_state_extents())

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
    default=1.0, type=float, help='total simulation time in seconds'
)
parser.add_argument(
    '-l', '--log_skip',
    default=1, type=int, help='logger skip in number of simu_step'
)
parser.add_argument(
    '-c', '--ctrl_skip',
    default=1, type=int, help='control skip in number of simu_step'
)
parser.add_argument(
    '-g', '--gps_skip',
    default=1, type=int, help='gps skip in number of simu_step'
)

arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
# Simulation parameters
simu_time = arg.simu_time
simu_step = arg.simu_step / 1000.0
n_steps = int(simu_time / simu_step)
time_steps = [simu_step * k for k in range(n_steps)]
log_skip = arg.log_skip
ctrl_skip = arg.ctrl_skip
gps_skip = arg.gps_skip

print(
    'Simulation Time: begin = {}, end = {}, step = {} sec'
    .format(0.0, simu_time, simu_step)
)
print(
    'Control step = {} sec'
    .format(ctrl_skip * simu_step)
)
print(
    'GPS step = {} sec'
    .format(gps_skip * simu_step)
)

# world parameters

n = 10
position = np.array([
    [9.085, 6.11],
    [17.441, 11.093],
    [30.955, 9.21],
    [18.693, 35.477],
    [24.375, 32.719],
    [37.217, 16.962],
    [11.93, 14.068],
    [27.428, 23.927],
    [23.436, 11.032],
    [22.484, 21.999]
])

print(position)

comm_range = 15.0
print('Communication range: {}'.format(comm_range))
graph = DiskGraph(realization=position, dmax=comm_range)
adjacency_matrix = graph.adjacency_matrix(float)
print(
    'Adjacency list: \n' +
    '\n'.join(
        '\t {}: {}'.format(key, val)
        for key, val in enumerate(core.adjacency_list(adjacency_matrix))
    )
)
edge_set = graph.edge_set(directed=False)
if not is_inf_distance_rigid(edge_set, position):
    raise ValueError('Framework should be infinitesimally rigid.')

world = World(
    dim=2,
    robot_dynamics=[EulerIntegrator(position[i]) for i in range(n)],
    graph=graph,
    gps_available=[6, 8],
    ctrl_action_stdev=0.0,
    range_meas_stdev=0.0,
    gps_meas_stdev=0.0
)

adjacency_matrix = graph.adjacency_matrix(float)
geodesics_matrix = core.geodesics(adjacency_matrix)
action_extents = minimum_distance_rigidity_extents(geodesics_matrix, position)
state_extents = superframework_extents(geodesics_matrix, action_extents)
print(
    'Action extents: \n' +
    '\n'.join(
        '\t node = {}, extent = {}'.format(i, r)
        for i, r in enumerate(action_extents) if r > 0
    )
)
print(
    'State extents: \n' +
    '\n'.join(
        '\t node = {}, extent = {}'.format(i, r)
        for i, r in enumerate(state_extents) if r > 0
    )
)

robots = Robots([
    Robot(
        node_id=i,
        position=np.random.normal(position[i],  0.0),
        comm_range=comm_range,
        action_extent=action_extents[i],
        state_extent=state_extents[i]
    )
    for i in range(n)
])

targets = Targets(
    n=30,
    dim=2,
    low_lim=(0.0, 0.0),
    up_lim=(100.0, 100.0),
    coverage=5.0
)
targets.data[:, :2] = np.array([
    [69.1877114, 31.5515631],
    [68.65009277, 83.46256719],
    [1.82882773, 75.01443149],
    [98.88610889, 74.81656544],
    [28.04439921, 78.92793285],
    [10.32260066, 44.78935262],
    [90.85955031, 29.36141484],
    [28.77753386, 13.00285721],
    [1.93669579, 67.88355329],
    [21.1628116, 26.55466594],
    [49.15731593,  5.33625451],
    [57.41176055, 14.67285749],
    [58.93055369, 69.975836],
    [10.23344288, 41.40559878],
    [69.44001577, 41.41792695],
    [4.99534589, 53.58964059],
    [66.37946452, 51.48891121],
    [94.4594756, 58.65550405],
    [90.34019153, 13.74747041],
    [13.92763473, 80.73912887],
    [39.7676837, 16.53541971],
    [92.75085804, 34.77658597],
    [75.08121031, 72.59979854],
    [88.33060912, 62.36722071],
    [75.0942434, 34.8898342],
    [26.99278918, 89.58862182],
    [42.80911899, 96.48400471],
    [66.34414978, 62.16957202],
    [11.4745973, 94.94892587],
    [44.99121335, 57.83896144],
])

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
logs = Logs(
    time=[],
    position=[],
    est_position=[],
    target_action=[],
    collision_action=[],
    rigidity_action=[],
    adjacency=[],
    action_extents=[],
    state_extents=[],
    targets=[]
)

simu_counter = 0
simu_counter = run_mission(simu_counter, end_counter=n_steps)

np.savetxt('simu_data/t.csv', logs.time, delimiter=',')
np.savetxt('simu_data/position.csv', logs.position, delimiter=',')
np.savetxt('simu_data/est_position.csv', logs.est_position, delimiter=',')
np.savetxt('simu_data/target_action.csv', logs.target_action, delimiter=',')
np.savetxt(
    'simu_data/collision_action.csv', logs.collision_action, delimiter=','
)
np.savetxt(
    'simu_data/rigidity_action.csv', logs.rigidity_action, delimiter=','
)
np.savetxt('simu_data/adjacency.csv', logs.adjacency, delimiter=',')
np.savetxt('simu_data/action_extents.csv', logs.action_extents, delimiter=',')
np.savetxt('simu_data/state_extents.csv', logs.state_extents, delimiter=',')
np.savetxt('simu_data/targets.csv', logs.targets, delimiter=',')
