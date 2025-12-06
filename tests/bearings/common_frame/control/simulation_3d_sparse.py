#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import collections
from dataclasses import dataclass
import time
import progressbar
import numpy as np
import copy
import transformations

from uvnpy.distances.core import minimum_distance
from uvnpy.bearings.common_frame.core import (
    is_bearing_rigid, minimum_bearing_rigidity_extents
)
from uvnpy.bearings.common_frame.localization import BearingBasedGradientFilter
from uvnpy.bearings.common_frame.control import RigidityMaintenance
from uvnpy.graphs.core import geodesics, as_undirected
from uvnpy.graphs.models import DiskGraph, ConeGraph
from uvnpy.dynamics.core import EulerIntegrator
from uvnpy.control.core import CollisionAvoidanceVanishing
from uvnpy.control.targets import Targets, TargetTracking
from uvnpy.network.token_passing import TokenPassing


# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
np.set_printoptions(
    suppress=True,
    precision=6
)


@dataclass
class Logs(object):
    time: list
    time_comm: list
    pose: list
    estimated_pose: list
    covariance: list
    target_action: list
    collision_action: list
    rigidity_action: list
    sens_adj: list
    comm_adj: list
    action_extents: list
    state_extents: list
    targets: list


@dataclass
class InterRobotMsg(object):
    node_id: int
    timestamp: float = None
    pose: np.ndarray = None
    covariance: np.ndarray = None
    bearings: np.ndarray = None
    in_balls: set = None
    action_tokens: dict = None
    state_tokens: dict = None


@dataclass
class NeighborData(object):
    node_id: int
    timestamp: float = None
    pose: np.ndarray = None
    covariance: np.ndarray = None
    bearing: np.ndarray = None
    is_isolated_edge: bool = None


def camera_axis(angle):
    axis = np.empty(angle.shape + (3,))
    axis[..., 0] = np.cos(angle)
    axis[..., 1] = np.sin(angle)
    axis[..., 2] = 0.0
    return axis


def camera_angle(axis):
    return np.arctan2(axis[..., 1], axis[..., 0])


class Robot(object):
    def __init__(
            self,
            node_id,
            pose,
            sens_range,
            comm_range,
            fov,
            action_extent=0,
            state_extent=1,
            t=0.0
            ):
        self.node_id = node_id
        self.dim = len(pose)
        self.action_extent = action_extent
        self.state_extent = state_extent
        self.current_time = t
        self.self_centered_ball = {node_id} if (action_extent > 0) else set()
        self.in_balls = self.self_centered_ball
        self.tracking = TargetTracking(
            tracking_radius=20.0,
            forget_radius=30.0,
            v_max=1.5
        )
        self.collision = CollisionAvoidanceVanishing(
            power=2.0,
            dmin=1.0,
            dmax=comm_range
        )
        self.maintenance = RigidityMaintenance(
            dim=3,
            range_lims=sens_range * np.array([0.8, 1.0]),
            cos_lims=np.cos(np.deg2rad(fov / 2)) * np.array([1.0, 1.4]),
            threshold=1e-4,
            eigenvalues='all',
            functional='log'
        )
        self.u_target = np.zeros(self.dim, dtype=float)
        self.u_collision = np.zeros(self.dim, dtype=float)
        self.u_rigidity = np.zeros(self.dim, dtype=float)
        self.control_action = np.zeros(self.dim, dtype=float)
        self.action = {}
        self.loc = BearingBasedGradientFilter(
            pose,
            pose_cov=0.0 * np.eye(self.dim),
            vel_meas_cov=0.0 * np.eye(self.dim),
            bearing_meas_cov=0.0 * np.eye(self.dim),
            gps_meas_cov=0.0 * np.eye(self.dim)
        )
        self.neighbors = {}
        self.routing = TokenPassing(self.node_id)

    def update_clock(self, t):
        self.current_time = t

    def create_msg(self, bearings):
        action_tokens, state_tokens = self.routing.set_tokens(
            timestamp=self.current_time,
            action=copy.deepcopy(self.action),
            state={
                'pose': self.loc.pose(),
                'covariance': None
            },
            action_extent=self.action_extent,
            state_extent=self.state_extent
        )
        msg = InterRobotMsg(
            node_id=self.node_id,
            timestamp=self.current_time,
            bearings=bearings,
            pose=self.loc.pose(),
            in_balls=self.in_balls.copy(),
            action_tokens=action_tokens,
            state_tokens=state_tokens
        )
        self.neighbors.clear()
        for node_id, bearing in bearings.items():
            self.neighbors[node_id] = NeighborData(
                node_id=node_id,
                timestamp=self.current_time,
                bearing=bearing
            )
        return msg

    def handle_received_msgs(self, msgs):
        # self.neighbors.clear()
        for msg in msgs:
            if self.node_id in msg.bearings:
                self.neighbors[msg.node_id] = NeighborData(
                    node_id=msg.node_id,
                    timestamp=msg.timestamp,
                    pose=msg.pose,
                    bearing=-msg.bearings[self.node_id],
                    is_isolated_edge=self.in_balls.isdisjoint(msg.in_balls)
                )
                self.routing.update_action(msg.action_tokens.values())
                self.routing.update_state(msg.state_tokens.values())

                self.in_balls = self.self_centered_ball.union(
                    self.routing.action_centers()
                )
            elif msg.node_id in self.neighbors:
                self.neighbors[msg.node_id].timestamp = msg.timestamp
                self.neighbors[msg.node_id].pose = msg.pose
                self.neighbors[msg.node_id].is_isolated_edge = \
                    self.in_balls.isdisjoint(msg.in_balls)

                self.routing.update_action(msg.action_tokens.values())
                self.routing.update_state(msg.state_tokens.values())

                self.in_balls = self.self_centered_ball.union(
                    self.routing.action_centers()
                )
            else:
                self.neighbors[msg.node_id] = NeighborData(
                    node_id=msg.node_id,
                    timestamp=msg.timestamp,
                    pose=msg.pose,
                )

    def update_state_extent(self):
        self.state_extent = max(1, self.routing.max_action_extent())

    def set_control_action(self, u):
        self.control_action = u

    def target_collection_control_action(self, target):
        if (target is not None):
            # go to allocated target
            v_tracking = self.tracking.update(
                self.loc.pose()[:3], target
            )
            self.u_target = np.append(v_tracking, 0.0)
        else:
            self.u_target = np.zeros(self.dim, dtype=float)

    def collision_avoidance_control_action(self):
        # get obstacles (other robots poses)
        obstacles = np.array([
            neighbor.pose[:3] for neighbor in self.neighbors.values()
        ])
        if len(obstacles) > 0:
            # obstacles_position = list(obstacles.values())
            u_collision = self.collision.update(
                self.loc.pose()[:3], obstacles
            ) * 0.5
            # collision control gain
            self.u_collision = np.append(u_collision, 0.0)
        else:
            self.u_collision = np.zeros(self.dim, dtype=float)

    def rigidity_maintenance_control_action(self):
        # get actions for ball subframework
        subframework = self.routing.extract_state('pose', self.action_extent)
        n_sub = len(subframework)
        if n_sub > 0:
            x = np.empty((n_sub + 1, self.dim), dtype=float)
            x[0] = self.loc.pose()
            x[1:] = list(subframework.values())
            # get rigidity maintenance control action
            u_sub = self.maintenance.update(x)
        else:
            u_sub = np.zeros((1, self.dim), dtype=float)

        # pack control action for other robots within ball
        self.action = {
            i: ui
            for i, ui in zip(subframework.keys(), u_sub[1:])
        }

        # compose all control actions from containing balls
        cmd = self.routing.extract_action()
        self.u_rigidity = u_sub[0] + sum(cmd.values())

        # add action for isolated edges
        for neighbor in self.neighbors.values():
            if neighbor.is_isolated_edge:
                x = np.vstack([self.loc.pose(), neighbor.pose])
                self.u_rigidity += self.maintenance.update(x)[0]

        # rigidity control gain
        self.u_rigidity *= (0.1, 0.1, 0.1, 0.01)

    def compose_actions(self):
        # compose control actions from different objectives and
        self.control_action = \
            (self.u_target + self.u_collision + self.u_rigidity) * 0.5

    def stop_motion(self):
        self.control_action = np.zeros(self.dim, dtype=float)

    def bearing_measurement_step(self):
        sensing_neighbors = [
            neighbor
            for neighbor in self.neighbors.values()
            if neighbor.bearing is not None
        ]
        if len(sensing_neighbors) > 0:
            z = np.array([
                neighbor.bearing for neighbor in sensing_neighbors
            ])
            xj = np.array([
                neighbor.pose for neighbor in sensing_neighbors
            ])
            Pj = np.array([
                neighbor.covariance for neighbor in sensing_neighbors
            ])
            self.loc.bearing_step(z, xj, Pj)

    def gps_measurement_step(self, gps_meas):
        self.loc.gps_step(gps_meas)


class Robots(list):
    def collect_estimated_poses(self):
        return np.hstack([robot.loc.pose() for robot in self])

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


class MultiRobotNetwork(object):
    def __init__(
            self,
            dim,
            robot_dynamics,
            sens_graph,
            comm_graph,
            gps_available,
            bearing_meas_stdev,
            gps_meas_stdev,
            queue=1
            ):
        """
            This class simulates a multi-robot network.
        """
        self.dim = dim
        self.robot_dynamics = robot_dynamics
        self.n = len(robot_dynamics)
        self.sens_graph = sens_graph
        self.comm_graph = comm_graph
        self.gps_available = gps_available

        self.bearing_meas_stdev = bearing_meas_stdev
        self.gps_meas_stdev = gps_meas_stdev

        self.cloud = collections.deque(maxlen=queue)

    def positions(self):
        return np.array([
            robot.x()[:self.dim - 1] for robot in self.robot_dynamics
        ])

    def angles(self):
        return np.array([
            robot.x()[self.dim - 1] for robot in self.robot_dynamics
        ])

    def collect_poses(self):
        return np.hstack([robot.x() for robot in self.robot_dynamics])

    def update_graphs(self):
        positions = self.positions()
        axes = camera_axis(self.angles())
        self.sens_graph.update(positions, axes)
        self.comm_graph.update(positions)

    def gps_measurement(self, node_index):
        if node_index in self.gps_available:
            pose = self.robot_dynamics[node_index].x()
            return pose
        else:
            return None

    def bearing_measurement(self, node_index):
        bearings = {}
        for neighbor_index in range(self.n):
            if self.sens_graph.is_edge(node_index, neighbor_index):
                noise = np.random.normal(
                        scale=self.bearing_meas_stdev, size=self.dim - 1
                )
                noisy_bearing = transformations.unit_vector(
                    self.robot_dynamics[neighbor_index].x()[:3] -
                    self.robot_dynamics[node_index].x()[:3] +
                    noise
                )
                bearings[neighbor_index] = noisy_bearing
        return bearings

    def upload_to_cloud(self, msg, sender_index):
        for receiver_index in self.comm_graph.out_neighbors(sender_index):
            self.cloud[-1][receiver_index].append(msg)

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
            robnet.cloud.append([[] for _ in robots])
            for robot in robots:
                node_index = index_map[robot.node_id]
                bearings = robnet.bearing_measurement(node_index)
                msg = robot.create_msg(bearings)
                robnet.upload_to_cloud(msg, node_index)
            for robot in robots:
                node_index = index_map[robot.node_id]
                msgs = robnet.download_from_cloud(node_index)
                robot.handle_received_msgs(msgs)
                robot.update_state_extent()
                robot.bearing_measurement_step()
            print('Communication event {} finished'.format(comm_events))

        # localization step
        for robot in robots:
            node_index = index_map[robot.node_id]

            gps_meas = robnet.gps_measurement(node_index)
            if (gps_meas is not None):
                robot.gps_measurement_step(gps_meas)

        # log data
        logs.time.append(t)
        logs.pose.append(robnet.collect_poses())
        logs.estimated_pose.append(robots.collect_estimated_poses())
        logs.covariance.append(robots.collect_covariances())
        logs.target_action.append(robots.collect_target_actions())
        logs.collision_action.append(robots.collect_collision_actions())
        logs.rigidity_action.append(robots.collect_rigidity_actions())
        logs.sens_adj.append(robnet.sens_graph.adjacency_matrix().ravel())
        logs.comm_adj.append(robnet.comm_graph.adjacency_matrix().ravel())
        logs.action_extents.append(robots.collect_action_extents())
        logs.state_extents.append(robots.collect_state_extents())
        logs.targets.append(targets.active.copy())

        for robot in robots:
            node_index = index_map[robot.node_id]
            robnet.robot_dynamics[node_index].step(t, robot.control_action)
        robnet.update_graphs()
        targets.update(robnet.positions())

        simu_counter += 1

    print(
        'Initialization completed after {} communication events.'
        .format(comm_events)
    )

    return simu_counter


def run_mission(simu_counter, end_counter):
    bar = progressbar.ProgressBar(maxval=arg.simu_t1).start()
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
            robnet.cloud.append([[] for _ in robots])
            for robot in robots:
                node_index = index_map[robot.node_id]
                bearings = robnet.bearing_measurement(node_index)
                msg = robot.create_msg(bearings)
                robnet.upload_to_cloud(msg, node_index)
            for robot in robots:
                node_index = index_map[robot.node_id]
                msgs = robnet.download_from_cloud(node_index)
                robot.handle_received_msgs(msgs)
                robot.bearing_measurement_step()
                #    why not apply control action at each time step
                #    (with latest positions)
                #    downside: computational load
                robot.rigidity_maintenance_control_action()

        # localization and control step
        # TODO: should be est poses
        alloc = targets.allocation(robnet.positions())

        for robot in robots:
            node_index = index_map[robot.node_id]

            robot.target_collection_control_action(alloc[node_index])
            robot.collision_avoidance_control_action()
            robot.compose_actions()

            gps_meas = robnet.gps_measurement(node_index)
            if (gps_meas is not None):
                robot.gps_measurement_step(gps_meas)

        # log data
        logs.time.append(t)
        logs.pose.append(robnet.collect_poses())
        logs.estimated_pose.append(robots.collect_estimated_poses())
        logs.covariance.append(robots.collect_covariances())
        logs.target_action.append(robots.collect_target_actions())
        logs.collision_action.append(robots.collect_collision_actions())
        logs.rigidity_action.append(robots.collect_rigidity_actions())
        logs.sens_adj.append(robnet.sens_graph.adjacency_matrix().ravel())
        logs.comm_adj.append(robnet.comm_graph.adjacency_matrix().ravel())
        logs.action_extents.append(robots.collect_action_extents())
        logs.state_extents.append(robots.collect_state_extents())
        logs.targets.append(targets.active.copy())

        for robot in robots:
            node_index = index_map[robot.node_id]
            robnet.robot_dynamics[node_index].step(t, robot.control_action)
        robnet.update_graphs()
        targets.update(robnet.positions())

        simu_counter += 1

        t_b = time.perf_counter()
        perf_time.append((t_b - t_a)/n)
        bar.update(np.round(t, 3))

    bar.finish()

    rt = arg.simu_t1
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
    '-t0', '--simu_t0',
    default=0.0, type=float, help='total simulation time in seconds'
)
parser.add_argument(
    '-t1', '--simu_t1',
    default=1.0, type=float, help='total simulation time in seconds'
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


def index_of(t): return int((t - simu_t0) / simu_step)


# Simulation parameters
np.random.seed(6)
simu_t0 = arg.simu_t0
simu_t1 = arg.simu_t1
simu_step = arg.simu_step / 1000.0
time_steps = [
    simu_step * k
    for k in range(int((simu_t1 - simu_t0) / simu_step))
]
comm_skip = arg.comm_skip

print(
    'Simulation Time: begin = {}, end = {}, step = {}'
    .format(simu_t0, simu_t1, simu_step)
)
print(
    'Communication Time: begin = {}, end = {}, step = {}'
    .format(simu_t0, simu_t1, comm_skip * simu_step)
)

# robnet parameters
n = 30
sens_range = 20.0
fov = 120.0
comm_range = 20.0
init_pos_lim_x = (0.0, 50.0)
init_pos_lim_y = (25.0, 75.0)
print('Communication range: {}'.format(comm_range))
print('Cameras\' range: {}'.format(sens_range))
print('Cameras\' fov: {} degrees'.format(fov))
print('Robots\' initial positions: {} '.format(fov))

while True:
    print('Looking for a valid initial framework...')
    positions = np.empty((n, 3))
    positions[:, 0] = np.random.uniform(*init_pos_lim_x, n)
    positions[:, 1] = np.random.uniform(*init_pos_lim_y, n)
    positions[:, 2] = 0.0

    if minimum_distance(positions) > 2.0:
        baricenter = np.mean(positions, axis=0)
        axes = transformations.unit_vector(baricenter - positions, axis=1)
        sens_graph = ConeGraph(
            positions,
            axes,
            dmax=sens_range,
            cmin=np.cos(np.deg2rad(fov / 2))
        )
        if is_bearing_rigid(sens_graph.edge_set(as_oriented=True), positions):
            angles = camera_angle(axes).reshape(-1, 1)
            pose = np.hstack([positions, angles])
            print(
                'Adjacency list: \n' +
                '\n'.join(
                    '\t {}: {}'.format(key, val)
                    for key, val in enumerate(sens_graph.adjacency_list())
                )
            )
            break


comm_graph = DiskGraph(
    positions,
    dmax=comm_range,
)

robnet = MultiRobotNetwork(
    dim=4,
    robot_dynamics=[EulerIntegrator(pose[i]) for i in range(n)],
    sens_graph=sens_graph,
    comm_graph=comm_graph,
    gps_available=range(n),
    bearing_meas_stdev=0.0,
    gps_meas_stdev=0.0,
    queue=arg.queue
)

robots = Robots([
    Robot(
        node_id=i,
        pose=np.random.normal(pose[i],  0.0),
        sens_range=sens_range,
        comm_range=comm_range,
        fov=fov,
        action_extent=1,
        # state_extent=2
    )
    for i in range(n)
])

index_map = {robots[i].node_id: i for i in range(n)}
np.random.seed(100)
targets = Targets(
    n=100,
    dim=3,
    low_lim=(0.0, 0.0, 10.0),
    up_lim=(100.0, 100.0, 50.0),
    collect_radius=5.0
)

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
logs = Logs(
    time=[],
    time_comm=[],
    pose=[],
    estimated_pose=[],
    covariance=[],
    target_action=[],
    collision_action=[],
    rigidity_action=[],
    sens_adj=[],
    comm_adj=[],
    action_extents=[],
    state_extents=[],
    targets=[]
)

simu_counter = 0
for t_break in [simu_t1]:
    adjacency_matrix = as_undirected(robnet.sens_graph.adjacency_matrix())
    edge_set = sens_graph.edge_set(as_oriented=True)
    positions = robnet.positions()
    geodesics_matrix = geodesics(adjacency_matrix.astype(float))
    action_extents = minimum_bearing_rigidity_extents(
        edge_set, geodesics_matrix, positions
    )
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
        robot.stop_motion()

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

np.savetxt('simu_data/t.csv', logs.time, delimiter=',')
np.savetxt('simu_data/tc.csv', logs.time_comm, delimiter=',')
np.savetxt('simu_data/pose.csv', logs.pose, delimiter=',')
np.savetxt('simu_data/est_pose.csv', logs.estimated_pose, delimiter=',')
np.savetxt('simu_data/covariance.csv', logs.covariance, delimiter=',')
np.savetxt('simu_data/target_action.csv', logs.target_action, delimiter=',')
np.savetxt(
    'simu_data/collision_action.csv', logs.collision_action, delimiter=','
)
np.savetxt(
    'simu_data/rigidity_action.csv', logs.rigidity_action, delimiter=','
)
np.savetxt('simu_data/sens_adj.csv', logs.sens_adj, delimiter=',')
np.savetxt('simu_data/comm_adj.csv', logs.comm_adj, delimiter=',')
np.savetxt('simu_data/action_extents.csv', logs.action_extents, delimiter=',')
np.savetxt('simu_data/state_extents.csv', logs.state_extents, delimiter=',')
np.savetxt('simu_data/targets.csv', logs.targets, delimiter=',')
np.savetxt('simu_data/targets_positions.csv', targets.positions, delimiter=',')
