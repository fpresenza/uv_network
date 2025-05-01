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

import uvnpy.distances.core as distances
import uvnpy.bearings.core as bearings
from uvnpy.network.core import geodesics, as_undirected
from uvnpy.dynamics.linear_models import Integrator
from uvnpy.network.disk_graph import DiskGraph
from uvnpy.network.cone_graph import ConeGraph
from uvnpy.control.core import Targets, CollisionAvoidanceVanishing
from uvnpy.routing.token_passing import TokenPassing
from uvnpy.bearings.localization import FirstOrderKalmanFilter
from uvnpy.bearings.control import RigidityMaintenance


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
    pose, \
    estimated_pose,  \
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
    bearing, \
    action_tokens, \
    state_tokens')


@dataclass
class NeighborData(object):
    node_id: int
    timestamp: float = None
    pose: np.ndarray = None
    covariance: np.ndarray = None
    bearing: np.ndarray = None
    is_isolated_edge: bool = None


def camera_axis(angle):
    axis = np.empty((len(angle), 3))
    axis[:, 0] = np.cos(angle)
    axis[:, 1] = np.sin(angle)
    axis[:, 2] = 0.0
    return axis


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
        self.maintenance = RigidityMaintenance(
            dim=3,
            range_lims=sens_range * np.array([0.8, 1.0]),
            cos_lims=np.cos(np.deg2rad(fov / 2)) * np.array([1.0, 1.8]),
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
        self.control_action = np.zeros(self.dim, dtype=float)
        self.action = {}
        self.loc = FirstOrderKalmanFilter(
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
        action_tokens, state_tokens = self.routing.broadcast(
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
            in_balls=self.in_balls.copy(),
            bearing=bearings,
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
            if self.node_id in msg.bearing:
                self.neighbors[msg.node_id] = NeighborData(
                    node_id=msg.node_id,
                    timestamp=msg.timestamp,
                    pose=msg.state_tokens[msg.node_id].data['pose'],
                    covariance=msg.state_tokens[msg.node_id].data[
                        'covariance'],
                    bearing=msg.bearing[self.node_id],
                    is_isolated_edge=self.in_balls.isdisjoint(msg.in_balls)
                )
                self.routing.update_action(msg.action_tokens.values())
                self.routing.update_state(msg.state_tokens.values())

                self.in_balls = self.self_centered_ball.union(
                    self.routing.action_centers()
                )
            elif msg.node_id in self.neighbors:
                self.neighbors[msg.node_id].timestamp = msg.timestamp
                self.neighbors[msg.node_id].pose = \
                    msg.state_tokens[msg.node_id].data['pose']
                self.neighbors[msg.node_id].covariance = \
                    msg.state_tokens[msg.node_id].data['covariance']
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
                    pose=msg.state_tokens[msg.node_id].data['pose'],
                    covariance=msg.state_tokens[msg.node_id].data['covariance']
                )

    def update_state_extent(self):
        self.state_extent = max(1, self.routing.max_action_extent())

    def set_control_action(self, u):
        self.control_action = u

    def target_collection_control_action(self, target):
        if (target is not None):
            # go to allocated target
            r = self.loc.pose()[:3] - target
            d = np.sqrt(np.square(r).sum())
            tracking_radius = 20.0    # radius
            forget_radius = 30.0     # radius
            v_collect_max = 1.0
            if d < tracking_radius:
                v_collect = v_collect_max
            elif d < forget_radius:
                factor = (forget_radius - d)/(forget_radius - tracking_radius)
                v_collect = v_collect_max * factor
            else:
                v_collect = 0.0
            self.u_target = np.append(- v_collect * r / d, 0.0)
        else:
            self.u_target = np.zeros(self.dim, dtype=float)

    def collision_avoidance_control_action(self):
        # get obstacles (other robots poses)
        # obstacles = self.routing.extract_state(
        #     'pose',
        #     1,
        #     wrapper=lambda x: np.take(x, range(3))
        # )
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
        self.u_rigidity *= 0.1

    def compose_actions(self):
        # compose control actions from different objectives and
        self.control_action = \
            self.u_target + self.u_collision + self.u_rigidity

    def stop_motion(self):
        self.control_action = np.zeros(self.dim, dtype=float)

    def velocity_measurement_step(self, vel_meas):
        self.loc.dynamic_step(self.current_time, vel_meas)

    def bearing_measurement_step(self):
        if len(self.neighbors) > 0:
            z = np.array([
                neighbor.bearing
                for neighbor in self.neighbors.values()
                if neighbor.bearing is not None
            ])
            xj = np.array([
                neighbor.pose
                for neighbor in self.neighbors.values()
                if neighbor.bearing is not None
            ])
            Pj = np.array([
                neighbor.covariance
                for neighbor in self.neighbors.values()
                if neighbor.bearing is not None
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
            vel_meas_stdev,
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

        self.vel_meas_stdev = vel_meas_stdev
        self.bearing_meas_stdev = bearing_meas_stdev
        self.gps_meas_stdev = gps_meas_stdev

        self.vel_meas_err = np.full((self.n, self.dim), np.nan)
        self.gps_meas_err = np.full((self.n, self.dim), np.nan)
        self.bearing_meas_err = np.full((self.n, self.n), np.nan)

        self.cloud = collections.deque(maxlen=queue)

    def positions(self):
        return np.array([
            robot.x[:self.dim - 1] for robot in self.robot_dynamics
        ])

    def angles(self):
        return np.array([
            robot.x[self.dim - 1] for robot in self.robot_dynamics
        ])

    def collect_poses(self):
        return np.hstack([robot.x for robot in self.robot_dynamics])

    def collect_vel_meas_err(self):
        return self.vel_meas_err.copy().ravel()

    def collect_gps_meas_err(self):
        return self.gps_meas_err.copy().ravel()

    def collect_bearing_meas_err(self):
        return self.bearing_meas_err.copy().ravel()

    def update_graphs(self):
        positions = self.positions()
        axes = camera_axis(self.angles())
        self.sens_graph.update_adjacency_matrix(positions, axes)
        self.comm_graph.update_adjacency_matrix(positions)

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
            pose = self.robot_dynamics[node_index].x
            self.gps_meas_err[node_index] = np.random.normal(
                scale=self.gps_meas_stdev, size=self.dim
            )
            return pose + self.gps_meas_err[node_index]
        else:
            self.gps_meas_err[node_index] = np.nan
            return None

    def bearing_measurement(self, node_index):
        bearings = {}
        for neighbor_index in range(self.n):
            if self.sens_graph.is_edge(node_index, neighbor_index):
                noise = np.random.normal(
                        scale=self.bearing_meas_stdev, size=self.dim - 1
                )
                noisy_bearing = transformations.unit_vector(
                    self.robot_dynamics[node_index].x[:3] -
                    self.robot_dynamics[neighbor_index].x[:3] +
                    noise
                )
                self.bearing_meas_err[node_index, neighbor_index] = \
                    np.dot(noise, noise)
                bearings[neighbor_index] = noisy_bearing
            else:
                self.bearing_meas_err[node_index, neighbor_index] = np.nan
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

            vel_meas = robnet.velocity_measurement(node_index)
            if (vel_meas is not None):
                robot.velocity_measurement_step(vel_meas)

        # log data
        logs.time.append(t)
        logs.pose.append(robnet.collect_poses())
        logs.estimated_pose.append(robots.collect_estimated_poses())
        logs.covariance.append(robots.collect_covariances())
        logs.target_action.append(robots.collect_target_actions())
        logs.collision_action.append(robots.collect_collision_actions())
        logs.rigidity_action.append(robots.collect_rigidity_actions())
        logs.vel_meas_err.append(robnet.collect_vel_meas_err())
        logs.gps_meas_err.append(robnet.collect_gps_meas_err())
        logs.bearing_meas_err.append(robnet.collect_bearing_meas_err())
        logs.adjacency.append(robnet.sens_graph.adjacency_matrix().ravel())
        logs.action_extents.append(robots.collect_action_extents())
        logs.state_extents.append(robots.collect_state_extents())
        logs.targets.append(targets.data.ravel().copy())

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
            # robot.compose_actions()

            gps_meas = robnet.gps_measurement(node_index)
            if (gps_meas is not None):
                robot.gps_measurement_step(gps_meas)

            vel_meas = robnet.velocity_measurement(node_index)
            if (vel_meas is not None):
                robot.velocity_measurement_step(vel_meas)

        # log data
        logs.time.append(t)
        logs.pose.append(robnet.collect_poses())
        logs.estimated_pose.append(robots.collect_estimated_poses())
        logs.covariance.append(robots.collect_covariances())
        logs.target_action.append(robots.collect_target_actions())
        logs.collision_action.append(robots.collect_collision_actions())
        logs.rigidity_action.append(robots.collect_rigidity_actions())
        logs.vel_meas_err.append(robnet.collect_vel_meas_err())
        logs.gps_meas_err.append(robnet.collect_gps_meas_err())
        logs.bearing_meas_err.append(robnet.collect_bearing_meas_err())
        logs.adjacency.append(robnet.sens_graph.adjacency_matrix().ravel())
        logs.action_extents.append(robots.collect_action_extents())
        logs.state_extents.append(robots.collect_state_extents())
        logs.targets.append(targets.data.ravel().copy())

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

np.random.seed(3)
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

# robnet parameters
n = 15
# positions = np.empty((n, 3))
# positions[:, 0] = np.random.uniform(0.0, 30.0, n)
# positions[:, 1] = np.random.uniform(0.0, 30.0, n)
# positions[:, 2] = 0.0
# print(positions)
# baricenter = np.mean(positions, axis=0)
# axes = transformations.unit_vector(baricenter - positions, axis=1)
# angles = np.arctan2(axes[:, 1], axes[:, 0]).reshape(-1, 1)
# print(angles)
positions = np.array([
    [16.74078127, 17.81264269, 0.],
    [21.25975608, 0.69126902, 0.],
    [8.70918232, 16.86743894, 0.],
    [15.72910327, 7.29325366, 0.],
    [26.76951052, 12.45178603, 0.],
    [26.88085721, 8.51942828, 0.],
    [3.87041333, 20.71506161, 0.],
    [5.73659268, 13.41440628, 0.],
    [1.56361411, 4.71457492, 0.],
    [13.14526584, 16.30023222, 0.],
    [0.7899329, 23.50186915, 0.],
    [13.19130069, 9.65597745, 0.],
    [19.62000619, 6.64034072, 0.],
    [8.68470161, 11.32315226, 0.],
    [20.28081445, 28.08292935, 0.]
])

angles = np.array([
    [-2.18071183],
    [2.03226461],
    [-0.6434728],
    [1.93774261],
    [3.13080092],
    [2.81615886],
    [-0.68535938],
    [0.04544866],
    [0.61925688],
    [-1.50800027],
    [-0.67965017],
    [1.62060603],
    [2.32684369],
    [0.23441121],
    [-1.9967138]
])


if distances.minimum_distance(positions) > 2.0:
    print('Yay! Robots\' positions are sufficiently separated.')
else:
    raise ValueError('Robots\' are too close.')

sens_range = 15.0
fov = 120.0
print('Camera\'s range: {}'.format(sens_range))
print('Camera\'s fov: {} degrees'.format(fov))
sens_graph = ConeGraph(
    positions,
    camera_axis(angles.ravel()),
    dmax=sens_range,
    cmin=np.cos(np.deg2rad(fov / 2))
)
print(
    'Adjacency list: \n' +
    '\n'.join(
        '\t {}: {}'.format(key, val)
        for key, val in sens_graph.adjacency_dict().items()
    )
)
adjacency_matrix = as_undirected(sens_graph.adjacency_matrix())
if bearings.is_inf_rigid(adjacency_matrix, positions):
    print('Yay! Sensing framework is infinitesimally rigid.')
    poses = np.hstack([positions, angles])
else:
    raise ValueError('Sensing framework should be infinitesimally rigid.')

comm_range = 15.0
print('Communication range: {}'.format(comm_range))
comm_graph = DiskGraph(
    positions,
    dmax=comm_range,
)

robnet = MultiRobotNetwork(
    dim=4,
    robot_dynamics=[Integrator(poses[i]) for i in range(n)],
    sens_graph=sens_graph,
    comm_graph=comm_graph,
    gps_available=range(n),
    vel_meas_stdev=0.0,
    bearing_meas_stdev=0.0,
    gps_meas_stdev=0.0,
    queue=arg.queue
)

robots = Robots([
    Robot(
        node_id=i,
        pose=np.random.normal(poses[i],  0.0),
        sens_range=sens_range,
        comm_range=comm_range,
        fov=fov,
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
    low_lim=(0.0, 0.0, -50.0),
    up_lim=(100.0, 100.0, 0.0),
    coverage=5.0
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
    adjacency_matrix = as_undirected(robnet.sens_graph.adjacency_matrix())
    positions = robnet.positions()
    geodesics_matrix = geodesics(adjacency_matrix.astype(float))
    action_extents = bearings.minimum_rigidity_extents(
        geodesics_matrix, positions
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

np.savetxt('data/t.csv', logs.time, delimiter=',')
np.savetxt('data/tc.csv', logs.time_comm, delimiter=',')
np.savetxt('data/pose.csv', logs.pose, delimiter=',')
np.savetxt('data/est_pose.csv', logs.estimated_pose, delimiter=',')
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
