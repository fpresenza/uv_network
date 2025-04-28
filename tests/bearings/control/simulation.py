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
from uvnpy.network.cone_graph import ConeGraph
from uvnpy.bearings.control import RigidityMaintenance
from uvnpy.control.core import Targets, CollisionAvoidanceVanishing
from uvnpy.bearings.core import is_inf_rigid, minimum_rigidity_extents


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
    action_tokens, \
    state_tokens')

NeigborhoodData = collections.namedtuple(
    'NeigborhoodData',
    'node_id, \
    timestamp, \
    pose, \
    covariance, \
    bearing, \
    is_isolated_edge')


class Neighborhood(dict):
    def update(
            self,
            node_id,
            timestamp,
            pose,
            covariance,
            bearing_measurement,
            is_isolated_edge
            ):
        self[node_id] = NeigborhoodData(
            node_id=node_id,
            timestamp=timestamp,
            pose=pose,
            covariance=covariance,
            bearing=bearing_measurement,
            is_isolated_edge=is_isolated_edge
        )


class Robot(object):
    def __init__(
            self,
            node_id,
            pose,
            comm_range,
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
        self.control_action = np.zeros(self.dim, dtype=float)
        self.action = {}
        self.loc = FirstOrderKalmanFilter(
            pose,
            pose_cov=0.0 * np.eye(self.dim),
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
                'pose': self.loc.pose(),
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
                pose=msg.state_tokens[msg.node_id].data['pose'],
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
        self.control_action = u

    def target_collection_control_action(self, target):
        if (target is not None):
            # go to allocated target
            r = self.loc.pose()[:3] - target
            d = np.sqrt(np.square(r).sum())
            tracking_radius = 20.0    # radius
            forget_radius = 30.0     # radius
            v_collect_max = 2.0
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
        obstacles = self.routing.extract_state(
            'pose',
            1,
            wrapper=lambda x: np.take(x, range(3))
        )
        if len(obstacles) > 0:
            obstacles_position = list(obstacles.values())
            u_collision = self.collision.update(
                self.loc.pose()[:3], obstacles_position
            )
            # collision control gain
            self.u_collision = 0.75 * np.append(u_collision, 0.0)
        else:
            self.u_collision = np.zeros(self.dim, dtype=float)

    def rigidity_maintenance_control_action(self):
        # get actions for ball subframework
        pose = self.routing.extract_state('pose', self.action_extent)
        n_sub = len(pose)
        if n_sub > 0:
            p = np.empty((n_sub + 1, self.dim), dtype=float)
            p[0] = self.loc.pose()
            p[1:] = list(pose.values())
            # get rigidity maintenance control action
            u_sub = self.maintenance.update(p)
        else:
            u_sub = np.zeros((1, self.dim), dtype=float)

        # pack control action for other robots within ball
        self.action = {
            i: ui
            for i, ui in zip(pose.keys(), u_sub[1:])
        }

        # compose all control actions from containing balls
        cmd = self.routing.extract_action()
        self.u_rigidity = u_sub[0] + sum(cmd.values())

        # add action for isolated edges
        for neighbor in self.neighborhood.values():
            if neighbor.is_isolated_edge:
                p = np.vstack([self.loc.pose(), neighbor.pose])
                self.u_rigidity += self.maintenance.update(p)[0]

        # rigidity control gain
        self.u_rigidity *= 0.125

    def compose_actions(self):
        # compose control actions from different objectives and
        self.control_action = \
            self.u_target + self.u_collision + self.u_rigidity

    def stop_motion(self):
        self.control_action = np.zeros(self.dim, dtype=float)

    def velocity_measurement_step(self, vel_meas):
        self.loc.dynamic_step(self.current_time, vel_meas)

    def bearing_measurement_step(self):
        if len(self.neighborhood) > 0:
            neighbors_data = self.neighborhood.values()
            z = np.array([
                neighbor.bearing for neighbor in neighbors_data
            ])
            xj = np.array([
                neighbor.pose for neighbor in neighbors_data
            ])
            Pj = np.array([
                neighbor.covariance for neighbor in neighbors_data
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
            graph,
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
        self.graph = graph
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

    def update_graph(self):
        positions = self.positions()
        angles = self.angles()
        axes = np.empty((n, 3))
        axes[:, 0] = np.cos(angles)
        axes[:, 1] = np.sin(angles)
        axes[:, 2] = 0.0
        self.graph.update_adjacency_matrix(positions, axes)

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

    def upload_to_cloud(self, msg, node_index):
        for neighbor_index in range(self.n):
            if self.graph.is_edge(node_index, neighbor_index):
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
            robnet.cloud.append([[] for _ in robots])
            for robot in robots:
                msg = robot.create_msg()
                node_index = index_map[robot.node_id]
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
        logs.adjacency.append(robnet.graph.adjacency_matrix().ravel())
        logs.action_extents.append(robots.collect_action_extents())
        logs.state_extents.append(robots.collect_state_extents())
        logs.targets.append(targets.data.ravel().copy())

        for robot in robots:
            node_index = index_map[robot.node_id]
            robnet.robot_dynamics[node_index].step(t, robot.control_action)
        robnet.update_graph()
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
                msg = robot.create_msg()
                node_index = index_map[robot.node_id]
                robnet.upload_to_cloud(msg, node_index)
            for robot in robots:
                node_index = index_map[robot.node_id]
                msgs = robnet.download_from_cloud(node_index)
                robot.handle_received_msgs(msgs)
                robot.bearing_measurement_step()
                # robot.rigidity_maintenance_control_action()

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
        logs.adjacency.append(robnet.graph.adjacency_matrix().ravel())
        logs.action_extents.append(robots.collect_action_extents())
        logs.state_extents.append(robots.collect_state_extents())
        logs.targets.append(targets.data.ravel().copy())

        for robot in robots:
            node_index = index_map[robot.node_id]
            robnet.robot_dynamics[node_index].step(t, robot.control_action)
        robnet.update_graph()
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
n = 20
positions = np.empty((n, 3))
positions[:, 0] = np.random.uniform(0.0, 40.0, n)
positions[:, 1] = np.random.uniform(0.0, 40.0, n)
positions[:, 2] = 0.0
print(positions)
baricenter = np.mean(positions, axis=0)
axes = transformations.unit_vector(baricenter - positions, axis=1)
angles = np.arctan2(axes[:, 1], axes[:, 0]).reshape(-1, 1)
print(angles)

comm_range = 15.0
print('Communication range: {}'.format(comm_range))
graph = ConeGraph(dmax=comm_range, fov=0.0)
directed_adjacency_matrix = graph.update_adjacency_matrix(positions, axes)
print(
    'Adjacency list: \n' +
    '\n'.join(
        '\t {}: {}'.format(key, val)
        for key, val in core.adjacency_dict(directed_adjacency_matrix).items()
    )
)
adjacency_matrix = core.as_undirected(directed_adjacency_matrix)
if not is_inf_rigid(adjacency_matrix, positions):
    raise ValueError('Framework should be infinitesimally rigid.')
else:
    print('Yay! Framework is infinitesimally rigid.')
    poses = np.hstack([positions, angles])

robnet = MultiRobotNetwork(
    dim=4,
    robot_dynamics=[Integrator(poses[i]) for i in range(n)],
    graph=graph,
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
    adjacency_matrix = core.as_undirected(robnet.graph.adjacency_matrix())
    positions = robnet.positions()
    geodesics_matrix = core.geodesics(adjacency_matrix.astype(float))
    action_extents = minimum_rigidity_extents(geodesics_matrix, positions)
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
np.savetxt('data/est_position.csv', logs.estimated_pose, delimiter=',')
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
