#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
import progressbar
import numpy as np
from transformations import unit_vector

from uvnpy.toolkit.functions import cosine_activation_derivative
from uvnpy.toolkit.data import write_json_file
from uvnpy.dynamics.lie_groups import EulerIntegratorOrtogonalGroup
from uvnpy.dynamics.quadrotor import VelocityControlledQuadrotor

from uvnpy.toolkit.geometry import rotation_matrix_from_vector_multiple_axes
from uvnpy.graphs.models import ConeGraph
from uvnpy.angles.local_frame.core import (
    angle_indices, is_angle_rigid, angle_rigidity_eigenvalue
)
from uvnpy.angles.local_frame.localization import AngleBasedPoseObserverCascade
from uvnpy.angles.local_frame.control import (
    weighted_angle_rigidity_eigenvalue, AngleRigidityMaintenance
)
from uvnpy.control.targets import MovingTargets


# ------------------------------------------------------------------
# Functions and Classes
# ------------------------------------------------------------------


@dataclass
class Logs(object):
    time: list
    position: list
    attitude: list
    orientation: list
    estimated_position: list
    estimated_orientation: list
    control_u: list
    control_w: list
    rigidity_val: list
    adjacency: list
    target_position: list


def aiming(p, target):
    a = unit_vector(target - p, axis=-1)
    x = np.random.normal(size=3)
    b = unit_vector(np.cross(a, x), axis=-1)
    c = np.cross(a, b)
    return np.dstack([a, b, c]).squeeze()


def dsigma_m_r(x):
    return cosine_activation_derivative(
        np.array([x]), 0.0, 100.0
    ).item()


def dsigma_m_f(x):
    return cosine_activation_derivative(
        np.array([x]), -1.0, 1.0
    ).item()


def projection_matrix(x):
    return np.eye(3) - np.outer(x, x)


def extract_quadrotor_position(quadrotors):
    return np.array([q.position() for q in quadrotors])


def extract_quadrotor_attitude(quadrotors):
    return np.array([q.attitude() for q in quadrotors])


def extract_quadrotor_linear_vel(quadrotors):
    return np.array([q.linear_vel() for q in quadrotors])


def extract_x(integrators):
    return np.array([p.x() for p in integrators])


def extract_u(integrators):
    return np.array([p.u() for p in integrators])


# ------------------------------------------------------------------
# Simulation loop inner functions
# ------------------------------------------------------------------


def simu_step():
    # Update the true poses and the sensing graph.
    p = np.empty((n, 3), dtype=np.float64)
    R = np.empty((n, 3, 3), dtype=np.float64)
    for i in nodes:
        R_int[i].step_left(t, wb[i])
        R[i] = R_int[i].x()
        p_int[i].closed_loop_step(t, R[i].dot(ub[i]), 0.0)
        p[i] = p_int[i].position()

    sensing_graph.update(p, R[:, :, 0])

    """Run formation control and pose estimation without estimate feedback."""

    # --- angle rigidity eigenvalue-vector --- #
    edge_set = sensing_graph.edge_set()
    angle_set = angle_indices(nodes, edge_set).astype(int)

    #    # --- unweighted part --- #
    rigidity_val[0] = angle_rigidity_eigenvalue(angle_set, p, scale_invariant=True)

    #    # --- weighted part --- #
    rigidity_val[:], rigidity_vec = weighted_angle_rigidity_eigenvalue(
        angle_set, p, R, range_lims, hfov_lims
    )
    rigidity_vec = np.squeeze(np.matmul(rigidity_vec[:, np.newaxis, :], R))

    # --- measurements and estimation --- #
    out_neighbors = {i: [] for i in nodes}
    distances = {i: {} for i in nodes}
    bearings, bearing_rates = {i: {} for i in nodes}, {i: {} for i in nodes}
    rotations = {i: {} for i in nodes}
    for i, j in edge_set:
        out_neighbors[i].append(j)
        pij = p[j] - p[i]

        #   # --- estimated values --- #
        distances[i][j] = np.sqrt(np.square(pij).sum())
        rotations[i][j] = R[i].T.dot(R[j])

        #   # --- measurements --- #
        bearings[i][j] = R[i].T.dot(pij / distances[i][j])
        bearing_rates[i][j] = projection_matrix(bearings[i][j]).dot(
            rotations[i][j].dot(ub[j]) - ub[i]
            ) / distances[i][j] + np.cross(bearings[i][j], wb[i])

    da, db = observer.distance_pair
    distance = np.sqrt(np.square(p[da] - p[db]).sum())
    observer.update(t, ub, wb, bearings, bearing_rates, distance, out_neighbors)

    # --- control ---#
    control_u_m = np.zeros((n, 3), dtype=np.float64)
    control_w_m = np.zeros((n, 3), dtype=np.float64)
    control_u_c = np.zeros((n, 3), dtype=np.float64)
    for i in nodes:
        # --- collision avoidance control --- #
        for j in out_neighbors[i]:
            dij = distances[i][j]
            bij = bearings[i][j]
            Rij = rotations[i][j]

            repulsion = 2 * sensing_range * (dij - sensing_range) * bij / dij**3
            control_u_c[i] += repulsion
            control_u_c[j] -= Rij.T.dot(repulsion)

        # --- target tracking control --- #
        k_m_r = 3.0
        k_m_f = 0.5
        if i in nodes:
            rit = targets[target_allocation[i]](t) - p[i]
            dit = np.sqrt(np.square(rit).sum())
            bit = R[i].T.dot(rit / dit)

            Pit = np.eye(3) - np.outer(bit, bit)

            nit = bit[0]
            e1_bit = np.array([0.0, -bit[2], bit[1]])

            control_u_m[i] = k_m_r * dsigma_m_r(dit) * bit \
                - k_m_f * dsigma_m_f(nit) * Pit[0] / dit
            control_w_m[i] = k_m_f * dsigma_m_f(nit) * e1_bit

            # target collision avoidance
            control_u_c[i] += 2 * sensing_range * (dit - sensing_range) * bit / dit**3

    # --- rigidity maintenance control --- #
    control_u_r, control_w_r = arm.update(
        out_neighbors, distances, bearings, rotations, rigidity_val[1], rigidity_vec
    )

    # define relative weights
    k_u_r = 5.0
    k_u_m = 50.0
    k_w_r = 0.5
    k_w_m = 5.0
    k_u_c = 1.0

    # Compose body-frame inputs using only the true states, as before.
    ub[:] = k_u_r * control_u_r + k_u_m * control_u_m + k_u_c * control_u_c
    wb[:] = k_w_r * control_w_r + k_w_m * control_w_m


def log_step():
    """Data log"""
    logs.time.append(t)
    logs.position.append(extract_quadrotor_position(p_int).ravel())
    logs.attitude.append(extract_quadrotor_attitude(p_int).ravel())
    logs.orientation.append(extract_x(R_int).ravel())
    logs.estimated_position.append(observer.position.ravel())
    logs.estimated_orientation.append(observer.orientation.ravel())
    logs.control_u.append(extract_quadrotor_linear_vel(p_int).ravel())
    logs.control_w.append(extract_u(R_int).ravel())
    logs.rigidity_val.append(rigidity_val.copy())
    logs.adjacency.append(sensing_graph.adjacency_matrix().ravel())
    logs.target_position.append(targets.position(t).ravel())


# ------------------------------------------------------------------
# Argument parse
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-s', '--simu_step_size',
    default=1, type=int, help='simulation step in milli seconds'
)
parser.add_argument(
    '-t', '--simu_length',
    default=1, type=int, help='total simulation time in milli seconds'
)
parser.add_argument(
    '-l', '--log_skip',
    default=1, type=int, help='logger skip in number of simu_step_size'
)
arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# --- simulation parameters --- #
if arg.simu_length % arg.simu_step_size != 0:
    print(
        'Simulation length is not a multiple of the step size. \
        Length will be truncated the closest multiple.\
    ')
simu_num_steps = int(arg.simu_length / arg.simu_step_size)

simu_length = arg.simu_length * 1e-3    # in seconds
simu_step_size = arg.simu_step_size * 1e-3    # in seconds
log_skip = arg.log_skip

print(
    'Simulation Time: begin = {} sec, end = {} sec, step = {} sec'
    .format(0.0, simu_length, simu_step_size)
)
print(
    'Logging Time: begin = {} sec, end = {} sec, step = {} sec'
    .format(0.0, simu_length, simu_step_size * log_skip)
)

np.set_printoptions(suppress=True, precision=10)

# --- world parameters --- #
found_IAR = False
seed = 0
while not found_IAR:
    np.random.seed(seed)
    seed += 1

    t = 0.0
    n = 5
    nodes = list(range(n))
    initial_position = np.random.uniform(
        [60.0, 0.0, 0.0],
        [100.0, 40.0, 40.0],
        size=(n, 3)
    )
    initial_orientation = aiming(initial_position, initial_position.mean(axis=0))

    sensing_range = 40.0
    range_lims = (0.8 * sensing_range, sensing_range)
    fov = 120.0
    cos_hfov = np.cos(np.deg2rad(fov / 2))
    hfov_lims = (cos_hfov, 1.4 * cos_hfov)
    sensing_graph = ConeGraph(
        initial_position,
        initial_orientation[:, :, 0],    # x-axes
        dmax=sensing_range,
        cmin=cos_hfov
    )

    initial_edge_set = sensing_graph.edge_set()
    initial_angle_set = angle_indices(nodes, initial_edge_set).astype(int)

    # # check if graph is complete
    # if sensing_graph.adjacency_matrix().sum() == n**2 - n:
    #     found_IAR = True
    #     print('seed = {}'.format(seed))
    # check if graph is angle rigid

    if is_angle_rigid(initial_angle_set, initial_position):
        found_IAR = True
        print('seed = {}'.format(seed))

p_int = [
    VelocityControlledQuadrotor(
        initial_position[i],
        np.eye(3),
        np.zeros(3),
        np.zeros(3),
        mass=1.5,
        inertia=np.diag([29e-3, 29e-3, 55e-3])
    )
    for i in nodes
]
R_int = [
    EulerIntegratorOrtogonalGroup(initial_orientation[i])
    for i in nodes
]

arm = AngleRigidityMaintenance(n, range_lims, hfov_lims)
ub = np.zeros((n, 3), dtype=np.float64)
wb = np.zeros((n, 3), dtype=np.float64)
rigidity_val = np.empty(2, dtype=np.float64)

# Independent initialization noise keeps the original simulation RNG unchanged.
position_error_std = 2.0             # metres per coordinate
orientation_error_std = 0.1         # radians per rotation-vector coordinate
hatp = np.random.normal(initial_position, position_error_std)
small_angle = np.random.normal(scale=orientation_error_std, size=3)
hatR = np.matmul(
    rotation_matrix_from_vector_multiple_axes(small_angle),
    initial_orientation
)
observer = AngleBasedPoseObserverCascade(
    n, hatp, hatR,
    k_alpha=2000.0, k_d=0.02, k_beta=2.0, k_v=2.0,
    distance_pair=(0, 1), time=t
)

# --- define targets --- #
targets = MovingTargets({
    0: lambda t: np.array([
        min(t, 100.0, 200.0 - t),
        min(t, 100.0, 200.0 - t),
        0.0
    ]),
    1: lambda t: np.array([
        min(t, 100.0, 200.0 - t),
        min(t, 100.0),
        min(t, 100.0)
    ]),
    2: lambda t: np.array([
        0.0,
        min(t, 100.0),
        0.0
    ])
})

target_allocation = [0, 1, 2, 1, 2]

write_json_file(
    'simu_data/targets.jsonlog',
    {
        'ids': list(targets.keys()),
        'tracking_nodes': nodes,
        'alloc': target_allocation
    }
)

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
# initialize logs
logs = Logs(
    time=[t],
    position=[extract_quadrotor_position(p_int).ravel()],
    attitude=[extract_quadrotor_attitude(p_int).ravel()],
    orientation=[extract_x(R_int).ravel()],
    estimated_position=[observer.position.ravel()],
    estimated_orientation=[observer.orientation.ravel()],
    control_u=[],
    control_w=[],
    rigidity_val=[],
    adjacency=[sensing_graph.adjacency_matrix().ravel()],
    target_position=[targets.position(0.0).ravel()]
)

# run simulation
simu_counter = 1
bar = progressbar.ProgressBar(maxval=simu_length).start()

while simu_counter < simu_num_steps:
    t = np.round(t + simu_step_size, 3)

    try:
        simu_step()
        if (simu_counter % log_skip == 0):
            log_step()
    except (ValueError, IndexError, ArithmeticError, KeyboardInterrupt) as e:
        print('Simulation interrupted at t={} sec due to <{}>.'.format(t, e))
        break

    simu_counter += 1

    bar.update(t)

bar.finish()

np.savetxt('simu_data/t.csv', logs.time, delimiter=',')
np.savetxt('simu_data/position.csv', logs.position, delimiter=',')
np.savetxt('simu_data/attitude.csv', logs.attitude, delimiter=',')
np.savetxt('simu_data/orientation.csv', logs.orientation, delimiter=',')
np.savetxt(
    'simu_data/estimated_position.csv', logs.estimated_position, delimiter=','
)
np.savetxt(
    'simu_data/estimated_orientation.csv', logs.estimated_orientation, delimiter=','
)
np.savetxt('simu_data/control_u.csv', logs.control_u, delimiter=',')
np.savetxt('simu_data/control_w.csv', logs.control_w, delimiter=',')
np.savetxt('simu_data/rigidity_val.csv', logs.rigidity_val, delimiter=',')
np.savetxt('simu_data/adjacency.csv', logs.adjacency, delimiter=',')
np.savetxt('simu_data/target_position.csv', logs.target_position, delimiter=',')
