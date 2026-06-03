#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
import progressbar
import numpy as np
from scipy.linalg import block_diag

from uvnpy.dynamics.core import EulerIntegrator
from uvnpy.dynamics.lie_groups import EulerIntegratorOrtogonalGroup
from uvnpy.toolkit.geometry import (
    rotation_matrix_from_vector,
    cross_product_matrix_multiple_axes as S,
    vector_angle_from_matrix
)

# ------------------------------------------------------------------
# Functions, Classes and Configurations
# ------------------------------------------------------------------
np.set_printoptions(suppress=True, precision=10)


@dataclass
class Logs(object):
    time: list
    position: list
    orientation: list
    estimated_position: list
    estimated_orientation: list
    covariance: list
    control_u: list
    control_w: list


def random_rotation_matrix(max_angle=2 * np.pi):
    v = np.random.normal(size=3)
    v /= np.sqrt(v.dot(v))
    a = np.random.uniform(0.0, max_angle)
    return rotation_matrix_from_vector(a * v)


def projection_matrix(x):
    return np.eye(3) - np.outer(x, x)


def extract_x(integrators):
    return np.array([p.x() for p in integrators])


def extract_u(integrators):
    return np.array([p.u() for p in integrators])


# ------------------------------------------------------------------
# Simulation loop inner functions
# ------------------------------------------------------------------


def simu_step():
    """Pose estimation algorithm"""
    # --- data ---#
    p = extract_x(p_int)
    dotp = extract_u(p_int)
    R = extract_x(R_int)
    dotR = extract_u(R_int)
    dt = simu_step_size

    # --- measurements --- #
    # velocity
    meas_lin_vel = np.random.normal(dotp, 0.25)
    meas_ang_vel = np.random.normal(dotR[a], 0.1)

    # distance
    noise_square_dist = np.random.normal(scale=1.0, size=n-1)
    meas_square_dist = 0.5 * np.square(
        p[neighbors] - p[a]
    ).sum(axis=1) + noise_square_dist

    # orientation
    noise_orient = np.random.normal(scale=0.5, size=3)
    meas_orient = R[a].dot(rotation_matrix_from_vector(noise_orient))

    # --- advance estimation --- #
    # prediction step
    hat_ai = (meas_lin_vel[neighbors] - meas_lin_vel[a]).dot(hatQ)

    F = np.kron(np.eye(n), np.eye(3) - dt * S(meas_ang_vel))
    F[:-3, -3:] = S(dt * hat_ai).reshape(3*n - 3, 3)

    V = np.diag([0.25**2] * 3*n + [0.1**2] * 3)
    G = np.zeros((3*n, 3*n + 3))
    G[:-3, :3] = np.kron(np.ones((n-1, 1)), hatQ.T)
    G[:-3, 3:-3] = np.kron(np.eye(n-1), -hatQ.T)
    G[:-3, -3:] = S(- hatq).reshape(3*n - 3, 3)
    G[-3:, -3:] = - np.eye(3)

    hatq[:] = hatq + dt * (hat_ai - np.cross(meas_ang_vel, hatq))
    hatQ[:] = hatQ.dot(rotation_matrix_from_vector(dt * meas_ang_vel))
    cov_matrix[:] = F.dot(cov_matrix).dot(F.T) + G.dot(V).dot(G.T) * dt**2

    # correction step
    hat_square_dist = 0.5 * np.square(hatq).sum(axis=1)
    hat_vector_angle = vector_angle_from_matrix(hatQ.T.dot(meas_orient))
    residual = np.hstack([meas_square_dist - hat_square_dist, hat_vector_angle])

    H = np.zeros((n + 2, 3*n))
    H[:-3, :-3] = block_diag(*hatq)
    H[-3:, -3:] = np.eye(3)

    N = np.diag([1.0**2] * (n - 1) + [0.5**2] * 3)
    K = cov_matrix.dot(H.T).dot(np.linalg.inv(H.dot(cov_matrix).dot(H.T) + N))

    correction = K.dot(residual)
    hatq[:] = hatq + correction[:-3].reshape(n - 1, 3)
    hatQ[:] = hatQ.dot(rotation_matrix_from_vector(correction[-3:]))

    X = np.eye(3*n) - K.dot(H)
    cov_matrix[:] = X.dot(cov_matrix).dot(X.T) + K.dot(N).dot(K.T)

    # advance pose
    ub = np.zeros((n, 3), dtype=np.float64)    # body-frame
    wb = np.zeros((n, 3), dtype=np.float64)    # body-frame

    for i in nodes:
        # --- Control inputs --- #
        ub[i] = control_u[i](t)
        wb[i] = control_w[i](t)

        # --- advance pose --- #
        p_int[i].step(t, R[i].dot(ub[i]))
        R_int[i].step_left(t, wb[i])


def log_step():
    """Data log"""
    logs.time.append(t)
    logs.position.append(extract_x(p_int).ravel())
    logs.orientation.append(extract_x(R_int).ravel())
    logs.estimated_position.append(hatq.copy().ravel())
    logs.estimated_orientation.append(hatQ.copy().ravel())
    logs.covariance.append(cov_matrix.copy().ravel())
    logs.control_u.append(extract_u(p_int).ravel())
    logs.control_w.append(extract_u(R_int).ravel())


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
    print('\
        Simulation length is not a multiple of the step size. \
        Length will be truncated the closest multiple.\
    ')
simu_num_steps = int(arg.simu_length / arg.simu_step_size)

simu_length = arg.simu_length * 1e-3    # in seconds
simu_step_size = arg.simu_step_size * 1e-3    # in seconds
log_skip = arg.log_skip

np.random.seed(2)

print(
    'Simulation Time: begin = {} sec, end = {} sec, step = {} sec'
    .format(0.0, simu_length, simu_step_size)
)
print(
    'Logging Time: begin = {} sec, end = {} sec, step = {} sec'
    .format(0.0, simu_length, simu_step_size * log_skip)
)

# --- world parameters --- #
t = 0.0
n = 5
nodes = np.arange(n)
p = np.random.uniform(0.0, 30.0, (n, 3))

R = np.array([random_rotation_matrix() for _ in nodes])
edge_set = np.array([
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
])
a = 0
neighbors = np.setdiff1d(nodes, a)

p_int = [
    EulerIntegrator(p[i])
    for i in nodes
]

R_int = [
    EulerIntegratorOrtogonalGroup(R[i])
    for i in nodes
]

# refer initial position to body frame a
q = (p[neighbors] - p[a]).dot(R[a])

hatq = np.random.normal(q, 2.0)

# refer initial orientation to body frame a
delta_theta = np.random.normal(scale=0.5, size=3)
hatQ = R[a].dot(rotation_matrix_from_vector(delta_theta))

# cov_matrixiance matrix
cov_matrix = np.eye(3*n)

# define velocities
control_u = {
    0: lambda t: np.array([0.0, 0.0, 0.0]),
    1: lambda t: np.array([np.cos(0.5*t), np.sin(0.5*t), 0.5]),
    2: lambda t: np.array([np.cos(0.5*t), np.sin(0.5*t), 0.5]),
    3: lambda t: np.array([np.cos(0.5*t), np.sin(0.5*t), 0.5]),
    4: lambda t: np.array([np.cos(0.5*t), np.sin(0.5*t), 0.5])
}

control_w = {
    0: lambda t: np.array([0.5, 0.0, 0.0]),
    1: lambda t: np.array([0.0, 0.0, 0.0]),
    2: lambda t: np.array([0.0, 0.0, 0.0]),
    3: lambda t: np.array([0.0, 0.0, 0.0]),
    4: lambda t: np.array([0.0, 0.0, 0.0])
}

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
# initialize logs
logs = Logs(
    time=[t],
    position=[extract_x(p_int).ravel()],
    orientation=[extract_x(R_int).ravel()],
    estimated_position=[hatq.copy().ravel()],
    estimated_orientation=[hatQ.copy().ravel()],
    covariance=[cov_matrix.copy().ravel()],
    control_u=[extract_u(p_int).ravel()],
    control_w=[extract_u(p_int).ravel()],
)

# run simulation
simu_counter = 1
bar = progressbar.ProgressBar(maxval=simu_length).start()

while simu_counter < simu_num_steps:
    t = np.round(t + simu_step_size, 3)

    simu_step()
    if (simu_counter % log_skip == 0):
        log_step()

    simu_counter += 1

    bar.update(t)

bar.finish()

np.savetxt('simu_data/t.csv', logs.time, delimiter=',')
np.savetxt('simu_data/position.csv', logs.position, delimiter=',')
np.savetxt('simu_data/orientation.csv', logs.orientation, delimiter=',')
np.savetxt(
    'simu_data/estimated_position.csv', logs.estimated_position, delimiter=','
)
np.savetxt(
    'simu_data/estimated_orientation.csv', logs.estimated_orientation, delimiter=','
)
np.savetxt('simu_data/covariance.csv', logs.covariance, delimiter=',')
np.savetxt('simu_data/control_u.csv', logs.control_u, delimiter=',')
np.savetxt('simu_data/control_w.csv', logs.control_w, delimiter=',')
