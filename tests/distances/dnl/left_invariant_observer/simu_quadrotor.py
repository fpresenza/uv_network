#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
import progressbar
import numpy as np
from scipy.linalg import block_diag
from transformations import unit_vector

from uvnpy.dynamics.quadrotor import Quadrotor
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


def extract_p(integrators):
    return np.array([p.position() for p in integrators])


def extract_R(integrators):
    return np.array([p.attitude() for p in integrators])


def extract_dotp(integrators):
    return np.array([p.linear_vel() for p in integrators])


def extract_dotR(integrators):
    return np.array([p.angular_vel() for p in integrators])


def extract_u(integrators):
    return np.array([p.u() for p in integrators])


def desired_attitude_from_yaw(b3d, yaw):
    b1c = np.array([np.cos(yaw), np.sin(yaw), 0.0])

    b2d = unit_vector(np.cross(b3d, b1c))
    b1d = np.cross(b2d, b3d)

    return np.column_stack((b1d, b2d, b3d))


def velocity_controller_quadrotor(
    quad,
    v_des,
    yaw_des,
    g=9.81,
    kv=1.5,
    kR=4.0,
    kOmega=1.5
):
    """
    Quadrotor velocity controller for the model:

        dot v = g e3 - (f/m) R e3
        dot R = R hat(Omega)
        J dot Omega + Omega x J Omega = tau

    Inputs:
        v      : current world-frame velocity, shape (3,)
        R      : current attitude, body-to-world rotation, shape (3,3)
        Omega  : body-frame angular velocity, shape (3,)
        v_des  : desired world-frame velocity, shape (3,)

    Returns:
        f      : scalar thrust
        tau    : body-frame torque, shape (3,)
        R_des  : desired attitude
    """
    R = quad.attitude()
    v = quad.linear_vel()
    Omega = quad.angular_vel()
    mass = quad.mass
    J = quad.inertia

    e3 = np.array([0.0, 0.0, 1.0])

    # Outer-loop desired acceleration.
    a_des = kv * (v_des - v)

    # For dot v = g e3 - (f/m) R e3,
    # choose f R e3 = m(g e3 - a_des).
    F_des = mass * (g * e3 + a_des)

    f = np.sqrt(np.square(F_des).sum())

    b3_des = F_des / f
    R_des = desired_attitude_from_yaw(b3_des, yaw_des)

    # Attitude error.
    e_R_mat = 0.5 * (R_des.T.dot(R) - R.T.dot(R_des))
    e_R = np.array([e_R_mat[2, 1], e_R_mat[0, 2], e_R_mat[1, 0]])

    # Simple attitude-rate reference: Omega_des = 0.
    tau = -kR * e_R - kOmega * Omega + np.cross(Omega, J.dot(Omega))

    return f, tau

# ------------------------------------------------------------------
# Simulation loop inner functions
# ------------------------------------------------------------------


def simu_step():
    """Pose estimation algorithm"""
    # --- data ---#
    p = extract_p(quad)
    dotp = extract_dotp(quad)
    R = extract_R(quad)
    dotR = extract_dotR(quad)
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
    for i in nodes:
        # --- Control inputs --- #
        force, torque = velocity_controller_quadrotor(
            quad[i],
            v_des=control_u[i](t),
            yaw_des=0.0
        )

        # --- advance pose --- #
        quad[i].step(t, force, torque)


def log_step():
    """Data log"""
    logs.time.append(t)
    logs.position.append(extract_p(quad).ravel())
    logs.orientation.append(extract_R(quad).ravel())
    logs.estimated_position.append(hatq.copy().ravel())
    logs.estimated_orientation.append(hatQ.copy().ravel())
    logs.covariance.append(cov_matrix.copy().ravel())
    logs.control_u.append(extract_dotp(quad).ravel())
    logs.control_w.append(extract_dotR(quad).ravel())


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

np.random.seed(3)

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

R = np.array([np.eye(3) for _ in nodes])
edge_set = np.array([
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
])
a = 0
neighbors = np.setdiff1d(nodes, a)

quad = [
    Quadrotor(
        p[i],
        R[i],
        np.zeros(3),
        np.zeros(3),
        mass=1.5,
        inertia=np.diag([29e-3, 29e-3, 55e-3])
    )
    for i in nodes
]

# refer initial position to body frame a
q = (p[neighbors] - p[a]).dot(quad[a].attitude())

hatq = np.random.normal(q, 2.0)

# refer initial orientation to body frame a
delta_theta = np.random.normal(scale=0.5, size=3)
hatQ = quad[a].attitude().dot(rotation_matrix_from_vector(delta_theta))

# cov_matrixiance matrix
cov_matrix = np.eye(3*n)

# define commanded velocities
control_u = {
    0: lambda t: np.array([1.0, 0.0, 0.0]),
    1: lambda t: np.array([0.0, 1.0, 0.0]),
    2: lambda t: np.array([0.0, 0.0, 1.0]),
    3: lambda t: np.array([np.cos(0.5*t), np.sin(0.5*t), 0.0]),
    4: lambda t: np.array([0.0, 0.0, 0.0])
}
# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
# initialize logs
logs = Logs(
    time=[t],
    position=[extract_p(quad).ravel()],
    orientation=[extract_R(quad).ravel()],
    estimated_position=[hatq.copy().ravel()],
    estimated_orientation=[hatQ.copy().ravel()],
    covariance=[cov_matrix.copy().ravel()],
    control_u=[extract_dotp(quad).ravel()],
    control_w=[extract_dotR(quad).ravel()],
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
