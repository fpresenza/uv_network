#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
import progressbar
import numpy as np
from transformations import unit_vector

from uvnpy.graphs.core import adjacency_matrix_from_edges
from uvnpy.dynamics.core import EulerIntegrator
from uvnpy.dynamics.lie_groups import EulerIntegratorOrtogonalGroup
from uvnpy.toolkit.geometry import rotation_matrix_from_vector
from uvnpy.angles.local_frame.core import is_angle_rigid, angle_indices

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
    gradient_q: list
    gradient_Q: list
    control_u: list
    control_w: list
    correction_u: list
    correction_w: list
    adjacency: list


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


def complete_angle_set(out_neighbors):
    i, j = np.triu_indices(out_neighbors.size, k=1)
    return np.column_stack([out_neighbors[i], out_neighbors[j]])


# ------------------------------------------------------------------
# Simulation loop inner functions
# ------------------------------------------------------------------


def simu_step():
    """Pose estimation algorithm"""
    # --- data ---#
    p = extract_x(p_int)
    dotp = extract_u(p_int)
    hatq = extract_x(hatq_int)
    R = extract_x(R_int)
    hatQ = extract_x(hatQ_int)

    grad_q[:] = 0.0
    grad_Q[:] = 0.0

    ub = np.zeros((n, 3), dtype=np.float64)    # body-frame
    wb = np.zeros((n, 3), dtype=np.float64)    # body-frame

    # --- similarity correction --- #
    # measurements
    qb = R[a].T.dot(p[b] - p[a])
    qc = R[a].T.dot(p[c] - p[a])

    # correction
    k_s = 10.0
    grad_q[a] += k_s * hatq[a]
    grad_q[b] += k_s * (hatq[b] - qb)
    grad_q[c] += k_s * (hatq[c] - qc)

    k_a = 500.0
    for i in nodes:
        # --- Control inputs --- #
        ub[i] = control_u[i](t)
        wb[i] = control_w[i](t)

        # --- advance pose --- #
        p_int[i].step(t, R[i].dot(ub[i]))
        R_int[i].step_left(t, wb[i])

        # --- angle correction --- #
        out_neighbors = edge_set[:, 1][edge_set[:, 0] == i]

        # estimated values
        hat_distances = {
            j: np.sqrt(np.square(hatq[j] - hatq[i]).sum()) for j in out_neighbors
        }
        hat_bearings = {j: unit_vector(hatq[j] - hatq[i]) for j in out_neighbors}

        # measurements
        distances = {
            j: np.sqrt(np.square(p[j] - p[i]).sum()) for j in out_neighbors
        }
        bearings = {
            j: R[i].T.dot(unit_vector(p[j] - p[i])) for j in out_neighbors
        }
        dot_bearings = {
            j: projection_matrix(bearings[j]).dot(
                R[i].T.dot(dotp[j] - dotp[i])
            ) / distances[j] - np.cross(wb[i], bearings[j])
            for j in out_neighbors
        }

        # position gradient
        for j, k in complete_angle_set(out_neighbors):

            dij = hat_distances[j]
            bij = hat_bearings[j]
            Pij = projection_matrix(bij)

            dik = hat_distances[k]
            bik = hat_bearings[k]
            Pik = projection_matrix(bik)

            # measured angles
            aijk = bearings[j].dot(bearings[k])

            eijk = bij.dot(bik) - aijk
            qijk = Pij.dot(bik) / dij
            qikj = Pik.dot(bij) / dik

            grad_q[i] -= k_a * eijk * (qijk + qikj)
            grad_q[j] += k_a * eijk * qijk
            grad_q[k] += k_a * eijk * qikj

        # orientation gradient
        if i in leaders:
            for j in out_neighbors:
                grad_Q[i] += np.cross(bearings[j], hatQ[i].T.dot(hatq[j] - hatq[i]))
                proj_ij = projection_matrix(hatQ[i].dot(bearings[j]))
                hat_dotQ_i_bij = hatQ[i].dot(
                    np.cross(wb[i] - hatQ[i].T.dot(wb[a]), bearings[j])
                )
                hat_Qi_dotbij = hatQ[i].dot(dot_bearings[j])
                hat_dotq_i = hatQ[i].dot(ub[i]) - ub[a] - np.cross(wb[a], hatq[i])
                aux_f[j]['num'] += hat_distances[j] * (
                    hat_dotQ_i_bij + hat_Qi_dotbij
                ) + proj_ij.dot(hat_dotq_i)
                aux_f[j]['den'] += proj_ij

    k_o = 2.0
    for i in nodes:
        # --- advance estimation --- #
        if i in leaders:
            hatq_int[i].step(
                t, hatQ[i].dot(ub[i]) - ub[a] - np.cross(wb[a], hatq[i]) - grad_q[i]
            )
            hatQ_int[i].step_left(
                t, wb[i] - hatQ[i].T.dot(wb[a]) + k_o * grad_Q[i]
            )
        else:
            hat_dotq_i = np.linalg.inv(aux_f[i]['den']).dot(aux_f[i]['num'])
            aux_f[i]['num'][:] = 0.0
            aux_f[i]['den'][:] = 0.0
            hat_Qui = hat_dotq_i + ub[a] + np.cross(wb[a], hatq[i])
            grad_Q[i] = np.cross(ub[i], hatQ[i].T.dot(hat_Qui))
            hatq_int[i].step(
                t, hat_dotq_i - grad_q[i]
            )
            hatQ_int[i].step_left(
                t, wb[i] - hatQ[i].T.dot(wb[a]) + k_o * grad_Q[i]
            )


def log_step():
    """Data log"""
    logs.time.append(t)
    logs.position.append(extract_x(p_int).ravel())
    logs.orientation.append(extract_x(R_int).ravel())
    logs.estimated_position.append(extract_x(hatq_int).ravel())
    logs.estimated_orientation.append(extract_x(hatQ_int).ravel())
    logs.gradient_q.append(grad_q.copy().ravel())
    logs.gradient_Q.append(grad_Q.copy().ravel())
    logs.control_u.append(extract_u(p_int).ravel())
    logs.control_w.append(extract_u(R_int).ravel())
    logs.correction_u.append(extract_u(hatq_int).ravel())
    logs.correction_w.append(extract_u(hatQ_int).ravel())


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
n = 4
nodes = np.arange(n)
p = np.random.uniform(0.0, 30.0, (n, 3))
R = np.array([random_rotation_matrix() for _ in nodes])
edge_set = np.array([
    [0, 1],
    [0, 2],
    [1, 0],
    [1, 2],
    [0, 3],
    [1, 3]
])
angle_set = angle_indices(nodes, edge_set).astype(int)
a, b, c = 0, 1, 2
leaders = np.array([0, 1])
followers = np.setdiff1d(nodes, leaders)

if not is_angle_rigid(angle_set, p):
    raise ValueError('The framework is not IAR.')

p_int = [
    EulerIntegrator(p[i])
    for i in nodes
]

R_int = [
    EulerIntegratorOrtogonalGroup(R[i])
    for i in nodes
]

# refer initial position to body frame a
q = (p - p[a]).dot(R[a])

hat_q = np.random.normal(q, 2.0)
hat_q[a] = 0.0
hatq_int = [
    EulerIntegrator(hat_q[i])
    for i in nodes
]

# refer initial orientation to body frame a
Q = np.matmul(R[a].T, R)

hatQ = [np.eye(3) for i in nodes]
hatQ[a] = np.eye(3)
hatQ_int = [
    EulerIntegratorOrtogonalGroup(hatQ[i])
    for i in nodes
]

# define velocities

control_u = {
    0: lambda t: np.array([0.0, 0.0, 1.0]),
    1: lambda t: np.array([np.cos(0.25*t), np.sin(0.25*t), 0.0]),
    2: lambda t: np.array([0.0, np.cos(1.0*t), np.sin(1.0*t)]),
    3: lambda t: np.array([np.cos(2.0*t), np.sin(2.0*t), 0.5])
}

control_w = {
    0: lambda t: np.array([0.5, 0.0, 0.0]),
    1: lambda t: np.array([0.0, 1.0, 0.0]),
    2: lambda t: np.array([0.0, 0.0, 1.0]),
    3: lambda t: np.array([0.0, 0.0, 0.0])
}

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
# initialize logs
grad_q = np.zeros((n, 3), dtype=np.float64)
grad_Q = np.zeros((n, 3), dtype=np.float64)
aux_f = {
    i: {
        'num': np.zeros(3, dtype=np.float64),
        'den': np.zeros((3, 3), dtype=np.float64)
    }
    for i in nodes
}

logs = Logs(
    time=[t],
    position=[extract_x(p_int).ravel()],
    orientation=[extract_x(R_int).ravel()],
    estimated_position=[extract_x(hatq_int).ravel()],
    estimated_orientation=[extract_x(hatQ_int).ravel()],
    gradient_q=[grad_q.copy().ravel()],
    gradient_Q=[grad_Q.copy().ravel()],
    control_u=[extract_u(p_int).ravel()],
    control_w=[extract_u(p_int).ravel()],
    correction_u=[extract_u(hatq_int).ravel()],
    correction_w=[extract_u(hatQ_int).ravel()],
    adjacency=[adjacency_matrix_from_edges(n, edge_set).ravel()]
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
np.savetxt('simu_data/gradient_q.csv', logs.gradient_q, delimiter=',')
np.savetxt('simu_data/gradient_Q.csv', logs.gradient_Q, delimiter=',')
np.savetxt('simu_data/control_u.csv', logs.control_u, delimiter=',')
np.savetxt('simu_data/control_w.csv', logs.control_w, delimiter=',')
np.savetxt('simu_data/correction_u.csv', logs.correction_u, delimiter=',')
np.savetxt('simu_data/correction_w.csv', logs.correction_w, delimiter=',')
np.savetxt('simu_data/adjacency.csv', logs.adjacency, delimiter=',')
