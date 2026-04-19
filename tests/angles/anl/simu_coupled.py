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
    hatq = extract_x(hatq_int)
    R = extract_x(R_int)
    hatQ = extract_x(hatQ_int)

    gradient = np.zeros((n, 3), dtype=np.float64)

    ub = np.zeros((n, 3), dtype=np.float64)    # body-frame
    wb = np.zeros((n, 3), dtype=np.float64)    # body-frame

    # --- Euclidean correction --- #
    # measurements
    dab = np.sqrt(np.square(p[b] - p[a]).sum())
    dac = np.sqrt(np.square(p[c] - p[a]).sum())
    bab = R[a].T.dot((p[b] - p[a]) / dab)
    bac = R[a].T.dot((p[c] - p[a]) / dac)

    # correction
    k_s = 10.0
    hat_bab = unit_vector(hatq[b] - hatq[a])
    hat_bac = unit_vector(hatq[c] - hatq[a])
    hat_Pab = np.eye(3) - np.outer(hat_bab, hat_bab)
    hat_Pac = np.eye(3) - np.outer(hat_bac, hat_bac)

    gradient[a] += k_s * hatq[a]
    gradient[a] += k_s * (hat_Pab.dot(bab) / dab + hat_Pac.dot(bac) / dac)
    gradient[b] -= k_s * hat_Pab.dot(bab) / dab
    gradient[c] -= k_s * hat_Pac.dot(bac) / dac

    k_a = 500.0
    for i in nodes:
        # --- angle correction --- #
        out_neighbors = edge_set[:, 1][edge_set[:, 0] == i]

        # estimated values
        hat_distances = {
            j: np.sqrt(np.square(hatq[j] - hatq[i]).sum()) for j in out_neighbors
        }
        hat_bearings = {j: unit_vector(hatq[j] - hatq[i]) for j in out_neighbors}

        # measurements
        bearings = {
            j: R[i].T.dot(unit_vector(p[j] - p[i])) for j in out_neighbors
        }

        # correction
        for j, k in complete_angle_set(out_neighbors):

            dij = hat_distances[j]
            bij = hat_bearings[j]
            Pij = np.eye(3) - np.outer(bij, bij)

            dik = hat_distances[k]
            bik = hat_bearings[k]
            Pik = np.eye(3) - np.outer(bik, bik)

            # measured angles
            aijk = bearings[j].dot(bearings[k])

            eijk = bij.dot(bik) - aijk
            qijk = Pij.dot(bik) / dij
            qikj = Pik.dot(bij) / dik

            gradient[i] -= k_a * eijk * (qijk + qikj)
            gradient[j] += k_a * eijk * qijk
            gradient[k] += k_a * eijk * qikj

    k_o = 2.0
    for i in nodes:
        # --- Control inputs --- #
        ub[i] = [np.cos(1.0*t), np.sin(1.0*t), np.sin(2.0*t)]
        wb[i] = [i / 10.0, 0, 0.5 - i / 10.0]

        # --- advance pose --- #
        p_int[i].step(t, R[i].dot(ub[i]))
        R_int[i].step_left(t, wb[i])

        # --- advance estimation --- #
        zi = gradient[i].copy()

        hatq_int[i].step(
            t, hatQ[i].dot(ub[i]) - ub[a] - np.cross(wb[a], hatq[i]) - zi
        )
        hatQ_int[i].step_left(
            t,  wb[i] - hatQ[i].T.dot(wb[a]) + k_o * np.cross(ub[i], hatQ[i].T.dot(-zi))
        )


def log_step():
    """Data log"""
    logs.time.append(t)
    logs.position.append(extract_x(p_int).ravel())
    logs.orientation.append(extract_x(R_int).ravel())
    logs.estimated_position.append(extract_x(hatq_int).ravel())
    logs.estimated_orientation.append(extract_x(hatQ_int).ravel())
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
hatq_int = [
    EulerIntegrator(hat_q[i])
    for i in nodes
]

# refer initial orientation to body frame a
# Q = np.matmul(R[a].T, R)

# est_R = [random_rotation_matrix(1.0).dot(Q[i]) for i in nodes]
hatQ_int = [
    EulerIntegratorOrtogonalGroup(np.eye(3))
    for i in nodes
]

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
# initialize logs
logs = Logs(
    time=[t],
    position=[extract_x(p_int).ravel()],
    orientation=[extract_x(R_int).ravel()],
    estimated_position=[extract_x(hatq_int).ravel()],
    estimated_orientation=[extract_x(hatQ_int).ravel()],
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
np.savetxt('simu_data/control_u.csv', logs.control_u, delimiter=',')
np.savetxt('simu_data/control_w.csv', logs.control_w, delimiter=',')
np.savetxt('simu_data/correction_u.csv', logs.correction_u, delimiter=',')
np.savetxt('simu_data/correction_w.csv', logs.correction_w, delimiter=',')
np.savetxt('simu_data/adjacency.csv', logs.adjacency, delimiter=',')
