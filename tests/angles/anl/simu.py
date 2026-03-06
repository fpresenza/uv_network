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
from uvnpy.angles.local_frame.core import is_angle_rigid

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
    control_u: list
    correction: list
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
    hatp = extract_x(hatp_int)
    R = extract_x(R_int)

    u = np.zeros((n, 3), dtype=np.float64)
    delta_hatp = np.zeros((n, 3), dtype=np.float64)

    # --- scale correction --- #
    # measurements
    dab2 = np.square(p[b] - p[a]).sum()
    dac2 = np.square(p[c] - p[a]).sum()

    # estimated values
    hat_dab2 = np.square(hatp[b] - hatp[a]).sum()
    hat_dac2 = np.square(hatp[c] - hatp[a]).sum()

    # correction
    k_s = 0.05
    sc_corr_ab = k_s * (hat_dab2 - dab2) * (hatp[a] - hatp[b])
    sc_corr_ac = k_s * (hat_dac2 - dac2) * (hatp[a] - hatp[c])
    delta_hatp[a] -= sc_corr_ab + sc_corr_ac
    delta_hatp[b] += sc_corr_ab
    delta_hatp[c] += sc_corr_ac

    # --- translational correction --- #
    k_t = 2.0
    delta_hatp[a] -= k_t * hatp[a]

    # --- rotational correction --- #
    # measurements
    bab = R[a].T.dot(unit_vector(p[b] - p[a]))
    bac = R[a].T.dot(unit_vector(p[c] - p[a]))

    # estimated values
    dab = np.sqrt(dab2)
    hat_bab = unit_vector(hatp[b] - hatp[a])
    hat_Pab = np.eye(3) - np.outer(hat_bab, hat_bab)

    dac = np.sqrt(dac2)
    hat_bac = unit_vector(hatp[c] - hatp[a])
    hat_Pac = np.eye(3) - np.outer(hat_bac, hat_bac)

    # correction
    k_r = 100.0
    delta_hatp[a] -= k_r * (hat_Pab.dot(bab) / dab + hat_Pac.dot(bac) / dac)
    delta_hatp[b] += k_r * hat_Pab.dot(bab) / dab
    delta_hatp[c] += k_r * hat_Pac.dot(bac) / dac

    # --- angle correction --- #
    k_a = 200.0
    for i in nodes:
        out_neighbors = edge_set[:, 1][edge_set[:, 0] == i]

        # --- estimated values --- #
        hat_distances = {
            j: np.sqrt(np.square(hatp[j] - hatp[i]).sum()) for j in out_neighbors
        }
        hat_bearings = {j: unit_vector(hatp[j] - hatp[i]) for j in out_neighbors}

        # --- measurements --- #
        bearings = {
            j: R[i].T.dot(unit_vector(p[j] - p[i])) for j in out_neighbors
        }

        # --- estimation law --- #
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

            delta_hatp[i] += k_a * eijk * (qijk + qikj)
            delta_hatp[j] -= k_a * eijk * qijk
            delta_hatp[k] -= k_a * eijk * qikj

    for i in nodes:
        # u[i] = i * np.exp(-t) * np.array([np.cos(0.2*t), np.sin(0.2*t), 0.0])
        p_int[i].step(t, u[i])
        hatp_int[i].step(t, delta_hatp[i])


def log_step():
    """Data log"""
    logs.time.append(t)
    logs.position.append(extract_x(p_int).ravel())
    logs.orientation.append(extract_x(R_int).ravel())
    logs.estimated_position.append(extract_x(hatp_int).ravel())
    logs.control_u.append(extract_u(p_int).ravel())
    logs.correction.append(extract_u(hatp_int).ravel())


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

np.random.seed(0)

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
init_pos = np.random.uniform(0.0, 30.0, (n, 3))
init_ori = np.array([random_rotation_matrix() for _ in nodes])
edge_set = np.array([
    [0, 1],
    [0, 2],
    [1, 0],
    [1, 2],
    [0, 3],
    [1, 3]
])
a, b, c = 0, 1, 2

if not is_angle_rigid(edge_set, init_pos):
    raise ValueError('The framework is not IAR.')

p_int = [
    EulerIntegrator(init_pos[i])
    for i in nodes
]

R_int = [
    EulerIntegratorOrtogonalGroup(init_ori[i])
    for i in nodes
]

# refer initial position to body-frame a
init_pos_a = (init_pos - init_pos[a]).dot(init_ori[a])

est_init_pos = np.random.normal(init_pos_a, 2.0)
hatp_int = [
    EulerIntegrator(est_init_pos[i])
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
    estimated_position=[extract_x(hatp_int).ravel()],
    control_u=[extract_u(p_int).ravel()],
    correction=[extract_u(hatp_int).ravel()],
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
np.savetxt('simu_data/control_u.csv', logs.control_u, delimiter=',')
np.savetxt('simu_data/correction.csv', logs.correction, delimiter=',')
np.savetxt('simu_data/adjacency.csv', logs.adjacency, delimiter=',')
