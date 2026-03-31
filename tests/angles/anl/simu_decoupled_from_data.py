#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
import progressbar
import numpy as np
from transformations import unit_vector

from uvnpy.toolkit.data import read_csv_numpy
from uvnpy.graphs.core import edges_from_adjacency
from uvnpy.dynamics.core import EulerIntegrator
from uvnpy.angles.local_frame.core import angle_indices

# ------------------------------------------------------------------
# Functions, Classes and Configurations
# ------------------------------------------------------------------
np.set_printoptions(suppress=True, precision=10)


@dataclass
class Logs(object):
    estimated_position: list
    correction_u: list


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
    t = time[simu_counter]
    p = position[simu_counter]
    R = orientation[simu_counter]
    edge_set = edges_from_adjacency(adjacency[simu_counter])
    angle_set = angle_indices(nodes, edge_set).astype(int)
    # ub = control_u[simu_counter - 1]
    # wb = control_w[simu_counter - 1]

    hatq = extract_x(hatq_int)

    gradient_descent = np.zeros((n, 3), dtype=np.float64)

    # # --- similarity correction --- #
    # # measurements
    # qb = R[a].T.dot(p[b] - p[a])
    # qc = R[a].T.dot(p[c] - p[a])

    # # correction
    # k_s = 30.0
    # gradient_descent[a] -= k_s * hatq[a]
    # gradient_descent[b] -= k_s * (hatq[b] - qb)
    # gradient_descent[c] -= k_s * (hatq[c] - qc)

    # --- scale correction --- #
    # measurements
    dab2 = np.square(p[b] - p[a]).sum()
    # dac2 = np.square(p[c] - p[a]).sum()

    # estimated values
    hat_dab2 = np.square(hatq[b] - hatq[a]).sum()
    # hat_dac2 = np.square(hatq[c] - hatq[a]).sum()

    # correction
    k_s = 0.05
    scale_correction_ab = k_s * (hat_dab2 - dab2) * (hatq[a] - hatq[b])
    # scale_correction_ac = k_s * (hat_dac2 - dac2) * (hatq[a] - hatq[c])
    gradient_descent[a] -= scale_correction_ab     # + scale_correction_ac
    gradient_descent[b] += scale_correction_ab
    # gradient_descent[c] += scale_correction_ac

    k_a = 1000.0
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
        for j, k in angle_set[angle_set[:, 0] == i][:, 1:]:

            dij = hat_distances[j]
            bij = hat_bearings[j]
            Pij = np.eye(3) - np.outer(bij, bij)

            dik = hat_distances[k]
            bik = hat_bearings[k]
            Pik = np.eye(3) - np.outer(bik, bik)

            # measured angles
            aijk = bearings[j].dot(bearings[k])

            eijk = bij.dot(bik) - aijk
            nijk = Pij.dot(bik) / dij
            nikj = Pik.dot(bij) / dik

            angle_correction_j = k_a * eijk * nijk
            angle_correction_k = k_a * eijk * nikj
            gradient_descent[i] += angle_correction_j + angle_correction_k
            gradient_descent[j] -= angle_correction_j
            gradient_descent[k] -= angle_correction_k

    for i in nodes:
        # --- advance estimation --- #
        # hatq_int[i].step(t, - ub[a] - np.cross(wb[a], hatq[i]) + gradient_descent[i])
        hatq_int[i].step(t, gradient_descent[i])


def log_step():
    """Data log"""
    logs.estimated_position.append(extract_x(hatq_int).ravel())
    logs.correction_u.append(extract_u(hatq_int).ravel())


# ------------------------------------------------------------------
# Argument parse
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-l', '--log_skip',
    default=1, type=int, help='logger skip in number of simu_step_size'
)
arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# --- simulation parameters --- #
log_skip = arg.log_skip

# --- world parameters --- #
time = read_csv_numpy('input_data/t.csv')
log_num_steps = len(time)

position = read_csv_numpy(
    'input_data/position.csv'
).reshape(log_num_steps, -1, 3)

n = position.shape[1]
nodes = np.arange(n)

orientation = read_csv_numpy(
    'input_data/orientation.csv'
).reshape(log_num_steps, n, 3, 3)

control_u = read_csv_numpy('input_data/control_u.csv').reshape(log_num_steps - 1, n, 3)
control_w = read_csv_numpy('input_data/control_w.csv').reshape(log_num_steps - 1, n, 3)

adjacency = read_csv_numpy(
    'input_data/adjacency.csv'
).reshape(log_num_steps, n, n)

a, b, c = 0, 1, 2

# refer initial position to body frame a
# position_a = (position[0] - position[0, a]).dot(orientation[0, a])

est_position = np.random.normal(position[0], 2.0)
hatq_int = [
    EulerIntegrator(est_position[i])
    for i in nodes
]

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
# initialize logs
logs = Logs(
    estimated_position=[extract_x(hatq_int).ravel()],
    correction_u=[extract_u(hatq_int).ravel()],
)

# run simulation
simu_counter = 1
bar = progressbar.ProgressBar(maxval=log_num_steps).start()

while simu_counter < log_num_steps:

    simu_step()
    if (simu_counter % log_skip == 0):
        log_step()

    simu_counter += 1

    bar.update(simu_counter)

bar.finish()

np.savetxt(
    'simu_data/estimated_position.csv', logs.estimated_position, delimiter=','
)
np.savetxt('simu_data/correction_u.csv', logs.correction_u, delimiter=',')
