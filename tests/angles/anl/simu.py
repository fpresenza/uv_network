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
from uvnpy.toolkit.geometry import rotation_matrix_from_quaternion
from uvnpy.angles.local_frame.core import angle_indices, is_angle_rigid

# ------------------------------------------------------------------
# Functions, Classes and Configurations
# ------------------------------------------------------------------
np.set_printoptions(suppress=True, precision=10)


@dataclass
class Logs(object):
    time: list
    position: list
    estimated_position: list
    adjacency: list


def random_rotation_matrix():
    q = np.random.normal(size=4)
    q /= np.sqrt(q.dot(q))
    return rotation_matrix_from_quaternion(q)


def extract(integrators):
    return np.array([p.x() for p in integrators])


def complete_angle_set(out_neighbors):
    i, j = np.triu_indices(out_neighbors.size, k=1)
    return np.column_stack([out_neighbors[i], out_neighbors[j]])


# ------------------------------------------------------------------
# Simulation loop inner functions
# ------------------------------------------------------------------


def simu_step():
    """Formation control algorithm"""
    # --- data ---#
    p = extract(p_int)
    hatp = extract(hatp_int)
    R = extract(R_int)

    hatu = np.zeros((n, 3), dtype=np.float64)

    for i in nodes:
        out_neighbors = edge_set[:, 1][edge_set[:, 0] == i]

        # --- estimated values --- #
        hat_distances = {
            j: np.sqrt(np.square(hatp[j] - hatp[i]).sum()) for j in out_neighbors
        }
        hat_bearings = {j: unit_vector(hatp[j] - hatp[i]) for j in out_neighbors}

        # --- measured values --- #
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

            # --- measured angles --- #
            aijk = bearings[j].dot(bearings[k])

            eijk = bij.dot(bik) - aijk
            qijk = Pij.dot(bik) / dij
            qikj = Pik.dot(bij) / dik

            hatu[i] += eijk * (qijk + qikj)
            hatu[j] -= eijk * qijk
            hatu[k] -= eijk * qikj

        gamma = 2.0
        dkl = np.square(p[kappa] - p[ell]).sum()
        hat_dkl = np.square(hatp[kappa] - hatp[ell]).sum()
        scale_correction = gamma * (hat_dkl - dkl) * (hatp[kappa] - hatp[ell])
        hatu[kappa] -= scale_correction
        hatu[ell] += scale_correction

    for i in nodes:
        hatp_int[i].step(t, hatu[i])


def log_step():
    """Data log"""
    logs.time.append(t)
    logs.estimated_position.append(extract(hatp_int).ravel())


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
edge_set = np.array([
    [0, 1],
    [0, 2],
    [1, 0],
    [1, 2],
    [0, 3],
    [1, 3]
])
angle_set = angle_indices(n, edge_set).astype(int)
# print(angle_set)

init_pos = np.random.uniform(0.0, 1.0, (n, 3))

kappa, ell = edge_set[0]

if not is_angle_rigid(edge_set, init_pos):
    raise ValueError('The framework is not IAR.')

p_int = [
    EulerIntegrator(init_pos[i])
    for i in nodes
]

hatp_int = [
    EulerIntegrator(np.random.normal(init_pos[i], 0.1, size=3))
    for i in nodes
]

R_int = [
    EulerIntegratorOrtogonalGroup(random_rotation_matrix())
    for _ in nodes
]

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
# initialize logs
logs = Logs(
    time=[t],
    position=[extract(p_int).ravel()],
    estimated_position=[extract(hatp_int).ravel()],
    adjacency=[adjacency_matrix_from_edges(n, edge_set).ravel()]
)

# run simulation
simu_counter = 1
bar = progressbar.ProgressBar(maxval=simu_length).start()

while simu_counter < simu_num_steps:
    t += simu_step_size

    simu_step()
    if (simu_counter % log_skip == 0):
        log_step()

    simu_counter += 1

    bar.update(np.round(t, 3))

bar.finish()

np.savetxt('simu_data/t.csv', logs.time, delimiter=',')
np.savetxt('simu_data/position.csv', logs.position, delimiter=',')
np.savetxt(
    'simu_data/estimated_position.csv', logs.estimated_position, delimiter=','
)
np.savetxt('simu_data/adjacency.csv', logs.adjacency, delimiter=',')
