#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
import progressbar
import numpy as np
from transformations import unit_vector

from uvnpy.graphs.core import adjacency_matrix_from_edges
from uvnpy.dynamics.core import EulerIntegrator
from uvnpy.toolkit.geometry import rotation_matrix_from_quaternion
from uvnpy.angles.local_frame.core import (
    angle_indices,
    angle_set_from_indices,
    is_angle_rigid
)

# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
np.set_printoptions(
    suppress=True,
    precision=10
)


@dataclass
class Logs(object):
    time: list
    position: list
    orientation: list
    desired_position: list
    adjacency: list


def random_position(*args, **kwargs):
    return np.random.uniform(*args, **kwargs)


def random_rotation_matrix():
    q = np.random.normal(size=4)
    q /= np.sqrt(q.dot(q))
    return rotation_matrix_from_quaternion(q)


def extract(integrators, wrapper=lambda x: x):
    return [wrapper(p.x()) for p in integrators]


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
    R = extract(o_int)
    u = np.zeros((n, 3), dtype=np.float64)

    for i in nodes:
        # --- measurements --- #
        out_neighbors = edge_set[:, 1][edge_set[:, 0] == i]
        distances = {
            j: np.sqrt(np.sum((p[j] - p[i])**2)) for j in out_neighbors
        }
        bearings = {
            j: R[i].T.dot(unit_vector(p[j] - p[i])) for j in out_neighbors
        }
        rotations = {
            j: R[i].T.dot(R[j]) for j in out_neighbors
        }

        # --- control law --- #
        # kp, kd = 1.0, 1.0
        for j, k in complete_angle_set(out_neighbors):
            dij = distances[j]
            bij = bearings[j]
            Pij = np.eye(3) - np.outer(bij, bij)
            Rij = rotations[j]

            dik = distances[k]
            bik = bearings[k]
            Pik = np.eye(3) - np.outer(bik, bik)
            Rik = rotations[k]

            eijk = bij.dot(bik) - desired_angles[(i, j, k)]
            qijk = Pij.dot(bik) / dij
            qikj = Pik.dot(bij) / dik

            u[i] += eijk * (qijk + qikj)
            u[j] -= eijk * Rij.T.dot(qijk)
            u[k] -= eijk * Rik.T.dot(qikj)

    for i in nodes:
        p_int[i].step(t, R[i].T.dot(u[i]))


def log_step():
    """Data log"""
    logs.time.append(t)
    logs.position.append(np.hstack(extract(p_int)))
    logs.orientation.append(np.hstack(extract(o_int, wrapper=np.ravel)))


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

desired_position = random_position(0.0, 1.0, (n, 3))
desired_angles = angle_set_from_indices(angle_set, desired_position)
# print(desired_angles)

if not is_angle_rigid(edge_set, desired_position):
    raise ValueError('The desired framework is not IAR.')

p_int = [
    EulerIntegrator(desired_position[i] + random_position(-0.1, 0.1, (3,)))
    for i in nodes
]
o_int = [EulerIntegrator(random_rotation_matrix()) for _ in nodes]

# initialize logs
logs = Logs(
    time=[t],
    position=[np.hstack(extract(p_int))],
    orientation=[np.hstack(extract(o_int, wrapper=np.ravel))],
    desired_position=[desired_position.ravel()],
    adjacency=[adjacency_matrix_from_edges(n, edge_set).ravel()]
)
# print(logs.position[0])
# print(logs.orientation[0])


# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------

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
np.savetxt('simu_data/orientation.csv', logs.orientation, delimiter=',')
np.savetxt(
    'simu_data/desired_position.csv', logs.desired_position, delimiter=','
)
np.savetxt('simu_data/adjacency.csv', logs.adjacency, delimiter=',')
