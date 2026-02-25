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
from uvnpy.angles.local_frame.core import (
    angle_indices,
    is_angle_rigid,
    angle_rigidity_matrix
)

# ------------------------------------------------------------------
# Functions and Classes
# ------------------------------------------------------------------


@dataclass
class Logs(object):
    time: list
    position: list
    # velocity: list
    orientation: list
    control: list
    rigidity_val: list
    adjacency: list


def random_rotation_matrix():
    q = np.random.normal(size=4)
    q /= np.sqrt(q.dot(q))
    return rotation_matrix_from_quaternion(q)


def extract(integrators, wrapper=lambda x: x):
    return [wrapper(p.x()) for p in integrators]


def extract_vel(integrators, wrapper=lambda x: x):
    return [wrapper(p.dotx()) for p in integrators]


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
    # v = extract_vel(p_int)
    R = extract(R_int)
    u = np.zeros((n, 3), dtype=np.float64)
    w = np.zeros((n, 3), dtype=np.float64)

    # --- angle rigidity eigenvalue-vector --- #
    A = angle_rigidity_matrix(edge_set, np.array(p))
    evals, evecs = np.linalg.eigh(A.T.dot(A))
    val = evals[7]
    vec = evecs[:, 7].reshape(n, 3)
    vec = [Ri.T.dot(veci) for Ri, veci in zip(R, vec)]

    for i in nodes:
        out_neighbors = edge_set[:, 1][edge_set[:, 0] == i]

        # --- measurements --- #
        distances = {
            j: np.sqrt(np.sum((p[j] - p[i])**2)) for j in out_neighbors
        }
        bearings = {
            j: R[i].T.dot(unit_vector(p[j] - p[i])) for j in out_neighbors
        }
        rotations = {
            j: R[i].T.dot(R[j]) for j in out_neighbors
        }
        # vi = R[i].T.dot(v[i])

        # --- control law --- #
        kp = 0.01
        for j, k in complete_angle_set(out_neighbors):
            dij = distances[j]
            bij = bearings[j]
            Pij = np.eye(3) - np.outer(bij, bij)
            Rij = rotations[j]

            dik = distances[k]
            bik = bearings[k]
            Pik = np.eye(3) - np.outer(bik, bik)
            Rik = rotations[k]

            qijk = Pij.dot(bik) / dij
            qikj = Pik.dot(bij) / dik

            vec_i = vec[i]
            vec_j = Rij.dot(vec[j])
            vec_k = Rik.dot(vec[k])

            sijk = qijk.dot(vec_j - vec_i) + qikj.dot(vec_k - vec_i)

            Dijk = np.outer(Pij.dot(bik), bij)
            Dijk += np.outer(bij, bik.dot(Pij))
            Dijk += bij.dot(bik) * Pij
            Dijk /= dij**2

            Dikj = np.outer(Pik.dot(bij), bik)
            Dikj += np.outer(bik, bij.dot(Pik))
            Dikj += bik.dot(bij) * Pik
            Dikj /= dik**2

            Eijk = Pij.dot(Pik) / (dij * dik)
            Eikj = Pik.dot(Pij) / (dik * dij)

            nijk_i = (Dijk - Eikj).dot(vec_j - vec_i) + (Dikj - Eijk).dot(vec_k - vec_i)
            nijk_j = - Dijk.dot(vec_j - vec_i) + Eijk.dot(vec_k - vec_i)
            nijk_k = Eikj.dot(vec_j - vec_i) - Dikj.dot(vec_k - vec_i)

            u[i] += 2 * kp * sijk * nijk_i
            u[j] += 2 * kp * sijk * Rij.T.dot(nijk_j)
            u[k] += 2 * kp * sijk * Rik.T.dot(nijk_k)

        # u[i] -= kd * vi
        w[i] = np.array([0.0, 0.0, 0.0])

    # print(p)
    # print(R)
    # print(evecs[:, 7].reshape(n, 3))
    # print(u)
    for i in nodes:
        p_int[i].step(t, R[i].dot(u[i]))
        R_int[i].step(t, R[i].dot(w[i]))

    control_action[:] = np.hstack([u, w])
    rigidity_val[:] = val


def log_step():
    """Data log"""
    logs.time.append(t)
    logs.position.append(np.hstack(extract(p_int)))
    # logs.velocity.append(np.hstack(extract_vel(p_int)))
    logs.orientation.append(np.hstack(extract(R_int, wrapper=np.ravel)))
    logs.control.append(control_action.copy().ravel())
    logs.rigidity_val.append(rigidity_val.copy())


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

print(
    'Simulation Time: begin = {} sec, end = {} sec, step = {} sec'
    .format(0.0, simu_length, simu_step_size)
)
print(
    'Logging Time: begin = {} sec, end = {} sec, step = {} sec'
    .format(0.0, simu_length, simu_step_size * log_skip)
)

np.set_printoptions(suppress=True, precision=10)
np.random.seed(0)

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
initial_position = np.random.uniform(0.0, 1.0, (n, 3))

if not is_angle_rigid(edge_set, initial_position):
    raise ValueError('The initial framework is not IAR.')

p_int = [
    EulerIntegrator(initial_position[i])
    for i in nodes
]
R_int = [
    EulerIntegratorOrtogonalGroup(random_rotation_matrix())
    for _ in nodes
]

control_action = np.empty((n, 6), dtype=np.float64)
rigidity_val = np.empty(1, dtype=np.float64)

# initialize logs
logs = Logs(
    time=[t],
    position=[np.hstack(extract(p_int))],
    # velocity=[np.hstack(extract_vel(p_int))],
    orientation=[np.hstack(extract(R_int, wrapper=np.ravel))],
    control=[],
    rigidity_val=[],
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

    # bar.update(np.round(t, 3))

bar.finish()

np.savetxt('simu_data/t.csv', logs.time, delimiter=',')
np.savetxt('simu_data/position.csv', logs.position, delimiter=',')
# np.savetxt('simu_data/velocity.csv', logs.velocity, delimiter=',')
np.savetxt('simu_data/orientation.csv', logs.orientation, delimiter=',')
np.savetxt('simu_data/control.csv', logs.control, delimiter=',')
np.savetxt('simu_data/rigidity_val.csv', logs.rigidity_val, delimiter=',')
np.savetxt('simu_data/adjacency.csv', logs.adjacency, delimiter=',')
