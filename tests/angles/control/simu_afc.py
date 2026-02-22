#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
import progressbar
import numpy as np
# from transformations import unit_vector

from uvnpy.graphs.core import adjacency_matrix_from_edges
from uvnpy.dynamics.core import EulerIntegrator
from uvnpy.toolkit.geometry import rotation_matrix_from_quaternion
from uvnpy.angles.local_frame.core import angle_indices, angle_function


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
    adjacency: list


def random_position(*args, **kwargs):
    return np.random.uniform(*args, **kwargs)


def random_rotation_matrix():
    q = np.random.normal(size=4)
    q /= np.sqrt(q.dot(q))
    return rotation_matrix_from_quaternion(q)


def extract(integrators):
    return [p.x().ravel() for p in integrators]


# ------------------------------------------------------------------
# Simulation loop inner functions
# ------------------------------------------------------------------


def simu_step():
    """Formation control algorithm"""
    for i in nodes:
        # --- measurements --- #
        # out_neighbors = edge_set[:, 1][edge_set[:, 0] == i]
        # bearings = [unit_vector(p[j] - p[i]) for j in out_neighbors]

        # --- control law --- #
        # kp, kd = 1.0, 1.0
        u = np.zeros(3, dtype=np.float64)
        # out_angles = angle_set[angle_set[:, 0] == i]
        # for angle in out_angles:
        #     u += 0.0
        # in_angles = angle_set[np.logical_or(
        #     angle_set[:, 1] == i, angle_set[:, 2] == i
        # )]
        # for angle in in_angles:
        #     u += 0.0
        p_int[i].step(t, u)


def log_step():
    """Data log"""
    if (simu_counter % log_skip == 0):
        logs.time.append(t)
        logs.position.append(np.hstack(extract(p_int)))


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
# Simulation parameters
simu_length = arg.simu_length
simu_step_size = arg.simu_step_size
simu_step_size_sec = simu_step_size * 1e-3

if simu_length % simu_step_size != 0:
    print('\
        Simulation length is not a multiple of the step size. \
        Length will be truncated the closest multiple.\
    ')

simu_step_num = int(simu_length / simu_step_size)
log_skip = arg.log_skip

np.random.seed(0)

print(
    'Simulation Time: begin = {} sec, end = {} sec, step = {} sec'
    .format(0.0, simu_length * 1e-3, simu_step_size_sec)
)

# World parameters
t = 0.0
n = 3
nodes = np.arange(n)
edge_set = np.array([
    [0, 1],
    [0, 2],
    [1, 0],
    [1, 2],
])
angle_set = angle_indices(n, edge_set).astype(int)
print(angle_set)

desired_position = random_position(0.0, 1.0, (n, 3))
desired_angles = angle_function(edge_set, desired_position)
# print(desired_angles)

p_int = [
    EulerIntegrator(desired_position[i] + random_position(0.0, 0.0, (3,)))
    for i in nodes
]
o_int = [EulerIntegrator(random_rotation_matrix()) for _ in nodes]

angles = angle_function(edge_set, np.vstack(extract(p_int)))
# print(angles)


# initialize logs
logs = Logs(
    time=[t],
    position=[np.hstack(extract(p_int))],
    orientation=[np.hstack(extract(o_int))],
    adjacency=adjacency_matrix_from_edges(n, edge_set)
)

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------

simu_counter = 1
bar = progressbar.ProgressBar(maxval=arg.simu_length).start()

while simu_counter <= simu_step_num:
    t += simu_step_size_sec

    simu_step()
    log_step()

    simu_counter += 1

    bar.update(np.round(t, 3))

bar.finish()

np.savetxt('simu_data/t.csv', logs.time, delimiter=',')
np.savetxt('simu_data/position.csv', logs.position, delimiter=',')
np.savetxt('simu_data/orientation.csv', logs.orientation, delimiter=',')
