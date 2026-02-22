#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
import progressbar
import numpy as np

from uvnpy.graphs.core import adjacency_matrix_from_edges
from uvnpy.dynamics.core import EulerIntegrator
from uvnpy.toolkit.geometry import rotation_matrix_from_quaternion


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
    pose: list
    estimated_pose: list
    adjacency: list


def random_position(lim):
    return np.random.uniform(-lim, lim, (3,))


def random_rotation_matrix():
    q = np.random.normal(size=4)
    q /= np.sqrt(q.dot(q))
    return rotation_matrix_from_quaternion(q)


def stack_pose(pose):
    return np.hstack([np.hstack([p.x(), R.x().ravel()]) for p, R in pose])


# ------------------------------------------------------------------
# Simulation loop inner functions
# ------------------------------------------------------------------


def simu_step():
    """Formation control algorithm"""
    for i in nodes:
        pass


def log_step():
    """Data log"""
    if (simu_counter % log_skip == 0):
        logs.time.append(t)
        logs.pose.append(stack_pose(pose))
        logs.estimated_pose.append(stack_pose(estimated_pose))


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

if simu_length % simu_step_size != 0:
    print('\
        Simulation length is not a multiple of the step size. \
        Length will be truncated the closest multiple.\
    ')

simu_step_num = int(simu_length / simu_step_size)
# time_steps = [simu_step_size * k for k in range(simu_step_num)]
log_skip = arg.log_skip

# np.random.seed(6)

print(
    'Simulation Time: begin = {} sec, end = {} sec, step = {} sec'
    .format(0.0, simu_length, simu_step_size)
)

# world parameters
t = 0.0
n = 3
nodes = np.arange(n)
edge_set = np.array([
    [0, 1],
    [1, 2],
])
pose = [
    (
        EulerIntegrator(random_position(1.0)),
        EulerIntegrator(random_rotation_matrix())
    )
    for i in nodes
]
estimated_pose = [
    (
        EulerIntegrator(p.x()),
        EulerIntegrator(R.x())
    )
    for p, R in pose
]

for (p, R) in pose:
    print(p.x())
    print(R.x())
    print('\n')

# initialize logs
logs = Logs(
    time=[t],
    pose=[stack_pose(pose)],
    estimated_pose=[stack_pose(estimated_pose)],
    adjacency=adjacency_matrix_from_edges(n, edge_set)
)

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------

simu_counter = 1
bar = progressbar.ProgressBar(maxval=arg.simu_length).start()

while simu_counter <= simu_step_num:
    t += simu_step_size

    simu_step()
    log_step()

    simu_counter += 1

    bar.update(np.round(t, 3))

bar.finish()

np.savetxt('simu_data/t.csv', logs.time, delimiter=',')
np.savetxt('simu_data/pose.csv', logs.pose, delimiter=',')
np.savetxt('simu_data/estimated_pose.csv', logs.estimated_pose, delimiter=',')
