#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
import progressbar
import numpy as np

from uvnpy.dynamics.core import EulerIntegrator
from uvnpy.dynamics.lie_groups import EulerIntegratorOrtogonalGroup
from uvnpy.toolkit.geometry import rotation_matrix_from_quaternion

# ------------------------------------------------------------------
# Functions, Classes and Configurations
# ------------------------------------------------------------------
np.set_printoptions(suppress=True, precision=10)


@dataclass
class Logs(object):
    time: list
    orientation: list
    estimated_orientation: list


def random_rotation_matrix():
    q = np.random.normal(size=4)
    q /= np.sqrt(q.dot(q))
    return rotation_matrix_from_quaternion(q)


def extract_x(integrators):
    return np.array([p.x() for p in integrators])


def extract_u(integrators):
    return np.array([p.u() for p in integrators])


# ------------------------------------------------------------------
# Simulation loop inner functions
# ------------------------------------------------------------------


def simu_step():
    """Orientation estimation algorithm"""
    # --- data ---#
    R = extract_x(R_int)
    hatQ = extract_x(hatQ_int)

    wb = np.zeros((n, 3), dtype=np.float64)    # body-frame

    # --- similarity correction --- #

    for i in nodes:
        # --- Control inputs --- #
        wb[i] = [0.0, 0.0, 0.0]

        # --- advance pose --- #
        R_int[i].step_left(t, wb[i])

        correction = 0.0
        for j in edge_set[edge_set[:, 0] == i, 1]:
            # --- measurement --- #
            Rij = R[i].T.dot(R[j])

            # --- compute update law --- #
            correction += np.dot(hatQ[j], Rij.T) - hatQ[i]

        hatQ_int[i].step(t, correction)


def log_step():
    """Data log"""
    logs.time.append(t)
    logs.orientation.append(extract_x(R_int).ravel())
    logs.estimated_orientation.append(extract_x(hatQ_int).ravel())


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
    # [0, 1],
    [1, 0],
    [1, 2],
    [1, 3],
    [2, 0],
    [2, 1],
    [2, 3],
    [3, 1],
    [3, 2]
])
a = 0

R_int = [
    EulerIntegratorOrtogonalGroup(random_rotation_matrix())
    for i in nodes
]

hatQ_int = [
    EulerIntegrator(np.eye(3) if i == a else random_rotation_matrix())
    for i in nodes
]

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
# initialize logs
logs = Logs(
    time=[t],
    orientation=[extract_x(R_int).ravel()],
    estimated_orientation=[extract_x(hatQ_int).ravel()],
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
np.savetxt('simu_data/orientation.csv', logs.orientation, delimiter=',')
np.savetxt(
    'simu_data/estimated_orientation.csv', logs.estimated_orientation, delimiter=','
)
