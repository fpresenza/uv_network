#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
import progressbar
import numpy as np

from uvnpy.dynamics.lie_groups import EulerIntegratorOrtogonalGroup
from uvnpy.toolkit.geometry import rotation_matrix_from_vector


# ------------------------------------------------------------------
# Functions and Classes
# ------------------------------------------------------------------


@dataclass
class Logs(object):
    time: list
    orientation: list
    estimated_orientation: list


def random_rotation_matrix(max_angle=2 * np.pi):
    v = np.random.normal(size=3)
    v /= np.sqrt(v.dot(v))
    a = np.random.uniform(0.0, max_angle)
    return rotation_matrix_from_vector(a * v)


# ------------------------------------------------------------------
# Simulation loop inner functions
# ------------------------------------------------------------------


def simu_step():
    """Orientation estimation algorithm"""
    # --- data ---#
    R = R_int.x()
    hatR = hatR_int.x()
    v = np.array([np.cos(t), np.sin(t), 0.5])

    # --- measurements --- #
    vb = R.T.dot(v)

    # --- estimation law --- #
    hatvb = hatR.T.dot(v)

    w = np.array([0.4, 1.0, -0.2])
    wb = R.T.dot(w)

    ub = wb + np.cross(vb, hatvb)

    R_int.step_right(t, w)
    hatR_int.step_left(t, ub)


def log_step():
    """Data log"""
    logs.time.append(t)
    logs.orientation.append(R_int.x().ravel())
    logs.estimated_orientation.append(hatR_int.x().ravel())


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
# Configuración
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
initial_error = random_rotation_matrix(max_angle=2*np.pi)
# initial_error = rotation_matrix_from_vector(np.array([0.0, 0.0, 1.0]))
R_int = EulerIntegratorOrtogonalGroup(random_rotation_matrix())
hatR_int = EulerIntegratorOrtogonalGroup(initial_error.dot(R_int.x()))

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
# initialize logs
logs = Logs(
    time=[t],
    orientation=[R_int.x().ravel()],
    estimated_orientation=[hatR_int.x().ravel()]
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
np.savetxt('simu_data/orientation.csv', logs.orientation, delimiter=',')
np.savetxt(
    'simu_data/estimated_orientation.csv',
    logs.estimated_orientation,
    delimiter=','
)
