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
    gradient_p: list
    gradient_R: list
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


def projection_matrix(x):
    return np.eye(3) - np.outer(x, x)


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
    v = extract_u(p_int)
    hatp = extract_x(hatp_int)
    R = extract_x(R_int)
    hatR = extract_x(hatR_int)

    grad_p[:] = 0.0
    grad_R[:] = 0.0

    ub = np.zeros((n, 3), dtype=np.float64)    # body-frame
    wb = np.zeros((n, 3), dtype=np.float64)    # body-frame

    # --- scale and angles correction --- #
    # correction
    k_s = 0.5
    dab2 = np.square(p[b] - p[a]).sum()
    hat_dab2 = np.square(hatp[b] - hatp[a]).sum()
    scale_correction_ab = k_s * (hat_dab2 - dab2) * (hatp[a] - hatp[b])
    grad_p[a] += scale_correction_ab
    grad_p[b] -= scale_correction_ab

    k_a = 2000.0
    for i in nodes:
        # --- Control inputs --- #
        ub[i] = control_u[i](t)
        wb[i] = control_w[i](t)

        # --- advance pose --- #
        p_int[i].step(t, R[i].dot(ub[i]))
        R_int[i].step_left(t, wb[i])

        # --- angle correction --- #
        out_neighbors = edge_set[:, 1][edge_set[:, 0] == i]

        # estimated values
        hat_distances = {
            j: np.sqrt(np.square(hatp[j] - hatp[i]).sum()) for j in out_neighbors
        }
        hat_bearings = {j: unit_vector(hatp[j] - hatp[i]) for j in out_neighbors}

        # measurements
        distances = {
            j: np.sqrt(np.square(p[j] - p[i]).sum()) for j in out_neighbors
        }
        bearings = {
            j: R[i].T.dot(unit_vector(p[j] - p[i])) for j in out_neighbors
        }
        dot_bearings = {
            j: projection_matrix(bearings[j]).dot(
                R[i].T.dot(v[j] - v[i])
            ) / distances[j] - np.cross(wb[i], bearings[j])
            for j in out_neighbors
        }

        # position gradient
        for j, k in complete_angle_set(out_neighbors):

            dij = hat_distances[j]
            bij = hat_bearings[j]
            Pij = projection_matrix(bij)

            dik = hat_distances[k]
            bik = hat_bearings[k]
            Pik = projection_matrix(bik)

            # measured angles
            aijk = bearings[j].dot(bearings[k])

            eijk = bij.dot(bik) - aijk
            Xijk = Pij.dot(bik) / dij
            Xikj = Pik.dot(bij) / dik

            grad_p[i] -= k_a * eijk * (Xijk + Xikj)
            grad_p[j] += k_a * eijk * Xijk
            grad_p[k] += k_a * eijk * Xikj

        # orientation gradient
        if i in leaders:
            for j in out_neighbors:
                grad_R[i] += np.cross(bearings[j], hatR[i].T.dot(hatp[j] - hatp[i]))
                hat_Mij = projection_matrix(hatR[i].dot(bearings[j]))
                hat_Oi_bij = hatR[i].dot(np.cross(wb[i], bearings[j]))
                hat_Ri_dotbij = hatR[i].dot(dot_bearings[j])
                hat_v_i = hatR[i].dot(ub[i])
                aux_f[j]['num'] += hat_distances[j] * (
                    hat_Oi_bij + hat_Ri_dotbij) + hat_Mij.dot(hat_v_i)
                aux_f[j]['den'] += hat_Mij

    k_o = 2.0
    for i in nodes:
        # --- advance estimation --- #
        if i in leaders:
            hatp_int[i].step(t, hatR[i].dot(ub[i]) - grad_p[i])
            hatR_int[i].step_left(t, wb[i] + k_o * grad_R[i])
        else:
            hat_v_i = np.linalg.inv(aux_f[i]['den']).dot(aux_f[i]['num'])
            aux_f[i]['num'][:] = 0.0
            aux_f[i]['den'][:] = 0.0
            grad_R[i] = np.cross(ub[i], hatR[i].T.dot(hat_v_i))
            hatp_int[i].step(t, hat_v_i - grad_p[i])
            hatR_int[i].step_left(t, wb[i] + k_o * grad_R[i])


def log_step():
    """Data log"""
    logs.time.append(t)
    logs.position.append(extract_x(p_int).ravel())
    logs.orientation.append(extract_x(R_int).ravel())
    logs.estimated_position.append(extract_x(hatp_int).ravel())
    logs.estimated_orientation.append(extract_x(hatR_int).ravel())
    logs.gradient_p.append(grad_p.copy().ravel())
    logs.gradient_R.append(grad_R.copy().ravel())
    logs.control_u.append(extract_u(p_int).ravel())
    logs.control_w.append(extract_u(R_int).ravel())
    logs.correction_u.append(extract_u(hatp_int).ravel())
    logs.correction_w.append(extract_u(hatR_int).ravel())


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
n = 5
nodes = np.arange(n)
# p = np.random.uniform(0.0, 30.0, (n, 3))
p = np.array([
    [13.07984706, 0.77778695, 16.48987434],
    [13.05967178, 8.61103406, 9.91004463],
    [6.13945902, 18.57812899, 8.98964021],
    [8.00481825, 12.63401498, 15.87426283],
    [15., 5., 5.]
])

R = np.array([random_rotation_matrix() for _ in nodes])
edge_set = np.array([
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [1, 0],
    [1, 2],
    [1, 3],
    [1, 4]
])
angle_set = angle_indices(nodes, edge_set).astype(int)
a, b = 0, 1
leaders = np.unique(angle_set[:, 0])
followers = np.setdiff1d(nodes, leaders)

if not is_angle_rigid(angle_set, p):
    raise ValueError('The framework is not IAR.')

p_int = [EulerIntegrator(p[i]) for i in nodes]

R_int = [EulerIntegratorOrtogonalGroup(R[i]) for i in nodes]

hatp_int = [EulerIntegrator(np.random.normal(p[i], 2.0)) for i in nodes]

hatR_int = [
    EulerIntegratorOrtogonalGroup(
        random_rotation_matrix(1.0).dot(R[i])
    )
    for i in nodes
]

# define velocities

control_u = {
    0: lambda t: np.array([0.0, 0.0, 1.0]),
    1: lambda t: np.array([0.0, np.cos(0.25*t), np.sin(0.25*t)]),
    2: lambda t: np.array([0.0, np.cos(1.0*t), np.sin(1.0*t)]),
    3: lambda t: np.array([np.cos(2.0*t), np.sin(2.0*t), 0.5]),
    4: lambda t: np.array([np.cos(1.0*t), np.sin(0.5*t), 0.0])
}

control_w = {
    0: lambda t: np.array([0.5, 0.0, 0.0]),
    1: lambda t: np.array([0.0, 1.0, 0.0]),
    2: lambda t: np.array([0.0, 0.0, 1.0]),
    3: lambda t: np.array([0.0, 0.0, 0.0]),
    4: lambda t: np.array([0.0, 0.0, 0.0])
}

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
# initialize logs
grad_p = np.zeros((n, 3), dtype=np.float64)
grad_R = np.zeros((n, 3), dtype=np.float64)
aux_f = {
    i: {
        'num': np.zeros(3, dtype=np.float64),
        'den': np.zeros((3, 3), dtype=np.float64)
    }
    for i in nodes
}

logs = Logs(
    time=[t],
    position=[extract_x(p_int).ravel()],
    orientation=[extract_x(R_int).ravel()],
    estimated_position=[extract_x(hatp_int).ravel()],
    estimated_orientation=[extract_x(hatR_int).ravel()],
    gradient_p=[grad_p.copy().ravel()],
    gradient_R=[grad_R.copy().ravel()],
    control_u=[extract_u(p_int).ravel()],
    control_w=[extract_u(p_int).ravel()],
    correction_u=[extract_u(hatp_int).ravel()],
    correction_w=[extract_u(hatR_int).ravel()],
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
np.savetxt('simu_data/position_gradient.csv', logs.gradient_p, delimiter=',')
np.savetxt('simu_data/orientation_gradient.csv', logs.gradient_R, delimiter=',')
np.savetxt('simu_data/control_u.csv', logs.control_u, delimiter=',')
np.savetxt('simu_data/control_w.csv', logs.control_w, delimiter=',')
np.savetxt('simu_data/correction_u.csv', logs.correction_u, delimiter=',')
np.savetxt('simu_data/correction_w.csv', logs.correction_w, delimiter=',')
np.savetxt('simu_data/adjacency.csv', logs.adjacency, delimiter=',')
