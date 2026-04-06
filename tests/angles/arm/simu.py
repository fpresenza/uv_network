#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
import progressbar
import numpy as np
from transformations import unit_vector

from uvnpy.toolkit.data import write_json_file
from uvnpy.dynamics.core import EulerIntegrator
from uvnpy.dynamics.lie_groups import EulerIntegratorOrtogonalGroup
from uvnpy.toolkit.geometry import rotation_matrix_from_quaternion
from uvnpy.toolkit.functions import cosine_activation, cosine_activation_derivative
from uvnpy.graphs.models import ConeGraph
from uvnpy.angles.local_frame.core import (
    angle_indices, angle_rigidity_matrix, is_angle_rigid
)
from uvnpy.control.targets import MovingTargets


# ------------------------------------------------------------------
# Functions and Classes
# ------------------------------------------------------------------


@dataclass
class Logs(object):
    time: list
    position: list
    # velocity: list
    orientation: list
    control_u: list
    control_w: list
    rigidity_val: list
    adjacency: list
    target_position: list


def random_rotation_matrix():
    q = np.random.normal(size=4)
    q /= np.sqrt(q.dot(q))
    return rotation_matrix_from_quaternion(q)


def aiming(p, target):
    a = unit_vector(target - p, axis=-1)
    x = np.random.normal(size=3)
    b = unit_vector(np.cross(a, x), axis=-1)
    c = np.cross(a, b)
    return np.dstack([a, b, c]).squeeze()


def sigma_r_r(x):
    return cosine_activation(
        np.array([x]), 0.8 * sensing_range, sensing_range
    ).item()


def dsigma_r_r(x):
    return cosine_activation_derivative(
        np.array([x]), 0.8 * sensing_range, sensing_range
    ).item()


def sigma_r_f(x):
    return cosine_activation(
        np.array([x]), cos_hfov, 1.4 * cos_hfov
    ).item()


def dsigma_r_f(x):
    return cosine_activation_derivative(
        np.array([x]), cos_hfov, 1.4 * cos_hfov
    ).item()


def dsigma_m_r(x):
    return cosine_activation_derivative(
        np.array([x]), 0.0, 200.0
    ).item()


def dsigma_m_f(x):
    return cosine_activation_derivative(
        np.array([x]), -1.0, 1.0
    ).item()


def distance_weights(angle_set, p):
    return np.array([
        np.sqrt(np.square(p[j] - p[i]).sum() * np.square(p[k] - p[i]).sum())
        for i, j, k in angle_set
    ])


def weight(indices, p, R):
    i, j, k = indices
    dij = np.sqrt(np.square(p[j] - p[i]).sum())
    bij = (p[j] - p[i]) / dij
    dik = np.sqrt(np.square(p[k] - p[i]).sum())
    bik = (p[k] - p[i]) / dik

    ei = R[i, :, 0]
    nij = ei.dot(bij)
    nik = ei.dot(bik)

    wrij = 1 - sigma_r_r(dij)
    wfij = sigma_r_f(nij)
    wrik = 1 - sigma_r_r(dik)
    wfik = sigma_r_f(nik)

    return dij * dik * wrij * wfij * wrik * wfik


def weights(angle_set, p, R):
    return np.array([weight((i, j, k), p, R) for i, j, k in angle_set])


def extract_x(integrators):
    return np.array([p.x() for p in integrators])


def extract_dotx(integrators):
    return np.array([p.dotx() for p in integrators])


def extract_u(integrators):
    return np.array([p.u() for p in integrators])


def complete_angle_set(out_neighbors):
    i, j = np.triu_indices(out_neighbors.size, k=1)
    return np.column_stack([out_neighbors[i], out_neighbors[j]])


# ------------------------------------------------------------------
# Simulation loop inner functions
# ------------------------------------------------------------------


def simu_step():
    """Formation control algorithm"""
    # --- data ---#
    p = extract_x(p_int)
    # v = extract_dotx(p_int)
    R = extract_x(R_int)
    control_u_r = np.zeros((n, 3), dtype=np.float64)
    control_w_r = np.zeros((n, 3), dtype=np.float64)
    control_u_m = np.zeros((n, 3), dtype=np.float64)
    control_w_m = np.zeros((n, 3), dtype=np.float64)
    control_u_c = np.zeros((n, 3), dtype=np.float64)

    # --- angle rigidity eigenvalue-vector --- #
    edge_set = sensing_graph.edge_set()
    angle_set = angle_indices(rigidity_nodes, edge_set).astype(int)
    A = angle_rigidity_matrix(angle_set, p)

    #    # --- unweighted part --- #
    W = distance_weights(angle_set, p)
    evals = np.linalg.eigvalsh(A.T.dot(W[:, np.newaxis] * A))
    rigidity_val[0] = evals[7]

    #    # --- weighted part --- #
    W = weights(angle_set, p, R)
    evals, evecs = np.linalg.eigh(A.T.dot(W[:, np.newaxis] * A))
    rigidity_val[1:] = evals[7:9]
    vec = evecs[:, 7].reshape(n, 3)
    vec = np.squeeze(np.matmul(vec[:, np.newaxis, :], R))

    for i in nodes:
        out_neighbors = edge_set[:, 1][edge_set[:, 0] == i]
        r = {j: p[j] - p[i] for j in out_neighbors}

        #   # --- estimated values --- #
        distances = {
            j: np.sqrt(np.square(r[j]).sum()) for j in out_neighbors
        }

        #   # --- measurements --- #
        bearings = {
            j: R[i].T.dot(r[j] / distances[j]) for j in out_neighbors
        }
        rotations = {
            j: R[i].T.dot(R[j]) for j in out_neighbors
        }

        # --- collision avoidance control --- #
        for j in out_neighbors:
            dij = distances[j]
            bij = bearings[j]
            Rij = rotations[j]

            repulsion = 2 * sensing_range * (dij - sensing_range) * bij / dij**3
            control_u_c[i] += repulsion
            control_u_c[j] += -Rij.T.dot(repulsion)

        # --- target tracking control --- #
        k_m_r = 3.0
        k_m_f = 2.5
        if i in tracking_nodes:
            rit = targets[target_allocation[i]](t) - p[i]
            dit = np.sqrt(np.square(rit).sum())
            bit = R[i].T.dot(rit / dit)

            Pit = np.eye(3) - np.outer(bit, bit)

            nit = bit[0]
            e1_bit = np.array([0.0, -bit[2], bit[1]])

            control_u_m[i] = k_m_r * dsigma_m_r(dit) * bit \
                - k_m_f * dsigma_m_f(nit) * Pit[0] / dit
            control_w_m[i] = k_m_f * dsigma_m_f(nit) * e1_bit

            # target collision avoidance
            control_u_c[i] += 2 * sensing_range * (dit - sensing_range) * bit / dit**3

        # --- rigidity maintenance control --- #
        if i in rigidity_nodes:
            for j, k in complete_angle_set(out_neighbors):
                # --- rigidity control law --- #
                dij = distances[j]
                bij = bearings[j]
                Pij = np.eye(3) - np.outer(bij, bij)
                Rij = rotations[j]

                dik = distances[k]
                bik = bearings[k]
                Pik = np.eye(3) - np.outer(bik, bik)
                Rik = rotations[k]

                # --- weight part --- #
                nij = bij[0]
                nik = bik[0]

                wrij = 1 - sigma_r_r(dij)
                wfij = sigma_r_f(nij)
                wrik = 1 - sigma_r_r(dik)
                wfik = sigma_r_f(nik)

                d = dij * dik
                wr = wrij * wrik
                wf = wfij * wfik
                w = wr * wf

                wijk_j = w * dik * bij + d * (
                    - wf * wrik * dsigma_r_r(dij) * bij
                    + wr * wfik * dsigma_r_f(nij) * Pij[0] / dij
                )

                wijk_k = w * dij * bik + d * (
                    - wf * wrij * dsigma_r_r(dik) * bik
                    + wr * wfij * dsigma_r_f(nik) * Pik[0] / dik
                )
                wijk_i = - wijk_j - wijk_k

                e1_bij = np.array([0.0, -bij[2], bij[1]])
                e1_bik = np.array([0.0, -bik[2], bik[1]])
                wijk_Ri = d * wr * (
                    wfik * dsigma_r_f(nij) * e1_bij
                    + wfij * dsigma_r_f(nik) * e1_bik
                )

                wijk = d * w

                # --- rigidity matrix part --- #
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

                sijk_j = - Dijk.dot(vec_j - vec_i) + Eijk.dot(vec_k - vec_i)
                sijk_k = Eikj.dot(vec_j - vec_i) - Dikj.dot(vec_k - vec_i)
                sijk_i = - sijk_j - sijk_k

                control_u_r[i] += sijk * (sijk * wijk_i + 2 * wijk * sijk_i)
                control_u_r[j] += sijk * Rij.T.dot(sijk * wijk_j + 2 * wijk * sijk_j)
                control_u_r[k] += sijk * Rik.T.dot(sijk * wijk_k + 2 * wijk * sijk_k)

                control_w_r[i] += sijk**2 * wijk_Ri

    # define relative weights
    k_u_r = 5.0 / evals[7]
    k_u_m = 50.0
    k_w_r = 0.5 / evals[7]
    k_w_m = 5.0
    k_u_c = 1.0

    # compose and apply control action
    for i in nodes:
        control_u = R[i].dot(
            k_u_r * control_u_r[i] + k_u_m * control_u_m[i] + k_u_c * control_u_c[i]
        )
        p_int[i].step(t, control_u)

        control_w = k_w_r * control_w_r[i] + k_w_m * control_w_m[i]
        R_int[i].step_left(t, control_w)

    p = extract_x(p_int)
    R = extract_x(R_int)
    sensing_graph.update(p, R[:, :, 0])


def log_step():
    """Data log"""
    logs.time.append(t)
    logs.position.append(extract_x(p_int).ravel())
    # logs.velocity.append(np.hstack(extract_dotx(p_int)))
    logs.orientation.append(extract_x(R_int).ravel())
    logs.control_u.append(extract_u(p_int).ravel())
    logs.control_w.append(extract_u(R_int).ravel())
    logs.rigidity_val.append(rigidity_val.copy())
    logs.adjacency.append(sensing_graph.adjacency_matrix().ravel())
    logs.target_position.append(targets.position(t).ravel())


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
    print(
        'Simulation length is not a multiple of the step size. \
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

# --- world parameters --- #
found_IAR = False
seed = 0
while not found_IAR:
    np.random.seed(seed)
    seed += 1

    t = 0.0
    n = 8
    nodes = np.arange(n)
    initial_position = np.random.uniform(
        [20.0, 20.0, 20.0],
        [80.0, 80.0, 80.0],
        size=(n, 3)
    )
    initial_orientation = aiming(initial_position, initial_position.mean(axis=0))

    sensing_range = 50.0
    fov = 120.0
    cos_hfov = np.cos(np.deg2rad(fov / 2))
    sensing_graph = ConeGraph(
        initial_position,
        initial_orientation[:, :, 0],    # x-axes
        dmax=sensing_range,
        cmin=cos_hfov
    )

    initial_edge_set = sensing_graph.edge_set()
    initial_angle_set = angle_indices(nodes, initial_edge_set).astype(int)

    # # check if graph is complete
    # if sensing_graph.adjacency_matrix().sum() == n**2 - n:
    #     found_IAR = True
    #     print('seed = {}'.format(seed))
    # check if graph is angle rigid

    if is_angle_rigid(initial_angle_set, initial_position):
        found_IAR = True
        print('seed = {}'.format(seed))

p_int = [
    EulerIntegrator(initial_position[i])
    for i in nodes
]
R_int = [
    EulerIntegratorOrtogonalGroup(initial_orientation[i])
    for i in nodes
]

rigidity_val = np.empty(3, dtype=np.float64)

# --- define targets --- #
targets = MovingTargets({
    0: lambda t: np.array([
        50.0 + 40 * np.cos(0.2 * np.pi * t),
        50.0 + 40 * np.sin(0.2 * np.pi * t) * np.cos(0.02 * np.pi * t),
        50.0 + 40 * np.sin(0.2 * np.pi * t) * np.sin(0.02 * np.pi * t)])
})

tracking_nodes = nodes
rigidity_nodes = nodes
target_allocation = np.zeros(n, dtype=int)

write_json_file(
    'simu_data/targets.jsonlog',
    {
        'ids': list(targets.keys()),
        'tracking_nodes': tracking_nodes.tolist(),
        'alloc': target_allocation.tolist()
    }
)

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
# initialize logs
logs = Logs(
    time=[t],
    position=[extract_x(p_int).ravel()],
    # velocity=[extract_dotx(p_int).ravel()],
    orientation=[extract_x(R_int).ravel()],
    control_u=[],
    control_w=[],
    rigidity_val=[],
    adjacency=[sensing_graph.adjacency_matrix().ravel()],
    target_position=[targets.position(0.0).ravel()]
)

# run simulation
simu_counter = 1
bar = progressbar.ProgressBar(maxval=simu_length).start()

while simu_counter < simu_num_steps:
    t = np.round(t + simu_step_size, 3)

    try:
        simu_step()
        if (simu_counter % log_skip == 0):
            log_step()
    except (ValueError, IndexError, ArithmeticError, KeyboardInterrupt) as e:
        print('Simulation interrupted at t={} sec due to <{}>.'.format(t, e))
        break

    simu_counter += 1

    bar.update(t)

bar.finish()

np.savetxt('simu_data/t.csv', logs.time, delimiter=',')
np.savetxt('simu_data/position.csv', logs.position, delimiter=',')
np.savetxt('simu_data/orientation.csv', logs.orientation, delimiter=',')
np.savetxt('simu_data/control_u.csv', logs.control_u, delimiter=',')
np.savetxt('simu_data/control_w.csv', logs.control_w, delimiter=',')
np.savetxt('simu_data/rigidity_val.csv', logs.rigidity_val, delimiter=',')
np.savetxt('simu_data/adjacency.csv', logs.adjacency, delimiter=',')
np.savetxt('simu_data/target_position.csv', logs.target_position, delimiter=',')
