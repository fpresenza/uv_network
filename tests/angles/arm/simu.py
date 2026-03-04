#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
import progressbar
import numpy as np
from transformations import unit_vector

from uvnpy.dynamics.core import EulerIntegrator
from uvnpy.dynamics.lie_groups import EulerIntegratorOrtogonalGroup
from uvnpy.toolkit.geometry import rotation_matrix_from_quaternion
from uvnpy.toolkit.functions import cosine_activation, cosine_activation_derivative
from uvnpy.graphs.models import ConeGraph
from uvnpy.angles.local_frame.core import angle_indices, angle_rigidity_matrix
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
    target_id: list
    target_position: list


def random_rotation_matrix():
    q = np.random.normal(size=4)
    q /= np.sqrt(q.dot(q))
    return rotation_matrix_from_quaternion(q)


def aim_to_target(p, target):
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
        np.array([x]), 0.0, 50.0
    ).item()


def dsigma_m_f(x):
    return cosine_activation_derivative(
        np.array([x]), -1.0, 1.0
    ).item()


def distance_weights(edge_set, p):
    return np.array([
        np.sqrt(np.square(p[j] - p[i]).sum() * np.square(p[k] - p[i]).sum())
        for i, j, k in angle_indices(n, edge_set).astype(int)
    ])


def weight(indices, p, R):
    i, j, k = indices
    dij = np.sqrt(np.square(p[j] - p[i]).sum())
    bij = unit_vector(p[j] - p[i])
    dik = np.sqrt(np.square(p[k] - p[i]).sum())
    bik = unit_vector(p[k] - p[i])

    ei = R[i, :, 0]
    nij = ei.dot(bij)
    nik = ei.dot(bik)

    wrij = 1 - sigma_r_r(dij)
    wfij = sigma_r_f(nij)
    wrik = 1 - sigma_r_r(dik)
    wfik = sigma_r_f(nik)

    return dij * dik * wrij * wfij * wrik * wfik


def weights(edge_set, p, R):
    return np.array([
        weight((i, j, k), p, R)
        for i, j, k in angle_indices(n, edge_set).astype(int)
    ])


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

    # --- angle rigidity eigenvalue-vector --- #
    edge_set = sensing_graph.edge_set()
    A = angle_rigidity_matrix(edge_set, p)

    #    # --- unweighted part --- #
    W = distance_weights(edge_set, p)
    evals = np.linalg.eigvalsh(A.T.dot(W[:, np.newaxis] * A))
    rigidity_val[0] = evals[7]

    #    # --- weighted part --- #
    W = weights(edge_set, p, R)
    evals, evecs = np.linalg.eigh(A.T.dot(W[:, np.newaxis] * A))
    rigidity_val[1:] = evals[7:9]
    vec = evecs[:, 7].reshape(n, 3)
    vec = np.squeeze(np.matmul(vec[:, np.newaxis, :], R))

    for i in nodes:
        # --- target tracking control --- #
        k_m_r = 2.0
        k_m_f = 1.0
        if i in targets.keys():
            rit = targets[i](t) - p[i]
            dit = np.sqrt(np.square(rit).sum())
            bit = R[i].T.dot(rit / dit)

            Pit = np.eye(3) - np.outer(bit, bit)

            nit = bit[0]
            e1_bit = np.array([0.0, -bit[2], bit[1]])

            control_u_m[i] = k_m_r * dsigma_m_r(dit) * bit \
                - k_m_f * dsigma_m_f(nit) * Pit[0] / dit
            control_w_m[i] = k_m_f * 0.5 * dsigma_m_f(nit) * e1_bit

        # --- rigidity maintenance control --- #
        out_neighbors = edge_set[:, 1][edge_set[:, 0] == i]
        r = {j: p[j] - p[i] for j in out_neighbors}

        #   # --- measurements --- #
        distances = {
            j: np.sqrt(np.square(r[j]).sum()) for j in out_neighbors
        }
        bearings = {
            j: R[i].T.dot(r[j] / distances[j]) for j in out_neighbors
        }
        rotations = {
            j: R[i].T.dot(R[j]) for j in out_neighbors
        }
        # vi = R[i].T.dot(v[i])

        #   # --- control law --- #
        for j, k in complete_angle_set(out_neighbors):
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

            w = wrij * wfij * wrik * wfik
            d = dij * dik
            wr = wrij * wrik
            wf = wfij * wfik

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
            wijk_Ri = 0.5 * d * wr * (
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

            sijk_i = (Dijk - Eikj).dot(vec_j - vec_i) + (Dikj - Eijk).dot(vec_k - vec_i)
            sijk_j = - Dijk.dot(vec_j - vec_i) + Eijk.dot(vec_k - vec_i)
            sijk_k = Eikj.dot(vec_j - vec_i) - Dikj.dot(vec_k - vec_i)

            control_u_r[i] += sijk * (sijk * wijk_i + 2 * wijk * sijk_i)
            control_u_r[j] += sijk * Rij.T.dot(sijk * wijk_j + 2 * wijk * sijk_j)
            control_u_r[k] += sijk * Rik.T.dot(sijk * wijk_k + 2 * wijk * sijk_k)

            control_w_r[i] += sijk**2 * wijk_Ri

        # control_u_r[i] -= kd * vi

    # define relative weights
    k_u_r = 1.0 / evals[7]
    k_u_m = 20.0
    k_w_r = 0.1 / evals[7]
    k_w_m = 2.0

    # compose and apply control action
    for i in nodes:
        control_u = R[i].dot(
            k_u_r * control_u_r[i] + k_u_m * control_u_m[i]
        )
        p_int[i].step(t, control_u)

        control_w = R[i].dot(
            k_w_r * control_w_r[i] + k_w_m * control_w_m[i]

        )
        R_int[i].step(t, control_w)

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

found_IAR = False
seed = 0
while not found_IAR:
    np.random.seed(seed)
    seed += 1

    # --- world parameters --- #
    t = 0.0
    n = 4
    nodes = np.arange(n)
    initial_position = np.random.normal(loc=15.0, scale=15.0, size=(n, 3))
    initial_orientation = aim_to_target(initial_position, initial_position.mean(axis=0))

    sensing_range = 30.0
    fov = 120.0
    cos_hfov = np.cos(np.deg2rad(fov / 2))
    sensing_graph = ConeGraph(
        initial_position,
        initial_orientation[:, :, 0],    # axes
        dmax=sensing_range,
        cmin=cos_hfov
    )

    initial_edge_set = sensing_graph.edge_set()
    initial_angle_set = angle_indices(n, initial_edge_set).astype(int)

    # check if graph is complete
    if sensing_graph.adjacency_matrix().sum() == n**2 - n:
        found_IAR = True
        print(initial_edge_set)
        print(initial_angle_set)

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
    2: lambda t: np.array([t, 0.0, 20.0])
})

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
    target_id=list(targets.keys()),
    target_position=[targets.position(0.0).ravel()]
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
# np.savetxt('simu_data/velocity.csv', logs.velocity, delimiter=',')
np.savetxt('simu_data/orientation.csv', logs.orientation, delimiter=',')
np.savetxt('simu_data/control_u.csv', logs.control_u, delimiter=',')
np.savetxt('simu_data/control_w.csv', logs.control_u, delimiter=',')
np.savetxt('simu_data/rigidity_val.csv', logs.rigidity_val, delimiter=',')
np.savetxt('simu_data/adjacency.csv', logs.adjacency, delimiter=',')
np.savetxt('simu_data/target_id.csv', logs.target_id, delimiter=',')
np.savetxt('simu_data/target_position.csv', logs.target_position, delimiter=',')
