#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import collections
import progressbar
import numpy as np

from uvnpy.dynamics.core import EulerIntegrator
from uvnpy.toolkit.geometry import rotation_matrix_from_vector


# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
np.set_printoptions(
    suppress=True,
    precision=10
)

Logs = collections.namedtuple(
    'Logs',
    'time, frames, est_frames'
)


def gram_schmidt(V):
    Q, _ = np.linalg.qr(V, mode='complete')
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1.0
    return Q


def random_rotation_vector(norm=np.pi):
    v_norm = np.inf
    while v_norm > norm:
        # random direction (uniform on sphere after normalization)
        v = np.random.uniform(-norm, norm, (3,))
        v_norm = np.sqrt(v.dot(v))

    return v


def Trinh2018CCTA(Rij, Ri_hat, Rj_hat):
    """
    2018 CCTA
    Trinh, Lee, Ye, Ahn
    Bearing-based Formation Control and Network Localization
    via Global Orientation Estimation
    """
    return np.dot(Rij, Rj_hat) - Ri_hat


def Li2019TCNS(Rij, Ri_hat, Rj_hat):
    """
    2019 IEEE T-CNS
    Li, Luo, Zhao
    Globally Convergent Distributed Network Localization
    Using Locally Measured Bearings
    """
    return np.dot(Rj_hat, Rij.T) - Ri_hat


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run_simu(simu_counter, end_counter):
    bar = progressbar.ProgressBar(maxval=arg.simu_time).start()
    while simu_counter < end_counter:
        t = time_steps[simu_counter]

        # -- Orientation estimation algorithm -- #
        for i in nodes:
            Ri = frames[i]
            Ri_hat = est_frames[i].x().copy()
            u = np.zeros((3, 3), dtype=float)
            for j in edge_set[edge_set[:, 0] == i, 1]:
                # for j in edge_set[edge_set[:, 1] == i, 0]:
                Rj = frames[j]
                Rj_hat = est_frames[j].x()
                Rij = np.dot(Ri.T, Rj)
                u += Trinh2018CCTA(Rij, Ri_hat, Rj_hat)
                # u += Li2019TCNS(Rij, Ri_hat, Rj_hat)
            est_frames[i].step(t, u)
            # est_frames[i].initialize(
            #     gram_schmidt(est_frames[i].x()), t=t, u=u
            # )

        # -- Data log -- #
        if (simu_counter % log_skip == 0):
            logs.time.append(t)
            logs.frames.append(np.hstack(
                [frame.ravel() for frame in frames]
            ))
            # logs.est_frames.append(np.hstack(
            #     [gram_schmidt(frame.x()).ravel() for frame in est_frames]
            # ))
            logs.est_frames.append(np.hstack(
                [frame.x().ravel() for frame in est_frames]
            ))

        simu_counter += 1

        bar.update(np.round(t, 3))

    bar.finish()

    return simu_counter


# ------------------------------------------------------------------
# Parseo de argumentos
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-s', '--simu_step',
    default=1, type=float, help='simulation step in milli seconds'
)
parser.add_argument(
    '-t', '--simu_time',
    default=1.0, type=float, help='total simulation time in seconds'
)
parser.add_argument(
    '-l', '--log_skip',
    default=1, type=int, help='logger skip in number of simu_step'
)

arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
# Simulation parameters
simu_time = arg.simu_time
simu_step = arg.simu_step / 1000.0
n_steps = int(simu_time / simu_step)
time_steps = [simu_step * k for k in range(n_steps)]
log_skip = arg.log_skip

# np.random.seed(6)

print(
    'Simulation Time: begin = {}, end = {}, step = {} sec'
    .format(0.0, simu_time, simu_step)
)

# world parameters

edge_set = np.array([
    [0, 1],
    # [1, 2],
])
nodes = np.unique(edge_set)
frames = [
    rotation_matrix_from_vector(random_rotation_vector())
    for i in nodes
]
est_frames = [
    EulerIntegrator(np.eye(3))
    for i in nodes
]
print([frame for frame in frames])

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
logs = Logs(
    time=[],
    frames=[],
    est_frames=[]
)

simu_counter = 1
simu_counter = run_simu(simu_counter, end_counter=n_steps)

np.savetxt('simu_data/t.csv', logs.time, delimiter=',')
np.savetxt('simu_data/frames.csv', logs.frames, delimiter=',')
np.savetxt('simu_data/est_frames.csv', logs.est_frames, delimiter=',')
