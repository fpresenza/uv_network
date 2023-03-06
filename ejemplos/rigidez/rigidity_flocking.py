#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import collections
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.rsn import rigidity
from uvnpy.network import disk_graph, plot
from uvnpy.network import edges_from_adjacency

import matplotlib
print(matplotlib.__version__)

# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple('Logs', 'position velocity energy frames')


def compute_energy(A, x, v):
    L = rigidity.symmetric_matrix(A, x)
    E = v.ravel().dot(L).dot(v.ravel())
    return E


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(timesteps, logs, adjacency_matrix):
    # iteración
    # x = logs.position[0]
    # L = rigidity.symmetric_matrix(adjacency_matrix, x)

    for k, t in enumerate(timesteps[1:], start=1):
        x = logs.position[k - 1]
        v = logs.velocity[k - 1]
        dt = timesteps[k] - timesteps[k - 1]

        L = rigidity.symmetric_matrix(adjacency_matrix, x)
        a = -0.5 * L.dot(v.reshape(-1, 1)).reshape(n, d)
        v = v + dt * a
        x = x + dt * v

        logs.position[k] = x.copy()
        logs.velocity[k] = v.copy()
        logs.energy[k] = compute_energy(adjacency_matrix, x, v)
        # logs.energy[k] = v.ravel() @ L @ v.ravel()
        logs.frames[k] = t, x.copy(), edges_from_adjacency(adjacency_matrix)

    return logs


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Parseo de argumentos
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-s', '--step',
        default=50e-3, type=float, help='simulation step')
    parser.add_argument(
        '-i', '--ti',
        default=0.0, type=float, help='initialization time')
    parser.add_argument(
        '-f', '--tf',
        default=1.0, type=float, help='finalization time')

    arg = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------
    n, d = 8, 2
    timesteps = np.arange(arg.ti, arg.tf, arg.step)
    logs = Logs(
        position=np.empty((len(timesteps), n, d)),
        velocity=np.empty((len(timesteps), n, d)),
        energy=np.empty(len(timesteps)),
        frames=np.empty(len(timesteps), dtype=np.ndarray)
    )

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    position = np.random.uniform(-1, 1, (n, d))
    velocity = np.random.uniform(-1, 1, (n, d))
    adjacency_matrix = disk_graph.adjacency(position, 1.5)
    edges = edges_from_adjacency(adjacency_matrix)

    energy = compute_energy(adjacency_matrix, position, velocity)
    if rigidity.algebraic_condition(adjacency_matrix, position) is False:
        raise ValueError('Flexible Framework')

    logs.position[0] = position
    logs.velocity[0] = velocity
    logs.energy[0] = energy
    logs.frames[0] = timesteps[0], position, edges

    logs = run(timesteps, logs, adjacency_matrix)

    # print(logs.frames[0], logs.frames[-1])
    print(logs.position[0])
    print(logs.position[-1])
    print(logs.velocity[0])
    print(logs.velocity[-1])
    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig, ax = plot.figure()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    anim = plot.Animate(fig, ax, arg.step, logs.frames)
    anim.run(file='/tmp/anim.mp4')

    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(timesteps, logs.energy)
    plt.show()
