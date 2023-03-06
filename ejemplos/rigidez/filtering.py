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


# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple(
    'Logs', 'position velocity estimation error frames')
# np.random.seed(0)


def graph_incidence(edges, position):
    n, d = position.shape
    e = len(edges)

    J = np.zeros((d*e, d*n))

    for k, (i, j) in enumerate(edges):
        grad = np.eye(d)
        J[d*k:d*k + d, d*i:d*i + d] = grad
        J[d*k:d*k + d, d*j:d*j + d] = -grad

    return J


def framework_incidence(edges, position, measurement):
    n, d = position.shape
    e = len(edges)
    J = np.empty(e, dtype=np.ndarray)

    if measurement == 'distance':
        grad = distance_gradient
    elif measurement == 'inner_product':
        grad = inner_product_gradient
    elif measurement == 'bearing':
        grad = bearing_gradient

    for k, (i, j) in enumerate(edges):
        J[k] = grad(i, j, position)

    return np.vstack(J)


def relative_function(edges, position, measurement):
    if measurement == 'distance':
        measure = distance
    elif measurement == 'inner_product':
        measure = inner_product
    elif measurement == 'bearing':
        measure = bearing
    return measure(edges, position)


def distance(edges, position):
    dist = np.empty(len(edges))

    for k, (i, j) in enumerate(edges):
        diff = position[i] - position[j]
        dist[k] = np.sqrt(diff.dot(diff))

    return dist


def distance_gradient(i, j, position):
    n, d = position.shape
    J = np.zeros(d*n)
    diff = position[i] - position[j]
    dist = np.sqrt(diff.dot(diff))
    grad = diff / dist
    J[d*i:d*i + d] = grad
    J[d*j:d*j + d] = -grad

    return J


def inner_product(edges, position):
    dot = np.empty(len(edges))

    for k, (i, j) in enumerate(edges):
        dot[k] = position[i].dot(position[j])

    return dot


def inner_product_gradient(i, j, position):
    n, d = position.shape
    J = np.zeros(d*n)
    J[d*i:d*i + d] = position[j]
    J[d*j:d*j + d] = position[i]

    return J


def bearing():
    return


def bearing_gradient():
    return


def linear_coefficients(t, r):
    return np.linspace(t, 1 - t, r)


def quadratic_coefficients(t, r):
    return np.linspace(t, 1 - t, r)**2


def apply_filter(signal, t, incidence):
    J = incidence
    L = J.T.dot(J)
    l, U = np.linalg.eigh(L)
    r = sum(l > 1e-8)
    h = linear_coefficients(t, r)
    V = U[:, -r:]
    H = V.dot(np.diag(h)).dot(V.T)
    return H.dot(signal)


def total_variation(signal):
    return np.sum(np.abs(np.diff(signal)))


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(timesteps, logs, adjacency_matrix, measurement, filter=None):
    # iteración
    # x = logs.position[0]
    # L = rigidity.symmetric_matrix(adjacency_matrix, x)
    N = len(timesteps)
    edges = edges_from_adjacency(adjacency_matrix)
    # n = len(adjacency_matrix)

    for k, t in enumerate(timesteps[1:], start=1):
        x = logs.position[k - 1]
        s = logs.estimation[k - 1]
        error_position = logs.error[k - 1]
        dt = timesteps[k] - timesteps[k - 1]

        Ig = graph_incidence(edges, s)
        If = framework_incidence(edges, s, measurement)

        phi = relative_function(edges, s, measurement)
        z = relative_function(edges, x, measurement)
        zn = np.random.normal(loc=z, scale=0.3)    # add noise

        v = If.T.dot(zn - phi)    # velocity
        if filter == 'graph':
            v = apply_filter(v, k/(N-1), Ig)
        elif filter == 'framework':
            v = apply_filter(v, k/(N-1), If)

        v = v.reshape(x.shape)
        s = s + 0.5 * dt * v
        # error_relative = np.sum((z - phi)**2)
        error_position = np.sum((x - s)**2)

        logs.position[k] = x.copy()
        logs.velocity[k] = v.copy()
        logs.estimation[k] = s.copy()
        logs.error[k] = error_position.copy()
        logs.frames[k] = t, s.copy(), edges

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
    n, d = 10, 2
    timesteps = np.arange(arg.ti, arg.tf, arg.step)

    logs_r = Logs(
        position=np.empty((len(timesteps), n, d)),
        velocity=np.empty((len(timesteps), n, d)),
        estimation=np.empty((len(timesteps), n, d)),
        error=np.empty((len(timesteps),)),
        frames=np.empty(len(timesteps), dtype=np.ndarray)
    )

    logs_g = Logs(
        position=np.empty((len(timesteps), n, d)),
        velocity=np.empty((len(timesteps), n, d)),
        estimation=np.empty((len(timesteps), n, d)),
        error=np.empty((len(timesteps),)),
        frames=np.empty(len(timesteps), dtype=np.ndarray)
    )

    logs_f = Logs(
        position=np.empty((len(timesteps), n, d)),
        velocity=np.empty((len(timesteps), n, d)),
        estimation=np.empty((len(timesteps), n, d)),
        error=np.empty((len(timesteps),)),
        frames=np.empty(len(timesteps), dtype=np.ndarray)
    )

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    # position = np.array([
    #     [ 0.08509, -0.71566],
    #     [-0.25332,  0.34827],
    #     [-0.11633, -0.13197],
    #     [ 0.23553,  0.02628],
    #     [ 0.30079,  0.20208],
    #     [ 0.61045,  0.04329],
    #     [ 0.8173 , -0.36153],
    #     [-0.81908, -0.3986 ],
    #     [-0.77203,  0.65736],
    #     [-0.90621,  0.25257]])

    position = np.random.uniform(-1, 1, (n, d))
    print(position)
    adjacency_matrix = disk_graph.adjacency(position, 0.8)
    print('RIGID: ', rigidity.algebraic_condition(adjacency_matrix, position))

    edges = edges_from_adjacency(adjacency_matrix)

    velocity = np.zeros((n, d))

    estimation = np.random.normal(loc=position, scale=0.2, size=(n, d))
    z = relative_function(edges, position, measurement='distance')
    phi = relative_function(edges, estimation, measurement='distance')

    error = (z - phi).dot(z - phi)

    logs_r.position[0] = position
    logs_r.velocity[0] = velocity
    logs_r.estimation[0] = estimation
    logs_r.error[0] = error
    logs_r.frames[0] = timesteps[0], estimation, edges

    logs_g.position[0] = position
    logs_g.velocity[0] = velocity
    logs_g.estimation[0] = estimation
    logs_g.error[0] = error
    logs_g.frames[0] = timesteps[0], estimation, edges

    logs_f.position[0] = position
    logs_f.velocity[0] = velocity
    logs_f.estimation[0] = estimation
    logs_f.error[0] = error
    logs_f.frames[0] = timesteps[0], estimation, edges

    logs_r = run(
        timesteps, logs_r, adjacency_matrix,
        measurement='distance', filter=None)
    logs_g = run(
        timesteps, logs_g, adjacency_matrix,
        measurement='distance', filter='graph')
    logs_f = run(
        timesteps, logs_f, adjacency_matrix,
        measurement='distance', filter='framework')

    # print(logs_g.position[-1])
    # print(logs_g.estimation[-1])
    # print(logs_g.error[0])
    # print(logs_g.error[-1])
    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    print('TOTAL VARIATION:')
    print('error-raw: ', total_variation(logs_r.error))
    print('error-graph: ', total_variation(logs_g.error))
    print('error-framework: ', total_variation(logs_f.error))

    fig, ax = plt.subplots()
    ax.grid()
    ax.semilogy(timesteps, logs_r.error, label='raw')
    ax.semilogy(timesteps, logs_g.error, label='graph')
    ax.semilogy(timesteps, logs_f.error, label='framework')
    ax.legend()
    fig.savefig('/tmp/error.pdf', format='pdf')

    fig, ax = plot.figure()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plot.graph(ax, position, adjacency_matrix)
    plot.graph(ax, estimation, adjacency_matrix)
    plot.graph(ax, logs_f.estimation[-1], adjacency_matrix)

    plt.show()

    # fig, ax = plot.figure()
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-2, 2)
    # plot.nodes(ax, position, marker='+', color='r')
    # plot.edges(ax, position, adjacency_matrix, color='r', linewidth=0.3)
    # anim = plot.Animate(fig, ax, arg.step, logs_g.frames)
    # anim.run(file='/tmp/anim.mp4')
