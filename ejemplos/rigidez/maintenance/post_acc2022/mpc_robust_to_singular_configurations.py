#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import collections
import progressbar
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from uvnpy.network import plot as netplot
from uvnpy.network import disk_graph
from uvnpy.rsn import distances, rigidity
from uvnpy.toolkit import functions


"""
Este script permite simular una ley de control con ventana de optimizacion
para poder sortear las singularidades locales que surgen de configuraciones
no regulares del grafo en R^d, donde el autovalor de rigidez se vuelve cero,
aun cuando el grafo es rigido. En el caso de una ley de control simple por
gradiente descendiente del inverso del autovalor de rigidez, dichas
configuraciones no regulares son evadidas como una colision, lo cual no es
de deseable ya que limita la movilidad de los vehiculos innecesariamente.

"""


# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple(
    'Logs',
    'position, adjacency, ctrl_eff, cmd_vel, rigidity_eigenvalue')

DebugLogs = collections.namedtuple(
    'DebugLogs',
    'rigidity_constraint')


def smooth_disk_proximity_adjacency(position):
    midpoint, steepness = 8, 5
    A = distances.matrix(position)
    A[A > 0] = functions.logistic(A[A > 0], midpoint, steepness)
    return A


class SimpleIntegrator(object):
    def __init__(self, position=None):
        self.position = position

    def step(self, velocity, step_size):
        self.position += velocity * step_size

    def prediction(self, velocity, window):
        W = window.reshape(-1, 1, 1)
        return self.position + W*velocity


class Control(object):
    def __init__(
            self,
            position,
            window,
            ctrl_eff_shape,
            agent_model,
            adjacency_model):
        self.n, self.d = position.shape
        delta = min(self.n - 1, self.d)
        self.rigidity_eigenvalue_index = int((delta + 1)*(2*self.d - delta)/2)
        self.window = window
        self.ctrl_eff_shape = ctrl_eff_shape
        self.ctrl_eff = np.zeros(ctrl_eff_shape)
        self.agent_model = agent_model(position)
        self.adjacency_model = adjacency_model
        self.constraint_threshold = 1e-2
        self.debuglogs = None

    def prediction(self, ctrl_eff):
        return self.agent_model.prediction(ctrl_eff, self.window)

    def functional(self, ctrl_eff, cmd_vel):
        cmd_err = ctrl_eff - cmd_vel.ravel()
        cmd_cost = cmd_err.dot(cmd_err)
        ctrl_cost = ctrl_eff.dot(ctrl_eff)
        return cmd_cost + 0.1*ctrl_cost

    def rigidity_constraint(self, ctrl_eff):
        ctrl_eff = ctrl_eff.reshape(self.ctrl_eff_shape)
        predicted_position = self.prediction(ctrl_eff)
        predicted_adjacency = self.adjacency_model(predicted_position)
        rigidity_matrix = rigidity.symmetric_matrix(
            predicted_adjacency, predicted_position)
        eigenvalues = np.linalg.eigvalsh(rigidity_matrix)
        lambdas = eigenvalues[:, self.rigidity_eigenvalue_index]
        constraint = sum(lambdas)
        self.debuglogs = constraint
        return constraint - self.constraint_threshold

    def update(self, position, cmd_vel):
        self.agent_model.__init__(position)

        constraints = {'type': 'ineq', 'fun': self.rigidity_constraint}
        self.debuglogs = None

        optimization = scipy.optimize.minimize(
            self.functional,
            self.ctrl_eff.ravel(),
            # cmd_vel.ravel(),
            args=(cmd_vel, ),
            constraints=constraints,
            method='SLSQP',
        )

        if not optimization.success:
            print(optimization)
        self.ctrl_eff[:] = optimization.x.reshape(self.ctrl_eff.shape)

# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(simulation_time, position, adjacency, control, logs, debuglogs, bar):
    bar.start()

    formation = SimpleIntegrator(position)

    for k in range(1, len(simulation_time)):
        t, t_prev = simulation_time[k], simulation_time[k-1]

        cmd_vel = np.array([
            [0, 0],
            [0, -formation.position[1, 1]],
            [0, 0]], dtype=float)
        control.update(position, cmd_vel)
        formation.step(control.ctrl_eff, t - t_prev)
        adjacency = disk_graph.adjacency(formation.position, 10)

        logs.position[k] = formation.position.copy()
        logs.adjacency[k] = adjacency.copy()
        logs.ctrl_eff[k] = control.ctrl_eff.copy()
        logs.cmd_vel[k] = cmd_vel.copy()
        logs.rigidity_eigenvalue[k] = rigidity.eigenvalue(
            adjacency, logs.position[k])

        debuglogs.rigidity_constraint[k] = control.debuglogs

        bar.update(np.round(t, 3))

    bar.finish()
    return logs, debuglogs


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
    parser.add_argument(
        '-r', '--arxiv',
        default='/tmp/config.yaml', type=str, help='config arxiv')
    parser.add_argument(
        '-a', '--save',
        default=False, action='store_true', help='flag to store data')

    arg = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------
    simulation_time = np.arange(arg.ti, arg.tf, arg.step)
    Nsteps = len(simulation_time)

    position = np.array([
        [-1, 0],
        [0, 2],
        [1, 0]], dtype=float)
    adjacency = 1 - np.eye(len(position))
    # window = np.array([0.05, 0.1, 0.2])
    window = np.array([0.2, 0.4, 0.6])

    control = Control(
        position,
        window,
        ctrl_eff_shape=position.shape,
        agent_model=SimpleIntegrator,
        adjacency_model=smooth_disk_proximity_adjacency)

    bar = progressbar.ProgressBar(maxval=arg.tf)

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    logs = Logs(
        position=np.empty((Nsteps,) + position.shape),
        adjacency=np.empty((Nsteps,) + adjacency.shape),
        ctrl_eff=np.empty((Nsteps,) + position.shape),
        cmd_vel=np.empty((Nsteps,) + position.shape),
        rigidity_eigenvalue=np.empty(Nsteps))
    logs.position[0] = position
    logs.adjacency[0] = adjacency
    logs.ctrl_eff[0][:] = 0
    logs.cmd_vel[0][:] = 0
    logs.rigidity_eigenvalue[0] = rigidity.eigenvalue(adjacency, position)

    debuglogs = DebugLogs(rigidity_constraint=np.empty(Nsteps))
    debuglogs.rigidity_constraint[0] = None

    logs, debuglogs = run(
        simulation_time, position, adjacency, control, logs, debuglogs, bar)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    fig.suptitle('Position')
    ax[0].grid()
    ax[1].grid()
    ax[0].set_xlabel('seconds')
    ax[1].set_xlabel('seconds')
    ax[0].set_ylabel('$x$-coordinate')
    ax[1].set_ylabel('$y$-coordinate')
    ax[0].set_prop_cycle('color', ('r', 'g', 'b'))
    ax[1].set_prop_cycle('color', ('r', 'g', 'b'))
    ax[0].plot(simulation_time, logs.position[:, :, 0])
    ax[1].plot(simulation_time, logs.position[:, :, 1])

    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    fig.suptitle('Control Effort')
    ax[0].grid()
    ax[1].grid()
    ax[0].set_xlabel('seconds')
    ax[1].set_xlabel('seconds')
    ax[0].set_ylabel('$x$-coordinate')
    ax[1].set_ylabel('$y$-coordinate')
    ax[0].set_ylim(-2, 2)
    ax[1].set_ylim(-2, 2)
    ax[0].set_prop_cycle('color', ('r', 'g', 'b'))
    ax[1].set_prop_cycle('color', ('r', 'g', 'b'))
    ax[0].plot(simulation_time, logs.ctrl_eff[:, :, 0])
    ax[0].plot(simulation_time, logs.cmd_vel[:, :, 0], ls='--')
    ax[1].plot(simulation_time, logs.ctrl_eff[:, :, 1])
    ax[1].plot(simulation_time, logs.cmd_vel[:, :, 1], ls='--')

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle('Rigidity Eigenvalue')
    ax.grid()
    ax.set_xlabel('seconds')
    ax.set_ylabel(r'$\lambda_{D + 1}$')
    ax.plot(simulation_time, logs.rigidity_eigenvalue, label='real')
    ax.plot(
        simulation_time, debuglogs.rigidity_constraint/len(window),
        label='predicted')
    ax.hlines(
        control.constraint_threshold/len(window), 0, simulation_time[-1],
        color='k', ls='--')
    ax.legend()

    fig, ax = netplot.figure()
    frames = 0, int(Nsteps/3)-1, 2*int(Nsteps/3)-1, 3*int(Nsteps/3)-1
    netplot.graph(ax, logs.position[frames[0]], logs.adjacency[frames[0]])
    netplot.graph(ax, logs.position[frames[1]], logs.adjacency[frames[1]])
    netplot.graph(ax, logs.position[frames[2]], logs.adjacency[frames[2]])
    netplot.graph(ax, logs.position[frames[3]], logs.adjacency[frames[3]])

    plt.show()
