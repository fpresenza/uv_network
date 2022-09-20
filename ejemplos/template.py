#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import collections
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple('Logs', 'data')


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(timesteps, logs, *args):
    # iteración
    # for t in timesteps[1:]:
    #     logs.data[t] = 0

    # return logs
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
    timesteps = np.arange(arg.ti, arg.tf, arg.step)
    logs = Logs(data=np.empty(len(timesteps)))

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    logs = run(timesteps, logs)
    data = logs.data

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(timesteps, data)

    if arg.save:
        fig.savefig('/tmp/ensayo.pdf', format='pdf')
    else:
        plt.show()
