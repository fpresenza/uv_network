#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import collections
import numpy as np
import matplotlib.pyplot as plt

from gpsic.plotting.planar import agregar_ax

# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple('Logs', 'x')


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(tiempo, logs, *args):
    # iteración
    # for t in tiempo[1:]:
    #     logs.x[t] = 0

    # return logs
    return logs


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Parseo de argumentos
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-s', '--step',
        dest='h', default=50e-3, type=float, help='paso de simulación')
    parser.add_argument(
        '-t', '--ti',
        metavar='T0', default=0.0, type=float, help='tiempo inicial')
    parser.add_argument(
        '-e', '--tf',
        default=1.0, type=float, help='tiempo final')
    parser.add_argument(
        '-r', '--arxiv',
        default='/tmp/config.yaml', type=str, help='arhivo de configuración')
    parser.add_argument(
        '-g', '--save',
        default=False, action='store_true',
        help='flag para guardar archivos')

    arg = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------
    tiempo = np.arange(arg.ti, arg.tf, arg.h)
    logs = Logs(np.empty(len(tiempo)))

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    logs = run(tiempo, logs)
    x = logs.x

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(13, 5))
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    gs = fig.add_gridspec(2, 2)

    ax = agregar_ax(
        gs[0, 0],
        title='x vs. t', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel='', label_kw={'fontsize': 10})
    ax.plot(tiempo, x, color='r', label='$x$')

    if arg.save:
        fig.savefig('/tmp/ensayo.pdf', format='pdf')
    else:
        plt.show()
