#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Thu May 21 18:34:16 2020
@author: fran
"""
import argparse
import numpy as np

from gpsic.modelos.multicoptero import Multicoptero


def run(arg):
    uav = (
        Multicoptero(
            ti=arg.ti, pi=(0., 3., 5.), vi=(0., 0., 0.),
            ei=(0., 0., 0.), f_ctrl=arg.f_ctrl),
        Multicoptero(
            ti=arg.ti, pi=(0., -3., 5.), vi=(0., 0., 0.),
            ei=(0., 0., np.pi/2), f_ctrl=arg.f_ctrl)
    )
    time = np.arange(arg.ti+arg.h, arg.tf, arg.h)

    P, V, A = ([], []), ([], []), ([], [])
    W, R, U = ([], []), ([], []), ([], [])
    G = ([], [])

    for t in time:
        # r = (2*np.sin(t/2), 0., 0., 0.) #(vx, vy, vz, yaw)
        r = (1, 0, 0, 0.5)
        uav[0].step(t, r)
        P[0].append(uav[0].p)
        V[0].append(uav[0].v)
        A[0].append(uav[0].euler)
        W[0].append(uav[0].w)
        R[0].append(r)
        U[0].append(uav[0].u)
        G[0].append(np.array([0, np.pi/4, 0]))

        r = (0, 2 * np.sin(t/2.), 0., 0.)  # (vx, vy, vz, yaw)
        # r = (0.,0.,0.,0.)
        uav[1].step(t, r)
        P[1].append(uav[1].p)
        V[1].append(uav[1].v)
        A[1].append(uav[1].euler)
        W[1].append(uav[1].w)
        R[1].append(r)
        U[1].append(uav[1].u)
        G[1].append(np.array([0, np.pi/7, 0]))

    return time, P, V, A, W, R, U, G


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Parseo de argumentos
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-s', '--step',
        dest='h', default=1e-3, type=float, help='paso de simulación')
    parser.add_argument(
        '-t', '--ti',
        metavar='T0', default=0.0, type=float, help='tiempo inicial')
    parser.add_argument(
        '-e', '--tf',
        default=1.0, type=float, help='tiempo final')
    parser.add_argument(
        '-f', '--f_ctrl',
        default=50.0, type=float, help='frecuencia del controlador')
    parser.add_argument(
        '-g', '--save',
        default=False, action='store_true',
        help='flag para guardar los videos')
    parser.add_argument(
        '-a', '--animate',
        default=False, action='store_true',
        help='flag para generar animaicion 3D')

    arg = parser.parse_args()

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    time, P, V, A, W, R, U, G = run(arg)
