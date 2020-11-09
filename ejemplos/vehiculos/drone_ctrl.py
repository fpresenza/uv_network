#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Thu Jul 7 17:21:34 2020
@author: fran
"""
import argparse
import numpy as np

from uvnpy.modelos.drone import Drone
# import gpsic.plotting.planar as planar
# from gpsic.plotting. import Animation3D


def run(arg):
    tsim = np.arange(arg.ti + arg.h, arg.tf, arg.h)

    P, V, A = ([], []), ([], []), ([], [])
    W, R, U = ([], []), ([], []), ([], [])
    G = ([], [])

    Tc = 1/arg.f_ctrl
    tc = tsim[0] + Tc
    tctrl = np.arange(arg.ti + Tc, arg.tf, Tc)

    for t in tsim:
        # simulation of dynamics
        wind = (0., 0., 0.)
        T = 5
        w = 2 * np.pi/T
        r = (-np.sin(w * t)/w, np.cos(w * t)/w, 0, 0)
        # r = (0,0,0,0)
        uav[0].sim_step(r, t, fw=wind)
        # r = (-np.sin(w*t)/w, np.cos(w*t)/w, 0.,0.)
        r = (0, 0, 0, 0)
        uav[1].sim_step(r, t, fw=wind)

        if t > tc:
            points = rover
            uav[0].ctrl_step(t, points)
            uav[0].filter.save()
            uav[1].ctrl_step(t, points)
            uav[1].filter.save()
            P[0].append(uav[0].motion.p())
            V[0].append(uav[0].motion.v())
            A[0].append(uav[0].motion.euler())
            W[0].append(uav[0].motion.w())
            R[0].append(uav[0].motion.ref())
            U[0].append(uav[0].motion.ctrl_eff())
            G[0].append(uav[0].cam.attitude)

            P[1].append(uav[1].motion.p())
            V[1].append(uav[1].motion.v())
            A[1].append(uav[1].motion.euler())
            W[1].append(uav[1].motion.w())
            R[1].append(uav[1].motion.ref())
            U[1].append(uav[1].motion.ctrl_eff())
            G[1].append(uav[1].cam.attitude)
            tc += Tc

    return tctrl, P, V, A, W, R, U, G, (uav[0].filter.log, uav[1].filter.log)


if __name__ == '__main__':
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
      default=20.0, type=float, help='frecuencia del controlador')
    parser.add_argument(
      '-g', '--save',
      default=False, action='store_true',
      help='flag para guardar los videos')
    parser.add_argument(
      '-a', '--animate',
      default=False, action='store_true',
      help='flag para generar animaicion 3D')

    arg = parser.parse_args()

    rover = [[0., 5., 0.], [0., -5., 0.]]

    uav = (
        Drone(
          1,
          motion_kw={
            'ti': arg.ti,
            'pi': (1., 5., 5.),
            'vi': (0, 0.5, 0),
            'ai': (0., 0., 0.)}),
        Drone(
          2,
          motion_kw={
            'ti': arg.ti,
            'pi': (0., -5., 5.),
            'vi': (0, 0, 0),
            'ai': (0., 0., 0.)})
    )

    tctrl, P, V, A, W, R, U, G, f = run(arg)

    # plot = []
    # # filtro del uav
    # color = ('r', 'g', 'b')
    # label = ('x', 'y', 'z')

    # f_x = np.vstack(f[0].x).T
    # lines = f_x[:3], f_x[3:]
    # plot += [planar.GridPlot(shape=(2,1), title='Multicopter 1 - filter')]
    # plot[0].draw(tctrl, lines, color=[color,color], label=[label,label])

    # dyh, dyv = np.vstack(f[0].dy).T
    # plot += [planar.GridPlot(shape=(1,1),
    #     title='Autocorrelación 1 - Innovación', xlabel='$\delta y$')]
    # plot[1].axes[0,0].acorr(dyh, usevlines=False,
    #     color='r', linestyle="-", marker=".", linewidth=0.75, label='h')
    # plot[1].axes[0,0].acorr(dyv, usevlines=False,
    #     color='g', linestyle="-", marker=".", linewidth=0.75, label='v')
    # plot[1].fig.legend()

    # f_x = np.vstack(f[1].x).T
    # lines = f_x[:3], f_x[3:]
    # plot += [planar.GridPlot(shape=(2,1), title='Multicopter 2 - filter')]
    # plot[2].draw(tctrl, lines, color=[color,color], label=[label,label])

    # dyh, dyv = np.vstack(f[1].dy).T
    # plot += [planar.GridPlot(shape=(1,1),
    #     title='Autocorrelación 2 - Innovación', xlabel='$\delta y$')]
    # plot[3].axes[0,0].acorr(dyh, usevlines=False, color='r',
    #     linestyle="-", marker=".", linewidth=0.75, label='h')
    # plot[3].axes[0,0].acorr(dyv, usevlines=False, color='g',
    #     linestyle="-", marker=".", linewidth=0.75, label='v')
    # plot[3].fig.legend()

    # if arg.save:
    #     plot[0].savefig('uav_1_filter')
    #     plot[1].savefig('uav_2_filter')
    #     plot[2].savefig('uav_1_acorr')
    #     plot[3].savefig('uav_2_acorr')
    # else:
    #     plot[0].show()

    # if arg.animate:
    #     ani = Animation3D(tctrl, step=1,
    #     save=True, plot_kw={'xlim':(-15,15), 'ylim':(-15,15), 'zlim':(0,15)})
    #     ani.add_drone(uav[0].id, P[0], A[0], (G[0],), camera=uav[0].cam)
    #     ani.add_drone(uav[1].id, P[1], A[1], (G[1],), camera=uav[1].cam)
    #     ani.add_sphere([rover[0] for k in P[0]], [1 for k in P[0]])
    #     ani.add_sphere([rover[1] for k in P[1]], [1.5 for k in P[1]])
    #     ani.run()
