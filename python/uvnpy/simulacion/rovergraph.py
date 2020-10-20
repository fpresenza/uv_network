#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
"""
import argparse
import numpy as np
import collections

import uvnpy.network.graph as graph
from uvnpy.navigation.metricas import metricas
# import gpsic.toolkit.linalg as linalg  # noqa
# from gpsic.plotting.planar import GraphPlotter
# from gpsic.plotting.spatial import Animation3D


def random_uniform_on_xy(dist):
    b = np.array([dist, dist, dist, 0., 0., 0.])
    return np.random.uniform(-b, b).reshape(-1, 1)


def run(arg):
    # creat graph
    g = graph.RoverGraph(directed=False, connect=graph.proximity)
    # g.add_robots(arg.n, deploy=lambda id: random_uniform_on_xy(arg.dist))
    g.add_robots(arg.n, deploy=lambda id: pos[id])

    # times and frequencies
    tsim = np.arange(arg.ti+arg.h, arg.tf, arg.h)
    Tc = 1./arg.f_ctrl
    tc = tsim[0] + Tc
    t_ctr = []

    P = collections.OrderedDict([(r.id, []) for r in g.robots()])
    hat_P = collections.OrderedDict([(r.id, []) for r in g.robots()])
    hat_Cov = collections.OrderedDict([(r.id, []) for r in g.robots()])

    L = []
    cmdv = collections.OrderedDict([(r.id, np.zeros(3)) for r in g.robots()])
    for k, t in enumerate(tsim):
        for i in g.get_robots():
            g.r(i).sim_step(0 * cmdv[i], t)
            if not k % (arg.h**-1):
                # update cmd vel every 1 second
                cmdv[i] = np.random.normal(0, [3, 3, 1])

        if t > tc:
            for i in g.get_robots():
                g.connect(i, arg.range)
                r = g.r(i)
                r.ctrl_step(t)
                P[i].append(r.p())
                r.filter.save()
                hat_P[i].append(r.hat_p())
                hat_Cov[i].append(metricas.sqrt_covar(r.cov_p()))
            g.share_msgs()
            L.append(g.get_links())
            tc += Tc
            t_ctr.append(t)

    # debug = dict([(r_i.id, {'f': r_i.filter.log,
    # 'r': r_i.imu.accel.bias}) for r_i in g.robots()])
    debug = {}

    return t_ctr, P, L, hat_P, hat_Cov, debug


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
      '-n', '--n',
      default=5, type=int, help='cantidad de robots')
    parser.add_argument(
      '-d', '--dist',
      default=20., type=float, help='distribución inicial')
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
      default=30.0, type=float, help='frecuencia del controlador')
    parser.add_argument(
      '-r', '--range',
      default=40., type=float, help='rango de la antena')
    parser.add_argument(
      '-g', '--save',
      default=False, action='store_true', help='flag para guardar los videos')
    parser.add_argument(
      '-a', '--animate',
      default=False, action='store_true',
      help='flag para generar animaicion 3D')

    arg = parser.parse_args()

    pos = {1: [0., 0, 0, 0, 0, 0],
           2: [-10., -10, 0, 0, 0, 0],
           3: [-10., 10, 15, 0, 0, 0],
           4: [10., -10, 10, 0, 0, 0],
           5: [5., -5, 20, 0, 0, 0]}

    # for i, p in pos.items():
    #     for j, q in pos.items():
    #         print('d({},{})={}'.format(i, j, linalg.distance(p,q)))

    t_ctr, P, L, hat_P, hat_Cov, debug = run(arg)

    plot = []

    p1 = np.vstack(P[1]).T
    p2 = np.vstack(P[2]).T
    p3 = np.vstack(P[3]).T
    p4 = np.vstack(P[4]).T

    f1 = np.vstack(hat_P[1]).T
    f2 = np.vstack(hat_P[2]).T
    f3 = np.vstack(hat_P[3]).T
    f4 = np.vstack(hat_P[4]).T

    # lines = (*p1,*f1), (*p2,*f2), (*p3,*f3), (*p4,*f4)
    # color = ('r', 'g', 'b')*2
    # color = (color,)*4
    # ls = ('-',)*3 + ('dotted',)*3
    # ls = (ls,)*4
    # label = ('x', 'y', 'z', '', '', '')
    # label = (label,)*4
    # plot += [GridPlot(shape=(2,2))]
    # plot[0].draw(t_ctr, lines, color=color, ls=ls, label=label)

    # *_, bx1, by1, bz1 = np.vstack(debug[1]['f'].x).T
    # *_, bx2, by2, bz2 = np.vstack(debug[2]['f'].x).T
    # *_, bx3, by3, bz3 = np.vstack(debug[3]['f'].x).T
    # *_, bx4, by4, bz4 = np.vstack(debug[4]['f'].x).T
    # bx, by, bz = list(zip(*[debug[1]['r'] for k in t_ctr]))
    # lines = (bx1, bx2, bx3, bx4, bx),
    #      (by1, by2, by3, by4, by), (bz1, bz2, bz3, bz4, bz)
    # color = ('r','g','b','y','k')
    # color = (color,)*4
    # ls = ('dotted', )*4 + ('--',)
    # ls = (ls,)*4
    # label = ('1','2','3','4','truth')
    # label = (label,)*4
    # plot += [GridPlot(shape=(3,1))]
    # plot[1].draw(t_ctr, lines, color=color, ls=ls, label=label)

    # dy = np.vstack(debug[1]['f'].dy).T
    # plot += [GridPlot(shape=(1,1), title='Autocorrelación 1 - Innovación')]
    # # color=[[('r','g')]], label=[[('horizontal','vertical')]],
    #      xlabel='$\delta y$')
    # label = 'x', 'y', 'z', 'range'
    # for lab, innov in zip(label, dy):
    #     plot[2].axes[0,0].acorr(innov, usevlines=False,
    #          linestyle="-", marker=".", linewidth=0.75, label=lab)
    # # ap1.axes[0,0].acorr(dyv, usevlines=False,
    #     color='g', linestyle="-", marker=".", linewidth=0.75)
    # plot[2].fig.legend()

    # if arg.save:
    #     plot[0].savefig('rover_filter_pos')
    #     # plot[1].savefig('rover_filter_param')
    # else:
    #     plot[0].show()
    #     # plot[1].show()

    # plotter = GraphPlotter(t_ctr, P, L, save=arg.save,
    #    landmarks=[(0, -10), (0, 10), (-10, 0)])
    # plotter.links()
    # if arg.animate:
    #     plotter.animation2d(step=2, plot_kw={'xlim':[-20,20],
    #    'ylim':[-20,20]})
    # ani = Animation3D(
    #   t_ctr, step=2, save=True, plot_kw={'xlim':(-50,50),
    #    'ylim':(-50,50), 'zlim':(0,30)})
    # ani.add_quadrotor((P[0],A[0]), (P[1],A[1]),
    #    camera=True, attached=True, gimbal=(G[0],G[1]))
    # for id in range(1, arg.n+1):
    #   ani.add_sphere(P[id], [1 for k in t_ctr])
    #   ani.add_ellipsoid(hat_P[id], hat_Cov[id])
    # ani.run()
