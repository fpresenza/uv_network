#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Thu Ago 5 11:00:29 2020
@author: fran
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gpsic.plotting.planar as planar
from uvnpy.vehicles.point import Point

def run(arg, points, landmarks):    
    # times and frequencies
    time = np.arange(arg.ti, arg.tf, arg.h)
    # cmd = dict([(point.id, [0., 0.]) for point in points])

    # logs
    x = [points[0].kin]
    u = [points[0].control.u]
    f_t, f_x, f_dvst, f_eig = points[0].filter.logs()
    f = ([f_x], [f_dvst], [np.flip(f_eig)])
    P = dict([(point.id, [point.kin[:2]]) for point in points])

    for t in time[1:]:
        for point in points:
            point.motion_step(t)
            point.control_step(t, landmarks=landmarks)
            P[point.id].append(point.kin[:2])
            if point.id == points[0].id:
                x.append(point.kin)
                u.append(point.control.u)
                f_t, f_x, f_dvst, f_eig = point.filter.logs()
                f[0].append(f_x)
                f[1].append(f_dvst)
                f[2].append(np.flip(f_eig))

    return time, P, x, u, f


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--step', dest='h', default=50e-3, type=float, help='paso de simulación')
    parser.add_argument('-t', '--ti', metavar='T0', default=0.0, type=float, help='tiempo inicial')
    parser.add_argument('-e', '--tf', default=1.0, type=float, help='tiempo final')
    parser.add_argument('-g', '--save', default=False, action='store_true', help='flag para guardar los videos')
    parser.add_argument('-a', '--animate', default=False, action='store_true', help='flag para generar animacion')
    parser.add_argument('-n', '--agents', default=1, type=int, help='cantidad de agentes')


    arg = parser.parse_args()

    # landmarks
    landmarks = [(0,-10), (0,10)]
    # landmarks = []
    # landmarks = [[-15.90949736,  11.74311878],
    #              [-5.21570337, -6.41701965],
    #              [-13.76694731,  -2.34360965],
    #              [-3.2733689 , 18.90361114]]

    points = [Point(i, motion_kw={'pi':np.random.uniform(-.1, .1, 2), 'freq':arg.h**(-1)}) for i in range(arg.agents)]

    t, P, x, u, f = run(arg, points, landmarks)

    # variables
    x = np.vstack(x).T
    f_x = np.vstack(f[0]).T
    f_dvst = np.vstack(f[1]).T
    f_eig = np.vstack(f[2]).T
    u = np.vstack(u).T

    # plot
    fig = plt.figure(figsize=(13, 5))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    gs = fig.add_gridspec(2, 3)
    posicion = (gs[0, 0],
        ('t [seg]', 'posición [m]'),
        [t]*4, [x[0], x[1], f_x[0], f_x[1]],
        {'color': ['r', 'g']*2, 'label':['$p_x$', '$p_y$', '', ''], 'ls':['-', '-', 'dotted', 'dotted']}
    )
    velocidad = (gs[0, 1], 
        ('t [seg]', r'velocidad [m/s]'),
        [t]*4, [x[2], x[3], f_x[2], f_x[3]],
        {'color': ['r', 'g']*2, 'label':['$v_x$', '$v_y$', '', ''], 'ls':['-', '-', 'dotted', 'dotted']}
    )
    control = (gs[0, 2],
        ('t [seg]', 'velocidad [m/s]'),
        [t]*2, u,
        {'color': ['r', 'g'], 'label':['$u_x$', '$u_y$']}
    )
    error_posicion = (gs[1, 0],
        ('t [seg]', 'posición [m]'),
        [t]*6, [*(x[:2]-f_x[:2]), *f_dvst[:2], *-f_dvst[:2]],
        {'color': ['r', 'g', 'r', 'g', 'r', 'g'], 'label':['$e_x$', '$e_y$', '', '', '', ''], 
         'ls':['-', '-', 'dotted', 'dotted', 'dotted', 'dotted']}
    )
    error_velocidad = (gs[1, 1],
        ('t [seg]', 'velocidad [m/s]'),
        [t]*6, [*(x[2:]-f_x[2:]), *f_dvst[2:], *-f_dvst[2:]],
        {'color': ['r', 'g', 'r', 'g', 'r', 'g'], 'label':['$e_{v_x}$', '$e_{v_y}$', '', '', '', ''], 
         'ls':['-', '-', 'dotted', 'dotted', 'dotted', 'dotted']}
    )
    autovalores = (gs[1, 2],
        ('t [seg]', ''),
        [t]*4, np.sqrt(f_eig),
        {'color': ['m', '0.5', '0.5' , 'c'], 'label':[r'$\sqrt{\lambda_{\rm{max}}}$', '', '', r'$\sqrt{\lambda_{\rm{min}}}$']}
    )
    axes = planar.grids(fig, 
        (posicion, velocidad, control, error_posicion, error_velocidad, autovalores), 
        label_kw={'fontsize':10.}, legend_kw={'fontsize':10.})
    axes[0].set_title('Pos. (verdadero vs. estimado)', fontsize=11)
    axes[1].set_title('Vel. (verdadero vs. estimado)', fontsize=11)
    axes[2].set_title('Esfuerzo de control', fontsize=11)
    axes[3].set_title('Pos. (error vs. std. dev.)', fontsize=11)
    axes[4].set_title('Vel. (error vs. std. dev.)', fontsize=11)
    axes[5].set_title('Autovalores', fontsize=11)

    for ax in axes:
        ax.tick_params(axis='both', labelsize=9., grid_linestyle='dashed', grid_linewidth=0.35)
    
    if arg.save:
        fig.savefig('/tmp/point_opt_ctrl.pdf', format='pdf')
    else:
        plt.show()

    L = [[] for _ in t]
    graph_plotter = planar.GraphPlotter(t, P, L, save=arg.save, landmarks=landmarks)
    if arg.animate:
        graph_plotter.animation2d(step=1, plot_kw={'xlim':[-40,40], 'ylim':[-40,40]})