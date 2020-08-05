#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Thu May 21 18:34:16 2020
@author: fran
"""
import argparse
import numpy as np
from gpsic.modelos.multicoptero import Multicoptero
from gpsic.modelos.camera import Camera
from gpsic.plotting.planar import GridPlot
from gpsic.plotting.spatial import Animation3D

def run(arg):
    uav = (
        Multicoptero(ti=arg.ti, pi=(0.,3.,5.), vi=(0.,0.,0.), ei=(0.,0.,0.), f_ctrl=arg.f_ctrl),
        Multicoptero(ti=arg.ti, pi=(0.,-3.,5.), vi=(0.,0.,0.), ei=(0.,0.,np.pi/2), f_ctrl=arg.f_ctrl)
    )
    time = np.arange(arg.ti+arg.h, arg.tf, arg.h)

    P, V, A, W, R, U = ([],[]), ([],[]), ([],[]), ([],[]), ([],[]), ([],[])
    G = ([],[])
    
    for t in time:
        wind = (0., 0., 0.)

        # r = (2*np.sin(t/2), 0., 0., 0.) #(vx, vy, vz, yaw)
        r = (1,0,0,0.5)
        uav[0].step(t, r, din_args=(wind,))
        P[0].append(uav[0].p)
        V[0].append(uav[0].v)
        A[0].append(uav[0].euler)
        W[0].append(uav[0].w)
        R[0].append(r)
        U[0].append(uav[0].u)
        G[0].append(np.array([0,np.pi/4,0]))

        r = (0,2*np.sin(t/2),0.,0.) #(vx, vy, vz, yaw)
        # r = (0.,0.,0.,0.)
        uav[1].step(t, r, din_args=(wind,))
        P[1].append(uav[1].p)
        V[1].append(uav[1].v)
        A[1].append(uav[1].euler)
        W[1].append(uav[1].w)
        R[1].append(r)
        U[1].append(uav[1].u)
        G[1].append(np.array([0,np.pi/7,0]))

    return time, P, V, A, W, R, U, G

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--step', dest='h', default=1e-3, type=float, help='paso de simulación')
    parser.add_argument('-t', '--ti', metavar='T0', default=0.0, type=float, help='tiempo inicial')
    parser.add_argument('-e', '--tf', default=1.0, type=float, help='tiempo final')
    parser.add_argument('-f', '--f_ctrl', default=50.0, type=float, help='frecuencia del controlador')
    parser.add_argument('-g', '--save', default=False, action='store_true', help='flag para guardar los videos')
    parser.add_argument('-a', '--animate', default=False, action='store_true', help='flag para generar animaicion 3D')

    arg = parser.parse_args()

    time, P, V, A, W, R, U, G = run(arg)

    plot = []
    p = np.vstack(P[0]).T
    v = np.vstack(V[0]).T
    e = np.vstack(A[0]).T
    w = np.vstack(W[0]).T
    r = np.vstack(R[0]).T
    lines = p, e, [*v, *r[:3]], [*w, r[3]]
    color = ['b', 'r', 'g'],['b', 'r', 'g'],['b', 'r', 'g', 'b', 'r', 'g'],['b', 'r', 'g', 'g']
    label = ['$x$', '$y$', '$z$'],['$\phi$', '$\Theta$', '$\psi$'],\
    ['$v_x$', '$v_y$', '$v_z$', '', '', ''],['$\omega_x$', '$\omega_y$', '$\omega_z$', '']
    ls = ['-', '-', '-'],['-', '-', '-'],['-', '-', '-', 'dotted', 'dotted', 'dotted'],['-', '-', '-', 'dotted']
    plot += [GridPlot(shape=(2,2), title='Multicopter 1 - state')]
    plot[0].draw(time, lines, color=color, label=label, ls=ls)

    p = np.vstack(P[1]).T
    v = np.vstack(V[1]).T
    e = np.vstack(A[1]).T
    w = np.vstack(W[1]).T
    r = np.vstack(R[1]).T
    lines = p, e, [*v, *r[:3]], [*w, r[3]]
    plot += [GridPlot(shape=(2,2), title='Multicopter 2 - state')]
    plot[1].draw(time, lines, color=color, label=label, ls=ls)

    Ft, Tx, Ty, Tz = np.vstack(U[0]).T
    lines = [Ft],[Tx],[Ty],[Tz]
    label = ['$F_t$'],['$T_x$'],['$T_y$'],['$T_z$']
    plot += [GridPlot(shape=(2,2), title='Multicopter 1 - control effort')]
    plot[2].draw(time, lines, label=label)

    Ft, Tx, Ty, Tz = np.vstack(U[1]).T
    lines = [Ft],[Tx],[Ty],[Tz]
    label = ['$F_t$'],['$T_x$'],['$T_y$'],['$T_z$']
    plot += [GridPlot(shape=(2,2), title='Multicopter 2 - control effort')]
    plot[3].draw(time, lines, label=label)

    if arg.save:
        plot[0].savefig('uav_1_s_robot')
        plot[1].savefig('uav_2_s_robot')
        plot[2].savefig('uav_1_u')
        plot[3].savefig('uav_2_u')
    else: 
        plot[0].show()

    if arg.animate:
        ani = Animation3D(time, step=100, save=True, plot_kw={'xlim':(-15,15), 'ylim':(-15,15), 'zlim':(0,15)})
        ani.add_drone(1, P[0], A[0], (A[0], G[0]), camera=Camera())
        ani.add_drone(2, P[1], A[1], (A[1], G[1]), camera=Camera())
        ani.add_sphere([np.array([0,10,0]) for k in P[0]], [1 for k in P[0]])
        ani.run()