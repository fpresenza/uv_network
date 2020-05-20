#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Thu May 21 18:34:16 2020
@author: fran
"""
import argparse
import numpy as np
from uvnpy.motion.multicopter import Multicopter
from uvnpy.tools.graphix_2d import TimePlot
from uvnpy.tools.graphix_3d import Animation3D

def run(arg):
    uav = (
        Multicopter(ti=arg.ti, pi=(2.,0.,5.), vi=(0.,0.,0.), ai=(0.,0.,0.), f_ctrl=arg.f_ctrl),
        Multicopter(ti=arg.ti, pi=(-2.,0.,5.), vi=(0.,0.,0.), ai=(0.,0.,0.), f_ctrl=arg.f_ctrl)
    )
    time = np.arange(arg.ti+arg.h, arg.tf, arg.h)

    P, V, A, W, R, U = ([],[]), ([],[]), ([],[]), ([],[]), ([],[]), ([],[])
    
    for t in time:
        wind = (0., 0., 0.)

        r = (-1*np.sin(t/2), 1*np.cos(t/2), 0., 1/2) #(vx, vy, vz, yaw)
        uav[0].step(r, t, fw=wind)
        P[0].append(uav[0].p())
        V[0].append(uav[0].v())
        A[0].append(uav[0].euler())
        W[0].append(uav[0].w())
        R[0].append(uav[0].ref())
        U[0].append(uav[0].ctrl_eff())

        r = (1*np.sin(t/2), -1*np.cos(t/2), 0., 1/2) #(vx, vy, vz, yaw)
        uav[1].step(r, t, fw=wind)
        P[1].append(uav[1].p())
        V[1].append(uav[1].v())
        A[1].append(uav[1].euler())
        W[1].append(uav[1].w())
        R[1].append(uav[1].ref())
        U[1].append(uav[1].ctrl_eff())

    return time, P, V, A, W, R, U

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--step', dest='h', default=1e-3, type=float, help='paso de simulación')
    parser.add_argument('-t', '--ti', metavar='T0', default=0.0, type=float, help='tiempo inicial')
    parser.add_argument('-e', '--tf', default=1.0, type=float, help='tiempo final')
    parser.add_argument('-f', '--f_ctrl', default=50.0, type=float, help='frecuencia del controlador')
    parser.add_argument('-g', '--save', default=False, action='store_true', help='flag para guardar los videos')
    arg = parser.parse_args()

    time, P, V, A, W, R, U = run(arg)

    # px, py, pz = np.hstack(P[0])
    # vx, vy, vz = np.hstack(V[0])
    # ar, ap, ay = np.hstack(A[0])
    # wx, wy, wz = np.hstack(W[0])
    # rvx, rvy, rvz, rwz = np.hstack(R[0])
    # lines = [[[px, py, pz],[ar, ap, ay]],[[vx, vy, vz, rvx, rvy, rvz],[wx, wy, wz, rwz]]]
    # color = [[['b', 'r', 'g'],['b', 'r', 'g']],[['b', 'r', 'g', 'b', 'r', 'g'],['b', 'r', 'g', 'g']]]
    # label = [[['$x$', '$y$', '$z$'],['$\phi$', '$\Theta$', '$\psi$']],
    #          [['$v_x$', '$v_y$', '$v_z$', '', '', ''],['$\omega_x$', '$\omega_y$', '$\omega_z$', '']]]
    # ls = [[['-', '-', '-'],['-', '-', '-']],[['-', '-', '-', 'dotted', 'dotted', 'dotted'],['-', '-', '-', 'dotted']]]
    # xp1 = TimePlot(time, lines, title='Multicopter 1 - state', color=color, label=label, ls=ls)
    # # xp1.show()
    # xp1.savefig('uav_1_s_robot')

    # px, py, pz = np.hstack(P[1])
    # vx, vy, vz = np.hstack(V[1])
    # ar, ap, ay = np.hstack(A[1])
    # wx, wy, wz = np.hstack(W[1])
    # rvx, rvy, rvz, rwz = np.hstack(R[1])
    # lines = [[[px, py, pz],[ar, ap, ay]],[[vx, vy, vz, rvx, rvy, rvz],[wx, wy, wz, rwz]]]
    # xp2 = TimePlot(time, lines, title='Multicopter 2 - state', color=color, label=label, ls=ls)
    # # xp2.show()
    # xp2.savefig('uav_2_s_robot')

    # Ft, Tx, Ty, Tz = np.hstack(U[0])
    # lines = [[[Ft],[Tx]],[[Ty],[Tz]]]
    # label = [[['$F_t$'],['$T_x$']],[['$T_y$'],['$T_z$']]]
    # up1 = TimePlot(time, lines, title='Multicopter 1 - control effort', label=label)
    # up1.show()

    # Ft, Tx, Ty, Tz = np.hstack(U[1])
    # lines = [[[Ft],[Tx]],[[Ty],[Tz]]]
    # up2 = TimePlot(time, lines, title='Multicopter 2 - control effort', label=label)
    # up2.show()

    ani = Animation3D(time[::50], xlim=(-5,5), ylim=(-5,5), zlim=(0,10), save=arg.save)
    ani.add_quadrotor((P[0][::50],A[0][::50]), (P[1][::50],A[1][::50]), camera=True)
    ani.run()