#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Thu May 21 18:34:16 2020
@author: fran
"""
import argparse
import numpy as np
from uvnpy.motion.cluster import R2DOF4
from uvnpy.tools.graphix_2d import TimePlot
from uvnpy.tools.graphix_3d import Animation3D

def run(arg):

    cluster = R2DOF4(
        ti=arg.ti,
        pi=((2.,0.,5.), (-2., 0., 5.)),
        f_ctrl=arg.f_ctrl,
        linear_model=False
    )
    time = np.arange(arg.ti+arg.h, arg.tf, arg.h)

    P, V, A, W, R, U = ([],[]), ([],[]), ([],[]), ([],[]), ([],[]), ([],[])
    C, Cv, Cr = [], [], []
    
    for t in time:
        wind = (0., 0., 0.)
        cmd_vc = (0.,0.,0.,0.5,0.,0.,0., 0.0)
        cluster.step(cmd_vc, t)
        C.append(cluster.pos())
        Cv.append(cluster.vel())
        Cr.append(np.array(cmd_vc).reshape(-1,1))

        P[0].append(cluster.uav_1.p())
        V[0].append(cluster.uav_1.v())
        A[0].append(cluster.uav_1.euler())
        W[0].append(cluster.uav_1.w())
        R[0].append(cluster.uav_1.ref())
        U[0].append(cluster.uav_1.ctrl_eff())

        P[1].append(cluster.uav_2.p())
        V[1].append(cluster.uav_2.v())
        A[1].append(cluster.uav_2.euler())
        W[1].append(cluster.uav_2.w())
        R[1].append(cluster.uav_2.ref())
        U[1].append(cluster.uav_2.ctrl_eff())

    return time, C, Cv, Cr, P, V, A, W, R, U

def plot_cluster(C, Cv, Cr):
    # Cluster state position plot
    #
    xc, yc, zc, phi_c, theta_c, phi_1, phi_2, d = np.hstack(C)

    lines = [[[xc, yc, zc],[phi_c, theta_c]],[[phi_1, phi_2],[d]]]
    color = [[['b', 'r', 'g'],['b', 'r']],[['b', 'r'],['b']]]
    label = [[['$x_c$', '$y_c$', '$z_c$'],['$\psi_c$', '$\Theta_c$']],
             [['$\psi_1$', '$\psi_2$'],['$d_c$']]]
    cp = TimePlot(time, lines, title='Cluster r2dof4 - position', color=color, label=label)
    # cp.show()
    cp.savefig('cluster_pos')

    # Cluster state velocity plot
    #
    xc, yc, zc, phi_c, theta_c, phi_1, phi_2, d = np.hstack(Cv)
    rxc, ryc, rzc, rphi_c, rtheta_c, rphi_1, rphi_2, rd = np.hstack(Cr)
    
    lines = [[[xc, yc, zc, rxc, ryc, rzc],[phi_c, theta_c, rphi_c, rtheta_c]],
             [[phi_1, phi_2, rphi_1, rphi_2],[d, rd]]]
    color = [[['b', 'r', 'g', 'b', 'r', 'g'],['b', 'r', 'b', 'r']],[['b', 'r', 'b', 'r'],['b', 'b']]]
    label = [[['$\dot{x}_c$', '$\dot{y}_c$', '$\dot{z}_c$', '', '', ''],['$\dot{\psi}_c$', '$\dot{\Theta}_c$', '', '']],
             [['$\dot{\psi}_1$', '$\dot{\psi}_2$', '', ''],['$\dot{d}_c$', '']]]
    ls = [[['-', '-', '-', 'dotted', 'dotted', 'dotted'],['-', '-', 'dotted', 'dotted']],
          [['-', '-', 'dotted', 'dotted'],['-', 'dotted']]]
    cv = TimePlot(time, lines, title='Cluster r2dof4 - velocity', color=color, label=label, ls=ls)
    # cv.show()
    cv.savefig('cluster_vel')

def plot_uav_state(id, P, V, A, W, R):
    # plot uav state
    px, py, pz = np.hstack(P)
    vx, vy, vz = np.hstack(V)
    ar, ap, ay = np.hstack(A)
    wx, wy, wz = np.hstack(W)
    rvx, rvy, rvz, rwz = np.hstack(R)
    lines = [[[px, py, pz],[ar, ap, ay]],[[vx, vy, vz, rvx, rvy, rvz],[wx, wy, wz, rwz]]]
    color = [[['b', 'r', 'g'],['b', 'r', 'g']],[['b', 'r', 'g', 'b', 'r', 'g'],['b', 'r', 'g', 'g']]]
    label = [[['$x$', '$y$', '$z$'],['$\phi$', '$\Theta$', '$\psi$']],
             [['$v_x$', '$v_y$', '$v_z$', '', '', ''],['$\omega_x$', '$\omega_y$', '$\omega_z$', '']]]
    ls = [[['-', '-', '-'],['-', '-', '-']],[['-', '-', '-', 'dotted', 'dotted', 'dotted'],['-', '-', '-', 'dotted']]]
    r1p = TimePlot(time, lines, title='Multicopter {} - state'.format(id), color=color, label=label, ls=ls)
    # r1p.show()
    r1p.savefig('uav_{}_s_cluster'.format(id))

def plot_uav_ctrl(id, U):
    # plot uav control effort
    Ft, Tx, Ty, Tz = np.hstack(U)
    lines = [[[Ft],[Tx]],[[Ty],[Tz]]]
    label = [[['$F_t$'],['$T_x$']],[['$T_y$'],['$T_z$']]]
    u1p = TimePlot(time, lines, title='Multicopter {} - control effort'.format(id), label=label)
    # u1p.show()
    u1p.savefig('uav_{}_u'.format(id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--step', dest='h', default=1e-3, type=float, help='paso de simulación')
    parser.add_argument('-t', '--ti', metavar='T0', default=0.0, type=float, help='tiempo inicial')
    parser.add_argument('-e', '--tf', default=1.0, type=float, help='tiempo final')
    parser.add_argument('-f', '--f_ctrl', default=50.0, type=float, help='frecuencia del controlador')
    parser.add_argument('-g', '--save', default=False, action='store_true', help='flag para guardar los videos')
    arg = parser.parse_args()

    time, C, Cv, Cr, P, V, A, W, R, U = run(arg)

    # cluster plot
    plot_cluster(C, Cv, Cr)
    # uav 1 plot
    #    
    plot_uav_state('1', P[0], V[0], A[0], W[0], R[0])
    plot_uav_ctrl('1', U[0])
    # uav 2 plot
    #
    plot_uav_state('2', P[1], V[1], A[1], W[1], R[1])
    plot_uav_ctrl('2', U[1])

    # Animation
    # ani = Animation3D(time[::50], xlim=(-5,5), ylim=(-5,5), zlim=(0,10), save=arg.save)
    # ani.add_quadrotor((P[0][::50],A[0][::50]), (P[1][::50],A[1][::50]))
    # ani.run()