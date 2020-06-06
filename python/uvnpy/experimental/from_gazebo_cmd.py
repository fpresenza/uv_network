#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Thu May 21 18:34:16 2020
@author: fran
"""
import argparse
import numpy as np
import gpsic.integradores as integradores
from uvnpy.model.multicopter import Multicopter
from uvnpy.graphix.planar import TimePlot
from uvnpy.graphix.spatial import Animation3D
from uvnpy.experimental.parser import interpolate

def run(arg, u, u_r, time):

    uav = (
        Multicopter(pi=(0.,5.,0.), f_ctrl=arg.f_ctrl),
        Multicopter(pi=(0.,-5.,0.), f_ctrl=arg.f_ctrl)
    )

    rover = integradores.EulerExplicito(lambda x,t,v: v, xi=(10,0,0.), ti=ti)

    P, A, G = ([],[]), ([],[]), ([],[])
    R = []
    
    for t in time:
        wind = (0., 0., 0.)
        # print(u[0](t))
        uav[0].step(u[0](t), t, fw=wind)
        P[0].append(uav[0].p())
        A[0].append(uav[0].euler())
        G[0].append(np.array([0., np.radians(20), 0.]))

        uav[1].step(u[1](t), t, fw=wind)
        P[1].append(uav[1].p())
        A[1].append(uav[1].euler())
        G[1].append(np.array([0., np.radians(20), 0.]))

        # # generate rover position
        try:
            R.append(rover.step(t, ([u_r(t)], )))
        except ValueError:
            if t<rover.x[0]:
                R.append(np.array(rover.xi))
            elif t>rover.x[-1]:
                R.append(R[-1])



    return P, A, G, R

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--step', dest='h', default=1e-3, type=float, help='paso de simulación')
    parser.add_argument('-t', '--ti', metavar='T0', default=0.0, type=float, help='tiempo inicial')
    parser.add_argument('-e', '--tf', default=1000., type=float, help='tiempo final')
    parser.add_argument('-f', '--f_ctrl', default=50.0, type=float, help='frecuencia del controlador')
    parser.add_argument('-g', '--save', default=False, action='store_true', help='flag para guardar los videos')
    parser.add_argument('-u', '--file1', default='uav1_mavros_setpoint_velocity_cmd_vel.csv',\
     type=str, help='nombre del archivo uav1')
    parser.add_argument('-v', '--file2', default='uav1_mavros_setpoint_velocity_cmd_vel.csv',\
     type=str, help='nombre del archivo uav2')
    parser.add_argument('-r', '--file3', default='rover_mavros_setpoint_velocity_cmd_vel.csv',\
     type=str, help='nombre del archivo target')
    parser.add_argument('-d', '--dir', default='./', type=str, help='nombre del directorio')
    parser.add_argument('-a', '--animate', default=False, action='store_true', help='flag para generar animaicion 3D')

    arg = parser.parse_args()

    # leer data de los archivos de texto
    data_1 = np.loadtxt(arg.dir+arg.file1, delimiter=',', usecols=(0,4,5,6,9), skiprows=1)
    data_2 = np.loadtxt(arg.dir+arg.file2, delimiter=',', usecols=(0,4,5,6,9), skiprows=1)
    data_r = np.loadtxt(arg.dir+arg.file3, delimiter=',', usecols=(0,4,5,6), skiprows=1)
    # convert to seconds and set init=0
    t0 = 90512000000 # min(data_1[0,0], data_2[0,0])
    to_secs = 1e-9
    data_1[:,0] = (data_1[:,0]-t0)*to_secs
    data_2[:,0] = (data_2[:,0]-t0)*to_secs
    data_r[:,0] = (data_r[:,0]-t0)*to_secs
    # for i, d in enumerate(data_1):
    #     print(i, np.round(d[[0,3]], 2))
    # interpolate data
    u = (interpolate.from_data(data_1),
         interpolate.from_data(data_2))
    u_rover = interpolate.from_data(data_r)
    # make time array
    ti = arg.ti #max(arg.ti, u[0].x[0], u[1].x[0])
    tf = min(arg.tf, u[0].x[-1], u[1].x[-1])
    time = np.arange(ti+arg.h, tf, arg.h)
    # run simulation
    P, A, G, R = run(arg, u, u_rover, time)

    px, py, pz = np.hstack(P[0])
    ar, ap, ay = np.hstack(A[0])
    lines = [[(px, py, pz)],[(ar, ap, ay)]]
    color = [[('b', 'r', 'g')],[('b', 'r', 'g')]]
    label = [[('$x$', '$y$', '$z$')],[('$\phi$', '$\Theta$', '$\psi$')]]
    xp1 = TimePlot(time, lines, title='Multicopter 1 - state', color=color, label=label)

    px, py, pz = np.hstack(P[1])
    ar, ap, ay = np.hstack(A[1])
    lines = [[(px, py, pz)],[(ar, ap, ay)]]
    color = [[('b', 'r', 'g')],[('b', 'r', 'g')]]
    label = [[('$x$', '$y$', '$z$')],[('$\phi$', '$\Theta$', '$\psi$')]]
    xp2 = TimePlot(time, lines, title='Multicopter 2 - state', color=color, label=label)

    if arg.save:
        xp1.savefig('uav_1_s_robot')
        xp2.savefig('uav_2_s_robot')
    else:
        xp1.show()
        xp2.show()
        
    if arg.animate:
        ani = Animation3D(time, xlim=(-15,15), ylim=(-15,15), zlim=(0,10), save=arg.save, slice=50)
        ani.add_quadrotor((P[0],A[0]), (P[1],A[1]), camera=True, attached=True, gimbal=G)
        ani.add_sphere(R)
        ani.run()