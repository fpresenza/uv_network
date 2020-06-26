#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Wed Jul 8 19:48:15 2020
@author: fran
"""
import argparse
import numpy as np
import collections
from uvnpy.vehicles.drone import Drone
from uvnpy.vehicles.rover import Rover
import uvnpy.graphix.planar as planar
import uvnpy.graphix.spatial as spatial
import uvnpy.network.graph as graph
from uvnpy.navigation.metrics import metrics
from uvnpy.graphix.spatial import Animation3D

def random_uniform_on_xy(dist):
    b = np.array([dist, dist, dist, 0., 0., 0.])
    return np.random.uniform(-b, b)

def run(arg):    
    # times and frequencies
    tsim = np.arange(arg.ti+arg.step, arg.tf, arg.step)
    Tc = 1/arg.f_ctrl
    tc = tsim[0]+Tc
    t_ctr = np.arange(arg.ti+Tc, arg.tf, Tc)

    P = collections.OrderedDict([(r.id, []) for r in g.robots()])
    hat_P = collections.OrderedDict([(r.id, []) for r in g.robots()])
    hat_Cov = collections.OrderedDict([(r.id, []) for r in g.robots()])
    A = collections.OrderedDict([(r.id, []) for r in g.robots()])
    Cam = collections.OrderedDict([(r.id, []) for r in g.robots() if r.type is 'Drone'])

    L = []
    cmdv = collections.OrderedDict([(r.id, np.zeros(3)) for r in g.robots()])
    for k, t in enumerate(tsim):
        # g.r(5).motion.step([0, 0, 2*np.sin(t/6), 0, 0, 0], t)

        # for r in g.robots():
        #     if r.type is 'Rover':
        #         r.sim_step(cmdv[r.id], t)
                # if not k%(arg.step**-1):
                #     # update cmd vel every 1 second
                #     cmdv[r.id] = np.random.normal(0, [3,3,1])
        if t > tc:
            print(t)
            points = []
            for r in g.robots():
                g.connect(r.id, arg.range)
            for r in g.robots():
                if r.type is 'Rover':
                    # print(r.id, r.N.neighbors.keys())
                    r.ctrl_step(t)
                    points.append(r.p())
                    r.filter.save()
                    P[r.id].append(r.p())
                    hat_P[r.id].append(r.hat_p())
                    hat_Cov[r.id].append(metrics.sqrt_covar(r.cov_p()))
                elif r.type is 'Drone':
                    r.ctrl_step(t, points)
                    r.filter.save()
                    P[r.id].append(r.p())
                    A[r.id].append(r.euler())
                    Cam[r.id].append(r.cam.attitude)
                    hat_P[r.id].append(r.hat_mp())
                    hat_Cov[r.id].append(metrics.sqrt_covar(r.cov_mp()))
                
            g.share_msgs()
            L.append(g.get_links())
            tc += Tc

    debug = dict([(r.id, r.filter.log) for r in g.robots()])

    return t_ctr, P, A, Cam, L, hat_P, hat_Cov, debug

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-g', '--rover', default=0, type=int, help='cantidad de rovers')
    parser.add_argument('-a', '--drone', default=0, type=int, help='cantidad de drones')
    parser.add_argument('-p', '--step', default=1e-3, type=float, help='paso de simulación')
    parser.add_argument('-t', '--ti', metavar='T0', default=0.0, type=float, help='tiempo inicial')
    parser.add_argument('-e', '--tf', default=1.0, type=float, help='tiempo final')
    parser.add_argument('-f', '--f_ctrl', default=30.0, type=float, help='frecuencia del controlador')
    parser.add_argument('-r', '--range', default=40., type=float, help='rango de la antena')
    parser.add_argument('-s', '--save', default=False, action='store_true', help='flag para guardar los videos')
    parser.add_argument('-v', '--animate', default=False, action='store_true', help='flag para generar animaicion 3D')

    arg = parser.parse_args()

    pos = {1:[ 50.,  50,  0, 0, 0, 0],
           2:[-50.,  50,  0, 0, 0, 0],
           3:[-50., -50,  0, 0, 0, 0],
           4:[ 50., -50,  0, 0, 0, 0],
           5:[ 30.,  20, 15, 0, 0, 0],
           6:[-40.,  50, 30, 0, 0, 0],
           7:[-30., -20, 20, 0, 0, 0],
           8:[ 40., -50, 10, 0, 0, 0]}

    # creat graph
    g = graph.UnmannedVehicleGraph(directed=False, connect=graph.proximity)
    # g = graph.UnmannedVehicleGraph(directed=False, connect=graph.selective_proximity)
    # g.add_robots(arg.rover, deploy=lambda id: random_uniform_on_xy(10.))
    g.add_robots(arg.rover, deploy=lambda id: pos[id], object=Rover)
    g.add_robots(arg.drone, deploy=lambda id: pos[id], object=Rover, sensor_kw={'gps':True})

    # g.add_robots(arg.drone, deploy=lambda id: pos[id], object=Drone)
    g.screen()

    t_ctr, P, A, Cam, L, hat_P, hat_Cov, debug = run(arg)

    g.screen()

    #############################################
    #                  PLOTS                    #
    #############################################
    plot = []

    # uav position
    e5 = np.vstack(P[5]).T - np.vstack(hat_P[5]).T
    e6 = np.vstack(P[6]).T - np.vstack(hat_P[6]).T
    e7 = np.vstack(P[7]).T - np.vstack(hat_P[7]).T
    e8 = np.vstack(P[8]).T - np.vstack(hat_P[8]).T

    lines = e5, e6, e7, e8
    color = [('r','g','b')]*4
    label = ('x','y','z'),
    # ls = [('--', '--', '--')]*arg.drone
    xlabel=['$t$ secs']*4
    ylabel=['$pos$ m']*4
    plot += [planar.GridPlot(shape=(4,1), figsize=(8,6), title='', 
        axtitle=['UAV {}'.format(i+1) for i in range(arg.rover)], xlabel=xlabel, ylabel=ylabel)]
    plot[0].draw(t_ctr, lines, color=color, label=label)
    plot[0].fig.tight_layout()

    # rover position
    e1 = np.vstack(P[1]).T - np.vstack(hat_P[1]).T
    e2 = np.vstack(P[2]).T - np.vstack(hat_P[2]).T
    e3 = np.vstack(P[3]).T - np.vstack(hat_P[3]).T
    e4 = np.vstack(P[4]).T - np.vstack(hat_P[4]).T

    lines = e1, e2, e3, e4
    color = [('r','g','b')]*arg.rover
    label = ('x','y','z'),# ('x','y','z'), ('x','y','z'), ('x','y','z')
    ls = [('dotted', 'dotted', 'dotted')]*arg.rover
    xlabel=['$t$ secs']*arg.rover
    ylabel=['$error$ m']*arg.rover
    plot += [planar.GridPlot(shape=(arg.rover,1), figsize=(8,6), title='', 
        axtitle=['UGV {}'.format(i+1) for i in range(arg.rover)], xlabel=xlabel, ylabel=ylabel)]
    plot[1].draw(t_ctr, lines, color=color, label=label, ls=ls)
    plot[1].fig.tight_layout()

    if arg.save:
        plot[0].savefig('uav_pos')
        plot[1].savefig('ugv_pos')
    else: 
        plot[0].show()


    # plotter = GraphPlotter(t_ctr, P, L, save=arg.save)
    # # plotter.links()
    if arg.animate:
        # plotter.animation2d(xlim=[-100,100], ylim=[-100,100], slice=2)
        ani = Animation3D(t_ctr, xlim=(-35,35), ylim=(-35,35), zlim=(0,20), save=arg.save, slice=3)
        ani.add_drone(g.r(5).id, P[5], A[5], (Cam[5],), camera=g.r(5).cam)
        for id in range(1, arg.g+1):
            # ani.add_sphere(P[id], [1 for k in t_ctr])
            ani.add_ellipsoid(hat_P[id], hat_Cov[id])
        ani.run()














