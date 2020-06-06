#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
import uvnpy.network.graph as graph
import uvnpy.controller.mpc as controller
from uvnpy.graphix.planar import TimePlot, GraphPlotter
from uvnpy.graphix.spatial import Animation3D
from uvnpy.toolkit.linalg import vector
from uvnpy.navigation.metrics import metrics

def random_uniform_on_xy(dist):
    b = np.array([dist, dist, dist, 0., 0., 0.])
    return np.random.uniform(-b, b).reshape(-1,1)

dep = {1:np.array([10,10,0,0,0,0]).reshape(-1,1),
       2:np.array([-10,-10,0,0,0,0]).reshape(-1,1),
       3:np.array([-10.,10.,15.,0.,0.,0.]).reshape(-1,1),
       4:np.array([10.,-10.,10.,0.,0.,0.]).reshape(-1,1),
       5:np.array([5.,-5.,20.,0.,0.,0.]).reshape(-1,1)}

def run(arg):
    freq = arg.h**-1
    
    g = graph.RoverGraph(directed=False, connect=graph.proximity)
    # g.add_robots(arg.n, deploy=lambda id: random_uniform_on_xy(arg.dist))
    g.add_robots(arg.n, deploy=lambda id: dep[id])
    time = np.arange(arg.ti, arg.tf, arg.h)

    P = dict([(r.id, []) for r in g.robots()])
    hat_P = dict([(r.id, []) for r in g.robots()])
    hat_Cov = dict([(r.id, []) for r in g.robots()])

    L = []
    debug = ([],[],[],[])
    # Cmd = dict([(r.id, np.random.normal(0, [3,3,0])) for r in g.robots()])
    cmdv = dict([(r.id, np.zeros((3,1))) for r in g.robots()])
    # dof = 2
    # size = arg.n*dof
    # c, p, v = np.empty((size, 1)), np.empty((size, 1)), np.empty((size, 1)) 
    # idx = dict(zip(g.get_robots(), np.arange(0, size, dof))) 

    # mpc = controller.MPC(c.size, weights=(5,1), window=(1, 5))
    # u = np.zeros_like(c)

    for k, t in enumerate(time):
        for i in g.get_robots():
            g.connect(i, arg.range)
                 # modificar para que cada robot i busque su propios vecinos
                # Link update
                # g.connect(i, j, arg.range)
                # g.share(i, j)
            if not k%100:
                # update cmd vel every 40 steps
                cmdv[i] = np.random.normal(0, [3,3,1])
            r = g.r(i)
            r.step(cmdv[i], t) # robots forced to be static
            P[i].append(r.xyz())
            hat_P[i].append(r.hat_xyz())
            hat_Cov[i].append(metrics.sqrt_covar(r.cov_xyz()))
        g.share_msgs()
        # debug[0].append(angvel)
        # debug[1].append(gyro)
        L.append(g.get_links())
        # print('{:.1f}'.format(t), [(i,j) for (i,j) in g.get_links()])

        # Control
            # c[m:m+2] = Cmd[i][:2]
            # p[m:m+2] = r.xy()
            # v[m:m+2] = r.vxy()
        # x = p
        # u = mpc.update(u, c, x)
        # print(np.round(c - u, 2))

    return time, P, L, hat_P, hat_Cov, debug

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--n', default=10, type=int, help='cantidad de robots')
    parser.add_argument('-d', '--dist', default=20., type=float, help='distribución inicial')
    parser.add_argument('-s', '--step', dest='h', default=20e-3, type=float, help='paso de simulación')
    parser.add_argument('-t', '--ti', metavar='T0', default=0.0, type=float, help='tiempo inicial')
    parser.add_argument('-e', '--tf', default=1.0, type=float, help='tiempo final')
    parser.add_argument('-f', '--f_ctrl', default=10.0, type=float, help='frecuencia del controlador')
    parser.add_argument('-r', '--range', default=40., type=float, help='rango de la antena')
    parser.add_argument('-g', '--save', default=False, action='store_true', help='flag para guardar los videos')
    parser.add_argument('-a', '--animate', default=False, action='store_true', help='flag para generar animaicion 3D')

    arg = parser.parse_args()

    time, P, L, hat_P, hat_Cov, debug = run(arg)
    # ax, ay, az = np.hstack(debug[0])
    # ix, iy, iz = np.vstack(debug[1]).T
    # lines = [[(ax, ix)],[(ay, iy)], [(az, iz)]]
    # dp = graphix.TimePlot(time, lines, title='Debug')
    # if arg.save:
    #     dp.savefig('debug')
    # else:
    #     dp.show()

    plotter = GraphPlotter(time, P, L, save=arg.save)
    plotter.links()
    if arg.animate:
        plotter.animation2d(xlim=[-100,100], ylim=[-100,100])
        # ani = Animation3D(time, xlim=(-50,50), ylim=(-50,50), zlim=(0,30), save=arg.save, slice=3)
        # ani.add_quadrotor((P[0],A[0]), (P[1],A[1]), camera=True, attached=True, gimbal=(G[0],G[1]))
        # for id in range(1, arg.n+1):
        #     ani.add_sphere(P[id])
        #     ani.add_ellipsoid(hat_P[id], hat_Cov[id])
        # ani.run()