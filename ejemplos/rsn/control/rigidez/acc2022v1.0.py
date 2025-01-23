#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mié sep  1 20:02:37 -03 2021
@author: fran
"""
import argparse
import collections
import time
import progressbar
import numpy as np
from numba import njit

from uvnpy.model import linear_models
from uvnpy.network import disk_graph
from uvnpy.rsn import distances, rigidity
from uvnpy.rsn.control import centralized_rigidity_maintenance
from uvnpy.toolkit import functions
from uvnpy.rsn.localization import distances_to_neighbors_kalman
from uvnpy.toolkit.calculus import gradient


# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple('Logs', 'x hatx u fre re adjacency hops')


def load(x, coeff, dmin, dmax):
    w = distances.matrix(x)
    w[w > 0] = functions.logistic(w[w > 0], dmax, 5/dmax)
    deg = w.sum(-1)
    return 0.5 * (coeff * deg).sum(-1)


def reduce_load(x, coeff, dmin, dmax):
    u = gradient(load, x, coeff, dmin, dmax)
    return -u


def collision_avoidance(x):
    w = distances.matrix(x)
    w[w > 0] = functions.power_derivative(w[w > 0], 1)
    u = distances.edge_potencial_gradient(w, x)
    return -u


def update_adjacency(A, Vi, x, hatx, i, dmin, dmax):
    """Toma los nuevos vecinos, y se fija si con ellos
    el framework resultante es rigido.
    Este protocolo ad-hoc en realidad hay que """
    Ni = A[i].astype(bool)
    _Ni = disk_graph.neighborhood_histeresis(
        x, i, Ni, dmin, dmax)
    _Vi = Vi + _Ni      # aca deberia reemplazar por los nuevos no un "or"
    _Ai = A[_Vi][:, _Vi]
    _xi = hatx[_Vi]
    _re = rigidity.eigenvalue(_Ai, _xi)
    if _re > 1e-3:
        A[i] = _Ni.astype(int)
    return A


def sat(u):
    """Funcion de saturacion"""
    return (0.5 - functions.logistic(u, steepness=0.5))*10


@njit
def geodesics(A):
    G = A.copy()
    As = np.eye(len(A)) + A
    h = 2
    while not np.all(As):
        Ah = np.linalg.matrix_power(A, h)
        for i, g in enumerate(G):
            idx = np.logical_and(Ah[i] > 0, As[i] == 0)
            g[idx] = h
        As += Ah
        h += 1
    return G

# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(steps, logs, t_perf, A, dinamica):
    # iteración
    bar = progressbar.ProgressBar(maxval=arg.tf).start()
    u = np.zeros(dinamica.x.shape)
    re = np.empty(n)

    for k, t in steps[1:]:
        x = dinamica.x

        u[:] = 0
        G = geodesics(A)

        hatx = np.array([localization[i].x for i in nodes])
        hatP = np.array([localization[i].P for i in nodes])
        err = np.linalg.norm(x - hatx, axis=1)
        if err.max() > 12:
            print('\n error: ', err.argmax(), err.max())

        t_a = time.perf_counter()
        """parte de control"""
        for i in nodes:
            if hops[i] > 1:
                hi = hops[i] - 1
                Vi = G[i] <= hi
                Ai = A[Vi][:, Vi]
                xi = x[Vi]
                if rigidity.eigenvalue(Ai, xi) > 0.05:
                    hops[i] = hi

            hi = hops[i]
            Vi = G[i] <= hi
            """ check si el nodo no esta solo """
            if not Vi.any():
                print('Desconexión. Nodo: {}'.format(i))
            # center = sum(Vi[:i])  # me dice que lugar ocupa i el vector x_i
            Ni = A[i].astype(bool)
            Ai = A[Vi][:, Vi]
            xi = x[Vi]
            hatxi = hatx[Vi]

            re[i] = rigidity.eigenvalue(Ai, xi)
            if re[i] < 1e-6:
                print('\n Flexibility. Node: {}, re = {}'.format(i, re[i]))
            u[Vi] += collision_avoidance(hatxi)
            coeff = G[i][Vi] < hi
            # coeff = hi - G[i][Vi]
            u[Vi] += 3*reduce_load(hatxi, coeff, dmin, dmax)
            u[Vi] += 3*maintenance[i].update(hatxi)

            A = update_adjacency(A, Vi, x, hatx, i, dmin, dmax)

        """parte de localizacion"""
        for i in nodes:
            hi = hops[i]
            Vi = G[i] <= hi
            Ni = A[i].astype(bool)

            xj = hatx[Ni]
            Pj = hatP[Ni]
            localization[i].update_neighbors(xj, Pj)
            di = distances.matrix_between(x[i], xj)
            zi = np.random.normal(di, range_sd)
            localization[i].step(t, u[i], zi)
            if i in [15, 41]:
                pi = np.random.normal(x[i], pos_sd)
                Pi = localization[i].P
                Ki = Pi.dot(np.linalg.inv(Pi + pos_sd**2 * np.eye(2)))
                localization[i]._x += Ki.dot(pi - hatx[i])

        t_b = time.perf_counter()
        u = 1.25*sat(u)
        x = dinamica.step(t, u)

        # Análisis
        logs.x[k] = x.ravel()
        logs.hatx[k] = hatx.ravel()
        logs.u[k] = u.ravel()
        logs.fre[k] = rigidity.eigenvalue(A, x)
        logs.re[k] = re
        logs.adjacency[k] = A.ravel()
        logs.hops[k] = hops

        t_perf.append((t_b - t_a)/n)
        bar.update(np.round(t, 3))

    bar.finish()

    # return
    return logs, t_perf


# ------------------------------------------------------------------
# Parseo de argumentos
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-s', '--step',
    dest='h', default=100e-3, type=float, help='paso de simulación')
parser.add_argument(
    '-t', '--ti',
    metavar='T0', default=0.0, type=float, help='tiempo inicial')
parser.add_argument(
    '-e', '--tf',
    default=1.0, type=float, help='tiempo final')

arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
tiempo = np.arange(arg.ti, arg.tf, arg.h)
steps = list(enumerate(tiempo))
t_perf = []

lim = 100
dim = 2
# np.random.seed(19)
# x = np.random.uniform(-0.9*lim, 0.9*lim, (80, dim))
# x[38] += (2, -4)
# rmv = (
#    3, 5, 7, 14, 25, 32, 33, 41, 43, 46, 50, 51, 62, 63, 68, 77, 78, 79)
# x = np.delete(x, rmv, axis=0)
x = np.array([
    [-72.26941441,  47.17917505],    # noqa
    [-45.60331138, -65.00815814],    # noqa
    [-30.62115446, -75.31073119],    # noqa
    [ 87.00179196,  24.37523031],    # noqa
    [ -5.590494  , -40.31927475],    # noqa
    [-62.28618703,   6.65708997],    # noqa
    [-14.25095638,  32.06935277],    # noqa
    [-23.29596559,  80.38786595],    # noqa
    [ 46.82716952,  18.65828373],    # noqa
    [ 14.98407233,  27.6650151 ],    # noqa
    [-42.30211817,  41.50276921],    # noqa
    [-78.43572452,  34.01971215],    # noqa
    [ 12.26277634,   9.25741989],    # noqa
    [-28.44919496, -57.92799531],    # noqa
    [  7.43613592, -80.81747392],    # noqa
    [ 58.59127795,  38.05985518],    # noqa
    # [-82.08299252, -76.22563786],    # noqa
    [-88.21786015,  88.85789947],    # noqa
    [ 27.66202478, -53.76242414],    # noqa
    [-23.8953201 ,  65.2702383 ],    # noqa
    [ -4.07860682,  83.66862899],    # noqa
    [ 75.3810178 ,  78.40838877],    # noqa
    [ 78.67800721,  31.07104951],    # noqa
    [ 79.10134955,   4.43487303],    # noqa
    [ -9.08910035, -84.36347773],    # noqa
    [ 84.03853519,  -6.12927553],    # noqa
    [ 41.23381654,  84.10803078],    # noqa
    [-60.41094967, -10.82328944],    # noqa
    [ 45.35057755,  -9.23386923],    # noqa
    # [-64.06210042, -55.27964615],    # noqa
    [ 90.04205222, -41.49352746],    # noqa
    [ 70.44332818, -64.7347392 ],    # noqa
    [ 59.94675015, -69.65317635],    # noqa
    [ 32.34276635,  40.08836599],    # noqa
    [-80.4754325 ,  -0.87646496],    # noqa
    [ 41.28368214, -34.89443335],    # noqa
    [ 23.53990366,  85.41110771],    # noqa
    [-75.96605296, -52.93918701],    # noqa
    [-38.46378631, -21.43357774],    # noqa
    [ 60.44859007, -74.73717344],    # noqa
    [ 87.10179988,  82.60807312],    # noqa
    [ 25.45064448,  71.99137778],    # noqa
    [-27.37260567,  23.65919189],    # noqa
    [-55.45897158, -28.4081608 ],    # noqa
    [ 87.3696433 ,  45.24036917],    # noqa
    [ 31.17720286, -64.72457412],    # noqa
    [-87.64522615,  64.29268726],    # noqa
    [ 38.61260062,  64.26865662],    # noqa
    [ 60.16256046,  51.35818519],    # noqa
    [ -4.19733307, -73.04753835],    # noqa
    [-79.56788196,  73.88558161],    # noqa
    [-37.6943308 , -32.26732964],    # noqa
    [-79.69849984, -84.6772994 ],    # noqa
    [-67.14614288, -72.20409514],    # noqa
    [-31.02210024,  10.00523958],    # noqa
    [-13.30338554, -34.31592192],    # noqa
    [ 35.51789649, -73.28973876],    # noqa
    [ 18.25378407, -34.03340603],    # noqa
    [-44.85045916,   8.36830118],    # noqa
    [  9.14022521,  17.85517328],    # noqa
    [ 25.25671716, -74.89141361],    # noqa
    [ 10.71130592, -20.26657499]])   # noqa

n = len(x)
nodes = np.arange(n)
# a = 5 / n

dmin = 0.4*lim
dmax = 1.4 * dmin

A = disk_graph.adjacency(x, dmin)
dinamica = linear_models.integrator(x, tiempo[0])
p = 3
hatx = x + np.random.normal(0, p, (n, 2))
Pi = p**2 * np.eye(2)
q = 0.05
Q = q**2 * np.eye(2)
range_sd = 3.
pos_sd = 3.
dt = arg.h

maintenance = [
    centralized_rigidity_maintenance(dim, dmin, 20/dmin, 1/3) for _ in nodes]
localization = [distances_to_neighbors_kalman(
    hatx[i], Pi, Q * dt, range_sd**2, tiempo[0]) for i in nodes]

hops = rigidity.minimum_hops(A, x)
print(hops)

logs = Logs(
    x=np.empty((tiempo.size, n*dim)),
    hatx=np.empty((tiempo.size, n*dim)),
    u=np.empty((tiempo.size, n*dim)),
    fre=np.zeros(tiempo.size),
    re=np.zeros((tiempo.size, n)),
    adjacency=np.empty((tiempo.size, n**2), dtype=int),
    hops=np.empty((tiempo.size, n), dtype=int))
logs.x[0] = x.ravel()
logs.hatx[0] = hatx.ravel()
logs.u[0] = np.zeros(n*dim)
logs.fre[0] = rigidity.eigenvalue(A, x)
logs.re[0] = [
    rigidity.subframework_eigenvalue(A, x, i, h) for i, h in enumerate(hops)]
logs.adjacency[0] = A.ravel()
logs.hops[0] = hops

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
logs, t_perf = run(steps, logs, t_perf, A, dinamica)

np.savetxt('/tmp/t.csv', tiempo, delimiter=',')
np.savetxt('/tmp/x.csv', logs.x, delimiter=',')
np.savetxt('/tmp/hatx.csv', logs.hatx, delimiter=',')
np.savetxt('/tmp/u.csv', logs.u, delimiter=',')
np.savetxt('/tmp/fre.csv', logs.fre, delimiter=',')
np.savetxt('/tmp/re.csv', logs.re, delimiter=',')
np.savetxt('/tmp/adjacency.csv', logs.adjacency, delimiter=',')
np.savetxt('/tmp/hops.csv', logs.hops, delimiter=',')

st = arg.tf - arg.ti
rt = sum(t_perf)
prompt = 'RT={:.3f} secs, ST={:.3f} secs  ==>  RTF={:.3f}'
print(prompt.format(rt, st, st / rt))
