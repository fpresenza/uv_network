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
import matplotlib.pyplot as plt

from uvnpy.model import linear_models
import uvnpy.network as network
from uvnpy.network import disk_graph, strength, subsets
from uvnpy.rsn import distances, rigidity
from uvnpy.toolkit.calculus import derivative_eval, gradient


plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple('Logs', 'x u re wre edges debug')


def rigidity_eigenvalue(A, x):
    if len(A) == 1:
        return 0.
    L = rigidity.laplacian(A, x)
    return np.linalg.eigvalsh(L)[3]


def weighted_rigidity_eigenvalue(x, midpoint, steepness):
    w = distances.matrix(x)
    w[w > 0] = strength.logistic(w[w > 0], midpoint, steepness)
    L = rigidity.laplacian(w, x)
    return np.linalg.eigvalsh(L)[..., 3]


def wighted_eigenvalue_product(x, M, midpoint, steepness):
    w = distances.matrix(x)
    w[w > 0] = strength.logistic(w[w > 0], midpoint, steepness)
    L = rigidity.laplacian(w, x)
    S = np.matmul(M.T, L.dot(M))
    return np.linalg.det(S)


def rigidity_maintenance(x, e, V):
    r = 1/5

    detS = e[3:].prod()
    grad = gradient(
        wighted_eigenvalue_product, x, V[:, 3:], midpoint[1], steepness[1])
    ur = -r * detS**(-r - 1) * grad

    return -100*ur


def rigidity_maintenance2(x, e, V):
    r = 1/10
    V = V[:, 3:]
    detS = e[3:].prod()
    dL_dx = derivative_eval(
        weighted_rigidity_matrix, x, midpoint[1], steepness[1])

    dS_dx = np.matmul(V.T, dL_dx.dot(V))
    tr = (dS_dx / e[3:].reshape(-1, 1)).trace(axis1=1, axis2=2)
    ur = -r * detS**(-r) * tr
    ur = ur.reshape(x.shape)

    return -50*ur


def rigidity_maintenance3(x, lambda4, v4):
    r = 1/3

    dL_dx = derivative_eval(
        weighted_rigidity_matrix, x, midpoint[1], steepness[1])
    dlambda4_dx = v4.dot(dL_dx).dot(v4).reshape(x.shape)
    ur = -r * lambda4**(-r - 1) * dlambda4_dx

    return -1*ur


def weighted_rigidity_matrix(x, midpoint, steepness):
    w = distances.matrix(x)
    w[w > 0] = strength.logistic(w[w > 0], midpoint, steepness)
    L = rigidity.laplacian(w, x)
    return L


def disconnect(x):
    w = distances.matrix(x)
    w[w > 0] = strength.logistic_derivative(
        w[w > 0], midpoint[0], steepness[0])
    u = distances.edge_potencial_gradient(w, x)
    return -1*u


def expand(x):
    w = distances.matrix(x)
    w[w > 0] = strength.power_derivative(w[w > 0], 1)
    u = distances.edge_potencial_gradient(w, x)
    return -1*u


def gamma(u):
    """Funcion de saturacion"""
    return (0.5 - strength.logistic(u, steepness=0.5))*10

# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(steps, logs, t_perf, A, dinamica, frames):
    # iteración
    bar = progressbar.ProgressBar(max_value=arg.tf).start()
    u = np.zeros(dinamica.x.shape)
    re = np.empty(n)
    wre = np.empty(n)
    hmax = hops.max()
    Ah = np.empty((hmax, n, n), dtype=bool)

    for k, t in steps[1:]:
        # step dinamica
        x = dinamica.x

        # Control
        # u_t = expand(x) + disconnect(x)
        # u[:] = u_t
        u[:] = 0
        R = subsets.reach(A, range(hmax+1))
        Ah[0] = (R[0] + R[1]).astype(bool)
        Ah[1] = Ah[0] + R[2].astype(bool)
        Ah[2] = Ah[1] + R[3].astype(bool)
        Ah[3] = Ah[2] + R[4].astype(bool)
        # Ah[4] = Ah[3] + R[5].astype(bool)

        t_a = time.perf_counter()
        for i in nodes:
            h = hops[i]
            # print('\n', h, i)
            Ni = Ah[h-1, i]
            if sum(Ni) > 1:
                """ check si el nodo no esta solo """
                # print(Ni)
                Ai = A[Ni][:, Ni]
                xi = x[Ni]
                Li = weighted_rigidity_matrix(xi, midpoint[1], steepness[1])
                e, V = np.linalg.eigh(Li)
                re[i] = rigidity_eigenvalue(Ai, xi)
                wre[i] = e[3]
                if re[i] < 1e-6:
                    print(
                        '\n Zero eigenvalue. Node: {}, re = {}, wre = {}'.format(   # noqa
                            i, re[i], wre[i]))
                ur = rigidity_maintenance3(xi, e[3], V[:, 3])
                # ur = rigidity_maintenance4(xi)
                # ur = 0
                u[Ni] += ur + expand(xi) + disconnect(xi)
                logs.debug[k, Ni] += ur
            else:
                print('Desconexión. Nodo: {}'.format(i))

        t_b = time.perf_counter()
        # print(np.where(eig < 1e-5))
        # u *= 2
        # u[30] = 1, 0
        u = gamma(u)
        x = dinamica.step(t, u)
        # print(t, x)

        # Análisis
        # print(distances.matrix(x))
        # A = disk_graph.adjacency(x, dmin)
        A = disk_graph.adjacency_histeresis(A, x, dmin, dmax)
        E = network.edges_from_adjacency(A)
        frames[k] = t, x, E

        logs.x[k] = x
        logs.u[k, :, 0] = u[:, 0]
        logs.u[k, :, 1] = u[:, 1]
        # logs.re[k, 0] = rigidity_eigenvalue(A, x)
        logs.re[k] = re
        # print(distances.from_adjacency(A, x))
        logs.wre[k] = wre
        logs.edges[k] = len(E)

        t_perf.append((t_b - t_a)/n)
        bar.update(np.round(t, 3))

    bar.finish()

    # return
    return logs, t_perf, frames


if __name__ == '__main__':
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
    parser.add_argument(
        '-r', '--arxiv',
        default='/tmp/config.yaml', type=str, help='arhivo de configuración')
    parser.add_argument(
        '-g', '--save',
        default=False, action='store_true',
        help='flag para guardar archivos')
    parser.add_argument(
        '-a', '--animate',
        default=False, action='store_true',
        help='flag para generar animacion')

    arg = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------
    tiempo = np.arange(arg.ti, arg.tf, arg.h)
    steps = list(enumerate(tiempo))
    t_perf = []

    lim = 100
    dof = 2
    # np.random.seed(19)
    # x = np.random.uniform(-0.9*lim, 0.9*lim, (80, dof))
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
        [-82.08299252, -76.22563786],    # noqa
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
        [-64.06210042, -55.27964615],    # noqa
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
    steepness = (5 / dmin, 20 / dmin)
    midpoint = (dmax, dmin)

    # x = np.random.uniform(-0.3*lim, 0.3*lim, (n, dof))
    # print(x, dmin)
    A = disk_graph.adjacency(x, dmin)
    E = network.edges_from_adjacency(A)
    # print(subsets.multihop_adjacency(A, hops).sum(1))
    dinamica = linear_models.integrator(x, tiempo[0])

    logs = Logs(
        x=np.empty((tiempo.size, n, dof)),
        u=np.empty((tiempo.size, n, dof)),
        re=np.zeros((tiempo.size, n)),
        wre=np.zeros((tiempo.size, n)),
        edges=np.zeros(tiempo.size),
        debug=np.zeros((tiempo.size, n, dof))
        )
    logs.x[0] = x
    logs.u[0] = np.zeros((n, dof))
    logs.edges[0] = len(E)    # cantidad de enlaces

    hops = np.empty(n, dtype=int)
    for i in nodes:
        subset_found = False
        h = 0
        while not subset_found:
            h += 1
            Ai, xi = subsets.multihop_subframework(A, x, i, h)
            re = rigidity_eigenvalue(Ai, xi)
            wre = weighted_rigidity_eigenvalue(xi, midpoint[1], steepness[1])
            if re > 1e-3:
                subset_found = True
                logs.re[0, i] = re
                logs.wre[0, i] = wre
                print(
                    'Node {}, hops = {}, RE = {} ~ {}'.format(i, h, re, wre))
        hops[i] = h

    frames = np.empty((tiempo.size, 3), dtype=np.ndarray)
    frames[0] = tiempo[0], x, E

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    logs, t_perf, frames = run(steps, logs, t_perf, A, dinamica, frames)

    x = logs.x
    u = logs.u
    re = logs.re
    wre = logs.wre
    edges = logs.edges
    debug = logs.debug
    print(re[3])
    print(wre[3])

    st = arg.tf - arg.ti
    rt = sum(t_perf)
    prompt = 'RT={:.3f} secs, ST={:.3f} secs  ==>  RTF={:.3f}'
    print(prompt.format(rt, st, st / rt))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(10, 4))

    axes[0, 0].set_xlabel(r'$t [seg]$')
    axes[0, 0].set_ylabel('x [m]')
    axes[0, 0].grid(1)
    # axes[0, 0].plot(tiempo, x[..., 0])
    axes[0, 0].plot(tiempo, logs.debug[..., 0])
    plt.gca().set_prop_cycle(None)

    axes[0, 1].set_xlabel(r'$t [seg]$')
    axes[0, 1].set_ylabel('y [m]')
    axes[0, 1].grid(1)
    # axes[0, 1].plot(tiempo, x[..., 1])
    axes[0, 1].plot(tiempo, logs.debug[..., 1])
    plt.gca().set_prop_cycle(None)

    axes[1, 0].set_xlabel(r'$t [seg]$')
    axes[1, 0].set_ylabel(r'$u_x [m/s]$')
    axes[1, 0].grid(1)
    axes[1, 0].plot(tiempo, u[..., 0])
    plt.gca().set_prop_cycle(None)

    axes[1, 1].set_xlabel(r'$t [seg]$')
    axes[1, 1].set_ylabel(r'$u_y [m/s]$')
    axes[1, 1].grid(1)
    axes[1, 1].plot(tiempo, u[..., 1])
    fig.savefig('/tmp/control.pdf', format='pdf')

    fig, axes = plt.subplots(3, 1)

    axes[0].set_xlabel(r'$t [seg]$')
    axes[0].set_ylabel(r'$\rho$')
    axes[0].grid(1)
    # axes[0].semilogy(
    #     tiempo, re[:, 0].clip(1e-6), color='purple', ls='--', zorder=10)
    axes[0].semilogy(tiempo, re)
    axes[0].set_ylim(bottom=1e-6)
    plt.gca().set_prop_cycle(None)

    axes[1].set_xlabel(r'$t [seg]$')
    axes[1].set_ylabel(r'$\rho_{w}$')
    axes[1].grid(1)
    axes[1].semilogy(tiempo, wre)
    axes[1].set_ylim(bottom=1e-6)
    plt.gca().set_prop_cycle(None)

    axes[2].set_xlabel(r'$t [seg]$')
    axes[2].set_ylabel(r'$m$')
    axes[2].grid(1)
    axes[2].plot(tiempo, edges)
    axes[2].hlines(2*n-2, tiempo[0], tiempo[-1], color='k', ls='--')
    axes[2].set_ylim(bottom=100)
    fig.savefig('/tmp/metricas.pdf', format='pdf')

    if arg.animate:
        fig, ax = network.plot.figure()
        ax.set_xlim(-2*lim, 2*lim)
        ax.set_ylim(-2*lim, 2*lim)
        anim = network.plot.Animate(fig, ax, arg.h/2, frames, maxlen=50)
        one_hop_rigid = hops == 1
        two_hop_rigid = hops == 2
        three_hop_rigid = hops == 3
        four_hop_rigid = hops == 4
        five_hop_rigid = hops == 5
        anim.set_teams(
            {'ids': nodes[one_hop_rigid], 'tail': True,
                'style': {'color': 'gray', 'marker': 'o', 'markersize': 5}},
            {'ids': nodes[two_hop_rigid], 'tail': True,
                'style': {'color': 'orange', 'marker': 'D', 'markersize': 5}},
            {'ids': nodes[three_hop_rigid], 'tail': True,
                'style': {'color': 'purple', 'marker': 's', 'markersize': 5}},
            {'ids': nodes[four_hop_rigid], 'tail': True,
                'style': {'color': 'red', 'marker': '^', 'markersize': 5}},
            {'ids': nodes[five_hop_rigid], 'tail': True,
                'style': {'color': 'lightpink', 'marker': 'X', 'markersize': 5}})  # noqa
        # anim.set_teams(
        # {'ids': np.delete(nodes, 28), 'tail': True,
        #     'style': {'color': 'gray', 'marker': 'o', 'markersize': 5}},
        # {'ids': np.array([28]), 'tail': True,
        #     'style': {'color': 'orange', 'marker': 'D', 'markersize': 6}})
        anim.set_edgestyle(color='0.4', alpha=0.6, lw=0.8)
        # anim.ax.legend(ncol=5)
        # anim.run()
        anim.run('/tmp/multihop.mp4')

    plt.show()
