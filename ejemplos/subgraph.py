#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on dom abr 11 18:34:34 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

import uvnpy.network.core as network
import uvnpy.network.disk_graph as disk_graph


N = 10
i = 4
V = np.arange(N)
L = 100
dmax = 65

p = np.array([
    [42.44679404, 44.08424234],
    [-9.99520811, -50.38700558],
    [-45.6380013, 57.12975081],
    [4.65664446, 28.77567225],
    [-13.99520811, -24.38700558],
    [-53.49395679, -12.66659477],
    [-3.89738138, 66.84358176],
    [-49.89541337, -69.58647407],
    [45.77263986, 59.81015092],
    [-53.60548283, 39.45235792]])

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax in axes:
    ax.set_aspect('equal')
    ax.grid(1)
    ax.set_xlim(-70, 65)
    ax.set_ylim(-80, 80)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.text(p[i, 0] + 5, p[i, 1] - 2, r'$i$', color='g')

E = disk_graph.edges(p, dmax)
# print(E)

Ri = disk_graph.neighborhood(p, i, dmax, inclusive=True)
pi = p[Ri]
qi = p[~Ri]
Ni = disk_graph.neighborhood(p, i, dmax)
Epi = disk_graph.edges(pi, dmax)
Epi_comp = network.complete_undirected_edges(V[Ri])
Eqi = disk_graph.edges(qi, dmax)
Epq = disk_graph.inter_edges(pi, qi, dmax)
Epq[:, 1] += len(pi)
# Ei = disk_graph.local_neighbors(p[i], dmax)
# print(Epq)
# print('pi = {}'.format(pi))
# print('qi = {}'.format(qi))
# print(Epi)
# print(Epi_comp)

# ax 1 : subgrafo i
network.plot_nodes(axes[0], qi, color='b', marker='o', alpha=0.5)
network.plot_edges(axes[0], qi, Eqi, color='0.2', lw=0.8)
network.plot_edges(axes[0], np.vstack([pi, qi]), Epq, color='0.2', lw=0.8)
network.plot_nodes(axes[0], p[Ni], color='g', marker='o', alpha=0.5)
network.plot_nodes(axes[0], p[i], color='g', marker='o')
network.plot_edges(axes[0], pi, Epi, color='g', lw=0.8)
axes[0].text(
    p[i, 0] - 19, p[i, 1] + 10, r'$\mathcal{G}_i$',
    color='g', fontsize=15)
axes[0].set_title(r'Subgrafo local $i$')

# ax 2 : subgrafo i completo
network.plot_nodes(axes[1], qi, color='b', marker='o', alpha=0.5)
network.plot_edges(axes[1], qi, Eqi, color='0.2', lw=0.8)
network.plot_edges(axes[1], np.vstack([pi, qi]), Epq, color='0.2', lw=0.8)
network.plot_nodes(axes[1], p[Ni], color='g', marker='o', alpha=0.5)
network.plot_nodes(axes[1], p[i], color='g', marker='o')
network.plot_edges(axes[1], p, Epi_comp, color='g', ls='--', lw=0.8)
axes[1].text(
    p[i, 0] - 19, p[i, 1] + 10, r'$\widehat{\mathcal{G}}_i$',
    color='g', fontsize=15)
axes[1].set_title(r'Subgrafo local completo $i$')

# ax 3 : interacción i
pni = np.delete(p, i, axis=0)
Epni = disk_graph.edges(pni, dmax)
Eini = disk_graph.inter_edges(p[[i]], pni, dmax)
Eini[:, 1] += 1
pNi = p[Ni]
network.plot_nodes(axes[2], p[i], color='g', marker='o')
network.plot_nodes(axes[2], p[Ni], color='g', marker='o', alpha=0.5)
network.plot_nodes(axes[2], qi, color='b', marker='o', alpha=0.5)
network.plot_edges(axes[2], pni, Epni, color='0.6', lw=0.8)
network.plot_edges(axes[2], np.vstack([p[i], pni]), Eini, color='g')

t_off = np.array([
    [-11, -2],
    [6, -2],
    [2, 3],
    [-11, 0]])
a_off = np.array([
    [9, -1],
    [-3, 0],
    [0, 0],
    [5, -1]])
for j, ni in enumerate(pNi):
    r = ni - p[i]
    t = p[i] + r/2 + t_off[j]
    a = t + a_off[j]
    ur = r / np.linalg.norm(r)
    idx = 'u_{i' + str(j + 1) + '}'
    axes[2].text(t[0], t[1], r'${}$'.format(idx), color='b')
    axes[2].arrow(a[0], a[1], 4 * ur[0], 4 * ur[1], head_width=2, color='b')

t_off = np.array([
    [-4, 0],
    [9, -2],
    [2, 11],
    [-4, 4]])
a_off = np.array([
    [1, 0],
    [-3, 0],
    [0, -8],
    [4, 0]])
for j, ni in enumerate(pNi):
    r = ni - p[i]
    t = p[i] + r/2 - t_off[j]
    a = t - a_off[j]
    ur = r / np.linalg.norm(r)
    axes[2].text(t[0], t[1], r'$x_{}$'.format(j+1), color='r')
    axes[2].arrow(a[0], a[1], -4 * ur[0], -4 * ur[1], head_width=2, color='r')

ax.text(pNi[0, 0] + 5, pNi[0, 1] - 5, r'$j_1$', color='g')
ax.text(pNi[1, 0] + 5, pNi[1, 1] - 5, r'$j_2$', color='g')
ax.text(pNi[2, 0] + 3, pNi[2, 1] + 3, r'$j_3$', color='g')
ax.text(pNi[3, 0] + 5, pNi[3, 1] - 4, r'$j_4$', color='g')


axes[2].set_title(r'Interacción local $i$')


# plt.show()
plt.savefig('/tmp/subgrafos.png')
