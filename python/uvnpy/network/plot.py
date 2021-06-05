#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date lun dic 14 15:37:42 -03 2020
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from transformations import unit_vector

from . import core


def figure(nrows=1, ncols=1, **kwargs):
    fig = plt.figure(**kwargs)
    gs = fig.add_gridspec(nrows, ncols)
    return gs


def xy(subplots):
    """ Grafico XY.

    args:
        subplots: matplotlib.gridspec.GridSpec o
            tupla de matplotlib.gridspec.SubplotSpec

    returns:
        axes: np.ndarray conteniendo los axes.
    """
    axes = np.empty(len(tuple(subplots)), dtype=object)
    for i, sp in enumerate(subplots):
        gs = sp.get_gridspec()
        axes[i] = gs.figure.add_subplot(sp)
        axes[i].set_aspect('equal')
        axes[i].grid(1)
        axes[i].set_xlabel(r'$x$')
        axes[i].set_ylabel(r'$y$')
    return axes


def nodes(ax, p, **kwargs):
    """Plotear nodos."""
    nodes = ax.scatter(p[..., 0], p[..., 1], **kwargs)
    return nodes


def edges(ax, p, E, **kwargs):
    """Plotear enlaces."""
    edges = mpl.collections.LineCollection(p[E], **kwargs)
    ax.add_artist(edges)
    return edges


def graph(ax, p, E, **kwargs):
    """Plotear grafo."""
    V = nodes(ax, p, **kwargs)
    kwargs.update({'color': V.get_facecolor()})
    E = edges(ax, p, E, **kwargs)
    return V, E


def motions(ax, p, m, **kwargs):
    """Plotear movimientos infintesimales centrados en los nodos.

    p, m = array (n, dof)
    """
    if not kwargs.get('width'):
        kwargs.update({'width': 0.003})
    if not kwargs.get('head_width'):
        kwargs.update({'head_width': 0.07})
    if not kwargs.get('color'):
        kwargs.update({'color': 'k'})
    kwargs.update({'length_includes_head': True})
    arrows = [ax.arrow(q[0], q[1], u[0], u[1], **kwargs) for q, u in zip(p, m)]
    return arrows


def displacements(ax, p, **kwargs):
    """Plotear desplazamientos centrados en el origen."""
    if not kwargs.get('width'):
        kwargs.update({'width': 0.003})
    if not kwargs.get('head_width'):
        kwargs.update({'head_width': 0.07})
    if not kwargs.get('color'):
        kwargs.update({'color': 'k'})
    kwargs.update({'length_includes_head': True})
    E = kwargs.pop('E', None)
    if E is None:
        E = core.complete_undirected_edges(range(len(p)))
    r = unit_vector(p[E[:, 0]] - p[E[:, 1]], axis=1)
    arrows = [ax.arrow(0, 0, d[0], d[1], **kwargs) for d in r]
    return arrows
