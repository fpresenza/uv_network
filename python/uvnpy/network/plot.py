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
import matplotlib.animation as ani
import collections
from transformations import unit_vector

from . import core


class Animate(object):
    def __init__(self, fig, ax, h, frames, maxlen=1, file=None):
        """ Animar grafo de posicion"""
        self.fig = fig
        self.ax = ax
        self.h = h
        self.frames = frames
        self.p_tail = collections.deque(maxlen=maxlen)
        _, Pk, _ = self.frames[0]
        self.p_tail.append(Pk)
        self.stamp = self.ax.text(
            0.01, 0.01, r'{:.2f} secs'.format(0.0),
            verticalalignment='bottom', horizontalalignment='left',
            transform=self.ax.transAxes, color='green', fontsize=10)
        self.anim = None
        self.teams = {}
        edgestyle = {'color': '0.2', 'linewidth': 0.7}
        self.edges = mpl.collections.LineCollection([], **edgestyle)
        self.ax.add_artist(self.edges)

    def set_teams(self, *teams):
        for i, team in enumerate(teams):
            name = team.get('name', 'Team {}'.format(i))
            style = team.get(
                'style', {'color': 'b', 'marker': 'o', 'markersize': '5'})
            self.teams[name] = {}
            self.teams[name]['ids'] = team['ids']
            style.update(label=name)
            line = self.ax.plot([], [], ls='', **style)
            self.teams[name]['points'] = line[0]
            if team.get('tail'):
                style.update(markersize=0.5, alpha=0.4)
                style.pop('label')
                line = self.ax.plot([], [], ls='', **style)
                self.teams[name]['tail'] = line[0]

    def set_edgestyle(self, **style):
        self.edges = mpl.collections.LineCollection([], **style)
        self.ax.add_artist(self.edges)

    def update(self, frame):
        tk, Pk, Ek = frame
        q = np.array(self.p_tail)
        self.p_tail.append(Pk)
        for team in self.teams.values():
            ids = team['ids']
            points = team['points']
            tail = team.get('tail')

            p = Pk[ids]
            points.set_data(p[:, 0], p[:, 1])
            if tail:
                tail.set_data(q[:, ids, 0], q[:, ids, 1])
        self.stamp.set_text('$t = {:.1f} s$'.format(tk))
        if len(Ek) > 0:
            self.edges.set_segments(Pk[Ek])
        else:
            self.edges.set_segments([])
        return self.ax.lines + self.ax.artists + self.ax.texts

    def run(self, file=None):
        self.anim = ani.FuncAnimation(
            self.fig,
            self.update,
            frames=self.frames,
            interval=1000 * self.h,
            blit=True
        )
        if file:
            self.anim.save(
                file,
                fps=1. / self.h,
                dpi=200,
                extra_args=['-vcodec', 'libx264'])


def figure(nrows=1, ncols=1, **kwargs):
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    _axes = np.asarray(axes)
    for ax in _axes.flat:
        ax.set_aspect('equal')
        ax.grid(1)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
    return fig, axes


def nodes(ax, p, **kwargs):
    """Plotear nodos."""
    nodes = ax.scatter(p[..., 0], p[..., 1], **kwargs)
    return nodes


def edges(ax, p, E, **kwargs):
    """Plotear edges."""
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
