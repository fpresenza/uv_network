#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date lun dic 14 15:37:42 -03 2020
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.animation as ani
import collections
from transformations import unit_vector

from . import core


class Animate(object):
    def __init__(self, fig, ax, timestep, frames, maxlen=1):
        """ Animar grafo de posicion"""
        self.fig = fig
        self.ax = ax
        self.h = timestep
        self.frames = frames
        self.x_tail = collections.deque(maxlen=maxlen)
        Xk = self.frames[0][1]
        self.x_tail.append(Xk)
        self.stamp = self.ax.text(
            0.01, 0.01, r'{:.2f} secs'.format(0.0),
            verticalalignment='bottom', horizontalalignment='left',
            transform=self.ax.transAxes, color='green', fontsize=10)
        self.anim = None
        self.teams = {}
        edgestyle = {'color': '0.2', 'linewidth': 0.7, 'zorder': 10}
        self.set_edgestyle(**edgestyle)
        self._extra_artists = None

    def set_teams(self, *teams):
        for i, team in enumerate(teams):
            name = team.get('name', 'Team {}'.format(i))
            style = team.get(
                'style', {'color': 'b', 'marker': 'o', 'markersize': '5'})
            style.update(ls='', zorder=1)
            self.teams[name] = {}
            self.teams[name]['ids'] = team['ids']
            style.update(label=name)
            line = self.ax.plot([], [], **style)
            self.teams[name]['points'] = line[0]
            if team.get('tail') is True:
                style.update(markersize=0.5, alpha=0.4)
                style.pop('label')
                line = self.ax.plot([], [], **style)
                self.teams[name]['tail'] = line[0]

    def set_edgestyle(self, **style):
        self.edges = LineCollection([], **style)
        self.ax.add_artist(self.edges)

    def _update_extra_artists(self, frame):
        pass

    def set_extra_artists(self, *artists):
        self._extra_artists = []
        for artist in artists:
            self._extra_artists.append(artist)

    def update(self, frame):
        tk, Xk, Ek = frame[:3]
        q = np.array(self.x_tail)
        self.x_tail.append(Xk)
        for team in self.teams.values():
            ids = team['ids']
            points = team['points']
            tail = team.get('tail')

            x = Xk[ids]
            points.set_data(x[:, 0], x[:, 1])
            if tail is not None:
                tail.set_data(q[:, ids, 0], q[:, ids, 1])
        self.stamp.set_text('$t = {:.1f} s$'.format(tk))
        if len(Ek) > 0:
            self.edges.set_segments(Xk[Ek])
        else:
            self.edges.set_segments([])
        self._update_extra_artists(frame)
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


class Animate2(object):
    def __init__(self, fig, ax, timestep, frames, maxlen=1):
        """ Animar grafo de posicion.
        Permite a los nodos cambiar de team"""
        self.fig = fig
        self.ax = ax
        self.h = timestep
        self.frames = frames
        self.x_tail = collections.deque(maxlen=maxlen)
        Xk = self.frames[0][1]
        self.x_tail.append(Xk)
        self.stamp = self.ax.text(
            0.01, 0.01, r'{:.2f} secs'.format(0.0),
            verticalalignment='bottom', horizontalalignment='left',
            transform=self.ax.transAxes, color='green', fontsize=10)
        self.anim = None
        self.teams = {}
        edgestyle = {'color': '0.2', 'linewidth': 0.7, 'zorder': 10}
        self.set_edgestyle(**edgestyle)
        self._extra_artists = None

    def set_teams(self, teams):
        self.teams = teams.copy()
        for name, data in teams.items():
            style = data.get(
                'style', {'color': 'b', 'marker': 'o', 'markersize': '5'})
            style.update(ls='', zorder=1, label=name)
            line = self.ax.plot([], [], **style)
            self.teams[name]['points'] = line[0]
            if data.get('tail') is True:
                style.update(markersize=0.5, alpha=0.4)
                style.pop('label')
                line = self.ax.plot([], [], **style)
                self.teams[name]['tail'] = line[0]
            else:
                self.teams[name]['tail'] = None

    def set_edgestyle(self, **style):
        self.edges = LineCollection([], **style)
        self.ax.add_artist(self.edges)

    def _update_extra_artists(self, frame):
        pass

    def set_extra_artists(self, *artists):
        self._extra_artists = []
        for artist in artists:
            self._extra_artists.append(artist)

    def update(self, frame):
        tk, Xk, Ek, Tk = frame[:4]
        q = np.array(self.x_tail)
        self.x_tail.append(Xk)
        for data in self.teams.values():
            inteam = Tk == data['id']
            points = data['points']
            tail = data.get('tail')

            x = Xk[inteam]
            points.set_data(x[:, 0], x[:, 1])
            if tail is not None:
                tail.set_data(q[:, inteam, 0], q[:, inteam, 1])
        self.stamp.set_text('$t = {:.1f} s$'.format(tk))
        if len(Ek) > 0:
            self.edges.set_segments(Xk[Ek])
        else:
            self.edges.set_segments([])
        self._update_extra_artists(frame)
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
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            pass
        ax.grid(1)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
    return fig, axes


def nodes(ax, x, **kwargs):
    """Plotear nodos."""
    d = x.shape[-1]
    if d == 2:
        nodes = ax.scatter(x[..., 0], x[..., 1], **kwargs)
    elif d == 3:
        nodes = ax.scatter(x[..., 0], x[..., 1], x[..., 2], **kwargs)
    return nodes


def edges(ax, x, E, **kwargs):
    """Plotear edges."""
    n, d = x.shape
    if E.shape[0] == n and E.shape[1] == n:
        E = core.edges_from_adjacency(E)
    if d == 2:
        edges = LineCollection(x[E], **kwargs)
    elif d == 3:
        edges = Line3DCollection(x[E], **kwargs)
    ax.add_collection(edges)
    return edges


def graph(ax, x, E, **kwargs):
    """Plotear grafo."""
    V = nodes(ax, x, **kwargs)
    kwargs.update({'color': V.get_facecolor()})
    E = edges(ax, x, E, **kwargs)
    return V, E


def motions(ax, x, u, **kwargs):
    """Plotear movimientos infintesimales centrados en los nodos.

    x, m = array (n, dof)
    """
    if not kwargs.get('color'):
        kwargs.update({'color': 'k'})
    d = x.shape[-1]
    if d == 2:
        arrows = ax.quiver(
            x[..., 0], x[..., 1],
            u[..., 0], u[..., 1], **kwargs)
    elif d == 3:
        arrows = ax.quiver(
            x[..., 0], x[..., 1], x[..., 2],
            u[..., 0], u[..., 1], u[..., 2], **kwargs)
    return arrows


def displacements(ax, x, **kwargs):
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
        E = core.complete_undirected_edges(range(len(x)))
    r = unit_vector(x[E[:, 0]] - x[E[:, 1]], axis=1)
    arrows = [ax.arrow(0, 0, d[0], d[1], **kwargs) for d in r]
    return arrows
