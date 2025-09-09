#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date lun dic 14 15:37:42 -03 2020
"""
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .geometry import triangle


def points(ax, p, **kwargs):
    """Plot graph nodes as points."""
    d = p.shape[-1]
    if d == 2:
        nodes = ax.scatter(p[..., 0], p[..., 1], **kwargs)
    elif d == 3:
        nodes = ax.scatter(p[..., 0], p[..., 1], p[..., 2], **kwargs)
    return nodes


def triangles(ax, x, height, **kwargs):
    """Plot graph nodes as triangles in 2D."""
    nodes = [
        ax.add_patch(Polygon(vert, **kwargs))
        for vert in triangle(x[..., :2], x[..., 2], height)
    ]
    return nodes


def quiver(ax, p, u, **kwargs):
    d = p.shape[-1]
    if d == 2:
        nodes = ax.quiver(
            p[..., 0], p[..., 1],
            u[..., 0], u[..., 1], **kwargs)
    elif d == 3:
        nodes = ax.quiver(
            p[..., 0], p[..., 1], p[..., 2],
            u[..., 0], u[..., 1], u[..., 2], **kwargs)
    return nodes


def bars(ax, p, edges, **kwargs):
    """Plot graph edges as bars."""
    d = p.shape[-1]
    if d == 2:
        lines = LineCollection(p[edges], **kwargs)
    elif d == 3:
        lines = Line3DCollection(p[edges], **kwargs)
    ax.add_collection(lines)
    return lines


def arrows(ax, p, edges, **kwargs):
    """Plot graph edges as arrows."""
    d = p.shape[-1]
    arrows = p[edges]
    if d == 2:
        x, y = arrows[..., 0, 0], arrows[..., 0, 1]
        u, v = arrows[..., 1, 0], arrows[..., 1, 1]
        lines = ax.quiver(
            x, y,
            u - x, v - y,
            angles='xy', scale_units='xy', **kwargs
        )
    elif d == 3:
        x, y, z = arrows[..., 0, 0], arrows[..., 0, 1], arrows[..., 0, 2]
        u, v, w = arrows[..., 1, 0], arrows[..., 1, 1], arrows[..., 1, 2]
        lines = ax.quiver(
            x, y, z,
            u - x, v - y, w - z,
            **kwargs
        )
    return lines


def point_bar_framework(ax, p, edges, **kwargs):
    """Plot point bar framework"""
    nodes = points(ax, p, **kwargs)
    kwargs.update(color=nodes.get_facecolor())
    lines = bars(ax, p, edges, **kwargs)
    return nodes, lines


def point_arrow_framework(ax, p, edges, **kwargs):
    """Plot point arrow framework"""
    nodes = points(ax, p, **kwargs)
    kwargs.update(color=nodes.get_facecolor())
    lines = arrows(ax, p, edges, **kwargs)
    return nodes, lines


def triangle_bar_framework(ax, x, edges, height, **kwargs):
    """Plot triangle bar framework"""
    nodes = triangles(ax, x, height, **kwargs)
    lines = bars(ax, x[:, :2], edges, **kwargs)
    return nodes, lines


def triangle_arrow_framework(ax, x, edges, height, **kwargs):
    """Plot triangle arrow framework"""
    nodes = triangles(ax, x, height, **kwargs)
    lines = arrows(ax, x[:, :2], edges, **kwargs)
    return nodes, lines
