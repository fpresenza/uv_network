#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import progressbar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from uvnpy.dynamics.unicycle import Unicycle


# ------------------------------------------------------------------
# Functions and Classes
# ------------------------------------------------------------------

def triangle_vertices(pose, scale=1.0):
    """Return coordinates of a triangle representing the robot."""
    # Triangle points in robot frame

    x = pose[..., 0]
    y = pose[..., 1]
    ct = scale * np.cos(pose[..., 2])
    st = scale * np.sin(pose[..., 2])

    vertices = np.empty((len(pose), 3, 2), dtype=float)
    vertices[..., 0, 0] = ct + x
    vertices[..., 0, 1] = st + y
    vertices[..., 1, 0] = 0.5 * (-ct - st) + x
    vertices[..., 1, 1] = 0.5 * (-st + ct) + y
    vertices[..., 2, 0] = 0.5 * (-ct + st) + x
    vertices[..., 2, 1] = 0.5 * (-st - ct) + y

    return vertices


# ------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-s', '--simu_step',
    default=1, type=float, help='simulation step in milli seconds'
)
parser.add_argument(
    '-t', '--simu_time',
    default=1.0, type=float, help='total simulation time in seconds'
)
arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
simu_time = arg.simu_time
simu_step = arg.simu_step / 1000.0
time_steps = np.arange(0.0, simu_time, simu_step)

unicycle = Unicycle(x=np.array([2.0, 1.0, 0.0 * np.pi]))
pose_log = np.empty((len(time_steps), 3), dtype=float)

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=arg.simu_time).start()
for k, t in enumerate(time_steps):

    unicycle.step(t, v=-1.0, w=0.5 if t < 5.0 else 0.0)
    pose_log[k] = unicycle.pose()

    bar.update(np.round(t, 3))

bar.finish()

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
# -- pose vs. time -- #
fig, ax = plt.subplots(3)

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[0].plot(
    time_steps, pose_log[:, 0], label=r'$x$', color='C0', ds='steps-post'
)
ax[1].plot(
    time_steps, pose_log[:, 1], label=r'$y$', color='C1', ds='steps-post'
)
ax[2].plot(
    time_steps, pose_log[:, 2], label=r'$\theta$', color='C2', ds='steps-post'
)

ax[0].legend()
ax[1].legend()
ax[2].legend()

# -- pose vs. xy -- #
fig, ax = plt.subplots()

ax.set_xlim(-10.0, 10.0)
ax.set_ylim(-10.0, 10.0)
ax.set_aspect('equal')
ax.grid()

triangles = [
    Polygon(vert) for vert in triangle_vertices(pose_log[::1000], scale=0.5)
]

ax.add_collection(
    PatchCollection(triangles, facecolor='none', edgecolor='blue', linewidth=1)
)


plt.show()
