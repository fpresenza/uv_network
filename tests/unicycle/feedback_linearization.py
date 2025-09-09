#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import progressbar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from uvnpy.dynamics.unicycle import Unicycle
from uvnpy.toolkit.geometry import triangle


# ------------------------------------------------------------------
# Functions and Classes
# ------------------------------------------------------------------


def feedback_linearization(u, theta, d=1.0):
    ct = np.cos(theta)
    st = np.sin(theta)

    v = u[0] * ct + u[1] * st
    w = (-u[0] * st + u[1] * ct) / d

    return v, w


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

unicycle = Unicycle(x=np.array([2.0, 1.0, 0.9 * np.pi]))
wmax = 0.5
pose_log = np.empty((len(time_steps), 3), dtype=float)
vel_log = np.empty((len(time_steps), 3), dtype=float)
ctrl_log = np.empty((len(time_steps), 2), dtype=float)

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=arg.simu_time).start()
for k, t in enumerate(time_steps):

    u = np.array([1.0, 0.0])
    pose = unicycle.pose()
    v, w = feedback_linearization(u, pose[2], 1.0 / wmax)
    unicycle.step(t, v, w)
    pose_log[k] = pose
    vel_log[k] = unicycle.vel()
    ctrl_log[k] = u.copy()

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

center = pose_log[::500, :2]
heading = pose_log[::500, 2]

triangles = [
    Polygon(vert) for vert in triangle(center, heading, height=0.5)
]

ax.add_collection(
    PatchCollection(triangles, facecolor='none', edgecolor='blue', linewidth=1)
)

# -- vel vs. time -- #
fig, ax = plt.subplots(3)

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[0].plot(
    time_steps, vel_log[:, 0],
    label=r'$\dot{x}$', color='C0', ds='steps-post'
)
ax[0].plot(
    time_steps, ctrl_log[:, 0],
    color='C0', ls='--', ds='steps-post'
)
ax[1].plot(
    time_steps, vel_log[:, 1],
    label=r'$\dot{y}$', color='C1', ds='steps-post'
)
ax[1].plot(
    time_steps, ctrl_log[:, 1],
    color='C1', ls='--', ds='steps-post'
)
ax[2].plot(
    time_steps, vel_log[:, 2],
    label=r'$\dot{\theta}$', color='C2', ds='steps-post'
)

ax[0].legend()
ax[1].legend()
ax[2].legend()


plt.show()
