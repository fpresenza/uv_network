#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mié 29 dic 2021 16:41:13 -03
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.toolkit.functions import logistic_saturation

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = np.loadtxt('data/t.csv', delimiter=',')
tc = np.loadtxt('data/tc.csv', delimiter=',')
x = np.loadtxt('data/position.csv', delimiter=',')
hatx = np.loadtxt('data/est_position.csv', delimiter=',')
cov = np.loadtxt('data/covariance.csv', delimiter=',')
u_t = np.loadtxt('data/target_action.csv', delimiter=',')
u_c = np.loadtxt('data/collision_action.csv', delimiter=',')
u_r = np.loadtxt('data/rigidity_action.csv', delimiter=',')
v = np.loadtxt('data/vel_meas_err.csv', delimiter=',')
g = np.loadtxt('data/gps_meas_err.csv', delimiter=',')
r = np.loadtxt('data/range_meas_err.csv', delimiter=',')
fre = np.loadtxt('data/fre.csv', delimiter=',')
re = np.loadtxt('data/re.csv', delimiter=',')
A = np.loadtxt('data/adjacency.csv', delimiter=',')
state_extents = np.loadtxt('data/state_extents.csv', delimiter=',')
targets = np.loadtxt('data/targets.csv', delimiter=',')

n = int(len(x[0])/2)
nodes = np.arange(n)
state_extents = state_extents.astype(int)

# reshapes
x = x.reshape(len(t), n, 2)
hatx = hatx.reshape(len(t), n, 2)
cov = cov.reshape(len(t), n, 2)
u_t = u_t.reshape(len(t), n, 2)
u_c = u_c.reshape(len(t), n, 2)
u_r = u_r.reshape(len(t), n, 2)
v = v.reshape(len(t), n, 2)
g = g.reshape(len(t), n, 2)
re = re.reshape(len(t), -1)
A = A.reshape(len(t), n, n)
targets = targets.reshape(len(t), -1, 3)

# ------------------------------------------------------------------
# Plot target control action
# ------------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(4.0, 4.0))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
ax[0].tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')

ax[0].set_xlabel('$t$ [$s$]', fontsize=8)
ax[0].set_ylabel(r'$u_{t, x}$ [$m/s$]', fontsize=8)
ax[0].grid(1)
ax[0].plot(t, u_t[..., 0], lw=0.8, ds='steps-post')

ax[1].tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')

ax[1].set_xlabel('$t$ [$s$]', fontsize=8)
ax[1].set_ylabel(r'$u_{t, y}$ [$m/s$]', fontsize=8)
ax[1].grid(1)
ax[1].plot(t, u_t[..., 1], lw=0.8, ds='steps-post')
fig.savefig('data/time/target_control.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot collision control action
# ------------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(4.0, 4.0))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
ax[0].tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')

ax[0].set_xlabel('$t$ [$s$]', fontsize=8)
ax[0].set_ylabel(r'$u_{c, x}$ [$m/s$]', fontsize=8)
ax[0].grid(1)
ax[0].plot(t, u_c[..., 0], lw=0.8, ds='steps-post')

ax[1].tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')

ax[1].set_xlabel('$t$ [$s$]', fontsize=8)
ax[1].set_ylabel(r'$u_{c, y}$ [$m/s$]', fontsize=8)
ax[1].grid(1)
ax[1].plot(t, u_c[..., 1], lw=0.8, ds='steps-post')
fig.savefig('data/time/collision_control.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot rigidity control action
# ------------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(4.0, 4.0))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
ax[0].tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')

ax[0].set_xlabel('$t$ [$s$]', fontsize=8)
ax[0].set_ylabel(r'$u_{r, x}$ [$m/s$]', fontsize=8)
ax[0].grid(1)
ax[0].plot(t, u_r[..., 0], lw=0.8, ds='steps-post')

ax[1].tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')

ax[1].set_xlabel('$t$ [$s$]', fontsize=8)
ax[1].set_ylabel(r'$u_{r, y}$ [$m/s$]', fontsize=8)
ax[1].grid(1)
ax[1].plot(t, u_r[..., 1], lw=0.8, ds='steps-post')
fig.savefig('data/time/rigidity_control.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot control action composition
# ------------------------------------------------------------------
u = logistic_saturation(u_t + u_c + u_r, limit=2.5)
fig, ax = plt.subplots(2, 1, figsize=(4.0, 4.0))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
ax[0].tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')

ax[0].set_xlabel('$t$ [$s$]', fontsize=8)
ax[0].set_ylabel(r'$u_{x}$ [$m/s$]', fontsize=8)
ax[0].grid(1)
ax[0].plot(t, u[..., 0], lw=0.8, ds='steps-post')

ax[1].tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')

ax[1].set_xlabel('$t$ [$s$]', fontsize=8)
ax[1].set_ylabel(r'$u_{y}$ [$m/s$]', fontsize=8)
ax[1].grid(1)
ax[1].plot(t, u[..., 1], lw=0.8, ds='steps-post')
fig.savefig('data/time/rigidity_control.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot state extents
# ------------------------------------------------------------------
# fig, ax = plt.subplots(figsize=(4.0, 2.0))
# fig.subplots_adjust(
#     bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
# ax.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='x-small')

# ax.set_xlabel('$t$ [$s$]', fontsize=8)
# ax.set_ylabel('state extents', fontsize=8)
# ax.grid(1)
# ax.plot(t, state_extents, lw=0.8, marker='.', ds='steps-post')
# fig.savefig('data/time/state_extents.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot velocity measurement error
# ------------------------------------------------------------------
# e_vel = np.sqrt(v[..., 0]**2 + v[..., 1]**2)
# fig, ax = plt.subplots(figsize=(4.0, 2.0))
# fig.subplots_adjust(
#     bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
# ax.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='x-small')

# ax.set_xlabel('$t$ [$s$]', fontsize=8)
# ax.set_ylabel(r'$\Vert e_{\mathrm{vel}} \Vert$ [$m/s$]', fontsize=8)
# ax.grid(1)
# ax.plot(
#     t, np.nanmedian(e_vel, axis=1),
#     lw=0.8, label='median', ds='steps-post'
# )
# ax.fill_between(
#     t,
#     np.nanmin(e_vel, axis=1),
#     np.nanmax(e_vel, axis=1),
#     alpha=0.3
# )
# ax.set_ylim(bottom=0.0)
# ax.legend(
#     fontsize=8, handlelength=1, labelspacing=0.4,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.,
#     ncol=2, columnspacing=1)
# fig.savefig('data/time/vel_meas_err.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot gps measurement error
# ------------------------------------------------------------------
# e_gps = np.sqrt(g[..., 0]**2 + g[..., 1]**2)
# fig, ax = plt.subplots(figsize=(4.0, 2.0))
# fig.subplots_adjust(
#     bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
# ax.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='x-small')

# ax.set_xlabel('$t$ [$s$]', fontsize=8)
# ax.set_ylabel(r'$\Vert e_{\mathrm{gps}} \Vert$ [$m/s$]', fontsize=8)
# ax.grid(1)
# ax.plot(
#     t, np.nanmedian(e_gps, axis=1),
#     lw=0.8, label='median', ds='steps-post'
# )
# ax.fill_between(
#     t,
#     np.nanmin(e_gps, axis=1),
#     np.nanmax(e_gps, axis=1),
#     alpha=0.3
# )
# ax.set_ylim(bottom=0.0)
# ax.legend(
#     fontsize=8, handlelength=1, labelspacing=0.4,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.,
#     ncol=2, columnspacing=1)
# fig.savefig('data/time/gps_meas_err.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot range measurement error
# ------------------------------------------------------------------
# e_range = np.abs(r)
# fig, ax = plt.subplots(figsize=(4.0, 2.0))
# fig.subplots_adjust(
#     bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
# ax.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='x-small')

# ax.set_xlabel('$t$ [$s$]', fontsize=8)
# ax.set_ylabel(r'$\Vert e_{\mathrm{range}} \Vert$ [$m$]', fontsize=8)
# ax.grid(1)
# ax.plot(
#     t, np.nanmedian(e_range, axis=1),
#     lw=0.8, label='median', ds='steps-post'
# )
# ax.fill_between(
#     t,
#     np.nanmin(e_range, axis=1),
#     np.nanmax(e_range, axis=1),
#     alpha=0.3
# )
# ax.set_ylim(bottom=0.0)
# ax.legend(
#     fontsize=8, handlelength=1, labelspacing=0.4,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.,
#     ncol=2, columnspacing=1)
# fig.savefig('data/time/range_meas_err.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot position x
# ------------------------------------------------------------------
# fig, ax = plt.subplots(figsize=(2.75, 1.75))
# fig.subplots_adjust(
#     bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.2)
# ax.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='x-small')
# ax.grid(1, lw=0.4)

# ax.set_xlabel(r'$t$ [$s$]', fontsize=8)
# ax.set_ylabel('position-$x$ [$m$]', fontsize=8)
# ax.plot(t, x[..., 0], lw=0.8, ds='steps-post')
# fig.savefig('data/time/pos_x.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot position y
# ------------------------------------------------------------------
# fig, ax = plt.subplots(figsize=(2.75, 1.75))
# fig.subplots_adjust(
#     bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.2)
# ax.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='x-small')
# ax.grid(1, lw=0.4)

# ax.set_xlabel(r'$t$ [$s$]', fontsize=8)
# ax.set_ylabel('position-$y$ [$m$]', fontsize=8)
# ax.plot(t, x[..., 1], lw=0.8, ds='steps-post')
# fig.savefig('data/time/pos_y.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot eigenvalues
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.0, 2.0))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')
ax.grid(1, lw=0.4)

ax.set_xlabel(r'$t$ [$s$]', fontsize=8)
ax.set_ylabel('Autovalores \n de Rigidez', fontsize=8)
ax.semilogy(
    t, np.median(re, axis=1),
    lw=0.8,
    marker='.',
    ds='steps-post',
    label='median'
)
ax.semilogy(
    t, fre,
    lw=0.8,
    ls='--',
    color='k',
    ds='steps-post',
    label=r'framework'
)
ax.set_ylim(bottom=1e-4, top=3)
ax.fill_between(
    t,
    np.min(re, axis=1),
    np.max(re, axis=1),
    alpha=0.3
)
# ax.vlines(tc, 1e-4, 1, lw=0.75, color='g', alpha=0.5)
ax.set_ylim(bottom=1e-4)
ax.legend(
    fontsize=8, handlelength=1, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=2, columnspacing=1)
fig.savefig('data/time/eigenvalues.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot position error
# ------------------------------------------------------------------
# err = np.sqrt(np.square(x - hatx).sum(axis=-1))
# fig, ax = plt.subplots(figsize=(4.0, 2.0))
# fig.subplots_adjust(
#     bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
# ax.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='x-small')
# ax.grid(1, lw=0.4)
# ax.set_xlabel(r'$t$ [$s$]', fontsize=8)
# ax.set_ylabel(r'$\Vert e_{\mathrm{pos}} \Vert$ [$m$]', fontsize=8)
# ax.plot(t, np.median(err, axis=1), lw=0.8, label='median', ds='steps-post')
# ax.fill_between(
#     t,
#     np.min(err, axis=1),
#     np.max(err, axis=1),
#     alpha=0.3
# )
# ax.set_ylim(bottom=0.0)
# ax.legend(
#     fontsize=8, handlelength=1, labelspacing=0.4,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.,
#     ncol=2, columnspacing=1)
# fig.savefig('data/time/pos_error.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot covariance
# ------------------------------------------------------------------
# fig, ax = plt.subplots(figsize=(4.0, 2.0))
# fig.subplots_adjust(
#     bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
# ax.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='x-small')
# ax.grid(1, lw=0.4)
# ax.set_xlabel(r'$t$ [$s$]', fontsize=8)
# ax.set_ylabel(
#     r'$\mathrm{tr}(\mathrm{cov}(e_{\mathrm{pos}}))$ [$m^2$]', fontsize=8
# )
# # ax.plot(
#     t, np.sqrt(cov[..., 0]**2 + cov[..., 1]**2), lw=0.8, ds='steps-post')
# ax.semilogy(
#     t, np.median(cov[..., 0] + cov[..., 1], axis=1),
#     lw=0.8, label='median', ds='steps-post'
# )
# ax.fill_between(
#     t,
#     np.min(cov[..., 0] + cov[..., 1], axis=1),
#     np.max(cov[..., 0] + cov[..., 1], axis=1),
#     alpha=0.3
# )
# ax.set_ylim(bottom=1e-2)
# ax.legend(
#     fontsize=8, handlelength=1, labelspacing=0.4,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.,
#     ncol=2, columnspacing=1)
# fig.savefig('data/time/pos_cov.png', format='png', dpi=360)

plt.show()
