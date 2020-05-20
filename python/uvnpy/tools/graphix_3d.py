#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 09 15:54:29 2020
@author: fran
"""
import numpy as np
import collections
import recordclass
import matplotlib
from uvnpy.tools.tools import Rotation
from gpsic.dibujo import dibujar_cono3, dibujar_cono


class Plot3D(object):
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'title'): self.title = kwargs.get('title', 'Plot3D')
        if not hasattr(self, 'color'): self.color = kwargs.get('color', 'b')
        if not hasattr(self, 'label'): self.label = kwargs.get('label', '')
        self.ls = kwargs.get('ls', '-')
        self.marker = kwargs.get('marker', '')
        self.markersize = kwargs.get('markersize', 5)
        self.fig = matplotlib.pyplot.figure()
        self.ax = self.fig.add_subplot(1,1,1, projection='3d')
        self.text = self.ax.text(0.01, 0.01, 4, r'%.2f secs'%0.0,
            verticalalignment='bottom', horizontalalignment='left',
            transform=self.ax.transAxes, color='green', fontsize=10)
        # self.ax.set_aspect('equal') # no implementado en la version 3.2.1 de matplotlib
        self.set_axis_label(
            kwargs.get('xlabel', '$X\,[\mathrm{m}]$'),
            kwargs.get('ylabel', '$Y\,[\mathrm{m}]$'),
            kwargs.get('zlabel', '$Z\,[\mathrm{m}]$')
        )
        self.set_axis_lim(
            kwargs.get('xlim', (-5,5)),
            kwargs.get('ylim', (-5,5)),
            kwargs.get('zlim', (-5,5)),
        )
        self.fig.suptitle(self.title, fontsize=16)

    def set_axis_label(self, x, y, z, **kwargs):
        fs = kwargs.get('fontsize', 13)
        self.ax.set_xlabel(x, fontsize=fs)
        self.ax.set_ylabel(y, fontsize=fs)
        self.ax.set_zlabel(z, fontsize=fs)

    def set_axis_lim(self, x, y, z, **kwargs):
        self.ax.set_xlim(*x)
        self.ax.set_ylim(*y)
        self.ax.set_zlim(*z)
        self.xlim = x
        self.ylim = y
        self.zlim = z

    def set_grid(self, ):
        # self.ax.set_xticks([-2, -1, 0, 1, 2])
        # self.ax.set_yticks([-2, -1, 0, 1, 2])
        # self.ax.set_zticks([0, 1])
        # self.ax.tick_params(axis='both', which='both', bottom=True,\
        # top=False, labelbottom=True, labelsize=13, grid_linewidth=0.35, pad=0.2)
        self.ax.minorticks_on()
        self.ax.grid(True)

    def clear(self):
        xlabel = self.ax.get_xlabel()
        ylabel = self.ax.get_ylabel()
        zlabel = self.ax.get_zlabel()
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()
        self.ax.clear()
        self.set_axis_label(xlabel, ylabel, zlabel)
        self.set_axis_lim(xlim, ylim, zlim)

    def add_line(self, X, Y, Z):
        self.ax.plot(X, Y, Z, color=self.color, ls=self.ls, marker=self.marker, markersize=self.markersize)


class ScatterPlot(Plot3D):
    def __init__(self, **kwargs):
        kwargs['ls'] = ''
        kwargs['marker'] = 'o'
        kwargs['title'] = '3D Scatter Plot'
        super(ScatterPlot, self).__init__(**kwargs)


class QuadrotorArtist(object):
    def __init__(self, ax, **kwargs):
        """ Draw simplistic quadcopter model in 3D """
        self.ax = ax
        self.armlen = kwargs.get('arm_length', 0.45)
        self.proplen = kwargs.get('prop_length', 0.15)
        # body
        self.body = self.__new_patch(('r', 'b', '0.3'))
        # shadow
        self.shadow = self.__new_patch(('0.5', '0.5', '0.5'))
        # camera
        self.with_cam = kwargs.get('camera', False)
        if self.with_cam:
            self.__add_camera(**kwargs)

    def __new_patch(self, colors):
        self.Patch = collections.namedtuple('Patch', 'arm propeller')
        p = self.Patch([],[])
        # arms
        p.arm.append(self.ax.plot([], [], [], color=colors[0])[0])
        p.arm.append(self.ax.plot([], [], [], color=colors[1])[0])
        # propellers
        p.propeller.append(self.ax.plot([], [], [], color=colors[2])[0])
        p.propeller.append(self.ax.plot([], [], [], color=colors[2])[0])
        p.propeller.append(self.ax.plot([], [], [], color=colors[2])[0])
        p.propeller.append(self.ax.plot([], [], [], color=colors[2])[0])
        return p

    def __draw_patch(self, patch, pos, tf, **kwargs):
        rx, ry, rz = tf.T
        vertex = (
            np.add(pos, self.armlen * rx),
            np.add(pos, self.armlen * ry),
            np.add(pos, self.armlen * -rx),
            np.add(pos, self.armlen * -ry),
        )
        patch.arm[0].set_data_3d(*zip(vertex[0], pos, vertex[1]))
        patch.arm[1].set_data_3d(*zip(vertex[2], pos, vertex[3]))
        circle = [self.proplen*np.cos(t)*rx + self.proplen*np.sin(t)*ry for t in np.arange(0, 2*np.pi, 0.2)]
        prop = [v + circle for v in vertex]
        patch.propeller[0].set_data_3d(*zip(*prop[0]))
        patch.propeller[1].set_data_3d(*zip(*prop[1]))
        patch.propeller[2].set_data_3d(*zip(*prop[2]))
        patch.propeller[3].set_data_3d(*zip(*prop[3]))

    def __add_camera(self, **kwargs):
        Cam = recordclass.recordclass('Cam', 'hfov zoom euler', defaults=(0.,0.,(0.,0.,0.)))
        self.cam = Cam(
            hfov=0.5 * kwargs.get('fov', np.radians(60.)),
            zoom=kwargs.get('zoom', 3.)
        )
        self.cone = [dibujar_cono3(self.ax, (0.,0.,0.), (0.,0.,-1.), self.cam.hfov, self.cam.zoom)]

    def draw(self, pos, euler, **kwargs):
        R = Rotation.Rzyx(*euler)
        # draw body of quadcopter
        self.__draw_patch(self.body, pos, R)
        # draw shadow of quadcopter
        Pxy = np.diag([1.,1.,0.])
        T = Pxy @ R
        pos_xy = (pos[0], pos[1], 0.)
        self.__draw_patch(self.shadow, pos_xy, T)       
        # draw camera fov
        if self.with_cam:
            self.cone[0].remove()
            self.cone = [dibujar_cono3(self.ax, pos, -R[:,2], self.cam.hfov, self.cam.zoom, alpha=0.3)]

        return self.collection()

    def collection(self):
        return self.body.arm + self.body.propeller + self.shadow.arm + self.shadow.propeller + self.cone


class Animation3D(object):
    def __init__(self, t, **kwargs):
        self.save = kwargs.get('save', False)
        if self.save: matplotlib.use("Agg")
        self.world = Plot3D(**kwargs)
        self.time = t
        self.collection = []
        self.frames = range(len(self.time))

    def add_quadrotor(self, *args, **kwargs):
        Q = collections.namedtuple('Q', 'artist p e')
        self.quads = [Q(QuadrotorArtist(self.world.ax, **kwargs), p, e) for p,e in args]
        
    def run(self, **kwargs):
        interval = self.time[1] - self.time[0]
        fps = interval**-1

        def init():
            self.collection += [self.world.text]
            self.collection += [artist for quad in self.quads for artist in quad.artist.draw(quad.p[0].flatten(), quad.e[0].flatten())]
            return self.collection

        def update(k):
            # self.world.ax.clear()
            self.world.text.set_text(r'%.2f secs'%(self.time[k]))
            self.collection = [self.world.text]
            self.collection += [artist for quad in self.quads for artist in quad.artist.draw(quad.p[k].flatten(), quad.e[k].flatten())]
            return self.collection

        animation = matplotlib.animation.FuncAnimation(
            self.world.fig, 
            update, 
            frames=self.frames, 
            init_func=init, 
            interval=1000*interval, 
            blit=True
        )

        if self.save:
            animation.save('/tmp/anim.mp4', fps=20, dpi=200, extra_args=['-vcodec', 'libx264'])
        else:
            matplotlib.pyplot.show()