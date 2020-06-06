#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 09 15:54:29 2020
@author: fran
"""
import numpy as np
import collections
import matplotlib as mpl
import matplotlib.animation
from uvnpy.toolkit.linalg import vector, rotation, projection
import uvnpy.sensor.camera

def circle3D(center, normal, radius, **kwargs):
    """ Get sample points that belong to a circle placed on a 3D space
    center: center of sphere
    normal: vector normal to the plane containing the circle 
    radius: positive real number
    """
    center = np.asarray(center)
    step = kwargs.get('step', 0.2)
    tx, ty = projection.orthogonal.complement(normal).T
    tx, ty = np.multiply(radius, [tx, ty])
    parameter = np.hstack((np.arange(0, 2*np.pi, step), 0))
    centered_circle = [np.cos(t)*tx + np.sin(t)*ty for t in parameter]
    return (center + centered_circle).T

def ellipse3D(center, basis, **kwargs):
    """ Get sample points that belong to an ellipse placed on a 3D space
    center: center of ellpise
    basis: tuple of two vectors that form an orthogonal basis of plane 
    containing the ellipse.
    Vector norm give semi-minor and semi-major axis length.
    Vector direction give rotation of ellipse
    """
    center = np.asarray(center)
    step = kwargs.get('step', 0.2)
    tx, ty = basis
    parameter = np.hstack((np.arange(0, 2*np.pi, step), 0))
    centered_ellipse = [np.cos(t)*tx + np.sin(t)*ty for t in parameter]
    return (center + centered_ellipse).T

def sphere(center, radius):
    """ Sample points that belong to a sphere: 
    center: center of sphere 
    radius: positive real number
    """
    n = 20
    center = np.asarray(center).reshape(-1,1)
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    unit_centered_sphere = np.vstack([np.outer(np.cos(u), np.sin(v)).flatten(),
                                      np.outer(np.sin(u), np.sin(v)).flatten(),
                                      np.outer(np.ones_like(u), np.cos(v)).flatten()])
    return center + np.dot(radius, unit_centered_sphere)

def ellipsoid(center, M):
    """ Sample points that belong to a ellipsoid: 
    center: center of ellipsoid 
    M: definite positive matrix
    """ 
    return sphere(center, M)

def surface_plot(ax, generator, *args, **kwargs):
    """ Add surface plot of given shape to ax:
    generator: 
        function that takes *args and **kwargs as parameter
        and returns a sample of points belonging to the surface.
    """
    x, y, z = generator(*args)
    n = int(np.sqrt(x.size))
    kwargs.update(shade=1)
    return ax.plot_surface(x.reshape(n,n), y.reshape(n,n), z.reshape(n,n), **kwargs)


class Plot3D(object):
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'title'): self.title = kwargs.get('title', 'Plot3D')
        if not hasattr(self, 'color'): self.color = kwargs.get('color', 'b')
        if not hasattr(self, 'label'): self.label = kwargs.get('label', '')
        self.ls = kwargs.get('ls', '-')
        self.marker = kwargs.get('marker', '')
        self.markersize = kwargs.get('markersize', 5)
        self.fig = mpl.pyplot.figure()
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
        self.arm_len = kwargs.get('arm_length', 0.235)
        self.prop_rad = kwargs.get('prop_rad', 0.08)
        # body
        self.body = self.__add_body(colors=('r', 'b', '0.3'))
        # shadow
        self.shadow = self.__add_body(colors=('0.3', '0.3', '0.3'), alpha=0.3, lw=2)
        # camera
        self.has_cam = kwargs.get('camera', False)
        self.attached_cam = kwargs.get('attached', False)
        self.view = []
        if self.has_cam:
            self.__add_camera(**kwargs)

    def __add_body(self, **kwargs):
        self.Patch = collections.namedtuple('Patch', 'arm propeller')
        p = self.Patch([],[])
        # plot style
        colors = kwargs.get('colors', ('b','b','b'))
        alpha = kwargs.get('alpha', 1.)
        lw = kwargs.get('lw', 1)
        # arms
        p.arm.append(self.ax.plot([], [], [], color=colors[0], alpha=alpha, lw=lw)[0])
        p.arm.append(self.ax.plot([], [], [], color=colors[1], alpha=alpha, lw=lw)[0])
        # propellers
        p.propeller.append(self.ax.plot([], [], [], color=colors[2], alpha=alpha, lw=lw)[0])
        p.propeller.append(self.ax.plot([], [], [], color=colors[2], alpha=alpha, lw=lw)[0])
        p.propeller.append(self.ax.plot([], [], [], color=colors[2], alpha=alpha, lw=lw)[0])
        p.propeller.append(self.ax.plot([], [], [], color=colors[2], alpha=alpha, lw=lw)[0])
        return p

    def __draw_body(self, patch, pos, R, **kwargs):
        P = kwargs.get('projection', np.eye(3))
        alpha = kwargs.get('alpha', None)
        tf = P @ R
        pos = P @ pos
        tx, ty, tz = tf.T
        vertex = (
            np.add(pos, self.arm_len * tx).reshape(-1,1),
            np.add(pos, self.arm_len * ty).reshape(-1,1),
            np.add(pos, self.arm_len * -tx).reshape(-1,1),
            np.add(pos, self.arm_len * -ty).reshape(-1,1),
        )
        patch.arm[0].set_data_3d(*zip(vertex[0], pos, vertex[1]))
        patch.arm[1].set_data_3d(*zip(vertex[2], pos, vertex[3]))
        ellipse = ellipse3D((0.,0.,0.), (self.prop_rad * tx, self.prop_rad * ty))
        patch.propeller[0].set_data_3d(*np.add(vertex[0], ellipse))
        patch.propeller[2].set_data_3d(*np.add(vertex[1], ellipse))
        patch.propeller[3].set_data_3d(*np.add(vertex[2], ellipse))
        patch.propeller[1].set_data_3d(*np.add(vertex[3], ellipse))

        if alpha is not None:
            patch.arm[0].set_alpha(alpha)
            patch.arm[1].set_alpha(alpha)
            patch.propeller[0].set_alpha(alpha)
            patch.propeller[1].set_alpha(alpha)
            patch.propeller[2].set_alpha(alpha)
            patch.propeller[3].set_alpha(alpha)

    def __add_camera(self, **kwargs):
        self.cam = uvnpy.sensor.camera.Camera(
            color=kwargs.get('color', 'b')
        )
        self.view = self.cam.view(self.ax)

    def __draw_camera(self, **kwargs):
        self.view[0].remove()
        self.view[1].remove()
        self.view = self.cam.view(self.ax)
        self.view[0]._facecolors2d = self.view[0]._facecolors3d
        self.view[0]._edgecolors2d = self.view[0]._edgecolors3d

    def draw(self, pos, euler, **kwargs):
        R = rotation.matrix.RZYX(*euler)
        # draw body of quadcopter
        self.__draw_body(self.body, pos, R)
        # draw shadow of quadcopter
        P = projection.oblique.to_xy((-0.15,-0.15,-1))
        a = np.clip(1-pos[2]/13., 0., 0.8) # alpha decreases with heigth
        self.__draw_body(self.shadow, pos, R, projection=P, alpha=a)
        # draw camera fov
        if self.has_cam:
            self.cam.position(pos)
            C = R if self.attached_cam else np.eye(3)
            self.cam.direction(C, **kwargs)
            self.__draw_camera()
        return self.collection()

    def collection(self):
        return self.body.arm + self.body.propeller + self.shadow.arm + self.shadow.propeller + self.view


class SurfaceArtist(object):
    def __init__(self, ax, generator, **kwargs):
        """ Draw sphere in 3D """
        self.ax = ax
        self.generator = generator
        self.color = kwargs.get('color', 'b')
        self.alpha = kwargs.get('alpha', 1)
        # self.radius = kwargs.get('radius', 0.5)
        self.body = []
        # self.body = [surface_plot(self.ax, self.generator, (0.,0.,0.), self.radius)]

    def draw(self, *args, **kwargs):
        try:
            self.body[0].remove()
        except IndexError:
            pass
        kwargs.update(color=self.color, alpha=self.alpha)
        self.body = [surface_plot(self.ax, self.generator, *args, **kwargs)]
        self.body[0]._facecolors2d = self.body[0]._facecolors3d
        self.body[0]._edgecolors2d = self.body[0]._edgecolors3d
        return self.collection()

    def collection(self):
        return self.body


class Animation3D(object):
    def __init__(self, t, **kwargs):
        self.save = kwargs.get('save', False)
        self.slice = slice(0, -1, kwargs.get('slice'))
        if self.save: mpl.use('Agg')
        print('Matplotlib backend: {}'.format(mpl.get_backend()))
        self.world = Plot3D(**kwargs)
        self.time = t[self.slice]
        self.collection = []
        self.frames = range(len(self.time))
        self.quads = []
        self.spheres = []
        self.ellipsoids = []
        self.colors = ('indianred', 'cornflowerblue', 'purple', 'orange', 'springgreen')

    def add_quadrotor(self, *args, **kwargs):
        Q = collections.namedtuple('Q', 'artist p e g')
        g = kwargs.get('gimbal', [[np.array((0.,np.pi/4,0.)) for _ in args[0][0]] for _ in range(2)])
        self.quads = [Q(QuadrotorArtist(self.world.ax, color=self.colors[i], **kwargs),\
         p[self.slice], e[self.slice], g[i][self.slice]) for i,(p,e) in enumerate(args)]

    def add_sphere(self, P, **kwargs):
        S = collections.namedtuple('S', 'artist p radius')
        radius = kwargs.get('radius', 1.)
        if not 'color' in kwargs:
            kwargs.update(color=self.colors[len(self.spheres)])
        self.spheres.append(S(SurfaceArtist(self.world.ax, sphere, **kwargs), P[self.slice], radius))

    def add_ellipsoid(self, P, M, **kwargs):
        E = collections.namedtuple('E', 'artist p cov')
        if not 'color' in kwargs:
            kwargs.update(color=self.colors[len(self.ellipsoids)])
        self.ellipsoids.append(E(SurfaceArtist(self.world.ax, ellipsoid, **kwargs), P[self.slice], M[self.slice]))

    def run(self, **kwargs):
        interval = np.diff(self.time).mean()
        fps = interval**-1
        print('Frames per second: {}'.format(fps))

        def init():
            return self.collection

        def update(k):
            self.world.text.set_text(r'%.2f secs'%(self.time[k]))
            self.collection = [self.world.text]
            self.collection += [artist for quad in self.quads for artist in\
             quad.artist.draw(quad.p[k].flatten(), quad.e[k].flatten(), gimbal=quad.g[k].flatten())]
            self.collection += [artist for sphere in self.spheres for artist in\
             sphere.artist.draw(sphere.p[k].flatten(), sphere.radius)]
            self.collection += [artist for ellipsoid in self.ellipsoids for artist in\
             ellipsoid.artist.draw(ellipsoid.p[k].flatten(), ellipsoid.cov[k])]
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
            animation.save('/tmp/anim.mp4', fps=fps, dpi=200, extra_args=['-vcodec', 'libx264'])
        else:
            mpl.pyplot.show()