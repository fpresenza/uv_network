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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import uvnpy.toolkit.linalg as linalg
from uvnpy.graphix.planar import CameraImage

def circle3D(center, normal, radius, **kwargs):
    """ Get sample points that belong to a circle placed on a 3D space
    center: center of sphere
    normal: vector normal to the plane containing the circle 
    radius: positive real number
    """
    center = np.asarray(center)
    step = kwargs.get('step', 0.2)
    tx, ty = linalg.orthogonal.complement(normal).T
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
    tx, ty = np.asarray(basis)
    step = kwargs.get('step', 0.2)
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

def quadrotor(center, attitude, split=False, **kwargs):
    """ Sample points that belong to a quadrotor
    center: center of geometry
    attitude: rotation matrix
    """
    center = np.asarray(center).reshape(-1,1)
    tx, ty, tz = np.matmul(attitude, linalg.rm.Rz(-np.pi/4)).T
    arm = kwargs.get('arm', 0.235)  # meters
    prop = kwargs.get('prop', 0.08) # meters
    v = (np.add(center, arm * tx.reshape(-1,1)),
         np.add(center, arm * ty.reshape(-1,1)),
         np.add(center, arm * -tx.reshape(-1,1)),
         np.add(center, arm * -ty.reshape(-1,1)))
    linsp = np.linspace(0,1,10)
    ellipse = ellipse3D((0.,0.,0.), (prop * tx, prop * ty))
    patches = [np.hstack([linalg.line(v[0], center, t) for t in linsp] + [linalg.line(center, v[1], t) for t in linsp]),
               np.hstack([linalg.line(v[2], center, t) for t in linsp] + [linalg.line(center, v[3], t) for t in linsp]),
               np.add(v[0], ellipse),
               np.add(v[1], ellipse),
               np.add(v[2], ellipse),
               np.add(v[3], ellipse)]
    if split:
        return patches
    else:
        return np.hstack(patches)

def surface_plot(ax, generator, *args, points=False, **kwargs):
    """ Add surface plot of given shape to ax:
    generator: 
        function that takes *args as parameter
        and returns a sample of points belonging to the surface.
    """
    x, y, z = generator(*args)
    n = int(np.sqrt(x.size))
    kwargs.update(shade=1)
    if points:
        return ax.plot_surface(x.reshape(n,n), y.reshape(n,n), z.reshape(n,n), **kwargs), np.vstack([x, y, z])
    else:
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
        kwargs.update(
            ls='',
            marker='o',
            title='3D Scatter Plot'
        )
        super(ScatterPlot, self).__init__(**kwargs)


class QuadrotorArtist(object):
    def __init__(self, ax, pos, attitude, quad_kw={}, **kwargs):
        """ Draw simplistic quadcopter model in 3D """
        self.ax = ax
        self.pos = pos
        self.attitude = attitude
        try:
            self.al = quad_kw['arm_length']
        except KeyError:
            self.al = 0.235
        try:
            self.pr = quad_kw['propeller_radius']
        except KeyError:
            self.pr = 0.08
        # body
        self.body = self.__add_body(colors=('r', 'b', '0.3'))
        # shadow
        self.shadow = self.__add_body(colors=('0.3', '0.3', '0.3'), alpha=0.3, lw=2)
        # points
        self.points = np.array([])

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

    def __draw_body(self, patch, pos, R, points=False, projection=np.eye(3), alpha=None):
        tf = np.matmul(projection, R)
        pos = np.matmul(projection, pos)
        arm_1, arm_2, *p = quadrotor(pos, tf, split=True, arm=self.al, prop=self.pr)
        patch.arm[0].set_data_3d(*arm_1)
        patch.arm[1].set_data_3d(*arm_2)
        patch.propeller[0].set_data_3d(*p[0])
        patch.propeller[1].set_data_3d(*p[1])
        patch.propeller[2].set_data_3d(*p[2])
        patch.propeller[3].set_data_3d(*p[3])
        if points:
            self.points = np.hstack([arm_1, arm_2, *p])
        if alpha is not None:
            patch.arm[0].set_alpha(alpha)
            patch.arm[1].set_alpha(alpha)
            patch.propeller[0].set_alpha(alpha)
            patch.propeller[1].set_alpha(alpha)
            patch.propeller[2].set_alpha(alpha)
            patch.propeller[3].set_alpha(alpha)

    def draw(self, k):
        pos = self.pos[k]
        R = linalg.rm.from_any(self.attitude[k])
        # draw body of quadcopter
        self.__draw_body(self.body, pos, R, points=True)
        # draw shadow of quadcopter
        P = linalg.oblique.projection_to_xy((-0.15,-0.15,-1))
        a = np.clip(1-pos[2]/13., 0., 0.8) # alpha decreases with heigth
        self.__draw_body(self.shadow, pos, R, projection=P, alpha=a)
        return self.collection(), self.collect_points()

    def collection(self):
        return self.body.arm + self.body.propeller + self.shadow.arm + self.shadow.propeller

    def collect_points(self):
        return self.points


class CameraArtist(object):
    def __init__(self, ax, pos, attitude, camera, **kwargs):
        self.ax = ax
        self.pos = pos
        self.attitude = attitude
        self.cam = camera
        self.color = kwargs.pop('color', 'gray')
        self.id = kwargs.get('id', 0)
        self.image = CameraImage(camera, **kwargs)
        self.alpha = 0.3
        self.footprint = self.field_of_view(3)

    def field_of_view(self, zoom):
        """ Plot a prisma representing field of field_of_view and footprint on xy plane """
        # params
        # prisma
        nx, ny, nz = self.cam.attitude.T
        tx = nx * zoom
        ty = ny * zoom * np.tan(np.radians(self.cam.hfov[0]))
        tz = nz * zoom * np.tan(np.radians(self.cam.hfov[1]))
        corners = self.cam.pos + tx + np.array([ty+tz, ty-tz, -ty-tz, -ty+tz])
        prisma = [[self.cam.pos, corners[0], corners[1]],
                  [self.cam.pos, corners[0], corners[3]],
                  [self.cam.pos, corners[1], corners[2]],
                  [self.cam.pos, corners[3], corners[2]]]
        verts = prisma

        # footprint
        def get_t(u, v):
            """ returns the scalar "t" that gives the a vector included in
            the rect that joins vectors "u" and "v" with z-value equal to 
            90% of self.cam.pos """
            if v[2] != u[2]:
                c = self.cam.pos[2]-0.1
                return np.divide(c-u[2], v[2]-u[2])
            else:
                return -1

        def below(v):
            """ return True if vector "v" is below self.cam.pos """
            return v[2] < self.cam.pos[2]

        vertices = []
        for i,j in [(0,1),(1,2),(2,3),(3,0)]:
            if below(corners[i]):
                vertices += [corners[i]]
            t = get_t(corners[i], corners[j])
            if (t>0 and t<1):
                vertices += [linalg.line(corners[i], corners[j], t)]

        def ftp(v):
            return np.matmul(linalg.oblique.projection_to_xy(v), self.cam.pos)

        if len(vertices)>0:
            vertices = np.subtract(vertices, self.cam.pos)
            footprint = [[ftp(v) for v in vertices]]
            verts += footprint
        field_of_view = Poly3DCollection(verts, color=self.color, lw=0.3, alpha=self.alpha)
        self.ax.add_collection3d(field_of_view)

        # center of image on plane xy
        if nx[2]<0:
            center = ftp(nx)
            circle = circle3D(center, (0,0,1), 0.35)
            circle = np.delete(circle, 2, axis=0).T
            
            yaw = linalg.heading(nx)
            Rz = linalg.rm.Rz(yaw)[:2,:2]
            x = Rz[:,1]
            points = [center[:2] + [x, 0.5*x], center[:2] + [-0.5*x, -x]] + \
            [circle[i:i+2] for i in range(len(circle)) if len(circle[i:i+2])==2]
            marker = mpl.collections.LineCollection(points, color=self.color, lw=0.8)
            self.ax.add_collection3d(marker)
        else:
            marker = mpl.collections.LineCollection([])
            self.ax.add_collection3d(marker)

        return [field_of_view, marker]

    def draw(self, k, zoom=3):
        self.footprint[0].remove()
        self.footprint[1].remove()
        self.cam.update_pose(self.pos[k], self.attitude[k])
        self.footprint = self.field_of_view(zoom)
        self.footprint[0]._facecolors2d = self.footprint[0]._facecolors3d
        self.footprint[0]._edgecolors2d = self.footprint[0]._edgecolors3d
        return self.collection()

    def render_image(self, time, points, k):
        self.cam.update_pose(self.pos[k], self.attitude[k])
        self.image.set_text((0,0), 't = {:.1f} secs'.format(time))
        self.image.render(*points)

    def collection(self):
        return self.footprint


class SurfaceArtist(object):
    def __init__(self, ax, generator, args, **kwargs):
        """ Draw sphere in 3D """
        self.ax = ax
        self.args = args
        self.generator = generator
        self.color = kwargs.get('color', 'b')
        self.alpha = kwargs.get('alpha', 1)
        self.surf = []
        self.points = np.array([])

    def draw(self, k, **kwargs):
        try:
            self.surf[0].remove()
        except IndexError:
            pass
        kwargs.update(color=self.color, alpha=self.alpha)
        surf, points = surface_plot(self.ax, self.generator, *self.args[k], points=True, **kwargs)
        self.surf = [surf]
        self.surf[0]._facecolors2d = self.surf[0]._facecolors3d
        self.surf[0]._edgecolors2d = self.surf[0]._edgecolors3d
        self.points = points
        return self.collection(), self.collect_points()

    def collection(self):
        return self.surf

    def collect_points(self):
        return self.points


class Animation3D(object):
    def __init__(self, t, **kwargs):
        self.save = kwargs.get('save', False)
        self.slice = slice(0, -1, kwargs.get('slice'))
        if self.save: mpl.use('Agg')
        print('Matplotlib backend: {}'.format(mpl.get_backend()))
        self.plot3d = Plot3D(**kwargs)
        self.time = t[self.slice]
        self.collection = []
        self.frames = range(len(self.time))
        self.quads = []
        self.cams = dict()
        self.spheres = []
        self.ellipsoids = []
        self.all_points_history = []
        self.colors = ('indianred', 'cornflowerblue', 'purple', 'orange', 'springgreen', 'r', 'b', 'g', 'y')

    def add_quadrotor(self, pos, attitude, quad_kw={}, **kwargs):
        Qart = QuadrotorArtist(self.plot3d.ax, pos[self.slice], attitude[self.slice], quad_kw, **kwargs)
        self.quads.append(Qart)

    def add_camera(self, pos, attitude, camera=None, extras={}, id=None):
        if id is None:
            id = len(self.cams)
        params = dict(
            id=id,
            color=self.colors[len(self.cams)],
            extras=extras
            )
        att_zipped = list(zip(*attitude))
        Cart = CameraArtist(self.plot3d.ax, pos[self.slice], att_zipped[self.slice], camera, **params)
        self.cams[id] = Cart

    def add_drone(self, id, pos, quad_att, cam_att, camera, extras={}, **kwargs):
        self.add_quadrotor(pos, quad_att, **kwargs)
        self.add_camera(pos, cam_att, camera=camera, extras=extras, id=id)

    def add_sphere(self, pos, radius, **kwargs):
        if not 'color' in kwargs:
            kwargs.update(color=self.colors[len(self.spheres)])
        args_zipped = list(zip(pos, radius))
        Sart = SurfaceArtist(self.plot3d.ax, sphere, args_zipped[self.slice], **kwargs)
        self.spheres.append(Sart)

    def add_ellipsoid(self, pos, cov, **kwargs):
        if not 'color' in kwargs:
            kwargs.update(color=self.colors[len(self.ellipsoids)])
        args_zipped = list(zip(pos, cov))
        Eart = SurfaceArtist(self.plot3d.ax, ellipsoid, args_zipped[self.slice], **kwargs)
        self.ellipsoids.append(Eart)

    def run_anim3d(self, **kwargs):
        interval = np.diff(self.time).mean()
        fps = interval**-1
        print('3D Animation\nFrames per second: {}'.format(fps))

        def init():
            return self.collection

        def update(k):
            self.plot3d.text.set_text(r'%.2f secs'%(self.time[k]))
            self.collection = [self.plot3d.text]
            all_points = []
            for q in self.quads:
                q_art, q_points = q.draw(k)
                self.collection += q_art
                all_points.append(q_points) 
            for s in self.spheres:
                s_art, s_points = s.draw(k)
                self.collection += s_art
                all_points.append(s_points) 
            for e in self.ellipsoids:
                e_art, e_points = e.draw(k)
                self.collection += e_art
                all_points.append(e_points)
            for c in self.cams.values():
                self.collection += c.draw(k)
            self.all_points_history.append(np.hstack(all_points).T)
            return self.collection

        animation3d = matplotlib.animation.FuncAnimation(
            self.plot3d.fig, 
            update, 
            frames=self.frames, 
            init_func=init, 
            interval=1000*interval, 
            blit=True
        )

        if self.save:
            animation3d.save('/tmp/world3d.mp4', fps=fps, dpi=200, extra_args=['-vcodec', 'libx264'])
        else:
            mpl.pyplot.show()

    def run_camera_image(self, id):
        interval = np.diff(self.time).mean()
        fps = interval**-1
        print('Camera {} View\nFrames per second: {}'.format(id, fps))
        cart = self.cams[id]

        def init():
            return cart.image.collection()

        def update(k):
            cart.render_image(self.time[k], self.all_points_history[k], k)
            return cart.image.collection()

        animation2d = matplotlib.animation.FuncAnimation(
            cart.image.fig, 
            update, 
            frames=self.frames, 
            init_func=init, 
            interval=1000*interval, 
            blit=True
        )

        if self.save:
            animation2d.save('/tmp/cam_{}_image.mp4'.format(id), fps=fps, dpi=200, extra_args=['-vcodec', 'libx264'])
        else:
            mpl.pyplot.show()

    def run(self, **kwargs):
        self.run_anim3d(**kwargs)
        for id in self.cams:
            self.run_camera_image(id)
