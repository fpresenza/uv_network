#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 09 15:54:29 2020
@author: fran
"""
import numpy as np
import collections
import matplotlib as mpl
import matplotlib.pyplot
import matplotlib.animation
import uvnpy.toolkit.linalg as linalg

def ellipse(ax, mu, sigma, **kwargs):
    """ Draw ellipse from xy mean and covariance matrix """
    vals, vecs = np.linalg.eigh(sigma)
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))
    w, h = 2 * np.sqrt(vals)
    ellipse = mpl.patches.Ellipse(mu, w, h, theta, **kwargs)
    ax.add_artist(ellipse)
    return ellipse


class GridPlot(object):
    def __init__(self, **kwargs):
        self.shape = kwargs.pop('shape', (1,1))
        self.title = kwargs.pop('title', 'Plot')
        self.aspect = kwargs.pop('aspect', 'auto')
        xlabel = kwargs.pop('xlabel', ['' for _ in range(np.dot(*self.shape))])
        ylabel = kwargs.pop('ylabel', ['' for _ in range(np.dot(*self.shape))])
        axtitle = kwargs.pop('axtitle', ['' for _ in range(np.dot(*self.shape))])
        xlim = kwargs.pop('xlim', False)
        ylim = kwargs.pop('ylim', False)
        self.grid = kwargs.pop('grid', True)
        self.fig, self.axes = matplotlib.pyplot.subplots(*self.shape, squeeze=False, **kwargs)
        self.fig.suptitle(self.title)
        for i, ax in enumerate(self.axes.flat):
                ax.set_aspect(self.aspect)
                ax.set_xlabel(xlabel[i])
                ax.set_ylabel(ylabel[i])
                ax.set_title(axtitle[i], fontsize=8)            
                if xlim: 
                    ax.set_xlim(*xlim)
                if ylim: 
                    ax.set_ylim(*ylim)
                ax.minorticks_on()
                ax.grid(self.grid)
        self.show = matplotlib.pyplot.show

    def _set_param(self, callable, param, i, j):
        try:
            callable(param[i][j])
        except IndexError:
            pass

    def draw(self, x, lines, color=[['b']], label=[['']], ls=[['-']], ds=[['default']], legend_kw={}):
        for i, ax in enumerate(self.axes.flat):
            ln = ax.plot(x, list(zip(*lines[i])))
            for j, l in enumerate(ln):
                self._set_param(l.set_color, color, i, j)
                self._set_param(l.set_label, label, i, j)
                self._set_param(l.set_ls, ls, i, j)
                self._set_param(l.set_ds, ds, i, j)
            ax.legend(**legend_kw)

    def clear(self):
        for ax in self.axes.flat:
            ax.clear()
            ax.minorticks_on()
            ax.grid(self.grid)

    def savefig(self, name, dir='/tmp/'):
        self.fig.savefig('{}{}'.format(dir, name))


class ScatterPlot(GridPlot):
    def __init__(self, **kwargs):
        super(ScatterPlot, self).__init__(**kwargs)
        self.lines = [[[] for c in range(self.shape[1])] for r in range(self.shape[1])]
        self.line_collection = np.full(self.shape, None)
        self.text = np.full(self.shape, None)

    def add_scatter(self, ax, **kwargs):
        r, c = ax
        color = kwargs.get('color', 'b')
        ls = kwargs.get('ls', '')
        marker = kwargs.get('marker', 'o')
        alpha = kwargs.get('alpha', 1)
        markersize = kwargs.get('markersize', 2)
        self.lines[r][c] += self.axes[ax].plot([],[], color=color, ls=ls, marker=marker, alpha=alpha, markersize=markersize)

    def add_line_collection(self, ax, **kwargs):
        rgba = kwargs.get('rgba', (0.,0.,0.,0.8))
        self.line_collection[ax] = mpl.collections.LineCollection([], color=rgba)
        self.axes[ax].add_artist(self.line_collection[ax])

    def add_text(self, ax, x=0., y=0., text='', **kwargs):
        self.text[ax] = self.axes[ax].text(x, y, text, **kwargs)

    def set_scatter(self, ax, i, X, Y, **kwargs):
        r, c = ax
        self.lines[r][c][i].set_data(X, Y)

    def set_line_collection(self, ax, lines):
        self.line_collection[ax].set_segments(lines)

    def set_text(self, ax, text):
        self.text[ax].set_text(text)

    def collection(self):
        lines = [line for row in self.lines for col in row for line in col] 
        texts = [text for text in self.text.flat if text!=None]
        line_collections = [line_col for line_col in self.line_collection.flat if line_col!=None]
        return [*lines, *texts, *line_collections]


class XYPlot(ScatterPlot):
    def __init__(self, **kwargs):
        nr, nc = kwargs.get('shape', (1,1))
        kwargs.update(
            title='XY Plot',
            xlabel=['$x$' for _ in range(nr*nc)],
            ylabel=['$y$' for _ in range(nr*nc)],
            aspect='equal'
        )
        super(XYPlot, self).__init__(**kwargs)


class ComplexPlane(ScatterPlot):
    def __init__(self, **kwargs):
        nr, nc = kwargs.get('shape', (1,1))
        kwargs.update(
            title='Complex Plane Plot',
            xlabel=['$\Re$' for _ in range(nr*nc)],
            ylabel=['$\Im$' for _ in range(nr*nc)],
            aspect='auto'
        )
        super(ComplexPlane, self).__init__(**kwargs)


class CameraImage(ScatterPlot):
    def __init__(self, camera, id=0, extras={}):
        self.cam = camera
        lim_x = self.cam.hres[0]
        lim_y = self.cam.hres[1]
        kwargs = dict(
            title='Camera Image',
            axtitle=['Cam {}'.format(id)],
            xlabel=['horizontal'],
            ylabel=['vertical'],
            xlim=(-lim_x, lim_x),
            ylim=(-lim_y, lim_y),
            aspect='equal',
            grid=False
            )
        super(CameraImage, self).__init__(**kwargs)
        self.add_scatter((0,0), marker='.')
        self.add_text((0,0), x=-0.9*lim_x, y=-0.9*lim_y, text='', color='green', fontsize=8)
        if extras != {}:
            try:
                fov = extras['fov']
                radius = np.tan(np.radians(fov/2)) * self.cam.f[1]
                self.axes[0,0].add_artist(matplotlib.pyplot.Circle((0,0), radius, color='0.5', fill=False))
            except KeyError:
                pass

        # floor markers
        self.add_scatter((0,0), color='g', marker='.', markersize=1.)
        grid = np.arange(-10, 11, 1)
        x_c, y_c, _ = self.cam.pos
        x, y = np.meshgrid(x_c+grid, y_c+grid)
        self.floor_markers = np.dstack([x, y, np.zeros_like(x)]).reshape(-1,3)
    
    def render(self, *points):
        pixels = self.cam.view(*points)
        self.set_scatter((0,0), 0, *pixels.T)
        # floor
        pixels = self.cam.view(*self.floor_markers, noisy=False)
        self.set_scatter((0,0), 1, *pixels.T)


class GraphPlotter(object):
    def __init__(self, t, P, L, **kwargs):
        self.time = t
        self.P = P
        self.L = L
        self.save = kwargs.get('save', False)
        if self.save: mpl.use('Agg')
        print('Matplotlib backend: {}'.format(mpl.get_backend()))
        self.show = lambda: matplotlib.pyplot.show()
        
    def links(self, **kwargs):
        """ Plot number of links of each robot per time """
        lines = [[np.count_nonzero(l==i) for l in self.L] for i in self.P.keys()]
        ds = ['steps' for i in self.P.keys()]
        color = [np.random.rand(3) for i in self.P.keys()]
        label = list(self.P.keys())
        lp = GridPlot(shape=(1,1), title='Number of links per robot')
        lp.draw(self.time, [lines], ds=[ds], color=[color], label=[label])
        if self.save:
            lp.savefig('link_plot')
        else: 
            lp.show()

    def animation2d(self, **kwargs):
        slc = kwargs.pop('slice', 1)
        xyplot = XYPlot(**kwargs)
        frames = range(0, len(self.time), slc)
        interval = np.diff(np.asarray(self.time)[frames]).mean()
        fps = interval**-1

        print('2D Animation\nFrames per second: {}'.format(fps))

        points = [linalg.vec3(*np.hstack([p[k].reshape(-1,1) for p in self.P.values()])) for k in range(len(self.time))]
        alphas = np.linspace(0.1, 1, 10) 
        X = collections.deque(maxlen=len(alphas))
        Y = collections.deque(maxlen=len(alphas))

        def init():
            for alpha in alphas: xyplot.add_scatter((0,0), alpha=alpha, markersize=4*alpha)
            xyplot.add_text((0,0), x=-80, y=80, text='')
            xyplot.add_line_collection((0,0))
            return xyplot.collection()

        def update(k):
            X.append(points[k].x)
            Y.append(points[k].y)
            for i, (x, y) in enumerate(zip(X, Y)):
                xyplot.set_scatter((0,0), i, x, y)
            xyplot.set_text((0,0), 't = {:.1f} secs'.format(self.time[k]))
            lines = [[self.P[i][k][:2], self.P[j][k][:2]] for (i,j) in self.L[k]]
            xyplot.set_line_collection((0,0), lines)
            return xyplot.collection()

        animation = matplotlib.animation.FuncAnimation(
            xyplot.fig, 
            update, 
            frames=frames,
            init_func=init, 
            interval=1000*interval , 
            blit=True
        )
        if self.save:
            animation.save('/tmp/xy_anim.mp4', fps=fps, dpi=200, extra_args=['-vcodec', 'libx264'])
        else:
            matplotlib.pyplot.show()
