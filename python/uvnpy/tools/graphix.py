#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 09 15:54:29 2020
@author: fran
"""
import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib.collections
import matplotlib.animation as ani
from matplotlib.patches import Ellipse
import uvnpy.tools.tools as tools

def ellipse(ax, mu, sigma, **kwargs):
	""" Draws ellipse from xy mean and covariance matrix """
	sigmas = kwargs.get('sigmas', 1.)
	color = kwargs.get('color', 'k')
	alpha = kwargs.get('alpha', 0.2)
	vals, vecs = np.linalg.eigh(sigma)
	x, y = vecs[:, 0]
	theta = np.degrees(np.arctan2(y, x))
	w, h = 2 * np.sqrt(vals) * sigmas
	ellipse = Ellipse(mu, w, h, theta, color=color)
	ellipse.set_alpha(0.2)
	ax.add_artist(ellipse)
	return ellipse

 
class Plot(object):
	def __init__(self, *args, **kwargs):
		self.nrows = kwargs.get('nrows', 1)
		self.ncols = kwargs.get('ncols', 1)
		title = kwargs.get('title', 'Plot')
		self.color = kwargs.get('color', 'b')
		self.ls = kwargs.get('ls', '-')
		self.marker = kwargs.get('marker', '')
		self.aspect = kwargs.get('aspect', 'auto')
		xlabel = kwargs.get('xlabel', np.full((self.nrows, self.ncols), '$t\,[s]$'))
		ylabel = kwargs.get('ylabel', np.full((self.nrows, self.ncols), ''))
		xlim = kwargs.get('xlim', False)
		ylim = kwargs.get('ylim', False)
		self.fig, axes = plt.subplots(self.nrows, self.ncols)
		self.axes = np.array(axes).reshape(self.nrows, self.ncols)
		self.fig.suptitle(title)
		for c in range(self.ncols):
			for r in range(self.nrows):
				self.axes[r][c].set_aspect(self.aspect)
				self.axes[r][c].set_xlabel(xlabel[r][c])
				self.axes[r][c].set_ylabel(ylabel[r][c])
				if xlim and ylim: 
					self.axes[r][c].set_xlim(*xlim)
					self.axes[r][c].set_ylim(*ylim)
				else:
					self.axes[r][c].autoscale()
				self.axes[r][c].minorticks_on()
				self.axes[r][c].grid(True)
		self.show = plt.show

	def clear(self):
		for c in range(self.ncols):
			for r in range(self.nrows):
				self.axes[r][c].clear()


class TimePlot(Plot):
	def __init__(self, time, line_array, **kwargs):
		line_array = np.array(line_array)
		kwargs['nrows'] = line_array.shape[0]
		kwargs['ncols'] = line_array.shape[1]
		kwargs['title'] = 'Time Plot'
		super(TimePlot, self).__init__(line_array, **kwargs)
		self.time = time
		self.draw(line_array)

	def draw(self, line_array):
		for c in range(self.ncols):
			for r in range(self.nrows):
				for line in line_array[r][c]:
					self.axes[r][c].plot(self.time, line, color=self.color)


class ScatterPlot(Plot):
	def __init__(self, *args, **kwargs):
		nrows = kwargs.get('nrows', 1)
		ncols = kwargs.get('ncols', 1)
		kwargs['ls'] = ''
		kwargs['marker'] = 'o'
		super(ScatterPlot, self).__init__(*args, **kwargs)
		self.lines = [[[] for c in range(ncols)] for r in range(nrows)]
		self.text = [[None for c in range(ncols)] for r in range(nrows)]
		self.line_collection = [[None for c in range(ncols)] for r in range(nrows)]

	def add_line(self, ax, **kwargs):
		r, c = ax
		color = kwargs.get('color', self.color)
		ls = kwargs.get('ls', self.ls)
		marker = kwargs.get('marker', self.marker)
		alpha = kwargs.get('alpha', 1)
		markersize = kwargs.get('markersize', 2)
		self.lines[r][c] += self.axes[r][c].plot([],[], color=color, ls=ls, marker=marker, alpha=alpha, markersize=markersize)

	def add_line_collection(self, ax, lines, **kwargs):
		r, c = ax
		rgba = kwargs.get('rgba', (0.,0.,0.,0.8))
		colors = [rgba for _ in lines]
		self.line_collection[r][c] = matplotlib.collections.LineCollection(lines, colors=colors)
		self.axes[r][c].add_collection(self.line_collection[r][c])

	def add_text(self, ax, **kwargs):
		r, c = ax
		x = kwargs.get('x', 0.)
		y = kwargs.get('y', 0.)
		string = kwargs.get('string', '')
		self.text[r][c] = self.axes[r][c].text(x, y, string)

	def set_data(self, ax, i, X, Y, **kwargs):
		r, c = ax
		self.lines[r][c][i].set_data(X, Y)

	def set_text(self, ax, text):
		r, c = ax
		self.text[r][c].set_text(text)

	def collection(self):
		return list(np.array(self.lines).flat) + list(np.array(self.text).flat) + list(np.array(self.line_collection).flat)


class XYPlot(ScatterPlot):
	def __init__(self, *args, **kwargs):
		nrows = kwargs.get('nrows', 1)
		ncols = kwargs.get('ncols', 1)
		kwargs['title'] = 'XY Plot'
		kwargs['xlabel'] = np.full((nrows, ncols), '$X\,[m]$')
		kwargs['ylabel'] = np.full((nrows, ncols), '$Y\,[m]$')
		kwargs['aspect'] = 'equal'
		super(XYPlot, self).__init__(*args, **kwargs)


class ComplexPlane(ScatterPlot):
	def __init__(self, *args, **kwargs):
		nrows = kwargs.get('nrows', 1)
		ncols = kwargs.get('ncols', 1)
		kwargs['title'] = 'Complex Plane'
		kwargs['xlabel'] = np.full((nrows, ncols), '$\Re$')
		kwargs['ylabel'] = np.full((nrows, ncols), '$\Im$')
		kwargs['aspect'] = 'auto'
		super(ComplexPlane, self).__init__(*args, **kwargs)


class GraphPlotter(object):
	def __init__(self, *args, **kwargs):
		self.time = args[0]
		self.frames = range(len(self.time))
		self.V = args[1]
		self.P = args[2]
		self.E = args[3]
		self.save = kwargs.get('save', False)
		self.points = [tools.Vec3(*tools.from_arrays([p[i] for p in self.P.values()])) for i in self.frames]
		self.show = lambda: plt.show()
		
	# def timeplot(self, time, *args, **kwargs):
	# 	fig = kwargs.get('fig', plt.figure())
	# 	ax = kwargs.get('ax', fig.add_subplot())
	# 	timeplot(ax, time, *args, **kwargs)

	# def xyplot(self, k, **kwargs): 
	# 	xy = XYPlot()
	# 	xyplot(ax, self.points[k].x, self.points[k].y, **kwargs)

	def animation2d(self, **kwargs):
		xyplot = XYPlot(**kwargs)
		interval = self.time[1] - self.time[0]
		fps = interval**-1
		alphas = np.linspace(0.1, 1, 10) 
		X = collections.deque(maxlen=len(alphas))
		Y = collections.deque(maxlen=len(alphas))

		def init():
			for alpha in alphas: xyplot.add_line((0,0), alpha=alpha, markersize=4*alpha)
			xyplot.add_text((0,0), x=-80, y=80, string='')
			xyplot.add_line_collection((0,0), [])
			return xyplot.collection()

		def update(k):
			X.append(self.points[k].x)
			Y.append(self.points[k].y)
			for i in range(len(X)):
				xyplot.set_data((0,0), i, X[i], Y[i])
			xyplot.set_text((0,0), 't = {:.1f} secs'.format(self.time[k]))
			lines = [[self.P[i][k].flatten(), self.P[j][k].flatten()] for (i,j) in self.E[k]]
			xyplot.add_line_collection((0,0), lines)
			return xyplot.collection()

		animation = ani.FuncAnimation(xyplot.fig, update, frames=self.frames, init_func=init, interval=1000*interval , blit=True)
		if self.save:
			animation.save('anim.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
		else:
			self.show()