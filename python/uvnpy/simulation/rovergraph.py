#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
import graph_tool as gt
from uvnpy.network.graph import UnmannedVehicleGraph, RoverGraph
from uvnpy.tools.ros import Vector3
from uvnpy.network.vehicles import UnmannedVehicle, Rover
import uvnpy.controller.mpc as controller
import uvnpy.tools.tools as tools
import uvnpy.tools.graphix as graphix

def run(arg):
	g = RoverGraph(directed=False)
	g.add_robots(arg.n, pi=lambda: np.random.uniform(-arg.dist, arg.dist, (3,1)))

	time = np.arange(arg.ti, arg.tf, arg.h)

	P = dict([(r.id, []) for r in g.robots()])
	V = dict([(r.id, []) for r in g.robots()])
	E = []
	Cmd = dict([(r.id, np.random.normal(0,3, (2,1))) for r in g.robots()])
	dof = 2
	size = arg.n*dof
	c, p, v = np.empty((size, 1)), np.empty((size, 1)), np.empty((size, 1)) 
	idx = dict(zip(g.get_robots(), np.arange(0, size, dof))) 

	mpc = controller.MPC(c.size, weights=(5,1), window=(1, 5))
	u = np.zeros_like(c)
	for t in time:
		# Link update
		for i in g.get_robots():
			m = idx[i]
			r = g.r(i)
			# cmdv = np.vstack((Cmd[i], 0))
			cmdv = np.vstack((u[m:m+2], 0))
			r.motion.step(cmdv, t)
			P[i].append(r.xy())
			V[i].append(r.vxy())
			c[m:m+2] = Cmd[i][:2]
			p[m:m+2] = r.xy()
			v[m:m+2] = r.vxy()

			for j in g.get_robots()[i:]:
				if g.dist(i,j) < arg.range and [i,j] not in g.get_links().tolist():
					g.add_link(i,j)
				elif g.dist(i,j) >= arg.range and [i,j] in g.get_links().tolist():
					g.remove_link(i,j)

		E.append(g.get_links()) 
		# print('{:.1f}'.format(t), [(i,j) for (i,j) in g.get_links()])

		# Control
		x = p
		u = mpc.update(u, c, x)
		# print(np.round(c - u, 2))

	return time, V, P, E

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-n', '--n', default=10, type=int, help='cantidad de robots')
	parser.add_argument('-d', '--dist', default=20., type=float, help='distribución inicial')
	parser.add_argument('-s', '--step', dest='h', default=50e-3, type=float, help='paso de simulación')
	parser.add_argument('-t', '--ti', metavar='T0', default=0.0, type=float, help='tiempo inicial')
	parser.add_argument('-e', '--tf', default=1.0, type=float, help='tiempo final')
	parser.add_argument('-f', '--f_ctrl', default=10.0, type=float, help='frecuencia del controlador')
	parser.add_argument('-r', '--range', default=40., type=float, help='rango de la antena')
	parser.add_argument('-g', '--save', default=False, action='store_true', help='flag para guardar los videos')
	arg = parser.parse_args()

	time, V, P, E = run(arg)
	plotter = graphix.GraphPlotter(time, V, P, E, save=arg.save)
	plotter.animation2d(xlim=[-100,100], ylim=[-100,100])
	# v = Vector3(*tools.from_arrays(V[1]))
	# p = Vector3(*tools.from_arrays(P[1]))
	# plotter.timeplot(time, v.x, v.y, p.x, p.y)
