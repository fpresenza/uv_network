#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from uvnpy.network.graph import UnmannedVehicleGraph, RoverGraph
from uvnpy.tools.ros import Vector3
from uvnpy.network.vehicles import UnmannedVehicle, Rover
import uvnpy.tools.tools as tools
import uvnpy.tools.graphix as graphix

def run():
	g = RoverGraph(directed=False)
	g.add_robots(20, xi=lambda: np.random.uniform(-100, 100, (3,1)))
	# g.add_robots(3, xi=lambda: np.zeros((3,1)))

	Ts = 0.2
	time = np.arange(0, 20, Ts)
	linkrange = 40.

	V = dict([(r.id, []) for r in g.robots()])
	P = dict([(r.id, []) for r in g.robots()])
	E = []
	Cmd = dict([(r.id, np.random.normal(0,3,(3,1))) for r in g.robots()])
	# Cmd = dict([(r.id, np.array([[2],[0],[0]])) for r in g.robots()])

	for t in time:
		for i in g.get_robots():
			r = g.r(i)
			v = Cmd[r.id]
			r.motion.step(v, t)
			V[r.id].append(r.motion.X[:3])
			P[r.id].append(r.motion.X[3:])

			for j in g.get_robots()[i:]:
				if g.dist(i,j) < linkrange and [i,j] not in g.get_links().tolist():
					g.add_link(i,j)
				elif g.dist(i,j) >= linkrange and [i,j] in g.get_links().tolist():
					g.remove_link(i,j)

		E.append(g.get_links()) 
		# print('{:.1f}'.format(t), [(i,j) for (i,j) in g.get_links()]) 

	return time, V, P, E

if __name__ == '__main__':
	time, V, P, E = run()
	plotter = graphix.Plotter(time, V, P, E)
	# v = Vector3(*tools.from_arrays(V[1]))
	# p = Vector3(*tools.from_arrays(P[1]))
	# plotter.timeplot(time, v.x, v.y, p.x, p.y)
	plotter.animation2d(xlim=[-100,100], ylim=[-100,100])
