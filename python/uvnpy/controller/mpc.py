#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Wed May 20 14:27:16 2020
@author: fran
"""
import numpy as np
import scipy.optimize
import scipy.integrate as integrate
import recordclass

Cost = recordclass.recordclass('Cost', 'cmd conn', defaults=(0,0))
Window = recordclass.recordclass('Window', 'step size', defaults=(1,1))

def quad_norm(v):
	return np.dot(v.T, v).item()

def quad_form(v, Q):
	return np.dot(v.T, np.dot(Q, v)).item()

def interdiff(x):
	p = np.hstack((x.reshape(-1,2)[:,0], x.reshape(-1,2)[:,1])).reshape(-1,1) 
	z = np.zeros((6,4))
	a = np.array([[1., -1,  0,  0],
				  [1.,  0, -1,  0],
				  [1.,  0,  0, -1],
				  [0.,  1, -1,  0],
				  [0.,  1,  0, -1],
				  [0.,  0,  1, -1]])
	A = np.block([[a, z],
				  [z, a]])
	return np.dot(A, p)

def interdist(x):
	d = interdiff(x)
	return quad_norm(d)

def plant_model(x, u, **kwargs):
	win = kwargs.get('window', Window())
	U = np.hstack([u]*(win.size+1))
	return x + integrate.cumtrapz(U, dx=win.step) # this should model the plant

class Constraint(object):
	def __init__(self, *args, **kwargs):
		self.bound = kwargs.get('bound', 1.)
		self.win = kwargs.get('window', Window())

	def __call__(self, *args):
		return {'type':'ineq', 'fun':self.fun, 'args': args}


class Connected(Constraint):
	def __init__(self, *args, **kwargs):
		super(Connected, self).__init__(*args, **kwargs)

	def fun(self, u, x):
		u = u.reshape(-1,1)
		X = plant_model(x, u, window=self.win)
		# print(X.shape)
		# d = interdiff(X.T)
		diff = map(interdiff, X.T)
		I = np.eye(6)
		I2 = np.block([I, I])
		r = [self.bound**2 - np.dot(I2, np.square(d)) for d in diff]
		# print(r)
		return np.array(r).flatten()


class MPC(object):
	def __init__(self, n, **kwargs):
		self.n = n
		self.info = kwargs.get('info', False)
		self.w = Cost(*kwargs.get('weights', (1,1)))
		self.win = Window(*kwargs.get('window', (0.2, 5)))
		self.Q = Cost(
			kwargs.get('Qcmd', np.eye(n)),
			kwargs.get('Qconn', np.eye(n))
		)
		self.cost = Cost(
			lambda c, u: quad_form(c-u, self.Q.cmd),
			lambda X: sum(map(interdist, X.T))
		)
		self.connected = Connected(bound=35., window=self.win)
		self.constraints = ()


	def functional(self, u, c, x):
		u = u.reshape(-1,1)
		# X = plant_model(x, u, window=self.win)
		# center = np.tile([50,25], int(0.5*self.n)).reshape(-1,1)
		J = (
			self.w.cmd * self.cost.cmd(c, u),
			# self.w.conn * self.cost.conn(X)
		)
		return sum(J)

	def update(self, ui, c, x):
		# self.constraints = (self.connected(x), )
		opt = scipy.optimize.minimize(
			self.functional,
			ui.flatten(),
			(c, x),
			method='SLSQP'
			# constraints=self.constraints
		)
		if not opt.success or self.info:
			print(opt)
		return opt.x.reshape(-1,1)