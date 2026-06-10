#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
"""
import numpy as np

from .core import EulerIntegrator
from .lie_groups import EulerIntegratorOrtogonalGroup


class Quadrotor(object):
    def __init__(
            self,
            pos,
            att,
            lin_vel,
            ang_vel,
            mass,
            inertia,
            t=0.0):
        """
        args:
        -----
            pose = (position, attitude)
            x = (pose, dotpose)
        """
        self.pos = EulerIntegrator(pos, t)
        self.att = EulerIntegratorOrtogonalGroup(att, t)
        self.lin_vel = EulerIntegrator(lin_vel, t)
        self.ang_vel = EulerIntegrator(ang_vel, t)
        self.mass = mass
        self.inertia = inertia
        self.g = np.array([0.0, 0.0, 9.81])

    def position(self):
        return self.pos.x()

    def attitude(self):
        return self.att.x()

    def linear_vel(self):
        return self.lin_vel.x()

    def angular_vel(self):
        return self.ang_vel.x()

    def step(self, t, force, torque):
        omega = self.ang_vel.x()
        J = self.inertia_matrix
        Jinv = np.linalg.inv(J)
        self.pos.step(t, self.lin_vel.x())
        self.att.step_left(t, omega)
        self.lin_vel.step(t, force / self.mass * self.att.x()[:, 2] - self.g)
        self.ang_vel.step(t, Jinv.dot(torque - np.cross(omega, J.dot(omega))))
