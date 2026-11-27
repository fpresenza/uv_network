#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
"""
import numpy as np
from transformations import unit_vector

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
        J = self.inertia
        Jinv = np.linalg.inv(J)
        self.pos.step(t, self.lin_vel.x())
        self.att.step_left(t, omega)
        self.lin_vel.step(t, force / self.mass * self.att.x()[:, 2] - self.g)
        self.ang_vel.step(t, Jinv.dot(torque - np.cross(omega, J.dot(omega))))


class VelocityControlledQuadrotor(Quadrotor):
    def __init__(
        self, pos, att, lin_vel, ang_vel, mass, inertia,
        t=0.0, kv=1.5, kR=4.0, kw=1.5
    ):
        super().__init__(pos, att, lin_vel, ang_vel, mass, inertia, t)
        self.kv = kv
        self.kR = kR
        self.kw = kw

    def _desired_attitude_from_yaw(self, b3d, yaw):
        b1c = np.array([np.cos(yaw), np.sin(yaw), 0.0])

        b2d = unit_vector(np.cross(b3d, b1c))
        b1d = np.cross(b2d, b3d)

        return np.column_stack((b1d, b2d, b3d))

    def compute_force_and_torque(self, v_des, yaw_des):
        """
        Quadrotor velocity controller for the model:

            dot v = g e3 - (f/m) R e3
            dot R = R hat(w)
            J dot w + w x J w = tau

        Inputs:
            v      : current world-frame velocity, shape (3,)
            R      : current attitude, body-to-world rotation, shape (3,3)
            w  : body-frame angular velocity, shape (3,)
            v_des  : desired world-frame velocity, shape (3,)

        Returns:
            f      : scalar thrust
            tau    : body-frame torque, shape (3,)
            R_des  : desired attitude
        """
        R = self.attitude()
        v = self.linear_vel()
        w = self.angular_vel()

        # Outer-loop desired acceleration.
        a_des = self.kv * (v_des - v)

        # For dot v = g e3 - (f/m) R e3,
        # choose f R e3 = m(g e3 - a_des).
        F_des = self.mass * (self.g + a_des)

        f = np.sqrt(np.square(F_des).sum())

        b3_des = F_des / f
        R_des = self._desired_attitude_from_yaw(b3_des, yaw_des)

        # Attitude error.
        e_R_mat = 0.5 * (R_des.T.dot(R) - R.T.dot(R_des))
        e_R = np.array([e_R_mat[2, 1], e_R_mat[0, 2], e_R_mat[1, 0]])

        # Simple attitude-rate reference: w_des = 0.
        tau = - self.kR * e_R - self.kw * w + np.cross(w, self.inertia.dot(w))

        return f, tau

    def closed_loop_step(self, t, v_des, yaw_des):
        force, torque = self.compute_force_and_torque(v_des, yaw_des)
        self.step(t, force, torque)
