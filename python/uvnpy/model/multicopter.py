#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue May 19 18:54:56 2020
@author: fran
"""
import numpy as np
import recordclass
from uvnpy.model.discrete import DiscreteModel
from uvnpy.controller.lqr import LQR
from gpsic.controladores.pid import PIDController

Equilibrium = recordclass.recordclass('Equilibrium', 'x u')
g = 9.81

Ctrl = recordclass.recordclass('Ctrl', 'attitude, velocity')

class Multicopter(DiscreteModel):
    def __init__(self, *args, **kwargs):
        """
        Esta clase implementa la dinámica a lazo cerrado de un
        multicótero. Tiene dos controles, un control de actitud
        y un control de velocidad lineal. 
        La implementación se hace con un pid. Tambíen hay un LQR
        pero no está implementado. 
        """
        # Variables generales de la dinámica del multicóptero
        pi = np.array(kwargs.get('pi', (0.,0.,0.))).reshape(-1,1)
        vi = np.array(kwargs.get('vi', (0.,0.,0.))).reshape(-1,1)
        ai = np.array(kwargs.get('ai', (0.,0.,0.))).reshape(-1,1)
        wi = np.array(kwargs.get('wi', (0.,0.,0.))).reshape(-1,1)
        xi = np.vstack((pi, vi, ai, wi))
        ui = np.zeros((4,1))
        kwargs.update({
            'xi':xi,
            'r':np.zeros((4,1)),
            'u':np.zeros((4,1))
        })
        super(Multicopter, self).__init__(*args, **kwargs)

        # Parámetros extra del modelo
        if kwargs.get('linear_model', False):
            self.model = self.linear_model
        else:
            self.model = self.nonlinear_model

        self.m = kwargs.get('m', 1.5)
        self.Ix = kwargs.get('Ix', 29e-3)
        self.Iy = kwargs.get('Iy', 29e-3)
        self.Iz = kwargs.get('Iz', 55e-3)
        self.eq = Equilibrium(np.vstack((pi, np.zeros((9,1)))), np.array([[self.m*g],[0],[0],[0]]))
        self.B = np.zeros_like(xi)

        # Variable generales de control
        f_ctrl = kwargs.get('f_ctrl', 1e3)
        self.Tc = 1./f_ctrl
        self.tc = self.ti + self.Tc
        self.ur, self.up, self.ut = 0., 0., 0.

        # Control LQR
        # A = np.array([[0, 0, 0,  0, g, 0, 0, 0, 0],
        #               [0, 0, 0, -g, 0, 0, 0, 0, 0],
        #               [0, 0, 0,  0, 0, 0, 0, 0, 0],
        #               [0, 0, 0,  0, 0, 0, 1, 0, 0],
        #               [0, 0, 0,  0, 0, 0, 0, 1, 0],
        #               [0, 0, 0,  0, 0, 0, 0, 0, 1],
        #               [0, 0, 0,  0, 0, 0, 0, 0, 0],
        #               [0, 0, 0,  0, 0, 0, 0, 0, 0],
        #               [0, 0, 0,  0, 0, 0, 0, 0, 0]])
        # B = np.array([[       0,         0,         0,         0],
        #               [       0,         0,         0,         0],
        #               [1/self.m,         0,         0,         0],
        #               [       0,         0,         0,         0],
        #               [       0,         0,         0,         0],
        #               [       0,         0,         0,         0],
        #               [       0, 1/self.Ix,         0,         0],
        #               [       0,         0, 1/self.Iy,         0],
        #               [       0,         0,         0, 1/self.Iz]])

        # self.ctrl = LQR(
        #     A, 
        #     B,
        #     Q=np.diag([1,1,1,3,3,3,4,4,4]), 
        #     R=np.diag([1,6,6,6])
        # )
        # self.ei = np.zeros((9,1))

        # Control PID
        # Empaqueto ambos controladores en un tuple
        self.ctrl = Ctrl(
            attitude=PIDController(
            kp=(0.2,0.2,0.1),
            ki=(0.05,0.05,0.05),
            kd=(0.05,0.05,0.005),
            t0=self.ti
            ),
            velocity=PIDController(
            kp=(0.7,0.7,0.9),
            ki=(0.2,0.2,0.5),
            kd=(0.1,0.1,0.1),
            t0=self.ti,
            usats=((-0.8,0.8),(-0.8,0.8),(-25.,25.))
            )
        )
        # para que el error en el primer paso no sea muy grande
        # y genere una accion derivativa excesiva
        self.ctrl.attitude.e = ai.flatten()
        self.ctrl.velocity.e = vi.flatten()

    def __str__(self):
        return 'Multicopter'

    def drag(self, v):
        """ air drag forces represented as a force
        proportional to the square of the velocity
        and in the opposite direction """
        return -0.2*np.multiply(v, np.abs(v))

    def linear_model(self, x, u, **kwargs):
        """ linear multicopter rigid body model 
        with wind forces and drag.
        input:  space state vector: x
                input vector: u

        returns dot_x
        """
        px, py, pz, vx, vy, vz, er, ep, ey, wx, wy, wz = (x-self.eq.x).flat
        Ft, Tx, Ty, Tz = u.flat
        Dx, Dy, Dz = self.drag((vx, vy, vz))
        Wx, Wy, Wz = kwargs.get('fw', (0.,0.,0.))
        m, Ix, Iy, Iz = self.m, self.Ix, self.Iy, self.Iz
        return np.array([[vx],
                         [vy],
                         [vz],
                         [g*ep + (Dx + Wx)/m],
                         [-g*er + (Dy + Wy)/m],
                         [-g + (Ft + Dz + Wz)/m],
                         [wx],
                         [wy],
                         [wz],
                         [Tx/Ix],
                         [Ty/Iy],
                         [Tz/Iz]])

    def nonlinear_model(self, x, u, **kwargs):
        """ nonlinear multicopter rigid body model 
        with wind forces and drag.
        input:  space state vector: x
                input vector: u

        returns dot_x
        """
        px, py, pz, vx, vy, vz, er, ep, ey, wx, wy, wz = x.flat
        Ft, Tx, Ty, Tz = u.flat
        Dx, Dy, Dz = self.drag((vx, vy, vz))
        Wx, Wy, Wz = kwargs.get('fw', (0.,0.,0.))
        m, Ix, Iy, Iz = self.m, self.Ix, self.Iy, self.Iz
        cr, sr = np.cos(er), np.sin(er)
        cp, sp = np.cos(ep), np.sin(ep)
        cy, sy = np.cos(ey), np.sin(ey)
        tp = np.tan(ep)
        return np.array([[vx],
                         [vy],
                         [vz],
                         [Ft*(sp*cy*cr + sy*sr + Dx + Wx)/m],
                         [Ft*(sp*sy*cr - sr*cy + Dy + Wy)/m],
                         [-g + (Ft*cp*cr + Dz + Wz)/m],
                         [wx + wy*sr*tp + wz*cr*tp],
                         [wy*cr - wz*sr],
                         [wy*sr/cp + wz*cr/cp],
                         [(Iy*wy*wz - Iz*wy*wz + Tx)/Ix],
                         [(-Ix*wx*wz + Iz*wx*wz + Ty)/Iy],
                         [(Ix*wx*wy - Iy*wx*wy + Tz)/Iz]])

    def lqr(self, x, r, **kwargs):
        err = x[3:]-r
        self.ei = self.ei + np.vstack((err[:3],np.zeros((6,1))))*self.Ts
        return self.ctrl.update(err + 0.5*self.ei) + self.eq.u

    def pid(self, x, r, **kwargs):
        """ This function implements one pid for the attitude
        control at each simulation step and another pid at ctrl
        time step for velocity.
        input:  space state vector: x
                input reference: r

        returns control_action
        """
        er, ep, ey = x[[6,7,8]].flat
        wz, = x[11]
        if self.t > self.tc:
            vx, vy, vz = x[[3,4,5]].flat
            ux, uy, uz = self.ctrl.velocity.update(r[:3], (vx,vy,vz), self.t)
            cr, sr = np.cos(er), np.sin(er)
            cp, sp = np.cos(ep), np.sin(ep)
            cy, sy = np.cos(ey), np.sin(ey)
            self.ur = ux*(-sp*sr*cy + sy*cr) + uy*(-sp*sy*sr - cy*cr) - uz*sr*cp
            self.up = ux*cp*cy + uy*sy*cp - uz*sp
            self.ut = ux*(sp*cy*cr + sy*sr) + uy*(sp*sy*cr - sr*cy) + uz*cp*cr
            self.tc += self.Tc
        Tx, Ty, Tz = self.ctrl.attitude.update((self.ur, self.up, r[3]), (er,ep,wz), self.t)
        Ft = np.clip(self.ut + self.eq.u[0][0], 0, 25)
        Tx, Ty, Tz = np.clip((Tx, Ty, Tz), -20, 20)
        return np.array([[Ft], [Tx], [Ty], [Tz]])

    def f(self, x, r, **kwargs):
        """ This function represents the 
        dynamics of the closed loop model """
        self.r = r
        self.u = self.pid(x, r, **kwargs)
        return self.model(x, self.u, **kwargs) 
        
    def dot_x(self, x, r, e, **kwargs):
        """ This function takes an input u and returns 
        a state derivative noisy sample """
        return self.f(x, r, **kwargs) + np.dot(self.B, self.e)

    def set(self, pi=(0.,0.,0.), vi=(0.,0.,0.,), ai=(0.,0.,0.), wi=(0.,0.,0.)):
        self.x = np.hstack([pi, vi, ai, wi]).reshape(-1,1)

    def xyzyaw(self):
        return self.x[[0,1,2,8]].flatten()

    def v_xyzyaw(self):
        return self.x[[3,4,5,11]].flatten()

    def p(self):
        return self.x[[0,1,2]]

    def v(self):
        return self.x[[3,4,5]]

    def euler(self):
        return self.x[[6,7,8]]

    def rp(self):
        return self.x[[6,7]]

    def w(self):
        return self.x[[9,10,11]]

    def ref(self):
        return np.asarray(self.r).reshape(-1,1)

    def ctrl_eff(self):
        return np.asarray(self.u).reshape(-1,1)
