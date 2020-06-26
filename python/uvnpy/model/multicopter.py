#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue May 19 18:54:56 2020
@author: fran
"""
import numpy as np
import quaternion
import uvnpy.toolkit.linalg as linalg
from uvnpy.model.discrete import DiscreteModel
from gpsic.controladores.pid import PIDController

g = 9.81

def drag(v, k=0.2):
    """ air drag forces represented as a force
    proportional to the square of the velocity
    and in the opposite direction """
    return -k*np.multiply(v, np.abs(v))

def linear_model(x, u, t, *args, fw=(0.,0.,0.)):
    """ linear multicopter rigid body model 
    with wind forces and drag.
    input:  space state vector: x
            input vector: u

    returns dot_x
    """
    px, py, pz, vx, vy, vz, er, ep, ey, wx, wy, wz = x
    Ft, Tx, Ty, Tz = u
    Dx, Dy, Dz = drag([vx, vy, vz])
    Wx, Wy, Wz = fw
    m, Ix, Iy, Iz = args
    return np.array([vx,
                     vy,
                     vz,
                     g*ep + (Dx + Wx)/m,
                     -g*er + (Dy + Wy)/m,
                     -g + (Ft + Dz + Wz)/m,
                     wx,
                     wy,
                     wz,
                     Tx/Ix,
                     Ty/Iy,
                     Tz/Iz])


class Multicopter(DiscreteModel):
    def __init__(self, **kwargs):
        """
        Esta clase implementa la dinámica a lazo cerrado de un
        multicótero. Tiene dos controles, un control de actitud
        y un control de velocidad lineal. 
        La implementación se hace con un pid.
        """
        # Variables generales de la dinámica del multicóptero
        ti = kwargs.get('ti', 0.)
        pi = kwargs.get('pi', [0.,0.,0.])
        vi = kwargs.get('vi', [0.,0.,0.])
        ai = kwargs.get('ai', [0.,0.,0.])
        wi = kwargs.get('wi', [0.,0.,0.])
        super(Multicopter, self).__init__(ti=ti, xi=np.hstack([pi, vi, ai, wi]))

        # Tipo de modelo utilizado
        self.model = kwargs.get('model', self.nonlinear_model)

        # Parámetros Físicos del modelo
        self.m = kwargs.get('m', 1.5)
        self.Ix = kwargs.get('Ix', 29e-3)
        self.Iy = kwargs.get('Iy', 29e-3)
        self.Iz = kwargs.get('Iz', 55e-3)
        self.eq_x = np.hstack([pi, np.zeros(9)])
        self.eq_u = np.array([self.m*g, 0, 0, 0])

        # Variable generales de control
        f_ctrl = kwargs.get('f_ctrl', 1e3)
        self.Tc = 1./f_ctrl
        self.tc = self.ti + self.Tc
        self.ur, self.up, self.ut = 0., 0., 0.

        # Control PID
        # Empaqueto ambos controladores en un tuple
        self.att_ctrl = PIDController(
            kp=(0.2,0.2,0.1),
            ki=(0.05,0.05,0.05),
            kd=(0.05,0.05,0.005),
            t0=self.ti
        )
        self.vel_ctrl = PIDController(
            kp=(0.7,0.7,0.9),
            ki=(0.2,0.2,0.5),
            kd=(0.1,0.1,0.1),
            t0=self.ti,
            usats=((-0.8,0.8),(-0.8,0.8),(-25.,25.))
        )
        
        # para que el error en el primer paso no sea muy grande
        # y genere una accion derivativa excesiva
        self.att_ctrl.e = ai
        self.vel_ctrl.e = vi

    def __str__(self):
        return 'Multicopter'

    def nonlinear_model(self, x, u, t, *args, fw=(0.,0.,0.)):
        """ nonlinear multicopter rigid body model 
        with wind forces and drag.
        input:  space state vector: x
                input vector: u

        returns dot_x
        """
        px, py, pz, vx, vy, vz, er, ep, ey, wx, wy, wz = x
        Ft, Tx, Ty, Tz = u
        Dx, Dy, Dz = drag([vx, vy, vz])
        Wx, Wy, Wz = fw
        m, Ix, Iy, Iz = args
        cr, sr = np.cos(er), np.sin(er)
        cp, sp = np.cos(ep), np.sin(ep)
        cy, sy = np.cos(ey), np.sin(ey)
        tp = np.tan(ep)
        return np.array([vx,
                         vy,
                         vz,
                         Ft*(sp*cy*cr + sy*sr + Dx + Wx)/m,
                         Ft*(sp*sy*cr - sr*cy + Dy + Wy)/m,
                         -g + (Ft*cp*cr + Dz + Wz)/m,
                         wx + wy*sr*tp + wz*cr*tp,
                         wy*cr - wz*sr,
                         wy*sr/cp + wz*cr/cp,
                         (Iy*wy*wz - Iz*wy*wz + Tx)/Ix,
                         (-Ix*wx*wz + Iz*wx*wz + Ty)/Iy,
                         (Ix*wx*wy - Iy*wx*wy + Tz)/Iz])

    def pid(self, x, r, t):
        """ This function implements one pid for the attitude
        control at each simulation step and another pid at ctrl
        time step for velocity.
        input:  space state vector: x
                input reference: r

        returns control_action
        """
        eu = x[[6,7,8]]
        wz = x[11]
        if t > self.tc:
            v = x[[3,4,5]]
            u = self.vel_ctrl.update(r[:3], v, t)
            qz = np.quaternion(0.707106781186548, 0, 0, 0.707106781186547)
            qwb = linalg.quat.ZYX(eu).conj()
            self.ur, self.up, self.ut = quaternion.rotate_vectors(qz*qwb, u)
            self.tc += self.Tc
        Tx, Ty, Tz = self.att_ctrl.update([self.ur, self.up, r[3]], [eu[0],eu[1],wz], t)
        Ft = np.clip(self.ut + self.eq_u[0], 0, 25)
        Tx, Ty, Tz = np.clip([Tx, Ty, Tz], -20, 20)
        return np.array([Ft, Tx, Ty, Tz])
        
    def dot_x(self, x, r, t, fw=(0.,0.,0.)):
        """ This function represents the dynamics of the closed
        loop model. It takes a reference input r and returns the 
        derivative of state x """
        self.r = r
        self.u = self.pid(x, r, t)
        return self.model(x, self.u, t, self.m, self.Ix, self.Iy, self.Iz, fw=fw) 

    def xyzyaw(self):
        return self.x[[0,1,2,8]]

    def v_xyzyaw(self):
        return self.x[[3,4,5,11]]

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
        return np.copy(self.r)

    def ctrl_eff(self):
        return np.copy(self.u)
