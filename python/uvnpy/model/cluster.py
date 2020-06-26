#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue Jun 2 19:45:33 2020
@author: fran
"""
import numpy as np
import uvnpy.toolkit.linalg as linalg
from uvnpy.model.multicopter import Multicopter
from uvnpy.model.discrete import DiscreteModel
import gpsic.cluster.r2dof4.kinematics as kinematics

class R2Dof4(kinematics.R2DOF4):
    def __init__(self, ti=0., pose=np.zeros(8)):
        """ Esta clase implementa la dinámica de un cluster de dos uav """
        super(R2Dof4, self).__init__()
        self.uav = (Multicopter(ti=ti, f_ctrl=50.),
                    Multicopter(ti=ti, f_ctrl=50.))
        self.set_pose(ti, pose)

    def step(self, cmd_vc, t, **kwargs):
        cmd_vr = self.inverse_velocity_kinematics(self.last_rc, cmd_vc)
        self.uav[0].step(cmd_vr[:4], t, d_kw=kwargs)
        self.uav[1].step(cmd_vr[4:], t, d_kw=kwargs)
        self.last_rs = np.hstack([self.uav[0].xyzyaw(), self.uav[1].xyzyaw()])
        self.update_fkin(self.last_rs)
        return self.last_rc

    def set_pose(self, t, c):
        self.last_rc = np.asarray(c)
        r = self.update_ikin(c)
        p = np.hstack([r[:3], 0, 0, 0, 0., 0., r[3], 0, 0, 0])
        self.uav[0].set(t, p)
        p = np.hstack([r[4:7], 0, 0, 0, 0., 0., r[7], 0, 0, 0])
        self.uav[1].set(t, p)

    def p(self):
        return self.last_rc

    def v(self):
        vr = np.hstack([self.uav[0].v_xyzyaw(), self.uav[1].v_xyzyaw()])
        return self.forward_velocity_kinematics(self.last_rs, vr)

    def pitch_and_roll(self):
        return np.hstack([self.uav[0].rp(), self.uav[1].rp()])


class R2Dof4Cam(object):
    def __init__(self, ti=0., pose=np.zeros(10)):
        """ Esta clase implementa la dinámica de un cluster de dos uav 
        con camaras montadas """
        self.r2dof4 = R2Dof4(ti, pose[:8])
        self.cam = DiscreteModel(ti=ti, xi=pose[8:])

    def step(self, cmd, t, fw=[0.,0.,0.]):
        cl = self.r2dof4.step(cmd[:8], t, fw=fw)
        c = self.cam.step(cmd[8:], t)
        return np.hstack([cl, c])

    def p(self):
        return np.hstack([self.r2dof4.p(), self.cam.x])

    def v(self):
        return np.hstack([self.r2dof4.v(), self.cam.u])

    def vision(self, fov):
        """ calcular el valor de la función de visión:
        si el valor devuelto es <1 entonces los uav no se 
        ven entre sí """
        c = self.p()
        phi_c, theta_c = c[[3,4]]
        """ vector normal a la camara n """
        def get_n(x, rp):
            phi, t = x
            roll, pitch = rp
            Rn = linalg.rm.ZYX([roll, pitch + t, phi_c + phi])
            return Rn[:,0]
        n1 = get_n(c[[5,8]], self.r2dof4.uav[0].rp())
        n2 = get_n(c[[6,9]], self.r2dof4.uav[1].rp())        
        """ vector en la recta que pasa por los uav """
        Ro = linalg.rm.ZYX((0, theta_c, phi_c - np.pi/2))
        o12 = Ro[:,0]
        o21 = -o12
        return np.cos(np.radians(fov)*0.5) - np.hstack([np.inner(n1, o12), np.inner(n2, o21)])


class HolR2Dof4(object):
    def __init__(self, ti=0., pose=np.zeros(8)):
        """ Esta clase implementa la dinámica de un cluster de dos uav """
        super(HolR2Dof4, self).__init__()
        self.uavs = DiscreteModel(xi=pose)

    def step(self, cmd_vc, t, **kwargs):
        self.uavs.step(cmd_vc, t)
        return self.uavs.x

    def p(self):
        return self.uavs.x

    def v(self):
        return self.uavs.u

    def pitch_and_roll(self):
        return np.zeros(4)


class HolR2Dof4Cam(object):
    def __init__(self, ti=0., pose=np.zeros(10)):
        """ Esta clase implementa la dinámica de un cluster de dos uav 
        con camaras montadas """
        self.r2dof4 = HolR2Dof4(ti, pose[:8])
        self.cam = DiscreteModel(ti=ti, xi=pose[8:])

    def step(self, cmd, t, fw=[0.,0.,0.]):
        cl = self.r2dof4.step(cmd[:8], t, fw=fw)
        c = self.cam.step(cmd[8:], t)
        return np.hstack([cl, c])

    def p(self):
        return np.hstack([self.r2dof4.p(), self.cam.x])

    def v(self):
        return np.hstack([self.r2dof4.v(), self.cam.u])

    def vision(self, fov):
        """ calcular el valor de la función de visión:
        si el valor devuelto es <1 entonces los uav no se 
        ven entre sí """
        c = self.p()
        phi_c, theta_c = c[[3,4]]
        """ vector normal a la camara n """
        def get_n(x, rp):
            phi, t = x
            roll, pitch = rp
            Rn = linalg.rm.ZYX([roll, pitch + t, phi_c + phi])
            return Rn[:,0]
        n1 = get_n(c[[5,8]], [0.,0.])
        n2 = get_n(c[[6,9]], [0.,0.])
        
        """ vector en la recta que pasa por los uav """
        Ro = linalg.rm.ZYX((0, theta_c, phi_c - np.pi/2))
        o12 = Ro[:,0]
        o21 = -o12
        return np.cos(np.radians(fov)*0.5) - np.hstack([np.inner(n1, o12), np.inner(n2, o21)])
