#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue Jun 2 19:45:33 2020
@author: fran
"""
import numpy as np
from uvnpy.model.multicopter import Multicopter
from gpsic.controladores.pid import PIDController
import gpsic.cluster.r2dof4.kinematics as gpsic

class R2DOF4(gpsic.R2DOF4):
    def __init__(self, **kwargs):
        pi = kwargs.get('pi', ((0.,0.,0.), (0.,0.,0.)))
        vi = kwargs.get('vi', ((0.,0.,0.), (0.,0.,0.)))
        ai = kwargs.get('ai', ((0.,0.,0.), (0.,0.,0.)))
        wi = kwargs.get('wi', ((0.,0.,0.), (0.,0.,0.)))
        ti = kwargs.get('ti', 0.)
        f_ctrl = kwargs.get('f_ctrl', 50.)

        r_s = np.hstack((pi[0], ai[0][2], pi[1], ai[1][2])) 
        super(R2DOF4, self).__init__(r_s=r_s)
        if kwargs.get('cluster', False):
            ci = kwargs.get('ci', np.zeros(8))
            self.last_rc = np.asarray(ci)
            self.update_ikin(ci)
            pi = (self.last_rs[:3], self.last_rs[4:7])
            ai = ((0.,0., self.last_rs[3]), (0.,0., self.last_rs[7]))

        self.uav = (Multicopter(ti=ti, pi=pi[0], vi=vi[0], ai=ai[0], f_ctrl=f_ctrl),
                    Multicopter(ti=ti, pi=pi[1], vi=vi[1], ai=ai[1], f_ctrl=f_ctrl))

    def step(self, cmd_vc, t, **kwargs):
        cmd_vc = np.asarray(cmd_vc)
        cmd_vr = self.inverse_velocity_kinematics(self.last_rc, cmd_vc)
        self.uav[0].step(cmd_vr[:4], t, **kwargs)
        self.uav[1].step(cmd_vr[4:], t, **kwargs)
        self.last_rs = np.hstack((self.uav[0].xyzyaw(), self.uav[1].xyzyaw()))
        self.update_fkin(self.last_rs)
        return self.last_rc

    def set_pos(self, c):
        self.last_rc = np.asarray(c)
        self.update_ikin(c)
        self.uav[0].set(pi=self.last_rs[:3], ai=(0.,0.,self.last_rs[3]))
        self.uav[1].set(pi=self.last_rs[4:7], ai=(0.,0.,self.last_rs[7]))

    def pos(self):
        return self.last_rc.reshape(-1,1)

    def vel(self):
        vr = np.hstack((self.uav[0].v_xyzyaw(), self.uav[1].v_xyzyaw()))
        vc =  self.forward_velocity_kinematics(self.last_rs, vr)
        return vc.reshape(-1,1)
