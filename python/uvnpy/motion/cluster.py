#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue Jun 2 19:45:33 2020
@author: fran
"""
import numpy as np
from uvnpy.motion.multicopter import Multicopter
from gpsic.controladores.pid import PIDController
import gpsic.cluster.r2dof4.kinematics as gpsic

class R2DOF4(gpsic.R2DOF4):
    def __init__(self, *args, **kwargs):
        pi = kwargs.get('pi', ((0.,0.,0.), (0.,0.,0.)))
        vi = kwargs.get('vi', ((0.,0.,0.), (0.,0.,0.)))
        ai = kwargs.get('ai', ((0.,0.,0.), (0.,0.,0.)))
        wi = kwargs.get('wi', ((0.,0.,0.), (0.,0.,0.)))
        ti = kwargs.get('ti', 0.)
        f_ctrl = kwargs.get('f_ctrl', 50.)
        linear_model = kwargs.get('linear_model', False)

        r_s = np.hstack((pi[0], ai[0][2], pi[1], ai[1][2])) 
        super(R2DOF4, self).__init__()
        self.uav_1 = Multicopter(ti=ti, pi=pi[0], vi=vi[0], ai=ai[0], f_ctrl=f_ctrl, linear_model=linear_model)
        self.uav_2 = Multicopter(ti=ti, pi=pi[1], vi=vi[1], ai=ai[1], f_ctrl=f_ctrl, linear_model=linear_model)

    def step(self, cmd_vc, t, **kwargs):
        cmd_vc = np.asarray(cmd_vc)
        cmd_vr = self.inverse_velocity_kinematics(self.last_rc, cmd_vc)
        self.uav_1.step(cmd_vr[:4], t, **kwargs)
        self.uav_2.step(cmd_vr[4:], t, **kwargs)
        self.last_rs = np.hstack((self.uav_1.xyzyaw(), self.uav_2.xyzyaw()))
        self.update_fkin(self.last_rs)

    def pos(self):
        return self.last_rc.reshape(-1,1)

    def vel(self):
        vr = np.hstack((self.uav_1.v_xyzyaw(), self.uav_2.v_xyzyaw()))
        vc =  self.forward_velocity_kinematics(self.last_rs, vr)
        return vc.reshape(-1,1)