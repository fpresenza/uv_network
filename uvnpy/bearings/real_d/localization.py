#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date lun sep 13 20:27:17 -03 2021
"""
import numpy as np

np.set_printoptions(precision=10)


class DecentralizedLocalization(object):
    def __init__(self, state, time=0.0):
        self.x = state.copy()
        self.t = time

    def state(self):
        return self.x.copy()


class FirstOrderKalmanFilter(DecentralizedLocalization):
    def __init__(
            self,
            pose,
            pose_cov,
            vel_meas_cov,
            bearing_meas_cov,
            gps_meas_cov,
            time=0.0):
        """
        Kalman Filter that neglects inter-agent cross-correlations.

        args:
        -----
            pose             : intial pose
            pose_cov         : initial pose covariance
            vel_meas_cov     : measured velocity covariance
            bearing_meas_cov : distance measurement variance
            gps_meas_cov     : gps covariance
        """
        super(FirstOrderKalmanFilter, self).__init__(pose, time)
        self.P = pose_cov.copy()
        self.vel_meas_cov = vel_meas_cov
        self.bearing_meas_cov = bearing_meas_cov
        self.gps_meas_cov = gps_meas_cov
        self.pose = self.state
        self.eye = np.eye(3)

    def covariance(self):
        return self.P.copy()

    def dynamic_step(self, time, vel_meas, *args):
        """
        Dynamic model.

        args:
        -----
            time     : current time
            vel_meas : velocity measurement
        """
        pass

    def bearing_step(
            self,
            bearing_meas,
            neighbors_pose,
            neighbors_cov):
        """
        Bearing measurements model.

        args:
        -----
            bearing_meas   : bearing measurements
            neighbors_pose : neighbors poses
            neighbors_cov  : neighbors poses covariance
        """
        R = bearing_meas[..., np.newaxis] * bearing_meas[..., np.newaxis, :]
        P = self.eye - R
        r = self.x[:3] - neighbors_pose[:, :3]
        Pr = np.matmul(r[:, np.newaxis], P)
        self.x[:3] -= 0.15 * Pr.sum(axis=0)[0]

    def gps_step(self, pose_meas):
        """
        Pose measurements model.

        args:
        -----
            pose_meas : pose measurements
        """
        self.x = pose_meas
