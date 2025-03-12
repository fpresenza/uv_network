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
            position,
            position_cov,
            vel_meas_cov,
            bearing_meas_cov,
            gps_meas_cov,
            time=0.0):
        """
        Kalman Filter that neglects inter-agent cross-correlations.

        args:
        -----
            position            : intial position
            position_cov        : initial position covariance
            vel_meas_cov   : measured velocity covariance
            bearing_meas_cov : distance measurement variance
            gps_meas_cov        : gps covariance
        """
        super(FirstOrderKalmanFilter, self).__init__(position, time)
        self.P = position_cov.copy()
        self.vel_meas_cov = vel_meas_cov
        self.bearing_meas_cov = bearing_meas_cov
        self.gps_meas_cov = gps_meas_cov
        self.position = self.state

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
            neighbors,
            neighbors_cov):
        """
        Bearing measurements model.

        args:
        -----
            bearing_meas : bearing measurements
            neighbors    : neighbors positions
            neighbors    : neighbors positions covariance
        """
        pass

    def gps_step(self, position_meas):
        """
        Position measurements model.

        args:
        -----
            position_meas : position measurements
        """
        self.x = position_meas
