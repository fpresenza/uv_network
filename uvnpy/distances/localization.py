#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date lun sep 13 20:27:17 -03 2021
"""
import numpy as np


class LocalizationFilter(object):
    def __init__(self, state, time=0.0):
        self.x = state.copy()
        self.t = time

    def state(self):
        return self.x.copy()


class DistanceBasedGradientFilter(LocalizationFilter):
    def __init__(self, position, stepsize, input_weights, time=0.0):
        """
        Gradient Descent of the cost function:
            V(x) = sum_{ij in E} (wij/2)*(||x_i-x_j|| - z_obs)^2 +
                (1/2)*sum_{i in V} ||x_i - f(x_i, u_i)||_{W_i}^2

        args:
        -----
            position      : initial position
            stepsize      : gradient descent stepsize
            input_weights : input weigth
            time          : initial time
        """
        super(DistanceBasedGradientFilter, self).__init__(position, time)
        self.stepsize = stepsize
        self.input_weights = input_weights

    def dynamic_step(self, time, vel_meas, *args):
        """
        Dynamic model.

        args:
        -----
            time     : current time
            vel_meas : velocity measurement
        """
        dt = time - self.t
        self.t = time

        self.x += self.input_weights.dot(vel_meas * dt)

    def batch_range_step(self, range_meas, anchors):
        """
        Range measurements model.

        args:
        -----
            range_meas : range measurements
            anchors    : anchors positions
        """
        r = self.x - anchors
        d = np.sqrt(np.square(r).sum(axis=1))
        m = (range_meas - d)/d
        self.x += self.stepsize * sum(r * m[:, None])

    def gps_step(self, position_meas):
        """
        GPS measurements model.

        args:
        -----
            position_meas : position measurements
        """
        self.x += self.stepsize * (position_meas - self.x)


class DistanceBasedKalmanFilter(LocalizationFilter):
    def __init__(
            self,
            position,
            position_cov,
            vel_meas_cov,
            range_meas_cov,
            gps_meas_cov,
            time=0.0):
        """
        Kalman Filter that neglects inter-agent cross-correlations.

        args:
        -----
            position       : intial position
            position_cov   : initial position covariance
            vel_meas_cov   : measured velocity covariance
            range_meas_cov : distance measurement variance
            gps_meas_cov   : gps covariance
        """
        super(DistanceBasedKalmanFilter, self).__init__(position, time)
        self.P = position_cov.copy()
        self.vel_meas_cov = vel_meas_cov
        self.range_meas_cov = range_meas_cov
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
        dt = time - self.t
        self.t = time

        self.x += vel_meas * dt
        self.P += self.vel_meas_cov * (dt**2)

    def range_step(
            self,
            range_meas,
            anchor,
            anchor_cov):
        """
        Range measurements model.

        args:
        -----
            range_meas : range measurement
            anchor     : anchor position
            anchor_cov : anchor covariance
        """
        r = self.x - anchor
        d = np.sqrt(np.square(r).sum())

        H = r.reshape(1, -1) / d

        PHt = self.P.dot(H.T)
        CHt = anchor_cov.dot(H.T)
        Pz = H.dot(PHt + CHt) + self.range_meas_cov
        K = PHt / Pz

        self.x += K.dot(range_meas - d)
        self.P -= K.dot(H).dot(self.P)

    def batch_range_step(
            self,
            range_meas,
            anchors,
            anchors_cov):
        """
        Range measurements model.

        args:
        -----
            range_meas  : range measurements
            anchors     : anchors positions
            anchors_cov : anchors covariance
        """
        r = self.x - anchors
        d = np.sqrt(np.square(r).sum(axis=-1, keepdims=True))
        d = d.ravel()
        dz = range_meas.ravel() - d
        H = r / d[:, None]
        Rj = np.matmul(
            H[:, None], np.matmul(anchors_cov, H[:, :,  None])
        )
        R = np.diag((Rj + self.range_meas_cov).ravel())

        PHt = self.P.dot(H.T)
        Pz = H.dot(PHt) + R
        K = PHt.dot(np.linalg.inv(Pz))

        self.x += K.dot(dz)
        self.P -= K.dot(H).dot(self.P)

    def gps_step(self, position_meas):
        """
        Position measurements model.

        args:
        -----
            position_meas : position measurements
        """
        dz = position_meas - self.x
        Pz = self.P + self.gps_meas_cov
        K = self.P.dot(np.linalg.inv(Pz))
        self.x += K.dot(dz)
        self.P -= K.dot(self.P)


class SecondOrderKalmanFilter(LocalizationFilter):
    def __init__(
            self,
            state,
            dim,
            state_cov,
            vel_meas_cov,
            range_meas_cov,
            gps_meas_cov,
            ti=0.0):
        """
        Kalman Filter that neglects inter-agent cross-correlations.

        args:
            state          : intial state (pos + vel)
            state_cov      : initial covariance (pos + vel)
            vel_meas_cov   : accelerometer covariance
            range_meas_cov : distance measurement variance
            gps_meas_cov   : gps covariance
        """
        super(SecondOrderKalmanFilter, self).__init__(state, ti)
        self.P = state_cov.copy()
        self.accel_cov = vel_meas_cov
        self.range_meas_cov = range_meas_cov
        self.gps_meas_cov = gps_meas_cov * np.eye(self.dim)

    def covariance(self):
        return self.P.copy()

    def distances_model(self, z, xj, Pj):
        """
        Distance with neighbors.

        args:
        -----
            z  : measurements
            xj : neighbors states
            Pj : covariance associated to each neighbor
        """
        r = self.x - xj
        d = np.sqrt(np.square(r).sum(axis=-1, keepdims=True))
        d = d.ravel()
        dz = z.ravel() - d
        H = r / d[:, None]
        Rj = np.matmul(H[:, None], np.matmul(Pj, H[:, :,  None]))
        R = np.diag((Rj + self.range_meas_cov).ravel())
        return dz, H, R

    def dynamic_step(self, t, u, *args):
        """
        Second Order Integrator.

        args:
        -----
            t : current time
            u : accelerometer measurement
        """
        dt = t - self.t
        self.t = t

        dt2 = dt**2
        Q = dt2 * self.accel_cov

        self.x[:self.dim] += self.x[self.dim:] * dt
        self.x[self.dim:] += u * dt

        Pxdx = self.P[:self.dim, self.dim:]
        Pdxx = self.P[self.dim:, :self.dim]
        Pdxdx = self.P[self.dim:, self.dim:]

        self.P[:self.dim, :self.dim] += dt * Pxdx + dt * Pdxx + dt2 * Pdxdx
        self.P[:self.dim, self.dim:] += dt * Pdxdx
        self.P[self.dim:, :self.dim] += dt * Pdxdx
        self.P[self.dim:, self.dim:] += dt2 * Q

    def range_step(self, z, xj, Pj, *args):
        """
        Distance measurements model step.

        args:
        -----
            z  : measurements
            xj : neighbors states
            Pj : weigth associated to each neighbor

        """
        dz, H, R = self.distances_model(z, xj, Pj, *args)
        Pz = H.dot(self.P).dot(H.T) + R
        K = self.P.dot(H.T).dot(np.linalg.inv(Pz))

        self.x += K.dot(dz)
        self.P -= K.dot(H).dot(self.P)

    def gps_step(self, position_meas):
        """
        Position measurements model step.

        args:
        -----
            z : measurements
        """
        dz = position_meas - self.x
        Pz = self.P + self.gps_meas_cov
        K = self.P.dot(np.linalg.inv(Pz))
        self.x += K.dot(dz)
        self.P -= K.dot(self.P)
