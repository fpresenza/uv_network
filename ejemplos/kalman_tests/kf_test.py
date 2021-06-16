#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date vie nov 6 15:24:28 -03 2020
"""
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.model import linear_models
from uvnpy.filtering import kalman


def dist(x, landmarks, sigma):
    p = x.ravel()
    d = np.sqrt(np.square(p - landmarks).sum(axis=-1))
    return np.random.normal(d, sigma)


def dist_model(x, landmarks, R_dist):
    p = x.reshape(-1, 2)
    r = p - landmarks
    hat_z = np.sqrt(np.square(r).sum(axis=-1)).reshape(-1, 1)
    H = r / hat_z.reshape(-1, 1)
    R = R_dist * np.eye(len(landmarks))
    return hat_z, H, R


if __name__ == '__main__':
    xi = np.random.uniform(-25, 25, 2)
    dxi = np.array([3., 3.])
    Pi = np.diag(dxi**2)
    hat_xi = np.random.multivariate_normal(xi, Pi)
    Q = 0.5 * np.eye(2)
    R_gps = np.array([[12, 0.], [0, 9.]])
    sigma_dist = 3.
    R_dist = sigma_dist ** 2

    landmarks = np.array([
        [0., 0.],
        [0., 50.],
        [50., 0.]])

    tiempo = np.arange(0, 1, 0.05)
    N = len(tiempo)
    planta = linear_models.random_walk(xi, Q)
    kf = kalman.KF(hat_xi, Pi)

    x = np.empty((N, 2))
    dot_x = np.empty((N, 2))
    x[0] = planta.x
    dot_x[0] = planta.dot_x

    for k, t in enumerate(tiempo[1:]):
        u = np.array([0.5, 1.])

        planta.step(t, u)

        kf.prediction(kalman.integrator, t, u.reshape(-1, 1), Q)

        z_gps = np.random.multivariate_normal(planta.x, R_gps)
        kf.correction(kalman.gps_model, z_gps.reshape(-1, 1), R_gps)

        z_dist = dist(planta.x, landmarks, sigma_dist)
        kf.correction(dist_model, z_dist.reshape(-1, 1), landmarks, R_dist)

        x[k+1] = planta.x
        dot_x[k+1] = planta.dot_x
        kf.save_data()

    kalman.plot(kf.summary(), ground_truth=x[..., None])

    plt.show()
