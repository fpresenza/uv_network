#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri June 05 19:16:34 2020
@author: fran
"""
import numpy as np
import quaternion
import yaml
import uvnpy.toolkit.linalg as linalg
from uvnpy.model.discrete import DiscreteModel

class Camera(object):
    """ This class implements model of a camera """
    def __init__(self, cnfg_file='../config/nikon.yaml',
        pos=[0.,0.,0.], attitude=(np.eye(3),),
        sc=(0.,0.), sigma=0.):
        # config
        config = yaml.load(open(cnfg_file))
        fov, res = config.values()
        self.hfov = np.divide([fov['x'], fov['y']], 2)
        self.hres = np.divide([res['x'], res['y']], 2)
        tanhfov = np.tan(np.radians(self.hfov))
        self.sigma = sigma
        # extrinsic camera params
        self.update_pose(pos, attitude)
        # intrinsic camera params
        self.sc = (cx, cy) = sc          # sensor center of image 
        self.f = (fx, fy) = self.hres/tanhfov
        K = np.array([[fx,  0, cx, 0],
                      [ 0, fy, cy, 0],
                      [ 0,  0,  1, 0],
                      [ 0,  0,  0, 1]])
        att2sensor = linalg.ht.matrix(linalg.rm.Ry(np.pi/2), [0, 0, 0])
        sensor2pixels = linalg.ht.matrix(linalg.rm.Rz(-np.pi/2), [0, 0, 0])
        self.intrinsic = linalg.multi_matmul(sensor2pixels, K, att2sensor)

    def update_pose(self, pos, rotation_seq):
        self.pos = np.copy(pos)
        self.attitude = linalg.rm.from_any(*rotation_seq)
        self.extrinsic = linalg.ht.inv_matrix(self.attitude, self.pos)

    def to_pixel(self, point, noisy=False):
        """ get the pixel coordinate (a,b) of a given point in space """
        matrix = np.matmul(self.intrinsic, self.extrinsic)
        x = linalg.ht.dot(matrix, point)
        if x[2]>=0:
            return np.array([np.inf, np.inf])
        if noisy:
            y = x[:2]/x[2] + np.random.normal(0., self.sigma, 2)
        else:
            y = x[:2]/x[2]
        return round(y[0]), round(y[1])

    def is_pixel_in_fov(self, pixels):
        pixels_normalized = np.abs(pixels-self.sc)
        return np.all(pixels_normalized < self.hres, axis=1)

    def view(self, *points, noisy=True):
        """ simulate a noisy measurment of a detected object """
        pixels = np.vstack([self.to_pixel(point, noisy) for point in points])
        return pixels[self.is_pixel_in_fov(pixels)]


class Gimbal(DiscreteModel):
    def __init__(self, ti=0., xi=quaternion.one):
        """ Clase para implementar un modelo de gimbal """
        super(Gimbal, self).__init__(ti=ti, xi=xi)

    def dot_x(self, x, r, t, n=[0.,0.,0.], k=3):
        """
        Modelo de control por medio de una acción proporcional
        a la velocidad angular.

        x: gimbal quaternion
        r: gimbal euler angles ref
        t: time
        n: additive euler angles output noise.
        """
        r_q = linalg.quat.ZYX(r)
        n_q = linalg.quat.ZYX(n)
        e_q = r_q * x.conj() * n_q.conj()
        w = k*quaternion.as_rotation_vector(e_q)
        w_q = np.quaternion(0, *w)
        return 0.5 * w_q * x

    def normalize(self):
        self.x = linalg.quat.normalize(self.x)
