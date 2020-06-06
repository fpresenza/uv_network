#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue May 06 14:15:45 2020
@author: fran
"""
import numpy as np
import scipy.linalg

def skew(w): 
    return np.array([[    0, -w[2],  w[1]], 
                     [ w[2],     0, -w[0]], 
                     [-w[1],   w[0],    0]])

def block_diag(*m):
    return scipy.linalg.block_diag(*m)

class vector(object):
    class vec3(object):
        def __init__(self, x=0., y=0., z=0.):
            self.x = x
            self.y = y
            self.z = z

        def __call__(self):
            return (self.x, self.y, self.z) 
     
        def __getitem__(self, key): 
           return self()[key] 

        def __add__(self, value):
            return np.add(self(), value)

        def __sub__(self, value):
            return np.subtract(self(), value)

        def __str__(self):
            return 'vec3(x={}, y={}, z={})'.format(self.x, self.y, self.z)

        def __repr__(self):
            return self.__str__()
    
    @staticmethod
    def norm(v):
        return np.sqrt(np.inner(v,v)) 

    @staticmethod
    def normalize(v):
        return np.divide(v, vector.norm(v))

    @staticmethod
    def distance(u, v):
        return vector.norm(np.subtract(u, v).flatten())

    @staticmethod
    def angle_between(u, v):
        i = np.inner(u, v)
        n  = vector.norm(u) * vector.norm(v)
        return np.arccos(i/n)

    @staticmethod
    def heading(v):
        return np.arctan2(v[1], v[0])

    @staticmethod
    def rect(u, v, t):
        return u*(1-t) + v*t

    @staticmethod
    def proj(u, v): 
        return np.divide(np.inner(u,v), np.inner(v, v)) * v


class rotation(object):
    class matrix(object):
        """ This class implements basic rotation matrix 
        algorithms in a 3d euclidean space """
        @staticmethod
        def Rx(roll):
            cr = np.cos(roll)
            sr = np.sin(roll)
            return np.array([[1,  0,   0],
                             [0, cr, -sr],
                             [0, sr,  cr]])

        @staticmethod
        def Ry(pitch):
            cp = np.cos(pitch)
            sp = np.sin(pitch)
            return np.array([[ cp,  0, sp],
                             [  0,  1,  0],
                             [-sp,  0, cp]])

        @staticmethod
        def Rz(yaw):
            cy = np.cos(yaw)
            sy = np.sin(yaw)
            return np.array([[cy, -sy, 0],
                             [sy,  cy, 0],
                             [ 0,   0, 1]])

        @classmethod
        def RZYX(cls, r, p, y):
            return cls.Rz(y) @ cls.Ry(p) @ cls.Rx(r)

        @staticmethod
        def from_vector_angle(t):
            """ Rodrigues rotation formula """
            angle = vector.norm(t)
            t_n = t/angle
            S = skew(t_n)
            return np.eye(3) + np.sin(angle)*S + (1-np.cos(angle))*np.dot(S,S)

        @staticmethod
        def from_quat(q):
            q = rotation.quaternion.normalize(q)
            S = skew(q[:3])
            return np.eye(3) + 2*q[3]*S + 2*np.dot(S, S)


    class quaternion(object):
        """ This class implements basic quaternion
        algorithms in a 3d euclidean space """
        @staticmethod
        def normalize(q):
            q = np.asarray(q) 
            # make the scalar part always positive 
            if q[3] < 0:  
                q *= -1 
            # norm equal to 1 
            return q/vector.norm(q)

        @staticmethod
        def from_euler(euler):
            chy, shy = np.cos(euler[2]/2), np.sin(euler[2]/2)
            chp, shp = np.cos(euler[1]/2), np.sin(euler[1]/2)
            chr, shr = np.cos(euler[0]/2), np.sin(euler[0]/2)
            q = (chy*chp*shr - shy*shp*chr,
                 chy*shp*chr + shy*chp*shr,
                 shy*chp*chr - chy*shp*shr,
                 chy*chp*chr + shy*shp*shr)
            return rotation.quaternion.normalize(q)

        @staticmethod
        def to_euler(q):
            qx, qy, qz, qw = rotation.quaternion.normalize(q)
            r = np.arctan2(2 * (qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
            p = np.arcsin(2 * (qw*qy - qz*qx))
            y = np.arctan2(2 * (qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
            return np.array([r, p, y])


class projection(object):
    class orthogonal(object):
        """ This class implements methods for orthogonal
        projections in R3 to to_subspace S (usually
        a plane) in the direction of n. """
        @staticmethod
        def complement(n):
            """ Orthogonal complement of a given vector """
            np.random.seed(0)
            p1 = np.cross(np.random.randn(3), n)
            p2 = np.cross(p1, n)
            return np.vstack([vector.normalize(p1), vector.normalize(p2)]).T

        @staticmethod
        def to_subspace(S):
            """ Orthogonal projection to a given Subspace """
            A = np.array(S).T
            return A @ np.linalg.inv(A.T@A) @ A.T

        @staticmethod
        def to_xy():
            """ Orthogonal projection to xy plane """
            return np.diag([1.,1.,0.])

        @staticmethod
        def to_yz():
            """ Orthogonal projection to yz plane """
            return np.diag([0.,1.,1.]) 

        @staticmethod
        def to_zx():
            """ Orthogonal projection to zx plane """
            return np.diag([1.,0.,1.])

    class oblique(object):
        """ This class implements methods for oblique
        projections in R3 to to_subspace S (usually
        a plane) in the direction of n. """
        @staticmethod
        def from_subspaces(S1, S2): 
            b1 = np.vstack(S1).T 
            b2 = np.vstack(S2).T 
            M = np.hstack([b1, b2]) 
            P = np.zeros_like(M) 
            r = len(S1) 
            P[:r,:r] = np.eye(r)            
            return M@P@np.linalg.inv(M)

        @staticmethod
        def to_xy(n):
            """ Projection to a given Subspace in
            the direction of n"""
            A = np.array([[1.,0.,0.],[0.,1.,0.]]).T
            B = projection.orthogonal.complement(n)
            return A @ np.linalg.inv(B.T@A) @ B.T

        @staticmethod
        def to_yz(n):
            """ Projection to a given Subspace in
            the direction of n"""
            A = np.array([[0.,1.,0.],[0.,0.,1.]]).T
            B = projection.orthogonal.complement(n)
            return A @ np.linalg.inv(B.T@A) @ B.T

        @staticmethod
        def to_zx(n):
            """ Projection to a given Subspace in
            the direction of n"""
            A = np.array([[1.,0.,0.],[0.,0.,1.]]).T
            B = projection.orthogonal.complement(n)
            return A @ np.linalg.inv(B.T@A) @ B.T