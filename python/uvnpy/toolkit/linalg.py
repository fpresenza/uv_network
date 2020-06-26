#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue May 06 14:15:45 2020
@author: fran
"""
import numpy as np
import scipy.linalg
import quaternion

def skew(w): 
    return np.array([[    0, -w[2],  w[1]], 
                     [ w[2],     0, -w[0]], 
                     [-w[1],   w[0],    0]])

def block_diag(*m):
    return scipy.linalg.block_diag(*m)

def multi_matmul(*mats): 
    M = mats[-1] 
    for mat in reversed(mats[:-1]): 
        M = np.matmul(mat, M) 
    return M 

def norm(v):
    return np.sqrt(np.inner(v,v)) 

def normalize(v):
    return np.divide(v, norm(v))

def distance(u, v):
    return norm(np.subtract(u, v))

def angle_between(u, v):
    i = np.inner(u, v)
    n  = norm(u) * norm(v)
    return np.arccos(i/n)

def heading(v):
    return np.arctan2(v[1], v[0])

def line(u, v, t):
    return u + t * np.subtract(v,u)

def proj(u, v):
    v = np.asarray(v)
    return np.divide(np.inner(u,v), np.inner(v, v)) * v


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


class rm(object):
    """ This class implements basic rotation matrix 
    algorithms in a 3d euclidean space """
    @staticmethod
    def check(R):
        """ returns True if R is an orthogonal matrix, False otherwise """
        try: 
            return np.allclose(np.matmul(R,R.T), np.eye(3)) 
        except (AttributeError, ValueError): 
            return False 

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
    def ZYX(cls, e):
        return np.matmul(cls.Rz(e[2]), np.matmul(cls.Ry(e[1]), cls.Rx(e[0])))

    @staticmethod
    def from_vec(n, t):
        """ Rodrigues rotation formula """
        S = skew(normalize(n))
        return np.eye(3) + np.sin(t)*S + (1-np.cos(t))*np.dot(S,S)

    @classmethod
    def from_any(cls, *rotations):
        matrix = np.eye(3)
        for rot in reversed(rotations):
            if cls.check(rot):
                matrix = np.matmul(rot, matrix)
            elif isinstance(rot, np.quaternion):
                matrix = np.matmul(quaternion.as_rotation_matrix(rot), matrix)
            elif len(rot)==3:
                matrix = np.matmul(cls.ZYX(rot), matrix)
            elif len(rot)==2:
                matrix = np.matmul(cls.from_vec(rot[0],rot[1]), matrix)
        return matrix


class quat(object):
    """ This class implements basic quaternion
    algorithms in a 3d euclidean space """
    @staticmethod
    def normalize(q):
        # make the scalar part always positive 
        if q.w < 0:  
            q *= -1
        return q.normalized()

    @staticmethod
    def qx(roll):
        chr = np.cos(roll/2)
        shr = np.sin(roll/2)
        return np.quaternion(chr, shr, 0, 0)
    
    @staticmethod
    def qy(pitch):
        chp = np.cos(pitch/2)
        shp = np.sin(pitch/2)
        return np.quaternion(chp, 0, shp, 0)
    
    @staticmethod
    def qz(yaw):
        chy = np.cos(yaw/2)
        shy = np.sin(yaw/2)
        return np.quaternion(chy, 0, 0, shy)

    @classmethod
    def ZYX(cls, e):
        return cls.qz(e[2]) * cls.qy(e[1]) * cls.qx(e[0])

    @classmethod
    def to_ZYX(cls, q):
        q = cls.normalize(q)
        return np.array([np.arctan2(2 * (q.w*q.x + q.y*q.z), 1 - 2*(q.x**2 + q.y**2)),
                         np.arcsin(2 * (q.w*q.y - q.z*q.x)),
                         np.arctan2(2 * (q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))])


class orthogonal(object):
    """ This class implements methods for orthogonal
    projections in R3 to subspace S (usually
    a plane). """
    @staticmethod
    def complement(n):
        """ Orthogonal complement of a 1-D subspace """
        p1 = np.cross([0.4236548 , 0.64589411, 0.43758721], n)
        p2 = np.cross(p1, n)
        return np.vstack([normalize(p1), normalize(p2)]).T

    @staticmethod
    def projection(S):
        """ Orthogonal projection to a given Subspace """
        A = np.array(S).T
        S = np.linalg.inv(np.matmul(A.T, A))
        return np.matmul(A, np.matmul(S, A.T))

    @staticmethod
    def projection_to_xy():
        """ Orthogonal projection to xy plane """
        return np.diag([1.,1.,0.])

    @staticmethod
    def projection_to_yz():
        """ Orthogonal projection to yz plane """
        return np.diag([0.,1.,1.]) 

    @staticmethod
    def projection_to_zx():
        """ Orthogonal projection to zx plane """
        return np.diag([1.,0.,1.])


class oblique(object):
    """ This class implements methods for oblique
    projections in R3 to to_subspace S (usually
    a plane) in the direction of n. """
    @staticmethod
    def projection(S1, S2): 
        M = np.vstack([*S1, *S2]).T
        P, r = np.zeros_like(M), len(S1)
        P[:r,:r] = np.eye(r)            
        return np.matmul(M, np.matmul(P, np.linalg.inv(M)))

    @classmethod
    def projection_to_xy(cls, n):
        """ Projection to xy plane in
        the direction of n"""
        return cls.projection(([1,0,0],[0,1,0]), (n,))

    @classmethod
    def projection_to_yz(cls, n):
        """ Projection to a given Subspace in
        the direction of n"""
        cls.projection(([0,1,0],[0,0,1]), (n,))

    @classmethod
    def projection_to_zx(cls, n):
        """ Projection to a given Subspace in
        the direction of n"""
        return cls.projection(([1,0,0],[0,0,1]), (n,))


class ht(object):
    """ This class implements basic functions of homogeneous
    transformation in R2 or R3. """
    @staticmethod
    def _aug(x):
        return np.hstack([*x, 1])

    @staticmethod
    def _red(x):
        return x[:-1]

    @classmethod
    def dot(cls, T, x):
        return cls._red(np.matmul(T, cls._aug(x)))

    @staticmethod
    def matrix(R, t):
        t = np.reshape(t, (-1,1))
        z = np.zeros_like(t).flatten()
        return np.block([[R, t],
                         [z, 1]])

    @classmethod
    def inv_matrix(cls, R, t):
        return cls.matrix(R.T, np.matmul(-R.T, t))

    @classmethod
    def apply(cls, R, t, x):
        return cls._red(np.matmul(cls.matrix(R, t), cls._aug(x)))

    @classmethod
    def inv_apply(cls, R, t, x):
        return cls._red(np.matmul(cls.inv_matrix(R, t), cls._aug(x)))

    @staticmethod
    def split(T):
        return T[:-1,:-1], T[:,-1][:-1]
