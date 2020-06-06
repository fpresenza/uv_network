#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri June 05 19:16:34 2020
@author: fran
"""
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from uvnpy.tools.linalg import vector, rotation
import uvnpy.graphix.spatial


class Camera(object):
    """ This class implements model of a camera """
    def __init__(self, **kwargs):
        self.hfov = 0.5 * kwargs.get('fov', np.radians(60.))
        self.tanhfov = np.tan(self.hfov)
        self.zoom = kwargs.get('zoom', 3.)
        self.center = np.asarray(kwargs.get('center', (0.,0.,0.)))
        self.ar = kwargs.get('aspect_ratio', 4./3)
        self.direction(**kwargs)
        self.color = kwargs.get('color', 'gray')
        self.alpha = kwargs.get('alpha', 0.3)

    def position(self, pos):
        self.center = np.asarray(pos)

    def direction(self, *rot, **kwargs):
        gimbal = kwargs.get('gimbal', (0.,0.,0.))
        if len(gimbal)==3:
            self.Rot = rotation.matrix.RZYX(*gimbal)
        elif len(gimbal)==4:
            self.Rot = rotation.matrix.from_quat(gimbal)

        for R in rot:
            self.Rot = R @ self.Rot
        self.normal = self.Rot[:,0]

    def view(self, ax, **kwargs):
        """ Plot a prisma representing field of view and
        footprint on xy plane """
        # prisma
        nx, ny, nz = self.Rot.T
        tx = nx * self.zoom
        tz = nz * self.zoom * self.tanhfov
        ty = ny * self.zoom * self.tanhfov * self.ar
        corners = self.center + tx + np.array([ty+tz, ty-tz, -ty-tz, -ty+tz])
        prisma = [[self.center, corners[0], corners[1]],
                  [self.center, corners[0], corners[3]],
                  [self.center, corners[1], corners[2]],
                  [self.center, corners[3], corners[2]]]
        verts = prisma

        # footprint
        def get_t(u, v):
            """ returns the scalar "t" that gives the a vector included in
            the rect that joins vectors "u" and "v" with z-value equal to 
            90% of self.center """
            if v[2] != u[2]:
                c = 0.9*self.center[2]
                return np.divide(c-u[2], v[2]-u[2])
            else:
                return -1

        def below(v):
            """ return True if vector "v" is below self.center """
            return v[2] < self.center[2]

        vertices = []
        for i,j in [(0,1),(1,2),(2,3),(3,0)]:
            if below(corners[i]):
                vertices += [corners[i]]
            t = get_t(corners[i], corners[j])
            if (t>0 and t<1):
                vertices += [vector.rect(corners[i], corners[j], t)]

        def ftp(v):
            vx, vy, vz = v
            cx, cy, cz = self.center
            if vz != 0:
                k = - cz / vz
                return np.array([cx+k*vx, cy+k*vy, 0.])
            else:
                return v 

        if len(vertices)>0:
            vertices = np.subtract(vertices, self.center)
            footprint = [[ftp(v) for v in vertices]]
            verts += footprint
        view = Poly3DCollection(verts, color=self.color, lw=0.3, alpha=self.alpha)
        ax.add_collection3d(view)

        # center of image on plane xy
        if nx[2]<0:
            center = ftp(nx)
            circle = uvnpy.graphix.spatial.circle3D(center, (0,0,1), 0.35)
            circle = np.delete(circle, 2, axis=0).T
            
            yaw = vector.heading(nx)
            Rz = rotation.matrix.Rz(yaw)
            x = Rz[:2,:2] @ (0.,1)
            points = [center[:2] + [x, 0.5*x], center[:2] + [-0.5*x, -x]] + \
            [circle[i:i+2] for i in range(len(circle)) if len(circle[i:i+2])==2]
            marker = mpl.collections.LineCollection(points, color=self.color, lw=0.8)
            ax.add_collection3d(marker)
        else:
            marker = mpl.collections.LineCollection([])
            ax.add_collection3d(marker)

        return [view, marker]