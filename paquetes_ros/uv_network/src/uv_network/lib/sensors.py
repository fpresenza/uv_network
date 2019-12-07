#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 5 14:07:45 2020
@author: fran
"""
import numpy as np
import scipy as sp
import scipy.optimize
import collections
import tf.transformations as transformations
import uv_network.lib.integradores as integradores
from uv_network.lib.tools import Vec3, Sen, GPSMeas
from uv_network.lib.constants import GRAVITY, EARTH_MAGNETIC_FIELD

class StackedSensor(object):
    """ This class stores info an modules of generic sensor.
    """
    def __init__(self, *sens, **kwargs):
        self.s = sens
        self.rate = sens[0].rate
        self.Q = np.empty(0)
    
    def f(self, mean, u):
        """ Dynamic propagiaton of stacked gaussian model 
        """
        #   split mean to each sensor with appropiate dimensions
        splitter =  np.cumsum([len(sen) for sen in self.s][:-1])
        split_mean = np.split(mean, splitter)
        #   dynamic
        dot_Xs, F_Xs, Bs, Qs = np.array([sen.f(mean, u) for mean, sen in zip(split_mean, self.s)]).T 
        dot_X = np.vstack(dot_Xs)
        F_X = sp.linalg.block_diag(*F_Xs)
        B = sp.linalg.block_diag(*Bs)
        self.Q = sp.linalg.block_diag(*Qs)
        return dot_X, F_X, B, self.Q
        

class ImuTools(object):
    """ This class contains info and modules to work with an IMU
    """
    def __init__(self, **kwargs):
        self.size = kwargs.get('size', 1)
        self.rate = kwargs.get('rate', 1.)
        #   Gyroscope params
        GYRO_SF = kwargs.get('GYRO_SF', 0.)
        GYRO_NSD = kwargs.get('GYRO_NSD', 0.)
        GYRO_BIAS = kwargs.get('GYRO_BIAS', 0.)
        #   Accelerometer params
        ACCEL_SF = kwargs.get('ACCEL_SF', 0.)
        ACCEL_NSD = kwargs.get('ACCEL_NSD', 0.)
        ACCEL_BIAS_XY = kwargs.get('ACCEL_BIAS_XY', 0.)
        ACCEL_BIAS_Z = kwargs.get('ACCEL_BIAS_z', 0.)
        #   Compass params
        MAG_SF = kwargs.get('MAG_SF', 0.)
        MAG_NSD = kwargs.get('MAG_NSD', 0.)
        MAG_BIAS = kwargs.get('MAG_BIAS', 0.)
        #   Convert raw imu parameters to appropiate units
        BW = 0.5*self.rate
        #   Gyroscore params
        gyro_bias_xyz = GYRO_BIAS * (np.pi / 180)
        self.gyro = Sen(
            sigma = GYRO_NSD * np.sqrt(2 * BW) * (np.pi / 180),   # rad/s
            bias_sample  = Vec3(np.random.normal(0, gyro_bias_xyz),   # rad/s
                                np.random.normal(0, gyro_bias_xyz),   # rad/s   
                                np.random.normal(0, gyro_bias_xyz)),   # rad/s
            bias = Vec3(gyro_bias_xyz,
                        gyro_bias_xyz,
                        gyro_bias_xyz),
            bias_drift = Vec3(0.1* gyro_bias_xyz,
                              0.1* gyro_bias_xyz,
                              0.1* gyro_bias_xyz),
            meas  = Vec3(0.,0.,0.)              
        )
        #   Accelerometer params
        accel_bias_xy_mks = ACCEL_BIAS_XY * GRAVITY * 0.001
        accel_bias_z_mks = ACCEL_BIAS_Z * GRAVITY * 0.001
        self.accel = Sen(
            sigma = ACCEL_NSD * np.sqrt(2 * BW) * GRAVITY * 0.001,  # m/s2
            bias_sample  = Vec3(np.random.normal(0, accel_bias_xy_mks),   # m/s2
                                np.random.normal(0, accel_bias_xy_mks),
                                np.random.normal(0, accel_bias_z_mks)),
            bias = Vec3(accel_bias_xy_mks,
                        accel_bias_xy_mks,
                        accel_bias_z_mks),
            bias_drift = Vec3(0.05 * accel_bias_xy_mks,
                              0.05 * accel_bias_xy_mks,
                              0.05 * accel_bias_z_mks),
            meas = Vec3(0.,0.,0.) 
        )
        #   Compass params
        self.mag = Sen(
            sigma = MAG_NSD * np.sqrt(2 * BW),     # microT
            meas = Vec3(0.,0.,0.)
        )
        #   EKF noise matrix 
        self.Q = np.diag([self.accel.sigma**2,
                          self.accel.sigma**2,
                          self.gyro.sigma**2,
                          self.accel.bias_drift.x**2,
                          self.accel.bias_drift.y**2,
                          self.gyro.bias_drift.z**2])
        self.R = np.diag([self.mag.sigma**2,
                          self.mag.sigma**2])

    def __call__(self, motion):
        """ Simulate imu_raw_data.
        This module takes as parameter accel, vel and pose
        of vehicle in a 9-dim list.
        """
        ae = Vec3(*motion[0:3])
        ve = Vec3(*motion[3:6])
        pe = Vec3(*motion[6:9])
        #   giroscopo
        self.gyro.meas.x = 0. + self.gyro.bias_sample.x + np.random.normal(0, self.gyro.sigma)
        self.gyro.meas.y = 0. + self.gyro.bias_sample.y + np.random.normal(0, self.gyro.sigma)
        self.gyro.meas.z = ve.z + self.gyro.bias_sample.z + np.random.normal(0, self.gyro.sigma)
        #   acelerometro
        Rb2e = transformations.euler_matrix(0.,0.,pe.z)[:3,:3]
        Re2b = Rb2e.T
        accel_earth = np.array([ae.x,
                                ae.y,
                                0.])
        axb, ayb, azb = np.dot(Re2b, accel_earth)
        self.accel.meas.x = axb + self.accel.bias_sample.x + np.random.normal(0, self.accel.sigma)
        self.accel.meas.y = ayb + self.accel.bias_sample.y + np.random.normal(0, self.accel.sigma)
        self.accel.meas.z = azb + self.accel.bias_sample.z + np.random.normal(0, self.accel.sigma)
        #   magnetometro
        Me = np.array([[0.],
                       [EARTH_MAGNETIC_FIELD],
                       [0.]])
        mxb, myb, mzb = np.dot(Re2b, Me)
        self.mag.meas.x = mxb + np.random.normal(0, self.mag.sigma)
        self.mag.meas.y = myb + np.random.normal(0, self.mag.sigma)
        self.mag.meas.z = 0. + np.random.normal(0, self.mag.sigma)
    
    def __len__(self):
        return self.size
    
    def f(self, mean, u):
        """ This function takes mean and covariance of a gaussian proccess,
        an IMU measurement to generate dot_X and matrices needed for prediction.
        """
        v = mean[:2]
        x = mean[2:4]
        yaw = mean[4][0]
        bf = mean[5:7]
        bw = mean[7][0]
        uf = u[:2]
        uw = u[2]
        xif = np.random.normal(0., np.array([[self.accel.bias_drift.x],
                                             [self.accel.bias_drift.y]]))
        xiw = np.random.normal(0., self.gyro.bias_drift.z)
        Rbe = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
        dot_v = np.dot(Rbe, uf-bf)
        dot_x = v
        dot_yaw = uw - bw
        dot_bf = xif
        dot_bw = xiw

        dot_X = np.block([[dot_v],
                          [dot_x],
                          [dot_yaw],
                          [dot_bf],
                          [dot_bw]])

        dRbe_dyaw = np.array([[-np.sin(yaw), -np.cos(yaw)],
                              [np.cos(yaw), -np.sin(yaw)]])

        F_X = np.zeros((mean.size, mean.size))
        F_X[0:2,4:5] = np.dot(dRbe_dyaw, uf-bf) 
        F_X[0:2,5:7] = -Rbe
        F_X[2:4,0:2] = np.eye(2)
        F_X[4,7] = -1

        B = np.zeros((mean.size, 6))
        B[0:2,0:2] = Rbe 
        B[4:8,2:6] = np.eye(4) 
        return dot_X, F_X, B, self.Q

    def h(self, mean):
        """ This function takes in mean to compute expected 
        magnetometer measurement, and jacobians needed for 
        the correction step.
        """
        yaw = mean[4][0]
        Reb = np.array([[np.cos(yaw), np.sin(yaw)],
                        [-np.sin(yaw), np.cos(yaw)]])
        dReb_dyaw = np.array([[-np.sin(yaw), np.cos(yaw)],
                              [-np.cos(yaw), -np.sin(yaw)]])
        Me = np.array([[0.],
                       [EARTH_MAGNETIC_FIELD]])
        hat_y = np.dot(Reb, Me)
        H = np.zeros((2, mean.size))
        H[:,4:6] = np.dot(dReb_dyaw, Me)
        return hat_y, H, self.R


class GPSTools(object):
    """ This class contains info and modules to work with an GPS
    """
    def __init__(self, **kwargs):
        self.rate = kwargs.get('rate', 1.)
        VEL_SIGMA = kwargs.get('VEL_SIGMA', 0.)
        POS_SIGMA = kwargs.get('POS_SIGMA', 0.)
        #   Convert raw imu parameters to appropiate units
        self.vel = Sen(
            sigma = VEL_SIGMA,
            meas = Vec3(0.,0.,0.)
        )
        self.pos = Sen(
            sigma = POS_SIGMA,
            meas = Vec3(0.,0.,0.)
        )
        #   EKF noise matrix
        self.R = np.diag([self.vel.sigma**2,
                          self.vel.sigma**2,
                          self.pos.sigma**2,
                          self.pos.sigma**2])

    def __call__(self, motion):
        """ This functions takes in a velocity, a position,
        and standard deviations and simulate gps_raw_data
        as normal random process.
        """
        # TODO: convertir las desviaciones standard
        ve = Vec3(*motion[3:6])
        pe = Vec3(*motion[6:9])
        self.vel.meas.x = ve.x + np.random.normal(0., self.vel.sigma)
        self.vel.meas.y = ve.y + np.random.normal(0., self.vel.sigma)
        self.vel.meas.z = 0. + np.random.normal(0., self.vel.sigma)
        #   LGV-ENU frame 
        self.pos.meas.x = pe.x + np.random.normal(0., self.pos.sigma)     #   east <--> x
        self.pos.meas.y = pe.y + np.random.normal(0., self.pos.sigma)      #   north <--> y
        self.pos.meas.z = 0. + np.random.normal(0., self.pos.sigma)                                              #   up <--> z

    def sequence_generator(self, uv_list, method=None, weights=None, seq=0):
        """ This function implements one of desired method to
        generate the sequence of vehicles that are reached 
        with gps signals.
        
        If 'method' kwarg not set (=None)
        this method generate a random sequence
        with probabilities specified in weights kwarg,
        but are all time invariant. If weights is not set, 
        then probabilities are all equal.

        If method='exponential':
        this method generate a random sequence
        with equal initial probabilities
        but all exponentially approaching 0 except
        for one uv which probability is 
        exponentially approaching 1.
        """
        if method is 'exponential':
            alpha = 0.02                                    #   dumping parameter
            N = len(uv_list)                                #   number of vehicles
            w0 = np.divide(np.ones(N), N)                   #   initial weights
            coeff = np.hstack((np.ones(1), np.zeros(N-1)))  #   coeficient array 
            weights = np.multiply(w0, np.exp(-alpha*seq)) + np.multiply(coeff, 1-np.exp(-alpha*seq)) 
        return np.random.choice(uv_list, p=weights)

    def h(self, mean):
        """ This function takes in mean to compute expected 
        gps measurement, and jacobians needed for the correction step
        """
        H = np.zeros((4, mean.size))
        H[0:4,0:4] = np.eye(4)
        hat_y = np.dot(H, mean)
        return hat_y, H, self.R

class RangeTools(object):
    """ This class contains info and modules to work with a range sensor
    """
    def __init__(self, **kwargs):
        self.rate = kwargs.get('rate', 1.)
        SIGMA = kwargs.get('SIGMA', 0.)
        self.sigma = SIGMA
        #   EKF noise matrix
        self.R = np.diag([self.sigma**2])

    def __call__(self, my_pos, partner_pos):
        """ Calculates the range based on client pose and 
        partner pose estimated by the EKF.
        """
        x = my_pos.x
        y = my_pos.y
        u = partner_pos.x
        v = partner_pos.y        
        return np.sqrt((x-u)**2 + (y-v)**2) + np.random.normal(0, self.sigma)

    def valid_measurements(self, partners):
        # TODO: estudiar un algoritmo para decidir cuanto es prudente
        # inicializar el filtro basado en mediciones de rango. 
        # Posiblemente haya que medir la covariancia de la
        # estimacion (x,y) y en base a ella decidir.

        # Get a list of al id's in measurements.
        # If at least 10 pose measurements (in total) 
        # from 3 different agents is received, 
        # then set valid_measurements to True
        ids_list = list(set([id for (k, id) in partners.keys()]))
        return len(partners.keys())>=10 and len(ids_list)>=3      

    def multilateration(self, partners):
        """ This module compute de optimal agent position xy
        that minimizes de sum of the squared distance to each 
        of the ranges measured by the receptor.
        """
        def sqr_dist_error(pos, points, ranges):
            sqr_err_sum = 0 
            for k, (x,y) in points.items():
                d = np.sqrt((pos[0]-x)**2 + (pos[1]-y)**2) - ranges[k]
                sqr_err_sum += d**2
            return sqr_err_sum
        points = {}
        ranges = {}
        for (k, id), partner in partners.items():
            points[k] = (partner['pose'].pose.position.x,
                         partner['pose'].pose.position.y)
            ranges[k] = partner['range']
        sol = scipy.optimize.minimize(sqr_dist_error, np.array([0.,0.]),\
            method='SLSQP', args=(points,ranges))
        return sol.x
   
    def h(self, mean, idx, p_m, p_c):
        """ This function takes in mean and and partner's mean and
        covariance to compute expected range measurement, and jacobians
        needed for the correction step
        """
        #   Expected measurement
        m = Vec3(mean[2][0], mean[3][0], 0.)        #   Own pose
        p = Vec3(mean[idx][0], mean[idx+1][0], 0.)        #   Partner's pose        
        rho = np.sqrt((m.x-p.x)**2 + (m.y-p.y)**2)
        hat_y = np.array([[p.x],
                          [p.y],
                          [rho]])
         #   Jacobians
        H = np.zeros((3, mean.size))
        H[0:2,idx:idx+2] = np.eye(2)
        H[2,2:4] = (m.x-p.x)/rho, (m.y-p.y)/rho
        H[2,idx:idx+2] = (p.x-m.x)/rho, (p.y-m.y)/rho
        #   Noise
        R = sp.linalg.block_diag(p_c, self.R)
        return hat_y, H, R