#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:09:53 2019
@author: fran
"""
import numpy as np
import rospy
import geometry_msgs.msg as gm
import tf.transformations as transformations
import collections
from uv_network.nodes.ros_node import ROSNode
import uv_network.lib.navigation as navigation
import uv_network.lib.sensors as sensors
from uv_network.lib.constants import MSG_QUEUE_MAXLEN
import uv_network.lib.graph_tools as gt
import uv_network.dyn.holonomic as holonomic
from uv_network.msg import NavFilter, PoseAndRange, FloatArrayStamped, ImuMag
from gps_common.msg import GPSFix

class EKFNode(ROSNode):
    """ Este nodo implementa un Filtro Extendido de Kalman para estimar
    la posicion y velocidad de los vehiculos tomando mediciones de
    una IMU, GPS y pseudo-rango entre vehiculos.
    """
    def __init__(self):
        ROSNode.__init__(self)
        rospy.on_shutdown(self.shutdown)    # TODO: agregar el shutdown en ROSNode 
        #   Topic names
        self.imu_topic = self.topics['subs'][0]
        self.gps_topic = self.topics['subs'][1]
        self.range_topic = self.topics['subs'][2]        
        self.cmd_vel_topic = self.topics['subs'][3]                
        self.nav_topic = self.topics['pubs'][0]
        #   Neighborhood
        self.nbh = gt.Neighborhood(
            size=rospy.get_param(rospy.search_param('size'), 0),
            dim=rospy.get_param(rospy.search_param('dim'), 1)
        )
        rospy.loginfo('[%s] Neighborhood size: %s', self.nodeID, self.nbh.size)
        #   Creo el mensaje que en algún momento se va a enviar        
        self.nav_msg = NavFilter()
        #   Creo el filtro
        self.ekf = navigation.EKFTools()
        self.ekf.rate = rospy.get_param('~rate', 10.)
        self.rate = rospy.Rate(self.ekf.rate)
        rospy.loginfo('[%s] Rate: %s Hz', self.nodeID, self.ekf.rate)
        #   IMU
        self.imu = sensors.ImuTools(
            size = 8,
            rate = rospy.get_param('~imu_rate', 1.),
            GYRO_NSD = rospy.get_param(rospy.search_param('gyro/nsd'), 0.),    # deg/sec*sqrt(hz)
            GYRO_BIAS = rospy.get_param(rospy.search_param('gyro/bias'), 0.),  # deg/sec
            ACCEL_NSD = rospy.get_param(rospy.search_param('accel/nsd'), 0.),  # microg/sqrt(hz),
            ACCEL_BIAS_XY = 0.1*rospy.get_param(rospy.search_param('accel/bias_xy'), 0.),    # mg
            ACCEL_BIAS_Z = rospy.get_param(rospy.search_param('accel/bias_z'), 0.),      # mg
            MAG_NSD = rospy.get_param(rospy.search_param('mag/nsd'), 0.)       # valor tentativo
        )
        # self.nbh.motion_model = navigation.VelocityRandomWalk(
        #     rate = rospy.get_param('~imu_rate', 1.),
        #     NSD = 3*rospy.get_param(rospy.search_param('accel/nsd'), 0.),
        #     N = 2*self.nbh.size 
        # )
        self.nbh.motion_model = holonomic.VelocityRandomWalk(
            rate = rospy.get_param('~imu_rate', 1.),
            NSD = 3*rospy.get_param(rospy.search_param('accel/nsd'), 0.),
            agents = self.nbh.size,
            dim = self.nbh.dim
        )
        self.aug_imu = sensors.StackedSensor(self.imu, self.nbh.motion_model)
        #   GPS
        self.gps = sensors.GPSTools(
            VEL_SIGMA = rospy.get_param(rospy.search_param('gps/sigma_v'), 0.),  # m/s
            POS_SIGMA = rospy.get_param(rospy.search_param('gps/sigma_p'), 0.)  # m
        )
        #   Range
        self.range = sensors.RangeTools(
            SIGMA = rospy.get_param(rospy.search_param('range/sigma_r'), 0.)  # m
        )
        #   Initialize filter
        x0, y0, _ = rospy.get_param(rospy.search_param(self.ns + '/pose'), [0., 0., 0.])
        p0 = np.array([[x0], [y0]])
        X = np.block([[0.],
                      [0.],
                      [np.random.normal(p0, self.gps.pos.sigma)],
                      [0.],
                      [0.],
                      [0.],
                      [0.],
                      [np.zeros((2*self.nbh.size, 1))],
                      [np.random.normal(np.tile(p0.T, self.nbh.size).T, 5*self.gps.pos.sigma)]])
        dX = np.block([[self.gps.vel.sigma],
                       [self.gps.vel.sigma],
                       [self.gps.pos.sigma],
                       [self.gps.pos.sigma],
                       [1.],
                       [self.imu.accel.bias.x],
                       [self.imu.accel.bias.y],
                       [self.imu.gyro.bias.z],
                       [self.gps.vel.sigma*np.ones((2*self.nbh.size, 1))],
                       [5*self.gps.pos.sigma*np.ones((2*self.nbh.size, 1))]])
        t = rospy.Time.now().to_sec()
        self.ekf.start(t, X, dX)
        #   Creo el publicador que envia los resultados del filtro
        self.nav_pub = rospy.Publisher(
            name=self.nav_topic,
            data_class=NavFilter,
            queue_size=MSG_QUEUE_MAXLEN
        )
        rospy.loginfo('[%s] Publishing to topic: %s', self.nodeID, self.nav_topic)
        #   Creo el suscriptor que recibe los mensajes del GPS
        self.gps_sub = rospy.Subscriber(
            name=self.gps_topic,
            data_class=GPSFix,
            callback=self.gps_received,
            queue_size=MSG_QUEUE_MAXLEN
        )
        rospy.loginfo('[%s] Subscribing to topic: %s', self.nodeID, self.gps_topic)
        #   Creo el suscriptor que recibe los mensajes de Range
        self.range_sub = rospy.Subscriber(
            name=self.range_topic,
            data_class=PoseAndRange,
            callback=self.range_received,
            queue_size=MSG_QUEUE_MAXLEN
        )
        rospy.loginfo('[%s] Subscribing to topic: %s', self.nodeID, self.range_topic)
        #   Creo el suscriptor que recibe los mensajes de la IMU
        self.imu_sub = rospy.Subscriber(
            name=self.imu_topic,
            data_class=ImuMag,
            callback=self.imu_received,
            queue_size=MSG_QUEUE_MAXLEN
        )
        rospy.loginfo('[%s] Subscribing to topic: %s', self.nodeID, self.imu_topic)
        self.nav_msg.pose = [gm.PoseWithCovariance() for _ in range(self.nbh.size+1)]
        self.nav_msg.twist = [gm.TwistWithCovariance() for _ in range(self.nbh.size+1)]
        rospy.loginfo("[%s] Waiting for GPS or Range measurements.", self.nodeID)
        rospy.logdebug('[%s] Node debugging.', self.nodeID)

    def gps_received(self, gps_msg):
        """ This module makes a correction of the estimated state of
        the process based on the gps measurements.
        """
        y = np.array([[gps_msg.speed],
                      [gps_msg.track],
                      [gps_msg.longitude],
                      [gps_msg.latitude]])
        self.ekf.correction(y, self.gps.h)
    
    def range_received(self, pose_and_range_msg):
        id = pose_and_range_msg.header.frame_id
        if self.nbh.incomplete() and id not in self.nbh():
                self.nbh.append(id)
                rospy.loginfo('[%s] Neighborhood ordered list: %s', self.nodeID, self.nbh())
        #   Neighbor state and measurement
        p_m = np.array([[pose_and_range_msg.pose.pose.position.x],
                        [pose_and_range_msg.pose.pose.position.y]])
        p_c = np.array([[pose_and_range_msg.pose.covariance[0], pose_and_range_msg.pose.covariance[1]],
                        [pose_and_range_msg.pose.covariance[6], pose_and_range_msg.pose.covariance[7]]])
        y = np.block([[p_m],
                      [pose_and_range_msg.range]])
        _, idx = self.nbh.index(id, self.imu.size)
        self.ekf.correction(y, self.range.h, idx, p_m, p_c)

    def imu_received(self, imu_msg):
        """ This module propagates the dynamic of the process based on the 
        intertial measurements and disturbance model.
        """
        u = np.array([[imu_msg.linear_acceleration.x],
                      [imu_msg.linear_acceleration.y],
                      [imu_msg.angular_velocity.z]])
        y = np.array([[imu_msg.magnetic_field.x],
                      [imu_msg.magnetic_field.y]])
        #   Update filter
        t = imu_msg.header.stamp.to_sec()
        self.ekf.prediction(u, self.aug_imu.f, t)
        self.ekf.correction(y, self.imu.h)

    def publish(self):
        """ This function takes parameters calculated by the ekf and
        packs them into custom-made ROS NavFilter msg.
        """
        while not rospy.is_shutdown():
            if self.ekf.is_enable:
                #   Time Stamp
                self.nav_msg.header.stamp = rospy.Time.now()
                #   Frame id
                self.nav_msg.header.frame_id = self.num
                #   Pack euler angles
                self.nav_msg.euler = gm.Vector3(0. ,0., self.ekf.X[4][0])
                #   Pack Pose
                self.nav_msg.pose[0].pose.position = gm.Point(self.ekf.X[2], self.ekf.X[3], 0.)
                q = transformations.quaternion_from_euler(0. ,0., self.ekf.X[4][0])
                self.nav_msg.pose[0].pose.orientation = gm.Quaternion(*q)
                self.nav_msg.pose[0].covariance[0] = self.ekf.P[2][2]
                self.nav_msg.pose[0].covariance[1] = self.ekf.P[2][3]
                self.nav_msg.pose[0].covariance[6] = self.ekf.P[3][2]
                self.nav_msg.pose[0].covariance[7] = self.ekf.P[3][3]
                #   Pack Twist
                self.nav_msg.twist[0].twist.linear = gm.Vector3(self.ekf.X[0], self.ekf.X[1], 0.)
                self.nav_msg.twist[0].covariance[0] = self.ekf.P[0][0]
                self.nav_msg.twist[0].covariance[1] = self.ekf.P[0][1]
                self.nav_msg.twist[0].covariance[6] = self.ekf.P[1][0]
                self.nav_msg.twist[0].covariance[7] = self.ekf.P[1][1]
                #   Pack Neighbors
                for i, id in enumerate(self.nbh()):
                    v_idx, p_idx = self.nbh.index(id, self.imu.size)
                    self.nav_msg.pose[i+1].pose.position = gm.Point(self.ekf.X[p_idx], self.ekf.X[p_idx+1], 0.)
                    self.nav_msg.twist[i+1].twist.linear = gm.Point(self.ekf.X[v_idx], self.ekf.X[v_idx+1], 0.)
                #   Pack IMU parameters
                self.nav_msg.accel_bias = gm.Point(self.ekf.X[5], self.ekf.X[6], 0.)
                self.nav_msg.gyro_bias = gm.Point(0., 0., self.ekf.X[7])                
                #   Filter's Performance metric
                self.nav_msg.cov_sv = np.sqrt(np.diagonal(self.ekf.P))
                self.nav_msg.cov_tr = np.sqrt(np.trace(self.ekf.P))
                #   Envio msgs
                self.nav_pub.publish(self.nav_msg)        
                rospy.logdebug("[%s] New nav odometry published.", self.nodeID)
                rospy.logdebug("[%s]:\n%s", self.nodeID, self.nav_msg)
                self.rate.sleep()

    def shutdown(self):
        """ Unregisters publishers and subscribers and shutdowns timers
        """
        try:
            self.imu_sub.unregister()
            self.gps_sub.unregister()
            self.range_sub.unregister()                
            self.nav_pub.unregister()
        except AttributeError:
            pass

def main():
    try:
        ekf = EKFNode()
    except KeyboardInterrupt, rospy.RosInterruptException:
        rospy.loginfo("[%s] Received Keyboard Interrupt (^C).", ekf.nodeID)
    try:
        ekf.publish()
    except KeyboardInterrupt, rospy.RosInterruptException:    
        rospy.loginfo("[%s] Received Keyboard Interrupt (^C).", ekf.nodeID)

if __name__ == '__main__':
    main()