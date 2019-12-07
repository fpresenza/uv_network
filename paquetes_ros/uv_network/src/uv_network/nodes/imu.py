#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:46:11 2019
@author: fran
"""
import rospy
from uv_network.nodes.ros_node import ROSNode
import uv_network.lib.sensors as sensors
from uv_network.msg import FloatArrayStamped, ImuMag
# from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Quaternion
from std_msgs.msg import Header
from uv_network.lib.constants import MSG_QUEUE_MAXLEN

class ImuNode(ROSNode):
    """ Este nodo publica en /imu_raw_data un msg Imu
    """
    def __init__(self):
        ROSNode.__init__(self)
        rospy.on_shutdown(self.shutdown)    # TODO: agregar el shutdown en ROSNode 

        self.imu = sensors.ImuTools(
            rate = rospy.get_param('~rate', 1.),
            GYRO_SF = rospy.get_param(rospy.search_param('gyro/sf'), 1.),  
            GYRO_NSD = rospy.get_param(rospy.search_param('gyro/nsd'), 0.),    # deg/sec*sqrt(hz)
            GYRO_BIAS = rospy.get_param(rospy.search_param('gyro/bias'), 0.),  # deg/sec
            ACCEL_SF = rospy.get_param(rospy.search_param('accel/sf'), 1.),     
            ACCEL_NSD = rospy.get_param(rospy.search_param('accel/nsd'), 0.),  # microg/sqrt(hz)
            ACCEL_BIAS_XY = rospy.get_param(rospy.search_param('accel/bias_xy'), 0.),   # mg
            ACCEL_BIAS_Z = rospy.get_param(rospy.search_param('accel/bias_z'), 0.),     # mg
            MAG_SF = rospy.get_param(rospy.search_param('mag/sf'), 1.),    
            MAG_BIAS = rospy.get_param(rospy.search_param('mag/bias'), 0.),     # microT
            MAG_NSD = rospy.get_param(rospy.search_param('mag/nsd'), 0.)       # valor tentativo
        )
        self.rate = rospy.Rate(self.imu.rate)
        rospy.loginfo('[%s] Rate: %s Hz', self.nodeID, self.imu.rate)
        rospy.loginfo("[%s] bias: ax=%s, ay=%s, gz=%s.", self.nodeID,
            self.imu.accel.bias_sample.x, self.imu.accel.bias_sample.y, self.imu.gyro.bias_sample.z)

        #   Subscriber and Pubishers
        self.motion = {'topic': self.topics['subs'][0]}
        self.imu_raw_data = {'topic': self.topics['pubs'][0],
            'msg': ImuMag(header=Header(frame_id=self.num))} 
        #   Get first msgs
        rospy.loginfo("[%s] Waiting for /%s msg to initialize...", self.nodeID, self.motion['topic'])
        motion = rospy.wait_for_message(self.motion['topic'], FloatArrayStamped)
        self.imu(motion.data)
        #   Creo el publicador que despacha los mensajes del imu
        self.imu_raw_data['pub'] = rospy.Publisher(
            name=self.imu_raw_data['topic'],
            data_class=ImuMag,
            queue_size=MSG_QUEUE_MAXLEN
        )
        rospy.loginfo('[%s] Publishing to topic: %s', self.nodeID, self.imu_raw_data['topic'])
        
        #   Creo el suscriptor que recibe los mensajes del modelo dinamico
        self.motion['sub'] = rospy.Subscriber(
            name=self.motion['topic'],
            data_class=FloatArrayStamped,
            callback=lambda motion: self.imu(motion.data),
            queue_size=MSG_QUEUE_MAXLEN
        )
        rospy.loginfo('[%s] Subscribing to topic: %s', self.nodeID, self.motion['topic'])
        rospy.loginfo('[%s] Node initialized.', self.nodeID)
        rospy.logdebug('[%s] Node debugging.', self.nodeID)

    def enable(self):
        """ Send imu msgs at a fixed rate
        """
        while not rospy.is_shutdown():
            #   Time stamp
            self.imu_raw_data['msg'].header.stamp = rospy.Time.now()
            #   Preparo msg
            self.imu_raw_data['msg'].angular_velocity = Vector3(*self.imu.gyro.meas)
            self.imu_raw_data['msg'].linear_acceleration = Vector3(*self.imu.accel.meas)
            self.imu_raw_data['msg'].magnetic_field = Vector3(*self.imu.mag.meas)
            # self.imu_raw_data['msg'].orientation = Quaternion(*self.imu.quaternion)
            #   Envio msg
            self.imu_raw_data['pub'].publish(self.imu_raw_data['msg'])
            rospy.logdebug("[%s] New imu_raw_data published.", self.nodeID)
            rospy.logdebug("[%s]:\n%s", self.nodeID, self.imu_raw_data['msg'])
            self.rate.sleep()

    def shutdown(self):
        """ Unregisters publishers and subscribers and shutdowns timers
        """
        try:
            self.motion['sub'].unregister()          
            self.imu_raw_data['pub'].unregister()
        except AttributeError:
            pass

def main():
    try:
        imu = ImuNode()
    except KeyboardInterrupt, rospy.RosInterruptException:
        rospy.loginfo("Received Keyboard Interrupt (^C).")
    try:
        imu.enable()
    except KeyboardInterrupt, rospy.RosInterruptException:
        rospy.loginfo("Received Keyboard Interrupt (^C).")

if __name__ == '__main__':
    main()