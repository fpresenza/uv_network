#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:36:47 2019

@author: fran
"""

# ROS Python API
import rospy
# Import numpy to process data
import numpy as np
# Import custom ROSNode class
from uv_network.nodes.ros_node import ROSNode
# Import the messages we're interested in sending and receiving
from uv_network.msg import FloatArrayStamped, NavFilter
#   Import constants
from uv_network.lib.constants import MSG_QUEUE_MAXLEN

class NormalRandomSignalNode(ROSNode):
    """
    Este nodo publica en /reference un FloatArrayStamped
    """
    def __init__(self):
        ROSNode.__init__(self)
        rospy.on_shutdown(self.shutdown)    # TODO: agregar el shutdown en ROSNode  

        #   Topic names
        self.nrs_topic = self.topics['pubs'][0]

        #   Creo el mensaje que en algún momento se va a enviar
        self.nrs_msg = FloatArrayStamped()
        self.nrs_msg.header.frame_id = self.num

        #   Fix rate to send normal random signal msgs
        self.nrs_rate = rospy.get_param('~rate', 1)   # Hz
        self.rate = rospy.Rate(self.nrs_rate)

        rospy.loginfo('[%s] Rate: %s Hz', self.nodeID, self.nrs_rate)

        #   normal random signal
        #   mean
        self.nrs_mean = rospy.get_param('~nrs_mean', [0., 0., 0.])        
        rospy.loginfo('[%s] nrs mean: %s', self.nodeID, self.nrs_mean)
        #   covariance
        self.nrs_covar = rospy.get_param('~nrs_covar', [[1., 0., 0.],[0.,1.,0.],[0.,0.,1.]])
        rospy.loginfo('[%s] nrs covariance: %s', self.nodeID, self.nrs_covar)        

        # Creo el publicador que despacha los mensajes de reference
        self.nrs_pub = rospy.Publisher(
            name=self.nrs_topic,
            data_class=FloatArrayStamped,
            queue_size=MSG_QUEUE_MAXLEN
            )
        rospy.loginfo('[%s] Publishing to topic: %s', self.nodeID, self.nrs_topic)

        #   Wait until filter is enabled before sending the signal
        rospy.loginfo("[%s] Waiting for %s msg to initialize...", self.nodeID, 'nav')
        rospy.wait_for_message('nav', NavFilter)

        rospy.loginfo('[%s] Node initialized.', self.nodeID)
        rospy.logdebug('[%s] Node debugging.', self.nodeID)


    def generate_nrs(self):
        """
        Send random nrs msgs at a fixed rate
        """
        while not rospy.is_shutdown():
            # Actualizo data
            self.nrs_msg.data = np.random.multivariate_normal(self.nrs_mean, self.nrs_covar)
            # Time stamp
            self.nrs_msg.header.stamp = rospy.Time.now()
            # Envio msg
            self.nrs_pub.publish(self.nrs_msg)            
            rospy.logdebug("[%s] New nrs published", self.nodeID)
            rospy.logdebug("[%s]:\n%s", self.nodeID, self.nrs_msg)

            self.rate.sleep()


    def shutdown(self):
        """
        Unregisters publishers and subscribers and shutdowns timers
        """
        try:
            self.nrs_pub.unregister()
        except AttributeError:
            pass


def main():
    nrs = NormalRandomSignalNode()
    nrs.generate_nrs()


if __name__ == '__main__':
    main()