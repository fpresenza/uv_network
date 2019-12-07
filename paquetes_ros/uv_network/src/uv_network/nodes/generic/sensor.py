#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:46:11 2019

@author: fran
"""
# ROS Python API
import rospy
import message_filters
# Import numpy to process data
import numpy as np
# Import custom ROSNode class
from uv_network.nodes.ros_node import ROSNode
# Import the messages we're interested in sending and receiving
from uv_network.msg import FloatArray, FloatArrayStamped
#   Import constants
from uv_network.lib.constants import MSG_QUEUE_MAXLEN

class SensorNode(ROSNode):
    """
    Este nodo publica en /measurement un FloatArrayStamped
    """
    def __init__(self):
        ROSNode.__init__(self)
        rospy.on_shutdown(self.shutdown)    # TODO: agregar el shutdown en ROSNode 

        # Topic names
        self.state_topic = self.topics['subs'][0]
        self.measurement_topic = self.topics['pubs'][0]

        # Creo el mensaje que en algún momento se va a enviar        
        self.measurement_msg = FloatArrayStamped()
        self.measurement_msg.header.frame_id = self.num

        # Fix rate to send measurement msgs
        self.sensor_rate = rospy.get_param(rospy.search_param('sensor/rate'), 10)   # Hz

        self.rate = rospy.Rate(self.sensor_rate)

        rospy.loginfo('[%s] Rate: %s Hz', self.nodeID, self.sensor_rate)

        rospy.loginfo("[%s] Waiting for /state msg to initialize...", self.nodeID,)
        state_msg = rospy.wait_for_message(self.state_topic, FloatArrayStamped)
        rospy.logdebug("[%s] First /state msg received.", self.nodeID)
        rospy.logdebug("[%s]:\n%s", self.nodeID, state_msg)
        
        # Guardo el estado inicial
        self.last_state_available = state_msg.data
        
        # Creo el publicador que despacha los mensajes del sensor
        self.measurement_pub = rospy.Publisher(
            name=self.measurement_topic,
            data_class=FloatArrayStamped,
            queue_size=MSG_QUEUE_MAXLEN
            )
        rospy.loginfo('[%s] Publishing to topic: %s', self.nodeID, self.measurement_topic)
        
        # Creo el suscriptor que recibe los mensajes del estado 
        self.state_sub = rospy.Subscriber(
            name=self.state_topic,
            data_class=FloatArrayStamped,
            callback=self.update,
            queue_size=MSG_QUEUE_MAXLEN
            )
        rospy.loginfo('[%s] Subscribing to topic: %s', self.nodeID, self.state_topic)

        rospy.loginfo('[%s] Node initialized.', self.nodeID)
        rospy.logdebug('[%s] Node debugging.', self.nodeID)


    def update(self, state):
        """
        Update measurement
        """
        self.last_state_available = state.data


    def transmit(self):
        """
        Send sensor msgs at a fixed rate
        """
        while not rospy.is_shutdown():
            # Actualizo data
            self.measurement_msg.data = self.last_state_available
            # Time stamp
            self.measurement_msg.header.stamp = rospy.Time.now()
            # Envio msg
            self.measurement_pub.publish(self.measurement_msg)
            rospy.logdebug("[%s] New measurement published.", self.nodeID)
            rospy.logdebug("[%s]:\n%s", self.nodeID, self.measurement_msg)

            self.rate.sleep()


    def shutdown(self):
        """Unregisters publishers and subscribers and shutdowns timers"""
        try:
            self.state_sub.unregister()
            self.measurement_pub.unregister()
        except AttributeError:
            pass

def main():
    try:
        sensor = SensorNode()
    except KeyboardInterrupt, rospy.RosInterruptException:
        rospy.loginfo("Received Keyboard Interrupt (^C).")

    try:
        sensor.transmit()
    except KeyboardInterrupt, rospy.RosInterruptException:
        rospy.loginfo("Received Keyboard Interrupt (^C).")


if __name__ == '__main__':
    main()