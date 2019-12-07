#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:34:28 2019

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

class SetPointNode(ROSNode):
    """
    Este nodo publica en /reference un FloatArrayStamped
    """
    def __init__(self):
        ROSNode.__init__(self)
        rospy.on_shutdown(self.shutdown)    # TODO: agregar el shutdown en ROSNode  

        # Topic names
        self.user_reference_topic = self.topics['subs'][0]
        self.reference_topic = self.topics['pubs'][0]

        # Creo el mensaje que en algún momento se va a enviar
        self.reference_msg = FloatArrayStamped()
        self.reference_msg.header.frame_id = self.num

        # initial setpoint
        # self.setpoint = rospy.get_param(rospy.search_param('pose/' + self.ns), [0., 0., 0.])
        self.init_key = rospy.get_param('~init_key', '')
        self.setpoint = rospy.get_param(rospy.search_param(self.ns + '/' + self.init_key), [0., 0., 0.])  
        rospy.loginfo('[%s] setpoint: %s', self.nodeID, self.setpoint)

        # Creo el publicador que despacha los mensajes de reference
        self.reference_pub = rospy.Publisher(
            name=self.reference_topic,
            data_class=FloatArrayStamped,
            queue_size=MSG_QUEUE_MAXLEN
            )
        rospy.loginfo('[%s] Publishing to topic: %s', self.nodeID, self.reference_topic)

        # Me suscribo al tópico donde el usuario envia reference
        self.user_reference_sub = rospy.Subscriber(
            name=self.user_reference_topic,
            data_class=FloatArray,
            callback=self.update_ref,
            queue_size=MSG_QUEUE_MAXLEN
            )
        rospy.loginfo('[%s] Subscribing to topic: %s', self.nodeID, self.user_reference_topic)

        rospy.loginfo('[%s] Node initialized.', self.nodeID)
        rospy.logdebug('[%s] Node debugging.', self.nodeID)

        # envio reference una vez
        rospy.sleep(1)
        self.send_msg()


    def update_ref(self, user_reference):
        """
        Actualiza /reference topic por referencia del usuario
        y envia el msg.
        """
        self.setpoint = user_reference.data
        self.send_msg()


    def send_msg(self):
        """
        Envia en un FloatArrayStamped el reference
        """
        # Envio reference
        self.reference_msg.data = self.setpoint
        # Time stamp
        self.reference_msg.header.stamp = rospy.Time.now()
        # Envio msg
        self.reference_pub.publish(self.reference_msg)
        rospy.logdebug("[%s] New reference published:\n%s", self.nodeID, self.reference_msg)


    def shutdown(self):
        """
        Unregisters publishers and subscribers and shutdowns timers
        """
        try:
            self.reference_pub.unregister()
        except AttributeError:
            pass

def main():
    try:
        SPNode = SetPointNode()
    except KeyboardInterrupt, rospy.ROSInterruptException:
        rospy.loginfo("[%s] Received Keyboard Interrupt (^C).", SPNode.nodeID)
    
    try:
        rospy.spin()
    except KeyboardInterrupt, rospy.ROSInterruptException:
        rospy.loginfo("[%s] Received Keyboard Interrupt (^C).", SPNode.nodeID)
        

if __name__ == '__main__':
    main()