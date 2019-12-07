#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:26:09 2019

@author: fran
"""

# ROS Python API
import rospy
# Import numpy to process data
import numpy as np
# Import custom ROSNode class
from uv_network.nodes.ros_node import ROSNode
# Import the messages we're interested in sending and receiving
from uv_network.msg import FloatArray, FloatArrayStamped
#   Import constants
from uv_network.lib.constants import MSG_QUEUE_MAXLEN

class SumNode(ROSNode):
    """
    Este nodo publica obtiene la suma de n FloatArrayStamped
    """
    def __init__(self):
        ROSNode.__init__(self)
        rospy.on_shutdown(self.shutdown)    # TODO: agregar el shutdown en ROSNode  

        # uv params
        self.dof = rospy.get_param(rospy.search_param('dyn/dof'), 3)

        # Espero a recibir msgs
        self.last_values = [[0.]*self.dof for _ in self.topics['subs']]
        # for topic_name in self.topics['subs']:
        #     rospy.loginfo("[%s] Waiting for /%s msg to initialize...", self.nodeID, topic_name)
        #     msg = rospy.wait_for_message(topic_name, FloatArrayStamped)
        #     self.last_values += [msg.data]
            # rospy.loginfo("[%s] First /%s received.", self.nodeID, topic_name)
            
        # Creo el mensaje que en algún momento se va a enviar
        self.sum_msg = FloatArrayStamped()
        self.sum_msg.header.frame_id = self.num

        # Creo el publicador que despacha los mensajes de cmd_vel
        self.sum_topic = self.topics['pubs'][0]
        self.sum_pub = rospy.Publisher(
            name=self.sum_topic,
            data_class=FloatArrayStamped,
            queue_size=MSG_QUEUE_MAXLEN
            )
        rospy.loginfo('[%s] Publishing to topic: %s', self.nodeID, self.sum_topic)

        #   Me suscribo a los tópicos donde se publican los valores a sumar
        self.subs = []
        for k, topic_name in enumerate(self.topics['subs']):
            self.subs += [rospy.Subscriber(
                name=topic_name,
                data_class=FloatArrayStamped,
                callback=self.update_sum,
                callback_args=k,
                queue_size=MSG_QUEUE_MAXLEN
                )]
            rospy.loginfo('[%s] Subscribing to topic: %s', self.nodeID, topic_name)

        rospy.loginfo('[%s] Node initialized.', self.nodeID)
        rospy.logdebug('[%s] Node debugging.', self.nodeID)

    
    def update_sum(self, msg, k):
        """
        Add new values to sum
        """
        #   Convierto a array
        self.last_values[k] = msg.data
        #   Actualizo data
        self.sum_msg.data = map(sum, zip(*self.last_values))

        #   Time stamp
        self.sum_msg.header.stamp = rospy.Time.now()
        #   Envio msg
        self.sum_pub.publish(self.sum_msg)            
        rospy.logdebug("[%s] New sum published", self.nodeID)
        rospy.logdebug("[%s]:\n%s", self.nodeID, self.sum_msg)


    def shutdown(self):
        """
        Unregisters publishers and subscribers and shutdowns timers
        """
        try:
            for sub in self.subs:
                sub.unregister()
            self.sum_pub.unregister()            
        except AttributeError:
            pass


def main():
    try:
        SNode = SumNode()
    except KeyboardInterrupt, rospy.ROSInterruptException:
        rospy.loginfo("[%s] Received Keyboard Interrupt (^C).", SNode.nodeID)
    
    try:
        rospy.spin()
    except KeyboardInterrupt, rospy.ROSInterruptException:
        rospy.loginfo("[%s] Received Keyboard Interrupt (^C).", SNode.nodeID)

if __name__ == '__main__':
    main()