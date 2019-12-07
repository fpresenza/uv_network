#!/usr/bin/env python
"""
Created on Wed 6 Dec 2019

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
# Import plant's dynamic model
from uv_network.dyn.holonomic import Holonomic
#   Import constants
from uv_network.lib.constants import MSG_QUEUE_MAXLEN

class FirstOrderIntegratorNode(ROSNode):
    """
    Este nodo lee /msg_in que es integrado y publicado en /msg_out
    """
    def __init__(self):
        ROSNode.__init__(self)
        rospy.on_shutdown(self.shutdown)    # TODO: agregar el shutdown en ROSNode

        # Topic names
        self.signal_in_topic = self.topics['subs'][0]
        self.signal_out_topic = self.topics['pubs'][0]

        # Creo el mensaje que en algun momento se va a enviar        
        self.signal_out = FloatArrayStamped()
        self.signal_out.header.frame_id = self.num

        # uv params
        self.dof_key = rospy.get_param('~dof_key', '')
        self.dof = rospy.get_param(rospy.search_param(self.dof_key + '/dof'), 3)
        # self.dof = rospy.get_param(rospy.search_param('dyn/dof'), 3)        
        
        # initial signal_out
        self.init_key = rospy.get_param('~init_key', '')
        self.init_signal_out = rospy.get_param(rospy.search_param(self.ns + '/' + self.init_key), [0., 0., 0.])
        
        self.signal_out_pub = rospy.Publisher(
            name=self.signal_out_topic,
            data_class=FloatArrayStamped,
            queue_size=MSG_QUEUE_MAXLEN,
            latch=True
        )
        rospy.loginfo('[%s] Publishing to topic: %s', self.nodeID, self.signal_out_topic)
        
        # Envia por msg el estado inicial 
        init_signal_out_msg = FloatArrayStamped()
        init_signal_out_msg.data = self.init_signal_out
        init_signal_out_msg.header.stamp = rospy.Time.now()
        init_signal_out_msg.header.frame_id = self.num        
        rospy.logdebug('[%s] Publishing initial /%s: %s', self.nodeID, self.signal_out_topic, init_signal_out_msg)
        self.signal_out_pub.publish(init_signal_out_msg)

        rospy.loginfo("[%s] Waiting for /%s msg to initialize...", self.nodeID, self.signal_in_topic)
        signal_in_msg = rospy.wait_for_message(self.signal_in_topic, FloatArrayStamped)
        rospy.loginfo("[%s] First /%s msg received.", self.nodeID, self.signal_in_topic)
        rospy.logdebug("[%s] /%s received = \n%s", self.nodeID, self.signal_in_topic, signal_in_msg)
            
        self.init_time =  init_signal_out_msg.header.stamp.to_sec()
        # self.previous_time = self.init_time       

        self.signal_in_sub = rospy.Subscriber(
            name=self.signal_in_topic,
            data_class=FloatArrayStamped,
            callback=self.update,
            queue_size=MSG_QUEUE_MAXLEN
        )
        rospy.loginfo('[%s] Subscribing to topic: %s', self.nodeID, self.signal_in_topic)
        
        self.model = Holonomic(
            dof=self.dof,
            init_time=self.init_time,
            init_state=self.init_signal_out
        )

        rospy.loginfo('[%s] Node initialized.', self.nodeID)
        rospy.logdebug('[%s] Node debugging.', self.nodeID)
        

    def update(self, signal_in):
        """
        Update system signal_out and Publish
        """
        rospy.logdebug("[%s] Current /%s = \n%s", self.nodeID, self.signal_out_topic, self.model.X)
        # dt = signal_in.header.stamp.to_sec() - self.previous_time
        curr_time = signal_in.header.stamp.to_sec()
        # rospy.logdebug("[%s] Time between samples = %s", self.nodeID, dt)        
        # self.signal_out.data = self.model.first_order_integrator(signal_in.data, dt)
        self.signal_out.data = self.model.first_order_integrator(signal_in.data, curr_time)        
        self.signal_out.header.stamp = rospy.Time.now()
        self.signal_out_pub.publish(self.signal_out)
        rospy.logdebug("[%s] New /%s published.", self.nodeID, self.signal_out_topic)
        rospy.logdebug("[%s] New %/s = \n%s", self.nodeID, self.signal_out_topic, self.signal_out)

        # self.previous_time = signal_in.header.stamp.to_sec()


    def shutdown(self):
        """
        Execute when shutting down
        """
        try:
            self.signal_in_sub.unregister()
            self.signal_out_pub.unregister()
        except AttributeError:
            pass


def main():
    try:
        FOINode = FirstOrderIntegratorNode()
    except rospy.ROSInterruptException:
        rospy.loginfo("[%s] Received Keyboard Interrupt (^C).", FOINode.nodeID)

    try:
        rospy.spin()
    except KeyboardInterrupt, rospy.ROSInterruptException:
        rospy.loginfo("[%s] Received Keyboard Interrupt (^C).", FOINode.nodeID)
        

if __name__ == '__main__':   
    main()