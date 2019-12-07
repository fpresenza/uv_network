#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 7 17:00:56 2019

@author: fran
"""

# ROS Python API
import rospy
# Import numpy to process data
import numpy as np
# Import ROSNode class for simplicity
from uv_network.nodes.ros_node import ROSNode
# Import the messages we're interested in sending and receiving
from uv_network.msg import FloatArrayStamped, NavFilter
# PID controller
from uv_network.ctrl.pid import PIDController
#   Import constants
from uv_network.lib.constants import MSG_QUEUE_MAXLEN

class PIDWaypointNode(ROSNode):
    """
    Agregar intro
    """
    def __init__(self):
        ROSNode.__init__(self)
        rospy.on_shutdown(self.shutdown)    # TODO: agregar el shutdown en ROSNode

        # topic names
        self.reference_topic = self.topics['subs'][0]
        self.nav_topic = self.topics['subs'][1]
        self.ctrl_eff_topic = self.topics['pubs'][0]        
        
        # Creo el mensaje que en algún momento se va a enviar        
        self.ctrl_eff_msg = FloatArrayStamped()
        self.ctrl_eff_msg.header.frame_id = self.num
        
        # uv params
        self.dof = rospy.get_param(rospy.search_param('dyn/dof'), 3)
        # controller params
        self.kp = rospy.get_param(rospy.search_param('pid/kp'), [1.,1.,1.])
        self.ki = rospy.get_param(rospy.search_param('pid/ki'), [0.,0.,0.])
        self.kd = rospy.get_param(rospy.search_param('pid/kd'), [0.,0.,0.])
        
        # log info
        rospy.loginfo('[%s] PID Waypoints information:', self.nodeID)
        rospy.loginfo('[%s] dof: %s', self.nodeID, self.dof)        
        rospy.loginfo('[%s] kp: %s', self.nodeID, self.kp)
        rospy.loginfo('[%s] ki: %s', self.nodeID, self.ki)
        rospy.loginfo('[%s] kd: %s', self.nodeID, self.kd)

        # wait for reference message
        rospy.loginfo("[%s] Waiting for /%s msg to initialize...", self.nodeID, self.reference_topic)
        first_reference_msg = rospy.wait_for_message(self.reference_topic, FloatArrayStamped)
        rospy.loginfo("[%s] First /%s msg received.", self.nodeID, self.reference_topic)
        # wait for nav message
        rospy.loginfo("[%s] Waiting for /%s msg to initialize...", self.nodeID, self.nav_topic)
        first_nav_msg = rospy.wait_for_message(self.nav_topic, NavFilter)
        rospy.loginfo("[%s] First /%s msg received.", self.nodeID, self.nav_topic)

        # Inicializo tiempo
        self.init_time = first_nav_msg.header.stamp.to_sec()
        self.curr_time = self.init_time  

        # Guardo reference inicial
        self.last_waypoint_available = first_reference_msg.data

        # Construyo el controlador que se va a utilizar
        self.ctrl = PIDController(
            kp=np.array(self.kp),
            ki=np.array(self.ki),
            kd=np.array(self.kd),
            t0=self.init_time
        )   

        # Creo el publicador que despacha los mensajes del control
        self.ctrl_eff_pub = rospy.Publisher(
            name=self.ctrl_eff_topic,
            data_class=FloatArrayStamped,
            queue_size=MSG_QUEUE_MAXLEN
        )
        rospy.loginfo('[%s] Publishing to topic: %s', self.nodeID, self.ctrl_eff_topic)

        # Me suscribo al tópico donde se envian los datos de navegacion
        self.nav_sub = rospy.Subscriber(
            name=self.nav_topic,
            data_class=NavFilter,
            callback=self.update_ctrl_eff,
            queue_size=MSG_QUEUE_MAXLEN
        )
        rospy.loginfo('[%s] Subscribing to topic: %s', self.nodeID, self.nav_topic)

        # Me suscribo al tópico donde se envía la referencia
        self.reference_sub = rospy.Subscriber(
            name=self.reference_topic,
            data_class=FloatArrayStamped,
            callback=self.update_reference,
            queue_size=MSG_QUEUE_MAXLEN
        )
        rospy.loginfo('[%s] Subscribing to topic: %s', self.nodeID, self.reference_topic)        

        rospy.loginfo('[%s] Node initialized.', self.nodeID)
        rospy.logdebug('[%s] Node debugging.', self.nodeID)


    def update_ctrl_eff(self, nav):
        """
        Compute control action based on nav and last reference available
        """
        # Creo arrays para obtener accion de control
        pose_array = np.array((
            nav.pose.pose.position.x,
            nav.pose.pose.position.y,
            nav.euler.z
            ),
            dtype='float'
        )
        ref_array = self.last_waypoint_available

        # Calculo el intervalo de tiempo entre las muestras
        delta_t = nav.header.stamp.to_sec() - self.curr_time
        self.curr_time = nav.header.stamp.to_sec()
        rospy.logdebug('[%s] Time between control samples: %s', self.nodeID, delta_t)

        # Genero la acción de control
        self.ctrl_eff_msg.data = self.ctrl.update(ref_array, pose_array, self.curr_time)

        # Time stamp
        self.ctrl_eff_msg.header.stamp = rospy.Time.now()
        self.ctrl_eff_pub.publish(self.ctrl_eff_msg)

        rospy.logdebug("[%s] New control effort published.", self.nodeID)
        rospy.logdebug("[%s]:\n%s", self.nodeID, self.ctrl_eff_msg)


    def update_reference(self, reference):
        """
        Actualiza el valor de waypoint
        """
        # rospy.loginfo('[%s] Update reference.', self.nodeID)        
        self.last_waypoint_available = reference.data
        

    def shutdown(self):
        """
        Unregisters publishers and subscribers and shutdowns timers
        """
        try:
            self.nav_sub.unregister()
            self.reference_sub.unregister()
            self.ctrl_eff_pub.unregister()
        except AttributeError:
            pass


def main():
    try:
        PIDNode = PIDWaypointNode()
    except KeyboardInterrupt, rospy.RosInterruptException:
        rospy.loginfo("[%s] Received Keyboard Interrupt (^C).", PIDNode.nodeID)

    try:
        rospy.spin()
    except KeyboardInterrupt, rospy.RosInterruptException:    
        rospy.loginfo("[%s] Received Keyboard Interrupt (^C).", PIDNode.nodeID)        
        

if __name__ == '__main__':
    main()
