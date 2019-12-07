#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 5 16:01:12 2020

@author: fran
"""

# ROS Python API
import rospy
# Import numpy to process data
import numpy as np
# Import ROSNode class for simplicity
from uv_network.nodes.ros_node import ROSNode
#   Import mission Tools
from uv_network.ctrl.mission_tools import MissionTools
# Import the messages we're interested in sending and receiving
from std_msgs.msg import Header
from uv_network.msg import FloatArrayStamped, NavFilter
#   Import constants
from uv_network.lib.constants import MSG_QUEUE_MAXLEN

class MissionNode(ROSNode):
    """
    Agregar intro
    """
    def __init__(self):
        ROSNode.__init__(self)
        rospy.on_shutdown(self.shutdown)    # TODO: agregar el shutdown en ROSNode

        #   ROS parameters
        #   normal random signal
        #   mean
        self.rnd_vel_mean = rospy.get_param('~rnd_vel_mean', [0., 0., 0.])        
        rospy.loginfo('[%s] random vel mean: %s', self.nodeID, self.rnd_vel_mean)
        #   covariance
        static = rospy.get_param('~static', [])
        if self.num in static:
            self.rnd_vel_covar = [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
        else:
            self.rnd_vel_covar = rospy.get_param('~rnd_vel_covar', [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])

        rospy.loginfo('[%s] random vel covariance: %s', self.nodeID, self.rnd_vel_covar)   
        #   motion limits
        self.limits = rospy.get_param('~bound', [100.,100.])   # m
        rospy.loginfo('[%s] XY bounds are %s.', self.nodeID, self.limits)     
        # Suscriptions & Publications
        self.nav = {
            'topic': self.topics['subs'][0]
        }
        self.action = {
            'topic': self.topics['pubs'][0],
            'msg': FloatArrayStamped(header=Header(frame_id=self.num))
        }
        #   Instancia de la clase MissionTools
        self.mission = MissionTools(self.rnd_vel_mean)
        # Creo el publicador que despacha los mensajes del control
        self.action['pub'] = rospy.Publisher(
            name=self.action['topic'],
            data_class=FloatArrayStamped,
            queue_size=MSG_QUEUE_MAXLEN
        )
        rospy.loginfo('[%s] Publishing to topic: %s', self.nodeID, self.action['topic'])

        # Me suscribo al tópico donde se envian los datos de navegacion
        self.nav['sub'] = rospy.Subscriber(
            name=self.nav['topic'],
            data_class=NavFilter,
            callback=self.send_action,
            queue_size=MSG_QUEUE_MAXLEN
        )
        rospy.loginfo('[%s] Subscribing to topic: %s', self.nodeID, self.nav['topic'])

        rospy.loginfo('[%s] Node initialized.', self.nodeID)
        rospy.logdebug('[%s] Node debugging.', self.nodeID)


    def send_action(self, nav):
        """ Compute mission action based on nav
        """
        # TODO: la accion de control deberia enviarse a una tasa fija
        # siempre y cuando se cuente con alguna informacion del estado
        # del vehiculo.
        #   Creo arrays para especificar la pose
        # pose = np.array((
        #     nav.pose[0].pose.position.x,
        #     nav.pose[0].pose.position.y,
        #     nav.euler.z
        #     ),
        #     dtype='float'
        # )
        #   Genero la accion de mision
        #limit_action = self.mission.limit_xy(pose, self.limits, gain=3.)
        random_vel = self.mission.random_vel_generator(self.rnd_vel_mean, self.rnd_vel_covar, wait_seq=60)
        
        self.action['msg'].data = random_vel #+ limit_action
        # Time stamp
        self.action['msg'].header.stamp = rospy.Time.now()
        self.action['pub'].publish(self.action['msg'])

        rospy.logdebug("[%s] New mission action published.", self.nodeID)
        rospy.logdebug("[%s]:\n%s", self.nodeID, self.action['msg'])
                

    def shutdown(self):
        """
        Unregisters publishers and subscribers and shutdowns timers
        """
        try:
            self.nav['sub'].unregister()
            self.action['pub'].unregister()
        except AttributeError:
            pass


def main():
    try:
        MissionNode()
    except KeyboardInterrupt, rospy.RosInterruptException:
        rospy.loginfo("[%s] Received Keyboard Interrupt (^C).", PIDNode.nodeID)

    try:
        rospy.spin()
    except KeyboardInterrupt, rospy.RosInterruptException:    
        rospy.loginfo("[%s] Received Keyboard Interrupt (^C).", PIDNode.nodeID)        
        

if __name__ == '__main__':
    main()
