#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:35:28 2019
@author: fran
"""
import rospy
from uv_network.nodes.ros_node import ROSNode
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Pose
from uv_network.msg import FloatArrayStamped, NavFilter
from uv_network.srv import PoseService
from datetime import datetime
from uv_network.lib.constants import MSG_QUEUE_MAXLEN

class PoseServiceNode(ROSNode):
    """ Put description here
    """
    def __init__(self):
        ROSNode.__init__(self)
        rospy.on_shutdown(self.shutdown)    # TODO: agregar el shutdown en ROSNode 

        #   Guardo mi posicion
        self.my_pose = Pose()
        #   Espero a Recibir msgs de motion y navegacion
        self.motion = {
            'topic': self.topics['subs'][0]
        }    
        self.nav = {
            'topic': self.topics['subs'][1]   
        }
        rospy.loginfo("[%s] Waiting for /%s msg to initialize...", self.nodeID, self.motion['topic'])
        motion = rospy.wait_for_message(self.motion['topic'], FloatArrayStamped)
        self.save_pos(motion)
        rospy.loginfo("[%s] Waiting for /%s msg to initialize...", self.nodeID, self.nav['topic'])
        self.nav['msg'] = rospy.wait_for_message(self.nav['topic'], NavFilter)
        #   Create Server
        try:
            service_name = self.topics['pubs'][0]
            self.server = rospy.Service(
                name=service_name,
                service_class=PoseService,
                handler=self.reply
            )
            rospy.loginfo('[%s] Service Server %s succesfully called.', self.nodeID, service_name)            
        except rospy.ServiceException, e:
            rospy.loginfo('[%s] Service Server %s call failed: %s', self.nodeID, service_name, e)

        #   Creo el suscriptor que recibe los mensajes de motion
        self.motion['sub'] = rospy.Subscriber(
            name=self.motion['topic'],
            data_class=FloatArrayStamped,
            callback=self.save_pos,
            queue_size=MSG_QUEUE_MAXLEN
        )
        rospy.loginfo('[%s] Subscribing to topic: %s', self.nodeID, self.motion['topic'])

        #   Creo el suscriptor que recibe los mensajes del ekf
        self.nav['sub'] = rospy.Subscriber(
            name=self.nav['topic'],
            data_class=NavFilter,
            callback=self.update_nav,
            queue_size=MSG_QUEUE_MAXLEN
        )
        rospy.loginfo('[%s] Subscribing to topic: %s', self.nodeID, self.nav['topic'])

       
    def reply(self, req):
        resp = {
            'header': Header(stamp=rospy.Time.now(),
                             frame_id=self.num),
            'pose': self.my_pose,
            'nav_pose': self.nav['msg'].pose[0]
        }
        return resp

    def save_pos(self, motion):
        self.my_pose.position = Point(*(motion.data[6], motion.data[7], 0))

    def update_nav(self, nav):
        self.nav['msg'] = nav

    def shutdown(self):
        """Unregisters publishers and subscribers and shutdowns timers"""
        try:
            self.server.shutdown('[%s] Service Shutting Down: %s', self.nodeID)
        except AttributeError:
            pass

def main():
    try:
        PoseServiceNode()
    except KeyboardInterrupt, rospy.RosInterruptException:
        rospy.loginfo("Received Keyboard Interrupt (^C).")
    try:
        rospy.spin()
    except KeyboardInterrupt, rospy.RosInterruptException:
        rospy.loginfo("Received Keyboard Interrupt (^C).")

if __name__ == '__main__':
    main()