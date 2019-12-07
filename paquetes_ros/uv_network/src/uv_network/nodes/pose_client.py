#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:16:45 2019

@author: fran
"""
import rospy
import numpy as np
from uv_network.nodes.ros_node import ROSNode
import uv_network.lib.sensors as sensors
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from uv_network.msg import FloatArrayStamped, PoseAndRange
from uv_network.srv import PoseService
from uv_network.lib.constants import MSG_QUEUE_MAXLEN
from uv_network.lib.tools import Vec3

class PoseClientNode(ROSNode):
    """ Put description here
    """
    def __init__(self):
        ROSNode.__init__(self)

        self.range = sensors.RangeTools(
            rate = rospy.get_param('~rate', 1.),
            SIGMA = rospy.get_param(rospy.search_param('range/sigma_r'), 0.)  # m
        )
        self.rate = rospy.Rate(self.range.rate)
        rospy.loginfo('[%s] Rate: %s Hz', self.nodeID, self.range.rate)
        #   Define servers
        self.topics['subs'].remove(self.num)
        self.servers = {id: {} for id in self.topics['subs']}
        #   Creo el publicador que envia los resultados del filtro
        self.pose_and_range = {
            'topic': self.topics['pubs'][0],
            'msg': PoseAndRange(),
            'pub': rospy.Publisher(
                name=self.topics['pubs'][0],
                data_class=PoseAndRange,
                queue_size=MSG_QUEUE_MAXLEN
            )
        }
        rospy.loginfo('[%s] Publishing to topic: %s', self.nodeID, self.pose_and_range['topic'])
        #   Espero primer msg de motion para saber mi posicion
        motion_msg = rospy.wait_for_message('motion', FloatArrayStamped)
        #   Guardo mi posicion
        self.save_pos(motion_msg)
        #   Creo el suscriptor que recibe los mensajes de motion
        self.motion = {
            'topic': 'motion',
            'sub': rospy.Subscriber(
                name='motion',
                data_class=FloatArrayStamped,
                callback=self.save_pos,
                queue_size=MSG_QUEUE_MAXLEN
            )
        }
        rospy.loginfo('[%s] Subscribing to topic: %s', self.nodeID, self.motion['topic'])
        #   Create one client for each server
        for id, server in self.servers.items():
            try:
                server['name'] = '/uv{}/request_pose'.format(id)
                server['service'] = rospy.ServiceProxy(
                    name=server['name'],
                    service_class=PoseService
                )
                rospy.loginfo('[%s] Service Client %s succesfully called.', self.nodeID, server['name'])            
            except rospy.ServiceException, e:
                rospy.logdebug('[%s] Service Client %s call failed: %s', self.nodeID, server['name'], e)

    def save_pos(self, motion):
        self.my_pos = Vec3(motion.data[6], motion.data[7], 0.)

    def pull(self):
        """ Continuously ask for other's agents pose
        and get range.
        """
        while not rospy.is_shutdown():
            for server in self.servers.values():
                try:
                    resp = server['service'](
                        Header(stamp=rospy.Time.now(),
                               frame_id=self.num)
                    )
                    #   Calculate range based on response
                    #   from other agents
                    server_pos = Vec3(resp.pose.position.x, resp.pose.position.y, 0.)
                    self.pose_and_range['msg'].range = self.range(self.my_pos, server_pos)
                    self.pose_and_range['msg'].pose = resp.nav_pose
                    self.pose_and_range['msg'].header.frame_id = resp.header.frame_id
                    self.pose_and_range['msg'].header.stamp = rospy.Time.now()                
                    #   Send PoseAndRange to ekf_nav node
                    self.pose_and_range['pub'].publish(self.pose_and_range['msg'])
                except rospy.ServiceException, e:
                    rospy.logdebug('[%s] Invalid Pose Request: %s', self.nodeID, e)
            self.rate.sleep()

def main():
    try:
        client = PoseClientNode()
    except KeyboardInterrupt, rospy.RosInterruptException:
        rospy.loginfo("Received Keyboard Interrupt (^C).")
    try:
        client.pull()
    except KeyboardInterrupt, rospy.RosInterruptException:
        rospy.loginfo("Received Keyboard Interrupt (^C).")

if __name__ == '__main__':
    main()
