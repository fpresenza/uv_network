#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:54:37 2019

@author: fran
"""
#   ROS Python API
import rospy
import message_filters
#   Import numpy to process data
import numpy as np
#   Import custom ROSNode class
from uv_network.nodes.ros_node import ROSNode
#   Import the messages we're interested in sending and receiving
from uv_network.msg import FloatArray, FloatArrayStamped, NavFilter
#   Import bool message for debugging
from std_msgs.msg import Bool
#   Import matplotlib library
import matplotlib.pyplot as plt
#   Import OS API
import os
#   Import systema pckgs
from datetime import datetime
#   Import csv library
import csv
#   Import constants
from uv_network.lib.constants import MSG_QUEUE_MAXLEN


class DataLoggerNode(ROSNode):
    """
    Este nodo (descripcion)
    """
    def __init__(self):
        ROSNode.__init__(self)
        rospy.on_shutdown(self.shutdown)    # TODO: agregar el shutdown en ROSNode  

        #   Tiempo de inicializacion del nodo
        self.init_time = self.node_init_time.to_sec()
        #   File to save data
        self.dir = rospy.get_param('~directory', "/home/user/")
        self.set_file_parameters()
        self.subs = dict()
        #   Me suscribo a donde se publican los datos a loggear
        for topic_name, data_class in self.topics['subs'].items():
            self.subs[topic_name] = message_filters.Subscriber(
                    topic_name,
                    eval(data_class)
                    )
            rospy.loginfo('[%s] Subscribing to topic: %s', self.nodeID, topic_name)
        #   Creo el sincronizador
        self.sync = message_filters.ApproximateTimeSynchronizer(
            self.subs.values(),
            queue_size=MSG_QUEUE_MAXLEN,
            slop=0.05
        )
        self.sync.registerCallback(self.save_data)
        
        rospy.loginfo('[%s] Node initialized.', self.nodeID)
        rospy.logdebug('[%s] Node debugging.', self.nodeID)
            
    def set_file_parameters(self):
        """
        Add date and name to data file
        """
        FORMAT = '%Y%m%d_%H:%M:%S'
        self.f_datetime = '%s' % (datetime.now().strftime(FORMAT))
        self.f_dir = '%s%s/' % (self.dir, self.f_datetime)
        os.mkdir(self.f_dir)
        self.f_name = '%s%s' % (self.num, '.csv')
        self.f_path = '%s%s' % (self.f_dir, self.f_name)

        self.fields = rospy.get_param('~fieldnames', ['id'])

        with open(self.f_path, 'a+') as csvfile:
            self.writer = csv.DictWriter(csvfile, fieldnames=self.fields)
            self.writer.writeheader()

    def save_data(self, motion, nav):
        """
        Save pose and nav estimates to csv file
        """
        with open(self.f_path, 'a+') as csvfile:
            self.writer = csv.DictWriter(csvfile, fieldnames=self.fields)            
            self.writer.writerow({
                'id': motion.header.frame_id,
                'stamp': motion.header.stamp.to_sec(),
                'x': motion.data[6],
                'y': motion.data[7],
                'yaw': motion.data[8],
                'nav_stamp': nav.header.stamp.to_sec(),
                'nav_x': nav.pose[0].pose.position.x,
                'nav_y': nav.pose[0].pose.position.y,
                'nav_yaw': nav.euler.z,
                'nav_x_cov': nav.pose[0].covariance[0],
                'nav_xy_cov': nav.pose[0].covariance[1],
                'nav_y_cov': nav.pose[0].covariance[7]              
            })
            
    def shutdown(self):
        """
        Unregisters publishers and subscribers and shutdowns timers
        """
        try:
            for sub in self.subs:
                sub.unregister()
        except AttributeError:
            pass

def main():
    try:
        data_logger = DataLoggerNode()
    except KeyboardInterrupt, rospy.ROSInterruptException:
        rospy.loginfo("[%s] Received Keyboard Interrupt (^C).", self.nodeID)

    rospy.spin()

if __name__ == '__main__':
    main()