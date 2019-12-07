#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:20:05 2019

@author: fran
"""

import rospy

class ROSNode(object):
    """
    This class defines a generic ROS Node.
    TODO: agregar shutdown routine  
    """
    def __init__(self):
        rospy.init_node('unknown_node', log_level=rospy.INFO)
        self.ns = rospy.get_namespace()[1:-1]   # e.g. 'uv1'
        self.num = str(rospy.get_param(rospy.search_param('num'), '0'))  # e.g. '1'

        try:
            self.nodeID = self.ns + '/' + rospy.get_param(rospy.search_param('node_id'), '')
        except TypeError:
            self.nodeID = 'UNK'
        
        self.topics = rospy.get_param('~topics', {'subs':['msg_in'], 'pubs':['msg_out']})

        self.node_init_time = rospy.Time.now()

# TODO:
# class ROSTopic(object):
#     """
#     This class defines a generic ROS Topic with
#     name, msg type, msg, and subscriber or publisher.
#     """
    # def __init__(self, name, type, subscriber):