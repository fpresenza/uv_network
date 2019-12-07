#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:35:28 2019
@author: fran
"""
import rospy
from uv_network.nodes.ros_node import ROSNode
from uv_network.lib.sensors import GPSTools
from std_msgs.msg import Header
from uv_network.msg import FloatArrayStamped
from gps_common.msg import GPSFix
import os
from datetime import datetime
import csv
from uv_network.lib.constants import MSG_QUEUE_MAXLEN

class GPSNode(ROSNode):
    """ Este nodo publica en /gps_raw_data un GPSFix msg
    """
    def __init__(self):
        ROSNode.__init__(self)
        rospy.on_shutdown(self.shutdown)    # TODO: agregar el shutdown en ROSNode 
        
        self.gps = GPSTools(
            rate = rospy.get_param('~rate', 1.),
            VEL_SIGMA = rospy.get_param(rospy.search_param('gps/sigma_v'), 0.),  # m/s
            POS_SIGMA = rospy.get_param(rospy.search_param('gps/sigma_p'), 0.)  # m
        )
        #   Fix rate to send gps_raw_data msgs
        self.rate = rospy.Rate(self.gps.rate)
        rospy.loginfo('[%s] Rate: %s Hz', self.nodeID, self.gps.rate)
        #   File to save data
        self.dir = rospy.get_param('~directory', "/home/user/")
        self.set_file_parameters()
        #   Topic names
        self.motion = {}
        self.gps_raw_data = {}
        self.weights = {}
        
        for id, weight in self.topics['subs'].items():
            ns = '/uv{}/'.format(id)
            #   Subscriber topics
            self.motion[id] = {
                'topic': '{}{}'.format(ns, 'motion')
            }
            #   Publisher topics
            self.gps_raw_data[id] = {
                'topic': '{}{}'.format(ns, self.topics['pubs'][0]),
                'msg': GPSFix(),
            }
            #   Save probability of sending a gps msg to each vechicle
            self.weights[id] = weight

        #   Get first msgs
        for id in self.motion.keys():
            rospy.loginfo("[%s] Waiting for %s msg to initialize...", self.nodeID, self.motion[id]['topic'])
            motion = rospy.wait_for_message(self.motion[id]['topic'], FloatArrayStamped)
            # self.gps_raw_data[id]['msg'].header.frame_id = motion.header.frame_id
            self.measurement(motion)
        
        #   Creo los publicadores y suscriptores
        for id in self.gps_raw_data.keys():
            #   GPS Publishers
            self.gps_raw_data[id]['pub'] = rospy.Publisher(
                    name=self.gps_raw_data[id]['topic'],                    
                    data_class=GPSFix,
                    queue_size=MSG_QUEUE_MAXLEN
                )
            rospy.loginfo('[%s] Publishing to topic: %s', self.nodeID, self.gps_raw_data[id]['topic'])
            
        for id in self.motion.keys():
            self.motion[id]['sub'] = rospy.Subscriber(
                name=self.motion[id]['topic'],
                data_class=FloatArrayStamped,
                callback=self.measurement,
                queue_size=MSG_QUEUE_MAXLEN
            )
            rospy.loginfo('[%s] Subscribing to topic: %s', self.nodeID, self.motion[id]['topic'])
        
        rospy.loginfo('[%s] Node initialized.', self.nodeID)
        rospy.logdebug('[%s] Node debugging.', self.nodeID)

    def set_file_parameters(self):
        """ Add date and name to data file
        """
        FORMAT = '%Y%m%d_%H:%M:%S'
        self.f_datetime = '%s' % (datetime.now().strftime(FORMAT))
        self.f_dir = '%s%s/' % (self.dir, self.f_datetime)
        os.mkdir(self.f_dir)
        self.f_name = '%s%s' % ('gps', '.csv')
        self.f_path = '%s%s' % (self.f_dir, self.f_name)

        self.fields = rospy.get_param('~fieldnames', ['id'])

        with open(self.f_path, 'a+') as csvfile:
            self.writer = csv.DictWriter(csvfile, fieldnames=self.fields)
            self.writer.writeheader()

    def save_data(self, id):
        """ Save gps msgs sent to agents.
        """
        with open(self.f_path, 'a+') as csvfile:
            self.writer = csv.DictWriter(csvfile, fieldnames=self.fields)            
            self.writer.writerow({
                'id': id,
                'gps_stamp': self.gps_raw_data[id]['msg'].header.stamp.to_sec(),
                'gps_x': self.gps_raw_data[id]['msg'].longitude,
                'gps_y': self.gps_raw_data[id]['msg'].latitude,
            })

    def measurement(self, motion):
        """ Simulate a measurement from gps based on
        motion of agent and gps error model.
        """
        id = motion.header.frame_id
        self.gps(motion.data)
        self.gps_raw_data[id]['msg'] = GPSFix(header = Header(frame_id=id),
                                              speed = self.gps.vel.meas.x,
                                              track = self.gps.vel.meas.y,
                                              climb = self.gps.vel.meas.z,
                                              longitude = self.gps.pos.meas.x,
                                              latitude = self.gps.pos.meas.y,
                                              altitude = self.gps.pos.meas.z)

    def enable(self):
        """ Send gps msgs at a fixed rate
        """
        #   counter
        k = 0
        while not rospy.is_shutdown():
            #   Determines which uv to send gps data
            uv_list, weights = zip(*self.weights.items())
            id = self.gps.sequence_generator(uv_list)            
            #   Time stamp
            self.gps_raw_data[id]['msg'].header.stamp = rospy.Time.now()            
            #   Envio msg
            self.gps_raw_data[id]['pub'].publish(self.gps_raw_data[id]['msg'])            
            rospy.loginfo("[%s] New gps_raw_data published to uv %s.", self.nodeID, id)
            rospy.logdebug("[%s]:\n%s", self.nodeID, self.gps_raw_data[id]['msg'])
            #   save data to file
            self.save_data(id)
            #   increment counter
            k += 1
            self.rate.sleep()

    def shutdown(self):
        """Unregisters publishers and subscribers and shutdowns timers"""
        try:
            self.vel_sub.unregister()
            self.gps_raw_data_pub.unregister()
        except AttributeError:
            pass

def main():
    try:
        gps = GPSNode()
    except KeyboardInterrupt, rospy.RosInterruptException:
        rospy.loginfo("Received Keyboard Interrupt (^C).")
    try:
        gps.enable()
    except KeyboardInterrupt, rospy.RosInterruptException:
        rospy.loginfo("Received Keyboard Interrupt (^C).")

if __name__ == '__main__':
    main()