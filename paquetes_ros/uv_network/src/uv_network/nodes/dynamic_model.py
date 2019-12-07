#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:46:11 2019
@author: fran
"""
import rospy
import numpy as np
from uv_network.nodes.ros_node import ROSNode
from uv_network.msg import FloatArrayStamped
from uv_network.lib.constants import MSG_QUEUE_MAXLEN
from uv_network.dyn.holonomic import MotionModelVelocity

class DynamicModelNode(ROSNode):
    """
    Este nodo publica en /measurement un FloatArrayStamped
    """
    def __init__(self):
        ROSNode.__init__(self)
        rospy.on_shutdown(self.shutdown)    # TODO: agregar el shutdown en ROSNode 

        #   Defino el modelo dinamico
        self.model = MotionModelVelocity(
            dof = rospy.get_param(rospy.search_param('dyn/dof'), 3),
            ctrl_gain=(1., 1., 0.2),
            alphas = rospy.get_param(rospy.search_param('dyn/alphas'), [[0.,0.,0.],[0.,0.,0.]])
        )
        # Fix rate to send measurement msgs
        self.sensor_rate = rospy.get_param('~rate', 1)   # Hz
        self.rate = rospy.Rate(self.sensor_rate)

        rospy.loginfo('[%s] Rate: %s Hz', self.nodeID, self.sensor_rate)
        
        # initial values
        self.init_acc  = [0.]*self.model.dof
        self.init_vel = rospy.get_param(rospy.search_param(self.ns + '/vel'), [0., 0., 0.])
        self.init_pose = rospy.get_param(rospy.search_param(self.ns + '/pose'), [0., 0., 0.])
        #   Subscriber cmd_vel
        self.cmd_vel = {
            'topic': self.topics['subs'][0],
        }
        #   Publishers
        self.motion = {
            'topic': self.topics['pubs'][0],
            'msg': FloatArrayStamped(),
            'pub': rospy.Publisher(
                name=self.topics['pubs'][0],
                data_class=FloatArrayStamped,
                queue_size=MSG_QUEUE_MAXLEN
            )
        }
        rospy.loginfo('[%s] Publishing to topic: %s', self.nodeID, self.motion['topic'])       

        #   Send motion msg once
        self.motion['msg'].data = self.init_acc + self.init_vel + self.init_pose
        self.motion['msg'].header.stamp = self.node_init_time
        self.motion['msg'].header.frame_id = self.num
        self.motion['pub'].publish(self.motion['msg'])
        rospy.loginfo('[%s] Publishing initial %s', self.nodeID, self.motion['topic'])
        
        #   Guardo el estado inicial como un vector columna
        self.last_cmd_vel = np.zeros((3,1))
        
        #   Creo el suscriptor que recibe los comandos de velocidad
        self.cmd_vel['sub'] = rospy.Subscriber(
            name=self.cmd_vel['topic'],
            data_class=FloatArrayStamped,
            callback=self.update_cmd_vel,
            queue_size=MSG_QUEUE_MAXLEN
        )
        rospy.loginfo('[%s] Subscribing to topic: %s', self.nodeID, self.cmd_vel['topic'])

        self.X = np.array([self.init_vel + self.init_pose + [0.,0.,0.]], dtype=float).T
        self.time = rospy.Time.now()

        rospy.loginfo('[%s] Node initialized.', self.nodeID)
        rospy.logdebug('[%s] Node debugging.', self.nodeID)

    def update_cmd_vel(self, cmd_vel):
        """
        Update measurement
        """
        self.last_cmd_vel = np.array([cmd_vel.data], dtype=float).T

    def run_model(self):
        """
        Get and send motion msg at a fixed rate
        """
        while not rospy.is_shutdown():
            t = rospy.Time.now()
            dt = t.to_sec() - self.time.to_sec()
            self.time = t
            self.X, acc = self.model.sample(self.X, self.last_cmd_vel, dt)
            motion = np.block([
                [acc],
                [self.X[:6]]
            ])
            # Actualizo data
            self.motion['msg'].data = motion.flatten()          
            # Time stamp
            self.motion['msg'].header.stamp = t            
            # Envio msg
            self.motion['pub'].publish(self.motion['msg'])            
            rospy.logdebug("[%s] New dynamic state published.", self.nodeID)

            self.rate.sleep()

    def shutdown(self):
        """Unregisters publishers and subscribers and shutdowns timers"""
        try:
            self.cmd_vel_sub.unregister()
            self.motion['pub'].unregister()            
        except AttributeError:
            pass


def main():
    try:
        dyn = DynamicModelNode()
    except KeyboardInterrupt, rospy.RosInterruptException:
        rospy.loginfo("Received Keyboard Interrupt (^C).")

    try:
        dyn.run_model()
    except KeyboardInterrupt, rospy.RosInterruptException:
        rospy.loginfo("Received Keyboard Interrupt (^C).")


if __name__ == '__main__':
    main()