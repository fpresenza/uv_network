#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue 02 dic 2021 14:27:42 -03
"""
import rospy
from geometry_msgs.msg import Vector3Stamped


class SingleIntegratorNode(object):
    """
    Este nodo lee un msg Vector3Stamped, integra la velocidad
    comandada (cmd_vel) y la publica otro Vector3Stamped (position)
    """
    def __init__(self):
        # Identificar vehiculo
        self.namespace = rospy.get_namespace()
        self.vehicle_id = rospy.get_param(
            rospy.search_param(self.namespace + 'vehicle_id'))

        # Modelo dinamico
        self.rostime = rospy.get_rostime()

        # Suscriber cmd_vel
        self.cmd_vel_msg = Vector3Stamped()
        self.cmd_vel_topic = 'cmd_vel'
        self.cmd_vel_sub = rospy.Subscriber(
            name=self.cmd_vel_topic,
            data_class=Vector3Stamped,
            callback=self.update_cmd_vel,
            queue_size=1)
        rospy.loginfo(
            '[SIN-%s] Subscribing to topic: %s',
            self.vehicle_id, self.cmd_vel_topic)

        # Publisher posicion
        initial_position = rospy.get_param(
            rospy.search_param(self.namespace + 'position'))
        self.position_topic = 'position'
        self.position_msg = Vector3Stamped()
        self.position_msg.vector.x = initial_position[0]
        self.position_msg.vector.y = initial_position[1]
        self.position_msg.vector.z = initial_position[2]
        self.position_pub = rospy.Publisher(
            name=self.position_topic,
            data_class=Vector3Stamped,
            queue_size=1)
        rospy.loginfo(
            '[SIN-%s] Publishing to topic: %s',
            self.vehicle_id, self.position_topic)

        # Fix rate to send position msgs
        self.rate = rospy.get_param('~rate', 1)   # Hz
        self.rate = 1
        self.sleeper = rospy.Rate(self.rate)

        rospy.loginfo('[SIN-%s] Node initialized.', self.vehicle_id)
        rospy.logdebug('[SIN-%s] Node debugging.', self.vehicle_id)

    def update_cmd_vel(self, cmd_vel):
        self.cmd_vel_msg.x = cmd_vel.vector.x
        self.cmd_vel_msg.vector.y = cmd_vel.vector.y
        self.cmd_vel_msg.vector.z = cmd_vel.vector.z

    def spin(self):
        """Send position msg at a fixed rate
        """
        while not rospy.is_shutdown():
            # Actualizo tiempo
            now = rospy.get_rostime()
            elapsed = now.to_sec() - self.rostime.to_sec()
            self.rostime = now

            # Preparo y publico msg
            self.position_msg.vector.x += elapsed * self.cmd_vel_msg.vector.x
            self.position_msg.vector.y += elapsed * self.cmd_vel_msg.vector.y
            self.position_msg.vector.z += elapsed * self.cmd_vel_msg.vector.z
            self.position_msg.header.stamp = now
            self.position_pub.publish(self.position_msg)
            rospy.logdebug(
                '[SIN-%s] Current position published.', self.vehicle_id)

            self.sleeper.sleep()

    def shutdown(self):
        """
        Execute when shutting down
        """
        try:
            self.cmd_vel_sub.unregister()
            self.position_pub.unregister()
            rospy.loginfo('[SIN-%s] Node shut down.', self.vehicle_id)
        except AttributeError:
            pass


def main():
    """Entrypoint del nodo"""
    rospy.init_node(
        'single_integrator_node',
        anonymous=True, log_level=rospy.INFO)
    node = SingleIntegratorNode()

    try:
        node.spin()
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        rospy.loginfo(
            '[SIN-%s] Received Keyboard Interrupt (^C).  Shutting down.',
            node.cluster_type)

    node.shutdown()


if __name__ == '__main__':
    main()
