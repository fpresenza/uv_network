#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue 02 dic 2021 14:27:42 -03
"""
import rospy
from geometry_msgs.msg import VectorStamped


# from uvnpy.linear_models import integrator


class SingleIntegratorNode(object):
    """
    Este nodo lee un msg Vector3Stamped, integra la posicion y
    publica otro Vector3Stamped
    """
    def __init__(self):
        # Identificar vehiculo
        self.namespace = rospy.get_namespace()
        self.vehicle_id = rospy.get_param('~vehicle_id')

        # Topicos
        self.position_topic = 'position'
        # initial_position = rospy.get_param('~')
        self.position_msg = VectorStamped(x=0., y=0., z=0.)
        self.position_pub = rospy.Publisher(
            name=self.position_msg,
            data_class=VectorStamped,
            queue_size=1)
        rospy.loginfo(
            '[SIN-%s] Publishing to topic: %s',
            self.vehicle_id, self.position_topic)

        # self.integrator = integrator()

        # Envia por msg el estado inicial
        # init_signal_out_msg = FloatArrayStamped()
        # init_signal_out_msg.data = self.init_signal_out
        # init_signal_out_msg.header.stamp = rospy.Time.now()
        # rospy.logdebug(
        #     '[%s] Publishing initial /%s: %s',
        #     self.nodeID, self.signal_out_topic, init_signal_out_msg)
        # self.position_pub.publish(init_signal_out_msg)

        # rospy.loginfo(
        #     '[%s] Waiting for /%s msg to initialize...',
        #     self.nodeID, self.signal_in_topic)
        # signal_in_msg = rospy.wait_for_message(
        #     self.signal_in_topic, FloatArrayStamped)
        # rospy.loginfo(
        #     '[%s] First /%s msg received.',
        #     self.nodeID, self.signal_in_topic)
        # rospy.logdebug(
        #     '[%s] /%s received = \n%s',
        #     self.nodeID, self.signal_in_topic, signal_in_msg)

        # self.init_time =  init_signal_out_msg.header.stamp.to_sec()
        # self.previous_time = self.init_time

        # self.signal_in_sub = rospy.Subscriber(
        #     name=self.signal_in_topic,
        #     data_class=FloatArrayStamped,
        #     callback=self.update,
        #     queue_size=MSG_QUEUE_MAXLEN
        # )
        # rospy.loginfo(
        #     '[%s] Subscribing to topic: %s',
        #     self.nodeID, self.signal_in_topic)

        # self.model = Holonomic(
        #     dof=self.dof,
        #     init_time=self.init_time,
        #     init_state=self.init_signal_out
        # )

        # rospy.loginfo('[%s] Node initialized.', self.nodeID)
        # rospy.logdebug('[%s] Node debugging.', self.nodeID)

    def update(self, signal_in):
        """
        Update system signal_out and Publish
        """
        rospy.logdebug(
            '%s] Current /%s = \n%s',
            self.nodeID, self.signal_out_topic, self.model.X)
        # dt = signal_in.header.stamp.to_sec() - self.previous_time
        curr_time = signal_in.header.stamp.to_sec()
        # rospy.logdebug('[%s] Time between samples = %s', self.nodeID, dt)
        # self.signal_out.data = self.model.first_order_integrator(signal_in.data, dt)  # noqa
        self.signal_out.data = self.model.first_order_integrator(
            signal_in.data, curr_time)
        self.signal_out.header.stamp = rospy.Time.now()
        self.position_pub.publish(self.signal_out)
        rospy.logdebug(
            '[%s] New /%s published.',
            self.nodeID, self.signal_out_topic)
        rospy.logdebug(
            '[%s] New %/s = \n%s',
            self.nodeID, self.signal_out_topic, self.signal_out)

        # self.previous_time = signal_in.header.stamp.to_sec()

    def shutdown(self):
        """
        Execute when shutting down
        """
        try:
            self.signal_in_sub.unregister()
            self.position_pub.unregister()
        except AttributeError:
            pass


def main():
    """Entrypoint del nodo"""
    rospy.init_node(
        'single_integrator_node',
        anonymous=True, log_level=rospy.INFO)
    node = SingleIntegratorNode()

    try:
        rospy.spin()
    except KeyboardInterrupt, rospy.ROSInterruptException:
        rospy.loginfo(
            '[%s] Received Keyboard Interrupt (^C).  Shutting down.',
            node.cluster_type)

    node.shutdown()


if __name__ == '__main__':
    main()
