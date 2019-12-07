#!/usr/bin/env python


# Import ROS pkgs
import rospy

# Import aux pkgs
import sys; sys.dont_write_bytecode = True


# Import my pkgs
from cluster_modules import *
from constraint_path import *

def make_connnected_path():

    # set point objective
    c_state = np.array([[0, 0, 0, 0, 0, 0, 50, 70, np.pi/3]], dtype=np.float64).T
    r_setpnt = three_robot_centroid_inverse_kin_v3(c_state)
    r_start = np.copy(r_setpnt)

    # connected_edges = -1

    # Sensor gain
    k = np.array([[600,600,600]]).T
    # R, d = get_connectivity_radius(k)


    # constrain to connect graph function
    # Objective function
    objective = Objective(function='norm_dist_diff', jac=True)

    # # Constraints
    constraint_12 = Constraint(edge=12, function='signal_intensity', jac=True)
    constraint_13 = Constraint(edge=13, function='signal_intensity', jac=True)
    constraint_23 = Constraint(edge=23, function='signal_intensity', jac=True)


    constraint_12.bound = 1
    constraint_13.bound = 1
    constraint_23.bound = 1

    objective.sensor_gain = k

    constraint_12.sensor_gain = k
    constraint_13.sensor_gain = k
    constraint_23.sensor_gain = k

    constraints = ([constraint_12.dict,constraint_13.dict],\
                    [constraint_12.dict,constraint_23.dict],\
                    [constraint_13.dict,constraint_23.dict])

    # constraints = None

    solution = constraint_path(r_setpnt, r_start, objective, constraints)
    r_sol = np.reshape(solution.x,(6,1))
    r_setpnt_connected = update_robot_state(r_setpnt, r_sol)
    print(r_setpnt_connected)

    goalpoint = unpack_robot_state(r_setpnt)


    lapl_setpnt = Laplacian()
    lapl_connct = Laplacian()

    lapl_setpnt.set_weigths(k, goalpoint)
    # print(lapl_setpnt.eigs())
    lapl_connct.set_weigths(k, r_sol)
    # print(lapl_connct.eigs())


def shutdown_cllbck():
    # do when shutting down
    print('\nShutting down!')


# Main 
if __name__ == '__main__':

    rospy.init_node('make_connected_path')
    rate = rospy.Rate(1) # 1hz

    while not rospy.is_shutdown():
        make_connnected_path()
        rate.sleep()
    
    rospy.on_shutdown(shutdown_cllbck)

    """ Spin """ 
    # rospy.spin()    