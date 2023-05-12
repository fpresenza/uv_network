import argparse
import numpy as np
from uvnpy.experimental.parser import interpolate
from uvnpy.graphix.planar import TimePlot
from uvnpy.graphix.spatial import Animation3D
import uvnpy.toolkit.linalg as la

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-s', '--step',
        dest='h', default=20e-3, type=float, help='paso de simulación')
    parser.add_argument(
        '-t', '--ti',
        metavar='T0', default=0.0, type=float, help='tiempo inicial')
    parser.add_argument(
        '-e', '--tf',
        default=1000., type=float, help='tiempo final')
    parser.add_argument(
        '-g', '--save',
        default=False, action='store_true',
        help='flag para guardar los videos')
    parser.add_argument(
        '-u', '--file1',
        default='r2dof4_vehicle1_position_local.csv',
        type=str, help='nombre del archivo uav1')
    parser.add_argument(
        '-v', '--file2',
        default='r2dof4_vehicle2_position_local.csv',
        type=str, help='nombre del archivo uav2')
    parser.add_argument(
        '-r', '--file3',
        default='r2dof4_target_position_local.csv',
        type=str, help='nombre del archivo target')
    parser.add_argument(
        '-d', '--dir',
        default='./', type=str, help='nombre del directorio')
    parser.add_argument(
        '-a', '--animate',
        default=False, action='store_true',
        help='flag para generar animaicion 3D')

    # initial offset
    uav_1_p0 = (0, 5, 0)
    uav_2_p0 = (0, -5, 0)
    rover_p0 = (10, 0, 0)
    # parseo de argumentos
    arg = parser.parse_args()
    # leer data de los archivos de texto
    data_1 = np.loadtxt(
        arg.dir + arg.file1,
        delimiter=',', usecols=(0, 4, 5, 6, 7, 8, 9, 10), skiprows=1)
    data_2 = np.loadtxt(
        arg.dir + arg.file2,
        delimiter=',', usecols=(0, 4, 5, 6, 7, 8, 9, 10), skiprows=1)
    data_r = np.loadtxt(
        arg.dir + arg.file3,
        delimiter=',', usecols=(0, 4, 5, 6), skiprows=1)
    # convert to seconds and set init=0
    t0 = min(data_1[0, 0], data_2[0, 0])
    to_secs = 1e-9
    data_1[:, 0] = (data_1[:, 0]-t0)*to_secs
    data_2[:, 0] = (data_2[:, 0]-t0)*to_secs
    data_r[:, 0] = (data_r[:, 0]-t0)*to_secs
    # refer to global system
    data_1[:, 1:4] = uav_1_p0 + data_1[:, 1:4]
    data_2[:, 1:4] = uav_2_p0 + data_2[:, 1:4]
    data_r[:, 1:4] = rover_p0 + data_r[:, 1:4]
    # interpolate data
    uav1 = interpolate.from_data(data_1)
    uav2 = interpolate.from_data(data_2)
    rover = interpolate.from_data(data_r)
    # make time array
    ti = max(data_1[0, 0], data_2[0, 0], arg.ti)
    tf = min(data_1[-1, 0], data_2[-1, 0], arg.tf)
    time = np.arange(ti, tf, arg.h)
    # pack all data
    P, A, G = ([], []), ([], []), ([], [])
    R = []
    for t in time:
        x, y, z, qx, qy, qz, qw = uav1(t)
        P[0].append(np.array([x, y, z]))
        A[0].append(la.quaternion.to_euler(np.quaternion(qw, qx, qy, qz)))
        G[0].append(np.array([0., np.radians(20), 0.]))
        x, y, z, qx, qy, qz, qw = uav2(t)
        P[1].append(np.array([x, y, z]))
        A[1].append(la.quaternion.to_euler(np.quaternion(qw, qx, qy, qz)))
        G[1].append(np.array([0., np.radians(20), 0.]))
        try:
            R.append(rover(t))
        except ValueError:
            if t < rover.x[0]:
                R.append(np.array(rover_p0))
            elif t > rover.x[-1]:
                R.append(np.array(rover(rover.x[-1])))

    px, py, pz = np.vstack(P[0]).T
    ar, ap, ay = np.vstack(A[0]).T
    lines = [[(px, py, pz)], [(ar, ap, ay)]]
    color = [[('b', 'r', 'g')], [('b', 'r', 'g')]]
    label = [[('$x$', '$y$', '$z$')], [(r'$\phi$', r'$\Theta$', r'$\psi$')]]
    xp1 = TimePlot(
        time, lines,
        title='Multicopter 1 - state', color=color, label=label)

    px, py, pz = np.vstack(P[1]).T
    ar, ap, ay = np.vstack(A[1]).T
    lines = [[(px, py, pz)], [(ar, ap, ay)]]
    color = [[('b', 'r', 'g')], [('b', 'r', 'g')]]
    label = [[('$x$', '$y$', '$z$')], [(r'$\phi$', r'$\Theta$', r'$\psi$')]]
    xp2 = TimePlot(
        time, lines,
        title='Multicopter 2 - state', color=color, label=label)

    if arg.save:
        xp1.savefig('uav_1_s_robot')
        xp2.savefig('uav_2_s_robot')
    else:
        xp1.show()
        xp2.show()

    # Animation
    if arg.animate:
        ani = Animation3D(
            time, xlim=(-15, 15), ylim=(-15, 15), zlim=(0, 10),
            save=arg.save, slice=3)
        ani.add_quadrotor(
            (P[0], A[0]), (P[1], A[1]), camera=True, attached=True, gimbal=G)
        ani.add_sphere(R)
        ani.run()
