#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import matplotlib.pyplot as plt

rospy.init_node('clock')

samples = int(1E4)
t = np.zeros(samples)
dt = np.zeros(samples-1)
for i in range(samples):
    now =  rospy.Time.now()
    t[i] = now.to_nsec()
    if i>0:
        dt[i-1] = t[i] - t[i-1]
        print(dt[i-1])

# print(dt)|
plt.figur|e()
plt.subplots_adjust(left=0.15, hspace = 0.3)
plt.suptitle('clock performance')
plt.subplot(211)
plt.plot(dt/1000)
plt.grid(1)
plt.xlabel('samples')
plt.ylabel('[$\mu sec$]')
plt.subplot(212)
plt.hist([val/1000 for val in dt if val<14000], bins=40, histtype='bar')
plt.grid(1)
plt.xlabel('[$\mu sec$]')
plt.ylabel('samples')
plt.savefig('/home/fran/repo/uv_network/simulaciones/clock/amdA9_7g.png')