#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 6 12:18:56 2020
@author: fran
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation

import csv

GPS_CIRCLE_RADIUS = 2.5

class XYGraph(object):
    """
    This class generate an animation of an XYGraph
    for multiple vehicles simulation
    """
    def __init__(self, arg):
        self.arg = arg
        self.directory = '../../../{}/{}'.format('sim', self.arg.directory)

        fields = [
            'id',\
            'stamp','x','y','yaw',\
            'nav_stamp','nav_x','nav_y','nav_yaw',\
            'nav_x_cov','nav_xy_cov','nav_y_cov'
        ]
        self.files = dict()
        for id in self.arg.agents:
            self.files[id] = {}
            self.files[id]['path'] = '{}/{}.csv'.format(self.directory, id)
            self.files[id]['fields'] = fields
        if self.arg.gps:
            self.files['gps'] = {}
            self.files['gps']['path'] = '{}/{}.csv'.format(self.directory, 'gps')            
            self.files['gps']['fields'] = ['id', 'gps_stamp', 'gps_x', 'gps_y']

        self.data = dict()
        self.agent = dict()
        
        """Set video parameters"""
        self.video_name = '{}/animation.mp4'.format(self.directory) 
        self.frame_step = 1#len(self.arg.agents) + 1  # plot only frames at defined step
        self.time = 0.
        FIG_LIMITS = 100
        self.fig = plt.figure(figsize=(8,8))
        self.fig.suptitle('UV-Network', fontsize=15)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-FIG_LIMITS,FIG_LIMITS)
        self.ax.set_ylim(-FIG_LIMITS,FIG_LIMITS)
        self.ax.set_xlabel('$X[m]$')
        self.ax.set_ylabel('$Y[m]$')        
        self.ax.minorticks_on()
        self.ax.grid(True)
        self.time_text = self.ax.text(-0.8*FIG_LIMITS, 0.8*FIG_LIMITS, '')
        self.colors = ['k','b','g','r','c','m','y']

    def read(self):
        for key, file in self.files.items():
            filepath = file['path']
            with open(filepath) as csvfile:
                    reader = csv.DictReader(csvfile, fieldnames=file['fields'])
                    next(reader)
                    for row in reader:
                        #   filter unwanted data
                        id = int(row[file['fields'][0]])
                        if id in self.arg.agents:
                            stamp = float(row[file['fields'][1]])
                            self.data[key, stamp] = {}
                            self.data[key, stamp]['id'] = id
                            for field in file['fields'][2:]:
                                self.data[key, stamp][field] = float(row[field])

        #   Sort data by time
        self.sorted_keys = sorted(self.data.keys(), key=lambda x: x[1])
        self.sorted_frames = self.sorted_keys[::self.frame_step]
        
        _, stamps = list(zip(*self.sorted_frames))
        self.interval = np.mean(np.diff(stamps))   # in seconds
        self.fps = self.interval**-1
        print('frames per second: {}'.format(self.fps))
    
    def collection(self):
        collections = [patch for agent in self.agent.values() for patch in agent.values()]
        collections += [self.time_text]
        if self.arg.gps: collections += [self.gps['circle']]
        return collections     

    def draw_circle(self, id, stamp):
        """ This functions draws a circle when gps data arrives """
        x = self.data['gps', stamp]['gps_x']
        y = self.data['gps', stamp]['gps_y']
        self.gps['circle'].center = (x,y)

    def draw_agent(self, id, stamp):
        """ This functions draws an ellipse for the uncertainty
        on the estimate of the filter """
        #   Draw exact center point
        if self.arg.exact:
            x = self.data[id, stamp]['x']
            y = self.data[id, stamp]['y']
            self.agent[id]['exact'].set_data(x, y)
        #   Draw estimated center point
        x = self.data[id, stamp]['nav_x']
        y = self.data[id, stamp]['nav_y']
        yaw = self.data[id, stamp]['nav_yaw']
        #   draw quiver
        self.agent[id]['quiver'].set_offsets((x,y)) 
        self.agent[id]['quiver'].set_UVC(np.cos(yaw),np.sin(yaw))
        #   get covariance ellipse
        x_cov = self.data[id, stamp]['nav_x_cov']
        xy_cov = self.data[id, stamp]['nav_xy_cov']
        y_cov = self.data[id, stamp]['nav_y_cov']
        pos_cov = np.array([
            [x_cov, xy_cov],
            [xy_cov, y_cov]
        ])
        #   Compute eigenvalues and associated eigenvectors
        vals, vecs = np.linalg.eigh(pos_cov)
        #   Compute "tilt" of ellipse using first eigenvector
        a, b = vecs[:, 0]
        theta = np.degrees(np.arctan2(b, a))
        #   Eigenvalues give length of ellipse along each eigenvector
        w, h = 2 * np.sqrt(vals) * 3    # mult by 3 for 99.7% of probability
        #   draw ellipse
        self.agent[id]['ellipse'].center = (x,y)
        self.agent[id]['ellipse'].width = w
        self.agent[id]['ellipse'].height = h
        self.agent[id]['ellipse'].angle = theta
    
    def animate(self):
        self.k = 0
        def init():
            self.time = 0.
            for id in self.arg.agents:
                self.agent[id] = {}
                self.agent[id]['quiver'] = self.ax.quiver(0, 0, 0.5, 0.5, angles='xy',\
                    scale=0.15, minshaft=1, scale_units='xy', pivot='mid', color=self.colors[id])
                self.agent[id]['ellipse'] = Ellipse((0,0),1,1,0,\
                    alpha=0.4, color=self.colors[id], fill=True)
                if self.arg.exact:
                    self.agent[id]['exact'], = self.ax.plot([], [], marker='+',\
                        markersize=10, color=self.colors[id])
            if self.arg.gps:
                self.gps = {}
                self.gps['circle'] = plt.Circle((0,0), GPS_CIRCLE_RADIUS,\
                    color=self.colors[0], fill=False)
            for patch in self.collection():
                self.ax.add_artist(patch)
            return self.collection()

        def update(frame):
            key, stamp = frame
            if key=='gps':
                #   Plot a circle over the vehicles when gps data arrives
                id = self.data[key, stamp]['id']
                self.draw_circle(id, stamp)
            else:
                #   Exact, Nav pos and covariance
               id = key
               self.draw_agent(id, stamp)
            #   Time Stamp 
            self.time_text.set_text('t = {:.1f} secs'.format(self.time))
            self.time += self.interval
            return self.collection()

        anim = matplotlib.animation.FuncAnimation(self.fig, update, frames=self.sorted_frames,
                    init_func=init, interval=1000*self.interval, blit=True, repeat=self.arg.repeat, repeat_delay=1000)
        
        if self.arg.save:
            anim.save(self.video_name, fps=self.fps, extra_args=['-vcodec', 'libx264'])
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d','--directory', action='store', help='Name of data files directory.', required=True)
    parser.add_argument('-a','--agents', nargs='+', type=int, help='Agents ids to be plotted.')
    parser.add_argument('-g','--gps', default=False, action='store_true', help='Plot GPS data.')
    parser.add_argument('-e','--exact', default=False, action='store_true', help='Plot exact agent data.')
    parser.add_argument('-r','--repeat', default=False, action='store_true', help='Repeat animation.')
    parser.add_argument('-s','--save', default=False, action='store_true', help='Save animation to mp4 file.')          
    arg = parser.parse_args()

    xy = XYGraph(arg)
    xy.read()
    xy.animate()

if __name__ == '__main__':
    main()