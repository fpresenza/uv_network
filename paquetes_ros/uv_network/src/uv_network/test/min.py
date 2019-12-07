#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.optimize import minimize
import numpy as np



def func(pos,p,r):
     
    d1 = np.sqrt((pos[0]-p['1'][0])**2 + (pos[1]-p['1'][1])**2) -r['1']
    d2 = np.sqrt((pos[0]-p['2'][0])**2 + (pos[1]-p['2'][1])**2) -r['2']
    d3 = np.sqrt((pos[0]-p['3'][0])**2 + (pos[1]-p['3'][1])**2) -r['3']
    
    return d1**2 + d2**2 + d3**2


def main():
    
    p = {'1':(20,10), '2':(0,10), '3':(10,20)}
    r = {'1':11, '2':17, '3':10}
    # p1 = (20,10)
    # p2 = (0,10)
    # p3 = (10,20)

    # r1 = 11
    # r2 = 14
    # r3 = 10

    sol = minimize(func, np.array([0,0]), method='SLSQP', args=(p,r))
    print(sol)


if __name__ == '__main__':
    main()
