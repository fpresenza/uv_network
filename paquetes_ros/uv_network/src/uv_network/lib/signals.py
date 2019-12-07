#!/usr/bin/env python

import numpy as np

def pulse(t, amplitude, init, end):
    """ This function implements a pulse signal. 
    Amplitude, init time and end time are parameters. """
    return amplitude*(np.sign(np.sign(t-init)-np.sign(t-end)))
