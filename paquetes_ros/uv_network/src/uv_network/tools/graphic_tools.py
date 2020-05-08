#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 09 15:54:29 2020
@author: fran
"""
import numpy as np
from matplotlib.patches import Ellipse

def plot_ellipse(ax, mean, covariance, sigmas=1, color='k'):
    """ Draws ellipse from xy of covariance matrix
    """
    #   Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(covariance)
    #   Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))
    #   Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals)*sigmas
    ellipse = Ellipse(mean, w, h, theta, color=color)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)