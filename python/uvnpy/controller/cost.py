#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Fran
"""
import numpy as np

def quad_form(ur, Q):
    d = ur.reshape(-1, 1)
    return np.dot(d.T, np.dot(Q, d))
