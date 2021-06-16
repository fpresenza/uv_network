#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date vie oct 30 09:37:21 -03 2020
"""
import numpy as np


def mean(samples, axis=-2):
    return samples.mean(axis)


def covariance(samples, axis=-2):
    N = samples.shape[axis]
    e = samples - samples.mean(axis)[..., None, :]
    cov = np.matmul(e.swapaxes(-2, -1), e) / (N - 1)
    return cov
