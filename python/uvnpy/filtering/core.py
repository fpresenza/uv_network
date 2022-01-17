#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date vie oct 30 09:37:21 -03 2020
"""
import numpy as np
from scipy.signal import butter


def mean(samples, axis=-2):
    return samples.mean(axis)


def covariance(samples, axis=-2):
    N = samples.shape[axis]
    e = samples - samples.mean(axis)[..., None, :]
    cov = np.matmul(e.swapaxes(-2, -1), e) / (N - 1)
    return cov


def butter_lpf(cutoff, fs, order=5):
    """Filtro pasabajos Butterworth.

    args:
        cutoff: frecuencia de corte (-3dB)
        fs: frecuencia de sampleo
        order: orden de la aproximacion
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)
