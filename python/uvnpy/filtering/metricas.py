#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Wed Jun 34 13:28:15 2020
@author: fran
"""
import numpy as np
from numpy import trace  # noqa
from numpy.linalg import norm, det, eigvalsh  # noqa
from scipy.linalg import svdvals, sqrtm  # noqa

norma = norm


def traza2(M):
    return M[0, 0] + M[1, 1]


def traza_inv2(M):
    a, b, c, d = M.flat
    return (a + d)/(a*d - b*c)


def traza3(M):
    return M[0, 0] + M[1, 1] + M[2, 2]


def traza(M):
    return M.diagonal().sum()


def det2(M):
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]


def inv2(M):
    a, b, c, d = M.flat
    return np.array([
        [d, -b],
        [-c, a]]) / (a*d - b*c)


def sqrt_diagonal(M):
    return np.sqrt(np.diagonal(M))


def variance(values):
    dispersion = values - values.mean()
    var = dispersion.dot(dispersion) / values.size
    return var


def variance_to_mean_ratio(values):
    """Variance to mean ratio.

        vmr = variance / mean

    Wikipedia:

        In probability theory and statistics, the index of dispersion,
        dispersion index, coefficient of dispersion, relative variance,
        or variance-to-mean ratio (VMR), is a normalized measure of the
        dispersion of a probability distribution: it is a measure used to
        quantify whether a set of observed occurrences are clustered or
        dispersed compared to a standard statistical model.
    """
    dispersion = values - values.mean()
    vmr = dispersion.dot(dispersion) / values.sum()
    return vmr


def relative_standard_deviation(values):
    """Relative Standard Deviation.

        rsd = std. deviation / mean

    Wikipedia:
        In probability theory and statistics, the coefficient of
        variation (CV), also known as relative standard deviation (RSD),
        is a standardized measure of dispersion of a probability distribution
        or frequency distribution. It is often expressed as a percentage,
        and is defined as the ratio of the standard deviation to the mean.
    """
    n = len(values)
    dispersion = values - values.mean()
    rsd = np.sqrt(n * dispersion.dot(dispersion)) / values.sum()
    return rsd
