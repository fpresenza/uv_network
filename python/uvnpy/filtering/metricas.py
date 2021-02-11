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
    n = values.shape[-1]
    mean = values.mean(axis=-1)
    dispersion = values - mean[..., np.newaxis]
    var = np.square(dispersion).sum(axis=-1) / n
    return var


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
    n = values.shape[-1]
    mean = values.mean(axis=-1)
    dispersion = values - mean[..., np.newaxis]
    std_dev = np.sqrt(np.square(dispersion).sum(axis=-1) / n)
    return std_dev / mean


def dispersion_index(values, q):
    """Índice de dispersión con exponente q

        index = (variance)^q / mean

        Si q = 0.5 equivale a relative standard deviation
        Si q = 1 equivale a variance to mean ratio.

    Wikipedia:

        In probability theory and statistics, the index of dispersion,
        dispersion index, coefficient of dispersion, relative variance,
        or variance-to-mean ratio (VMR), is a normalized measure of the
        dispersion of a probability distribution: it is a measure used to
        quantify whether a set of observed occurrences are clustered or
        dispersed compared to a standard statistical model.
    """
    n = values.shape[-1]
    mean = values.mean(axis=-1)
    dispersion = values - mean[..., np.newaxis]
    var = np.square(dispersion).sum(axis=-1) / n
    return var**q / mean
