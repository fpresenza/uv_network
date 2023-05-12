import numpy as np
from scipy.interpolate import interp1d


def unique_time_stamp(data):
    keepme = np.diff(data[:, 0]) != 0
    # interp1d no admite tiempos repetidos
    if not np.all(keepme):
        data = data[np.insert(keepme, 0, True), ...]
    return data


class interpolate(object):
    @staticmethod
    def from_data(data):
        data = unique_time_stamp(data)
        return interp1d(
            data[:, 0], data[:, 1:],
            axis=0, kind='cubic')    # kind puede ser 'zero'

    @staticmethod
    def from_file(file, **kwargs):
        data = np.loadtxt(file, delimiter=',', **kwargs)
        return interpolate.from_data(data)
