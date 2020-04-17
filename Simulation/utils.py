"""Utility functions"""

import numpy as np


def getUnitVector(v):
    l2norm = np.sqrt(np.sum(np.square(v)))
    if l2norm > 1e-6:
        return v / l2norm
    else:
        return v


def getL2Norm(v):
    return np.sqrt(np.sum(np.square(v)))


def getNTBmatrixfromN(n):
    N = getUnitVector(n)
    T = np.cross(n, [0, 1, 0])
    B = np.cross(n, T)
    return np.transpose(np.vstack((N, T, B)))


def sample_hemisphere(limit):
    v = np.asarray([0, 0, -1])
    phi_limit = np.pi*limit/180
    phi = np.pi
    while v[2] < 0 or phi > phi_limit:
        v = np.random.standard_normal(3)
        v = v / np.sum(v ** 2) ** 0.5
        phi = np.arctan(np.sqrt(v[0]**2 + v[1]**2)/v[2])

    print ("            phi = ", phi*180/np.pi)
    return v

