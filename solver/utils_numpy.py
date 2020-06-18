import numpy as np


"""
Generate mm-spaces an their euclidean metric
"""


def euclid_dist(x, y):
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)

