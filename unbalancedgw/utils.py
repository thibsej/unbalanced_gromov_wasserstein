import numpy as np
import torch

"""
Generate mm-spaces an their euclidean metric
"""


def euclid_dist(x, y):
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)


def dist_matrix(x_i, y_j, p=2):
    if p == 1:
        return (x_i[:, :, None, :] - y_j[:, None, :, :]).norm(p=2, dim=3)
    elif p == 2:
        return (x_i[:, :, None, :] - y_j[:, None, :, :]).norm(p=2, dim=3) ** 2
    else:
        c_e = (x_i[:, :, None, :] - y_j[:, None, :, :]).norm(p=2, dim=3)
        return c_e ** p


def generate_measure(n_batch, n_sample, n_dim, equal=False):
    """
    Generate a batch of probability measures in R^d sampled over the unit
    square.
    :param n_batch: Number of batches
    :param n_sample: Number of sampling points in R^d
    :param n_dim: Dimension of the feature space
    :param equal: Weights equal to 1
    :return: A (Nbatch, Nsample, Ndim) torch.Tensor
    """
    m = torch.distributions.exponential.Exponential(1.0)
    a = m.sample(torch.Size([n_batch, n_sample]))
    a = a / a.sum(dim=1)[:, None]
    m = torch.distributions.uniform.Uniform(0.0, 1.0)
    x = m.sample(torch.Size([n_batch, n_sample, n_dim]))
    Cx = dist_matrix(x, x, 2)
    if equal:
        a = torch.ones_like(a)
    return a, Cx, x
