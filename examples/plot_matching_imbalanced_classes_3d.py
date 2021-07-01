"""
======================================
Comparing imbalanced pointclouds in 3D
======================================

This example shows how UGW matches two 3D pointclouds with imbalanced weights.

"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import torch
from sklearn.cluster import KMeans

from solver.utils import euclid_dist
from ot.gromov import gromov_wasserstein
from unbalancedgw.vanilla_ugw_solver import log_ugw_sinkhorn

path = os.getcwd() + "/output"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/plots"
if not os.path.isdir(path):
    os.mkdir(path)


def generate_data(nsample, ratio):
    # Generate first ellipse
    s = np.random.uniform(size=(nsample, 3))
    x1 = np.zeros_like(s)
    x1[:, 0] = (
        np.sqrt(s[:, 0])
        * np.cos(2 * np.pi * s[:, 1])
        * np.cos(2 * np.pi * s[:, 2])
    )
    x1[:, 1] = 2 * np.sqrt(s[:, 0]) * np.sin(2 * np.pi * s[:, 1])
    x1[:, 2] = (
        np.sqrt(s[:, 0])
        * np.cos(2 * np.pi * s[:, 1])
        * np.sin(2 * np.pi * s[:, 2])
    )
    rot = 0.5 * np.sqrt(2) * np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]])
    x1 = x1.dot(rot)

    # Generate second circle
    s = np.random.uniform(size=(nsample, 3))
    x2 = np.zeros_like(s)
    x2[:, 0] = (
        np.sqrt(s[:, 0])
        * np.cos(2 * np.pi * s[:, 1])
        * np.cos(2 * np.pi * s[:, 2])
    )
    x2[:, 1] = np.sqrt(s[:, 0]) * np.sin(2 * np.pi * s[:, 1])
    x2[:, 2] = (
        np.sqrt(s[:, 0])
        * np.cos(2 * np.pi * s[:, 1])
        * np.sin(2 * np.pi * s[:, 2])
    )
    x2 = x2 + np.array([5.0, 0.0, 0.0])
    x = np.concatenate((x1, x2)) + np.array([0.0, 0.0, 5.0])

    # Generate second data drom translation
    y = np.concatenate((x1[:, :2], s[:, :2] + np.array([4.0, 0.0])))
    angle = -np.pi / 4
    x[:nsample] = x[:nsample].dot(
        np.array(
            [
                [np.cos(angle), np.sin(angle), 0],
                [-np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
    )
    y[nsample:] = (y[nsample:] - np.mean(y[nsample:], axis=0)).dot(
        np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
    ) + np.mean(y[nsample:], axis=0)

    # Generate weights
    a, b = np.ones(x.shape[0]) / x.shape[0], np.ones(y.shape[0]) / y.shape[0]
    b[:n1], b[n1:] = (1 - ratio) * b[:n1], ratio * b[n1:]
    b = b / np.sum(b)
    return a, x, b, y


def plot_density_matching(pi, a, x, b, y, idx, alpha, linewidth):
    cmap1 = get_cmap("Blues")
    cmap2 = get_cmap("Reds")
    plt.figure(figsize=(6.0, 6.0))
    ax = plt.axes(projection="3d")
    ax.set_xlim(-2, 5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_zlim(-1, 6)
    ax.scatter(
        x[:, 0],
        x[:, 1],
        x[:, 2],
        c=cmap1(0.3 * (a - np.amin(b)) / np.amin(b) + 0.4),
        s=10 * (a / a) ** 2,
        zorder=1,
    )
    ax.scatter(
        y[:, 0],
        y[:, 1],
        0.0,
        c=cmap2(0.3 * (b - np.amin(b)) / np.amin(b) + 0.4),
        s=10 * (b / a) ** 2,
        zorder=1,
    )

    # Plot argmax of coupling
    for i in idx:
        m = np.sum(pi[i, :])
        ids = (-pi[i, :]).argsort()[:30]
        for j in ids:
            w = pi[i, j] / m
            t = [x[i][0], y[j][0]]
            u = [x[i][1], y[j][1]]
            v = [x[i][2], 0.0]
            ax.plot(
                t, u, v, c="k", alpha=w * alpha, linewidth=linewidth, zorder=0
            )
    # plt.xticks([])
    # plt.yticks([])
    plt.tight_layout()


if __name__ == "__main__":
    n1 = 1000
    dim = 2
    rho = 0.5
    eps = 0.01
    n_clust = 20
    ratio = 0.7
    compute_balanced = True

    # Generate gaussian mixtures translated from each other
    a, x, b, y = generate_data(n1, ratio)
    clf = KMeans(n_clusters=n_clust)
    clf.fit(x)
    idx = np.zeros(n_clust)
    for i in range(n_clust):
        d = clf.transform(x)[:, i]
        idx[i] = np.argmin(d)
    idx = idx.astype(int)

    # Generate costs and transport plan
    dx, dy = euclid_dist(x, x), euclid_dist(y, y)

    if compute_balanced:
        pi_b = gromov_wasserstein(dx, dy, a, b, loss_fun="square_loss")
        plot_density_matching(pi_b, a, x, b, y, idx, alpha=1.0, linewidth=0.5)
        plt.legend()
        plt.savefig(path + f"/fig_matching_plan_balanced_ratio{ratio}.png")
        plt.show()

    dx, dy = torch.from_numpy(dx), torch.from_numpy(dy)

    rho_list = [0.1]
    peps_list = [2, 1, 0, -1, -2, -3]
    for rho in rho_list:
        pi = None
        for p in peps_list:
            eps = 10 ** p
            print(f"Params = {rho, eps}")
            a, b = torch.from_numpy(a), torch.from_numpy(b)
            pi = log_ugw_sinkhorn(
                a,
                dx,
                b,
                dy,
                init=pi,
                eps=eps,
                rho=rho,
                rho2=rho,
                nits_plan=1000,
                tol_plan=1e-5,
                nits_sinkhorn=1000,
                tol_sinkhorn=1e-5,
            )
            print(f"Sum of transport plans = {pi.sum().item()}")

            # Plot matchings between measures
            a, b = a.data.numpy(), b.data.numpy()
            pi_ = pi.data.numpy()
            plot_density_matching(
                pi_, a, x, b, y, idx, alpha=1.0, linewidth=1.0
            )
            plt.legend()
            plt.savefig(
                path + f"/fig_matching_plan_ugw_"
                f"rho{rho}_eps{eps}_ratio{ratio}.png"
            )
            plt.show()
