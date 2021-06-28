"""
==================================
Using a custom density with Picard
==================================

This example shows how to use custom densities using Picard

"""

import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from solver.utils import euclid_dist
from ot.gromov import gromov_wasserstein
from solver.vanilla_sinkhorn_solver import VanillaSinkhornSolver

path = os.getcwd() + "/output"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/plots"
if not os.path.isdir(path):
    os.mkdir(path)

torch.set_default_tensor_type(torch.cuda.FloatTensor)


def generate_data(nsample, nout, noise, normalize=False):
    z, w = np.linspace(0, np.pi, nsample), np.linspace(0., 1., nout)
    x = np.transpose(np.stack((np.cos(z), np.sin(z))))
    y = np.transpose(np.stack((1 - np.cos(z), 1 - np.sin(z) - .5)))
    outlier = np.transpose(np.stack((-1 - w, -w)))
    x = np.concatenate((x, outlier))

    # Generate weights
    if normalize:
        a, b = np.ones(x.shape[0]) / x.shape[0], np.ones(y.shape[0]) / y.shape[0]
    else:
        a, b = np.ones(x.shape[0]) / x.shape[0], np.ones(y.shape[0]) / x.shape[0]
    return a, x + noise * np.random.normal(size=x.shape), b, y + noise * np.random.normal(size=y.shape)


def plot_density_matching(pi, a, x, b, y, ids, alpha, linewidth, fontsize, ftitle='', fname=None):
    n1, n2 = b.shape[0], a.shape[0] - b.shape[0]
    marg1, marg2 = np.sum(pi, axis=1), np.sum(pi, axis=0)
    plt.figure(figsize=(6., 6.))
    plt.scatter(x[:n1, 0], x[:n1, 1], c='b', s=(marg1 / a)[:n1] ** 2 * 25., zorder=1)
    plt.scatter(x[n1:, 0], x[n1:, 1], c='c', s=(marg1 / a)[n1:] ** 2 * 25., zorder=1)
    plt.scatter(y[:, 0], y[:, 1], c='r', s=(marg2 / b) ** 2 * 25., zorder=1)

    # Plot argmax of coupling
    for i in ids:
        ids = (-pi[i, :]).argsort()[:5]
        for j in ids:
            w = pi[i, j] / a[i]
            t, u = [x[i][0], y[j][0]], [x[i][1], y[j][1]]
            plt.plot(t, u, c='k', alpha=w * alpha, linewidth=linewidth, zorder=0)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.xlim(-2.3, 2.3)
    plt.ylim(-1.5, 1.5)
    plt.title(ftitle, fontsize=fontsize)
    if fname is not None:
        plt.savefig(fname)


if __name__ == '__main__':
    dim = 2
    rho = 0.01
    eps = 0.001
    nsample, nout = 300, 50
    compute_balanced = True
    solver = VanillaSinkhornSolver(nits_plan=500, nits_sinkhorn=1000, gradient=False, tol_plan=1e-3, tol_sinkhorn=1e-3)
    ids = np.concatenate((np.arange(start=0, stop=nsample, step=10),
                          np.arange(start=nsample, stop=nsample + nout, step=10)))

    for noise in [0.1]:
        print(f"NOISE = {noise}")
        a, x, b, y = generate_data(nsample=nsample, nout=nout, noise=noise, normalize=True)

        # Generate costs and transport plan
        Cx, Cy = euclid_dist(x, x), euclid_dist(y, y)

        if compute_balanced:
            pi_b = gromov_wasserstein(Cx, Cy, a, b, loss_fun='square_loss')
            plot_density_matching(pi_b, a, x, b, y, ids, alpha=1., linewidth=0.5, fontsize=16,
                                  ftitle='Balanced GW matching')
            plt.legend()
            plt.savefig(f"matching_outlier_balanced_noise{noise}.png")
            plt.show()

        # Lowercase for numpy, Uppercase for torch
        CX, CY = torch.from_numpy(Cx).cuda(), torch.from_numpy(Cy).cuda()
        A, B = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()

        # Compute the matching
        for rho in [0.01, 1.]:
            PI = None
            for eps in [1., 0.1, 0.01, 0.001]:
                print(f"        PARAMS = {rho, eps}")
                solver.rho, solver.eps = rho, eps
                PI = solver.ugw_sinkhorn(A, CX, B, CY, init=PI)
            print(f"Mass of transport plan = {PI.sum().item()}")

            # Plot matchings between measures
            pi = PI.cpu().data.numpy()
            plot_density_matching(pi, a, x, b, y, ids, alpha=1., linewidth=1., fontsize=16,
                                  ftitle=f'Unbalanced GW matching, $\\rho$={rho}, $\epsilon$={eps}', fname=None)
            plt.legend()
            plt.savefig(f"matching_outlier_unbalanced_noise{noise}_rho{rho}_eps{eps}.png")
            plt.show()
