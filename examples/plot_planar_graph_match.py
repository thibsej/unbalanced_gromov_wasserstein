"""
==================================
Using a custom density with Picard
==================================

This example shows how to use custom densities using Picard

"""
# TODO: DEbug it, graphs are not consistent

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import torch
import networkx as nx
import scipy as sp
from scipy.spatial import Delaunay

from unbalancedgw.utils import euclid_dist
from ot.gromov import gromov_wasserstein
from ot.lp import emd
from ot.partial import (
    partial_gromov_wasserstein,
    partial_gromov_wasserstein2,
    entropic_partial_gromov_wasserstein,
    partial_wasserstein,
)

# from solver.vanilla_sinkhorn_solver import VanillaSinkhornSolver
from unbalancedgw.vanilla_ugw_solver import log_ugw_sinkhorn

path = os.getcwd() + "/output"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/plots_graph"
if not os.path.isdir(path):
    os.mkdir(path)

torch.set_default_tensor_type(torch.cuda.FloatTensor)


def generate_data_target(sig=0.05):
    # Generate circles
    n1, n2 = 100, 25
    t1, t2 = np.linspace(0, 2 * np.pi, n1), np.linspace(0, 2 * np.pi, n2)
    x1 = np.vstack((np.cos(t1), np.sin(t1))).transpose()
    x1[::2] = 0.9 * x1[::2]
    x2 = np.vstack(
        (
            0.2 * np.cos(t2) + 1.2 * x1[int(0.8 * n1), 0],
            0.2 * np.sin(t2) + 1.2 * x1[int(0.8 * n1), 1],
        )
    ).transpose()

    # Generate lines
    nu = 20
    u = np.linspace(0, 1, nu)
    x3 = np.vstack(
        (
            np.cos(t1[int(0.4 * n1)]) * (1 + u),
            np.sin(t1[int(0.4 * n1)]) * (1 + u),
        )
    ).transpose()
    x4 = np.vstack(
        (
            np.cos(t1[int(0.5 * n1)]) * (1 + u),
            np.sin(t1[int(0.5 * n1)]) * (1 + u),
        )
    ).transpose()
    x5 = np.vstack((np.zeros_like(u), u)).transpose()
    x5 = 0.5 * x5 + x4[int(0.7 * nu)] - np.array([0.0, 0.3])

    # Add random clusters
    n6, n8 = 30, 30
    x6 = 0.05 * np.random.normal(size=(n6, 2)) + x3[-2]
    x8 = 0.1 * np.random.normal(size=(n8, 2)) + x1[9]
    x = np.concatenate((x1, x2, x3, x4, x5, x6, x8)) + np.array([0.3, 0.0])

    # Generate a pre-graph with some substructures
    G = nx.Graph()
    G.add_nodes_from(range(x.shape[0]))
    ntot = 0

    tri = Delaunay(x1)
    for i, j, k in tri.vertices:
        if np.abs((i - j) % n1) < 5:
            G.add_edge(i, j)
        if np.abs((i - k) % n1) < 5:
            G.add_edge(i, k)
        if np.abs((k - j) % n1) < 5:
            G.add_edge(k, j)
    ntot += n1

    for i in range(ntot, ntot + n2 - 1):
        G.add_edge(i, (i + 1), weight=0.0)
    G.add_edge(ntot, ntot + n2 - 1)
    ntot += n2

    for i in range(ntot, ntot + nu - 1):
        G.add_edge(i, i + 1)
    ntot += nu

    for i in range(ntot, ntot + nu - 1):
        G.add_edge(i, i + 1)
    ntot += nu

    for i in range(ntot, ntot + nu - 1):
        G.add_edge(i, i + 1)
    ntot += nu

    tri = Delaunay(x6)
    for i, j, k in tri.vertices:
        G.add_edge(ntot + i, ntot + j)
        G.add_edge(ntot + i, ntot + k)
        G.add_edge(ntot + k, ntot + j)
    ntot += n6

    tri = Delaunay(x8)
    for i, j, k in tri.vertices:
        G.add_edge(ntot + i, ntot + j)
        G.add_edge(ntot + i, ntot + k)
        G.add_edge(ntot + k, ntot + j)
    return (
        np.ones(shape=x.shape[0]),
        x + sig * np.random.normal(size=x.shape),
        G,
    )


def generate_data_source(sig=0.05):
    # generate Second graph
    n1 = 100
    t = np.linspace(0, 2 * np.pi, n1)
    x1 = np.vstack((np.cos(t), np.sin(t))).transpose()
    x1[::2] = 0.9 * x1[::2]

    # Generate lines
    nu = 25
    u = np.linspace(0, 1, nu)
    x4 = 0.7 * (np.vstack((u, -u)).transpose()) + x1[int(0.75 * n1)]
    x5 = 0.7 * (np.vstack((-u, -u)).transpose()) + x1[int(0.75 * n1)]

    # Generate random clusters
    n9, n10, n11 = 45, 25, 25
    x9 = 0.1 * np.random.normal(size=(n9, 2)) + x1[int(0.75 * n1)]
    x10 = 0.05 * np.random.normal(size=(n10, 2)) + x4[-1]
    x11 = 0.05 * np.random.normal(size=(n11, 2)) + x5[-1]

    x = np.concatenate((x1, x4, x5, x9, x10, x11))
    angle = 1.35 * np.pi
    rot = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    x = x.dot(rot)

    G = nx.Graph()
    G.add_nodes_from(range(x.shape[0]))
    ntot = 0

    tri = Delaunay(x1)
    for i, j, k in tri.vertices:
        if np.abs((i - j) % n1) < 5:
            G.add_edge(i, j)
        if np.abs((i - k) % n1) < 5:
            G.add_edge(i, k)
        if np.abs((k - j) % n1) < 5:
            G.add_edge(k, j)
    ntot += n1

    for i in range(ntot, ntot + nu - 2):
        G.add_edge(i, i + 1)
    ntot += nu

    for i in range(ntot, ntot + nu - 2):
        G.add_edge(i, i + 1)
    ntot += nu

    tri = Delaunay(x9)
    for i, j, k in tri.vertices:
        G.add_edge(ntot + i, ntot + j)
        G.add_edge(ntot + i, ntot + k)
        G.add_edge(ntot + k, ntot + j)
    ntot += n9

    tri = Delaunay(x10)
    for i, j, k in tri.vertices:
        G.add_edge(ntot + i, ntot + j)
        G.add_edge(ntot + i, ntot + k)
        G.add_edge(ntot + k, ntot + j)
    ntot += n10

    tri = Delaunay(x11)
    for i, j, k in tri.vertices:
        G.add_edge(ntot + i, ntot + j)
        G.add_edge(ntot + i, ntot + k)
        G.add_edge(ntot + k, ntot + j)
    return (
        np.ones(shape=x.shape[0]),
        x + sig * np.random.normal(size=x.shape),
        G,
    )


def convert_points_to_graph(x, G):
    C = euclid_dist(x, x)
    idx = np.argsort(C, axis=1)

    # Preprocess graph
    for i, j in G.edges:
        G[i][j]["weight"] = C[i, j]

    # Add all edges of the circle
    for i in range(100):
        G.add_edge(i, (i + 1) % 100, weight=C[i, (i + 1) % 40])

    # Add all edges of the knn
    for i in range(C.shape[0]):
        for j in range(1, 3):
            G.add_edge(i, idx[i, j], weight=C[i, idx[i, j]])

    if not nx.is_connected(G):
        comp = list(nx.connected_components(G))
        for c in range(1, nx.number_connected_components(G)):
            for j in range(3, 10):
                for n in comp[c]:
                    if idx[n, j] in comp[0]:
                        G.add_edge(n, idx[n, j], weight=C[n, idx[n, j]])
                        break
                if nx.is_connected(G):
                    break
    print("CHECK CONNECTION = ", nx.is_connected(G))
    dist = nx.floyd_warshall_numpy(G)
    return dist, G


def draw_graph(x, G):
    plt.scatter(x[:, 0], x[:, 1])
    for (i, j) in G.edges:
        t, u = [x[i][0], x[j][0]], [x[i][1], x[j][1]]
        plt.plot(t, u, c="k", zorder=0)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()


def plot_density_matching(pi, a, x, b, y, Gx, Gy, titlefile):
    marg1, marg2 = np.sum(pi, axis=1), np.sum(pi, axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(titlefile, fontsize=20)
    col1 = np.cumsum(a) / np.sum(a)
    col2 = pi.transpose().dot(col1) / np.maximum(
        np.sum(pi, axis=0), 10 ** (-6)
    )  # max for stability with partial GW
    cmap = get_cmap("hsv")

    # Display first graph
    ax[0].scatter(
        x[:, 0], x[:, 1], c=cmap(col1), s=(marg1 / a) ** 2 * 50.0, zorder=1
    )
    for (i, j) in Gx.edges:
        w = np.minimum((marg1[i] * marg1[j]) / (a[i] * a[j]), 1.0)
        t, u = [x[i][0], x[j][0]], [x[i][1], x[j][1]]
        ax[0].plot(t, u, c="k", alpha=0.5 * w, zorder=0)

    ax[0].axis("off")
    extent = (
        ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    )
    fname = path + f"/pic_graph_source_{titlefile}.pdf"
    fig.savefig(fname, format="pdf", bbox_inches=extent)

    # Display second graph
    ax[1].scatter(
        y[:, 0], y[:, 1], c=cmap(col2), s=(marg2 / b) ** 2 * 50.0, zorder=1
    )
    for (i, j) in Gy.edges:
        w = np.minimum((marg2[i] * marg2[j]) / (b[i] * b[j]), 1.0)
        t, u = [y[i][0], y[j][0]], [y[i][1], y[j][1]]
        ax[1].plot(t, u, c="k", alpha=0.5 * w, zorder=0)

    ax[1].axis("off")
    extent = (
        ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    )
    fig.savefig(
        path + f"/pic_graph_target_{titlefile}.pdf",
        format="pdf",
        bbox_inches=extent,
    )

    ax[0].set_title("Source graph")
    ax[1].set_title("target graph")
    plt.tight_layout()


if __name__ == "__main__":
    np.random.seed(42)
    sig = 0.04
    normalize_proba = True
    compare_with_gw = True

    # Check the graph is fully connex before running the code.
    a, x, Gx = generate_data_source(sig)
    cx, Gx = convert_points_to_graph(x, Gx)
    assert ~np.isinf(cx).any()

    b, y, Gy = generate_data_target(sig)
    cy, Gy = convert_points_to_graph(y, Gy)
    assert ~np.isinf(cy).any()

    # Defaults to true to allow comparison with balanced GW
    if normalize_proba:
        a, b = a / np.sum(a), b / np.sum(b)

    if torch.cuda.is_available():  # Uppercase are used for torch tensors
        A, B = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
        CX, CY = torch.from_numpy(cx).cuda(), torch.from_numpy(cy).cuda()
    else:
        A, B = torch.from_numpy(a), torch.from_numpy(b)
        CX, CY = torch.from_numpy(cx), torch.from_numpy(cy)

    # Keep masses of optimal plans to compare with Partial GW
    list_mass_pgw = []
    for rho in [10 ** e for e in [-1.0, 0.0, 1.0]]:  # Compute UGW plans
        PI = None
        for eps in [10 ** e for e in [2.0, 1.5, 1, 0.5, 0.0, -0.5, -1.0]]:
            PI = log_ugw_sinkhorn(
                A,
                CX,
                B,
                CY,
                init=PI,
                nits_plan=500,
                nits_sinkhorn=1000,
                tol_plan=1e-7,
                tol_sinkhorn=1e-5,
            )
        # Save mass for comparison with Partial-GW
        list_mass_pgw.append(PI.sum().item())
        print(f"Sum of transport plans = {PI.sum().item()}")

        # Plot matchings between measures --> UGW
        pi = PI.cpu().data.numpy()
        plot_density_matching(
            pi, a, x, b, y, Gx, Gy, titlefile=f"UGW_rho{rho}_eps{eps}"
        )
        plt.legend()
        plt.show()

    # Comparison with Partial GW from POT
    for m in list_mass_pgw:
        # TRIAL 1 - initialized with simulated annealing
        pi = a[:, None] * b[None, :]
        # Simulated annealing loop
        for eps in [10 ** e for e in [2.0, 1.5, 1]]:
            pi = entropic_partial_gromov_wasserstein(
                cx, cy, a, b, eps, m=m, G0=pi
            )
        pi = partial_gromov_wasserstein(cx, cy, a, b, m=m, G0=pi)
        cost = partial_gromov_wasserstein2(cx, cy, a, b, m=m, G0=pi)
        # TRIAL 2 - Initialize with partial OT plan
        M = sp.spatial.distance.cdist(x, y)
        gam = partial_wasserstein(a, b, M, m=m)
        gam = partial_gromov_wasserstein(cx, cy, a, b, m=m, G0=gam)
        if partial_gromov_wasserstein2(cx, cy, a, b, m=m, G0=gam) < cost:
            pi = gam
        # TRIAL 3 - Initialize with OT plan
        gam = emd(a, b, M)
        gam = partial_gromov_wasserstein(cx, cy, a, b, m=m, G0=gam)
        if partial_gromov_wasserstein2(cx, cy, a, b, m=m, G0=gam) < cost:
            pi = gam

        # Plot matchings between measures --> Partial GW
        plot_density_matching(
            pi, a, x, b, y, Gx, Gy, titlefile=f"PGW_mass{m:.3f}"
        )
        plt.legend()
        plt.show()

    # Plot the behaviour of GW as reference
    if normalize_proba & compare_with_gw:
        pi = a[:, None] * b[None, :]
        pi_gw = gromov_wasserstein(cx, cy, a, b, loss_fun="square_loss")
        plot_density_matching(pi_gw, a, x, b, y, Gx, Gy, titlefile="GW")
        plt.legend()
        plt.show()
