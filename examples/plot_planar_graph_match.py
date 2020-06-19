import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import torch
import networkx as nx
from scipy.spatial import Delaunay

from solver.utils_numpy import euclid_dist
from ot.gromov import gromov_wasserstein
from solver.tlb_kl_sinkhorn_solver import TLBSinkhornSolver

path = os.getcwd() + "/output"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/plots"
if not os.path.isdir(path):
    os.mkdir(path)

torch.set_default_tensor_type(torch.cuda.FloatTensor)

def generate_data_target(sig=0.05):
    # Generate circles
    n1, n2 = 100, 25
    t1, t2 = np.linspace(0, 2 * np.pi, n1), np.linspace(0, 2 * np.pi, n2)
    x1 = np.vstack((np.cos(t1), np.sin(t1))).transpose()
    x1[::2] = 0.9 * x1[::2]
    x2 = np.vstack((0.2 * np.cos(t2) + 1.2 * x1[int(0.8 * n1), 0], 0.2 * np.sin(t2)
                    + 1.2 * x1[int(0.8 * n1), 1])).transpose()

    # Generate lines
    nu = 20
    u = np.linspace(0, 1, nu)
    x3 = np.vstack((np.cos(t1[int(0.4 * n1)]) * (1 + u), np.sin(t1[int(0.4 * n1)]) * (1 + u))).transpose()
    x4 = np.vstack((np.cos(t1[int(0.5 * n1)]) * (1 + u), np.sin(t1[int(0.5 * n1)]) * (1 + u))).transpose()
    x5 = 0.5 * np.vstack((np.zeros_like(u), u)).transpose() + x4[int(0.7 * nu)] - np.array([0.0, 0.3])

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
    return np.ones(shape=x.shape[0]), x + sig * np.random.normal(size=x.shape), G


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
    rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
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
    return np.ones(shape=x.shape[0]), x + sig * np.random.normal(size=x.shape), G


def convert_points_to_graph(x, G):
    C = euclid_dist(x, x)
    idx = np.argsort(C, axis=1)

    # Preprocess graph
    for i, j in G.edges:
        G[i][j]['weight'] = C[i, j]

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
    print(f"CHECK CONNECTION = ", nx.is_connected(G))
    dist = nx.floyd_warshall_numpy(G)
    return dist, G


def draw_graph(x, G):
    plt.scatter(x[:, 0], x[:, 1])
    for (i, j) in G.edges:
        t, u = [x[i][0], x[j][0]], [x[i][1], x[j][1]]
        plt.plot(t, u, c='k', zorder=0)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()


def plot_density_matching(pi, a, x, b, y, Gx, Gy, titlename=None):
    marg1, marg2 = np.sum(pi, axis=1), np.sum(pi, axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    col1 = np.cumsum(a) / np.sum(a)
    col2 = pi.transpose().dot(col1) / np.sum(pi, axis=0)
    cmap = get_cmap('hsv')

    # Display first graph
    ax[0].scatter(x[:, 0], x[:, 1], c=cmap(col1), s=(marg1 / a) ** 2 * 50., zorder=1)
    for (i, j) in Gx.edges:
        w = np.minimum( (marg1[i] * marg1[j]) / (a[i] * a[j]), 1. )
        t, u = [x[i][0], x[j][0]], [x[i][1], x[j][1]]
        ax[0].plot(t, u, c='k', alpha=0.5 * w, zorder=0)
    ax[0].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False)
    ax[0].set_title('Source graph')
    ax[0].set_aspect('equal')

    # Display second graph
    ax[1].scatter(y[:, 0], y[:, 1], c=cmap(col2), s=(marg2 / b) ** 2 * 50., zorder=1)
    for (i, j) in Gy.edges:
        w = np.minimum( (marg2[i] * marg2[j]) / (b[i] * b[j]), 1. )
        t, u = [y[i][0], y[j][0]], [y[i][1], y[j][1]]
        ax[1].plot(t, u, c='k', alpha=0.5 * w, zorder=0)
    ax[1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False)
    ax[1].set_title('target graph')
    ax[1].set_aspect('equal')

    fig.suptitle(titlename, fontsize=20)
    plt.tight_layout()


if __name__ == '__main__':
    np.random.seed(42)
    sig = 0.04
    normalize_proba = True
    solver = TLBSinkhornSolver(nits=500, nits_sinkhorn=1000, gradient=False, tol=1e-3, tol_sinkhorn=1e-3)
    rho = .1

    a, x, Gx = generate_data_source(sig)
    Cx, Gx = convert_points_to_graph(x, Gx)
    assert ~np.isinf(Cx).any()

    b, y, Gy = generate_data_target(sig)
    Cy, Gy = convert_points_to_graph(y, Gy)
    assert ~np.isinf(Cy).any()

    if normalize_proba:
        a, b = a / np.sum(a), b / np.sum(b)

    if torch.cuda.is_available():
        a, b = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
        Cx, Cy = torch.from_numpy(Cx).cuda(), torch.from_numpy(Cy).cuda()
    else:
        a, b = torch.from_numpy(a), torch.from_numpy(b)
        Cx, Cy = torch.from_numpy(Cx), torch.from_numpy(Cy)

    pi = None
    for eps in [10 ** e for e in [2., 1.5, 1, 0.5, 0., -0.5, -1., -1.5, -2]]:
        pi, _ = solver.tlb_sinkhorn(a, Cx, b, Cy, rho=rho, eps=eps, init=pi)
    print(f"Sum of transport plans = {pi.sum().item()}")

    # Plot matchings between measures
    a, b = a.cpu().data.numpy(), b.cpu().data.numpy()
    pi = pi.cpu().data.numpy()
    plot_density_matching(pi, a, x, b, y, Gx, Gy, titlename=f'UGW matching, ($\\rho$,$\epsilon$)={rho, eps}')
    plt.legend()
    plt.show()

    if normalize_proba:
        pi_b = gromov_wasserstein(Cx.cpu().numpy(), Cy.cpu().numpy(), a, b, loss_fun='square_loss')
        plot_density_matching(pi_b, a, x, b, y, Gx, Gy, titlename='GW matching')
        plt.legend()
        plt.show()
