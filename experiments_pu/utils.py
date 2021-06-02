#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:03:33 2020

@author: lchapel
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from sklearn.datasets import make_moons
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
from torchvision import datasets
import scipy.io as sio



def plot_dataset(P, U, y, dim='2d', ax=None, y_hat=None, transp=None):
    """Plot a dataset

    Parameters
    ----------
    P: pandas dataframe, shape=(n_p, d_p)
        Positive dataset

    U: pandas dataframe, shape=(n_u, d_u)
        Unlabeled dataset

    y: array, len=n_u
        Labels on the unlabeled dataframe. Should be 0 (negatives) or 1 (pos)

    dim: string (default: '2d')
        Choose between a '2d' or '3d' plot

    y_hat: array, len=n_u (default: None)
        Predicted labels of the unlabeled dataframe.
        If None, no labels are displayed.
    """
    if ax is None:
        fig = plt.figure()
        if dim == '3d':
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
    if dim == '3d':
        ax.scatter(P.feature1, P.feature2, P.feature3, c='k', marker='o',
                   linewidth=1, label='P')
    elif dim == '2d':
        ax.scatter(
            P.feature1, P.feature2, c='k', marker='o', linewidth=1, s=50,
            label='P')
    else:
        raise ValueError("dim argument takes either '2d' or '3d' argument")
    if y_hat is None:
        pos_1 = np.where(y == 1)
        pos_0 = np.where(y == 0)
        if dim == '3d':
            ax.view_init(elev=40., azim=100)
            ax.scatter(U.feature1, U.feature2, zs=0, zdir='z', label='U')
        else:
            ax.scatter(
                U.iloc[pos_1].feature1, U.iloc[pos_1].feature2,
                s=70, facecolors='none', edgecolors='b', label='P+U')
            ax.scatter(
                U.iloc[pos_0].feature1, U.iloc[pos_0].feature2,
                c='b', marker='+', linewidth=1, s=70, alpha=0.5, label='N+U')
    else:
        if dim == '3d':
            ax.view_init(elev=40., azim=100)
            ax.scatter(U.feature1, U.feature2,
                       zs=0, zdir='z', label='U', c='b')
        else:
            for i, y_i in enumerate(y):
                if (y_i == 1 and y_hat[i] == 1):
                    ax.scatter(U.iloc[i].feature1, U.iloc[i].feature2, s=70,
                               facecolors='none', edgecolors='b', label='P+U')
                if (y_i == 0 and y_hat[i] == 0):
                    ax.scatter(U.iloc[i].feature1, U.iloc[i].feature2, s=70,
                               c='b', marker='+', linewidth=1, alpha=0.5,
                               label='N+U')
                if (y_i == 0 and y_hat[i] == 1):
                    ax.scatter(U.iloc[i].feature1, U.iloc[i].feature2, s=70,
                               facecolors='none', edgecolors='r')
                if (y_i == 1 and y_hat[i] == 0):
                    ax.scatter(U.iloc[i].feature1, U.iloc[i].feature2, c='r',
                               marker='+', linewidth=1, s=70, alpha=0.5)

    ax.tick_params(axis='x', which='both', bottom=False, top=False,
                   labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False,
                   labelleft=False)
    if dim == '3d':
        ax.tick_params(axis='z', which='both', left=False, right=False,
                       labelleft=False)
    if transp is not None:
        for i in range(transp.shape[0]):
            for j in range(transp.shape[1]):
                if transp[i, j] > 1e-5:
                    if dim == '2d':
                        ax.plot([P.iloc[i].feature1, U.iloc[j].feature1],
                                [P.iloc[i].feature2, U.iloc[j].feature2], 'k',
                                alpha=0.15)
                    elif dim == '3d':
                        ax.plot([P.iloc[i].feature1, U.iloc[j].feature1],
                                [P.iloc[i].feature2, U.iloc[j].feature2],
                                [P.iloc[i].feature3, 0], 'k', alpha=0.05)


def annotate_transp_matrix(ax):
    ax.text(0.3, -0.05, 'Unl. positives', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes, c='b', size=20)
    ax.text(0.85, -0.05, 'Unl. negatives', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes, c='b', size=20)
    ax.text(-0.05, 0.51, 'Positives', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes, rotation=90,
            size=20)
    ax.text(-0.05, 0.06, 'D', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes, rotation=90,
            size=20)
    ax.axvline(x=5.5, color='k', linewidth=2)
    ax.axhline(y=9.5, color='k')
    ax.tick_params(axis='x', which='both', bottom=False, top=False,
                   labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False,
                   labelleft=False)


def make_data(dataset='mnist'):
    """Load a dataset (need to be stored into the folder /data)

    Parameters
    ----------
    dataset: name of the dataset

    Returns
    -------
    np_array that contains the data

    list that contains the labels
    """
    # Piece of code for the mnist dataset
    def make_environment(images, labels, e):
        def torch_bernoulli(p, size):
            torch.manual_seed(0)
            return (torch.rand(size) < p).float()

        def torch_xor(a, b):
            return b.abs()  # Assumes both inputs are either 0 or 1
        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit; flip label with proba 0.25
        labels[labels > 1] = 0.0  # positives: class1, negatives: others
        labels = labels.float()
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
        return {
            'images': (images.float() / 255.),
            'labels': labels[:, None]
        }

    if dataset == "mushrooms":
        x, t = load_svmlight_file("data/mushrooms")
        x = x.toarray()
        x = np.delete(x, 77, 1)  # contains only one value
        t[t == 1] = 1
        t[t == 2] = 0
    elif dataset == "shuttle":
        x_train, t_train = load_svmlight_file('data/shuttle.scale')
        x_train = x_train.toarray()
        x_test, t_test = load_svmlight_file('data/shuttle.scale.t')
        x_test = x_test.toarray()
        x = np.concatenate([x_train, x_test])
        t = np.concatenate([t_train, t_test])
        t[~(t == 1)] = 0
    elif dataset == "pageblocks":
        data = np.loadtxt('data/page-blocks.data')
        x, t = data[:, :-1], data[:, -1]
        t[~(t == 1)] = 0
    elif dataset == "usps":
        x_train, t_train = load_svmlight_file('data/usps')
        x_train = x_train.toarray()
        x_test, t_test = load_svmlight_file('data/usps.t')
        x_test = x_test.toarray()
        x = np.concatenate([x_train, x_test])
        t = np.concatenate([t_train, t_test])
        t[t == 1] = 1
        t[t > 1] = 0
    elif dataset == "connect-4":
        x, t = load_svmlight_file('data/connect-4')
        x = x.toarray()
        t[t == -1] = 0
    elif dataset == "spambase":
        data = np.loadtxt('data/spambase.data', delimiter=',')
        x, t = data[:, :-1], data[:, -1]
    elif dataset[:5] == "mnist":
        mnist = datasets.MNIST('~/data/mnist', train=True, download=True)
        mnist = (mnist.data, mnist.targets)
        if dataset == "mnist":
            envs = [make_environment(mnist[0][::2], mnist[1][::2], 0)]
        elif dataset == "mnist_color_change_p":
            envs = [make_environment(mnist[0][::2], mnist[1][::2], 0.1)]
        elif dataset == "mnist_color_change_u":
            envs = [make_environment(mnist[0][::2], mnist[1][::2], 1)]
        data = envs[0]['images']
        x = np.zeros((data.shape[0], 2*14*14))
        for i in range(x.shape[0]):
            x[i] = data[i].flatten()
        t = np.array(envs[0]['labels']).flatten()
    elif dataset.startswith("surf"):
        domain = dataset[5:]
        mat = sio.loadmat("data/" + domain + "_zscore_SURF_L10.mat")
        if domain == "dslr":
            x = mat['Xs']
            t = mat['Ys']
        else:
            x = mat['Xt']
            t = mat['Yt']
        t[t != 1] = 0
        t = t.flatten()
        pca = PCA(n_components=10, random_state=0)
        pca.fit(x.T)
        x = pca.components_.T
    elif dataset.startswith("decaf"):
        domain = dataset[6:]
        mat = sio.loadmat("data/" + domain + "_decaf.mat")
        x = mat['feas']
        t = mat['labels']
        t[t != 1] = 0
        t = t.flatten()
        pca = PCA(n_components=40, random_state=0)
        pca.fit(x.T)
        x = pca.components_.T
    else:
        raise ValueError("Check the name of the dataset")
    return x, t

 
def draw_p_u_dataset_scar(dataset_p, dataset_u, size_p, size_u, prior, seed_nb=None):
    """Draw a Positive and Unlabeled dataset "at random""

    Parameters
    ----------
    dataset_p: name of the dataset among which the positives are drawn

    dataset_u: name of the dataset among which the unlabeled are drawn

    size_p: number of points in the positive dataset

    size_u: number of points in the unlabeled dataset

    prior: percentage of positives on the dataset (s)

    seed_nb: seed

    Returns
    -------
    pandas.DataFrame of shape (n_p, d_p)
        Positive dataset

    pandas.DataFrame of shape (n_u, d_u)
        Unlabeled dataset

    pandas.Series of len (n_u)
        labels of the unlabeled dataset
    """
    # Normalize data
    x, t = make_data(dataset=dataset_p)
    div = np.max(x, axis=0) - np.min(x, axis=0)
    div[div == 0] = 1 # Avoid division by zero
    x = (x - np.min(x, axis=0)) / div

    # Set size of datasets
    size_u_p = int(prior * size_u)
    size_u_n = size_u - size_u_p

    # Build splits for positive and unlabeled-positive
    xp_t = x[t == 1]
    tp_t = t[t == 1]

    xp, xp_other, _, tp_o = train_test_split(xp_t, tp_t, train_size=size_p, random_state=seed_nb)
    if dataset_u == dataset_p:
        xup, _, _, _ = train_test_split(xp_other, tp_o, train_size=size_u_p, random_state=seed_nb)
    else:
        x, t = make_data(dataset=dataset_u)
        div = np.max(x, axis=0) - np.min(x, axis=0)
        div[div == 0] = 1
        x = (x - np.min(x, axis=0)) / div
        xp_other = x[t == 1]
        tp_o = t[t == 1]
        xup, _, _, _ = train_test_split(xp_other, tp_o, train_size=size_u_p, random_state=seed_nb)

    # Build splits for negative-unlabeled
    xn_t = x[t == 0]
    tn_t = t[t == 0]
    xun, _, _, _ = train_test_split(xn_t, tn_t, train_size=size_u_n, random_state=seed_nb)

    # Merge samples and labels of all unlabeled
    xu = np.concatenate([xup, xun], axis=0)
    yu = np.concatenate((np.ones(len(xup)), np.zeros(len(xun))))

    return pd.DataFrame(xp), pd.DataFrame(xu), pd.Series(yu)
