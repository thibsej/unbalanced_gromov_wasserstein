#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:58:27 2020

Experiments where one marginal is fixed
"""

import os
import numpy as np
from joblib import Parallel, delayed
import torch
import ot

from unbalancedgw.batch_stable_ugw_solver import log_batch_ugw_sinkhorn
from unbalancedgw._batch_utils import compute_batch_flb_plan
import utils
from partial_gw import compute_cost_matrices

folder = "marginals_without_rescaling"
path = os.getcwd() + "/saved_plans"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/" + folder
if not os.path.isdir(path):
    os.mkdir(path)


def euclid_dist(x, y):
    """
    Computes the euclidean distance between two pointclouds, returning a
    matrix whose coordinates are the distance between two points.

    Parameters
    ----------

    x: torch.Tensor of size [size_X, dim]
    coordinates of the first group of vectors of R^d.

    y: torch.Tensor of size [size_Y, dim]
    coordinates of the second group of vectors of R^d.

    Returns
    -------
    torch.Tensor of size [size_X, size_Y]
    Matrix of all pairwise distances.
    """
    return (x[:, None, :] - y[None, :, :]).norm(p=2, dim=2)


def prepare_initialisation(dataset_p, dataset_u, n_pos, n_unl, prior, nb_try):
    """
    Compute the tensor used as initialization for UGW.
    The init is obtained by solving partial EMD as in Chapel et al. when the
    domains are the same.

    Parameters
    ----------

    dataset_p: string
    name of the dataset used for positive data

    dataset_u: string
    name of the dataset used for unlabeled data

    n_pos: int
    number of positives samples

    n_unl: int
    number of unlabeled samples

    prior: float
    proportion of positive samples in the unlabeled dataset

    nb_try: int
    number of folds to perform PU learning

    Returns
    -------
    init_plan: torch.Tensor of size [nb_try, n_pos, n_unl]
    Set of initialization plans used to init UGW.
    """
    init_plan = torch.zeros([nb_try, n_pos, n_unl])
    for i in range(nb_try):
        # Draw dataset
        P, U, _ = utils.draw_p_u_dataset_scar(dataset_p, dataset_u, n_pos,
                                              n_unl, prior, seed_nb=i)
        Ctot, C1, C2, mu, nu = compute_cost_matrices(P, U, prior,
                                                     nb_dummies=10)
        # Compute init
        init_plan[i] = torch.tensor(ot.emd(mu, nu, Ctot)[:n_pos, :])
    return init_plan


def compute_plan_ugw(dataset_p, dataset_u, n_pos, n_unl, prior, eps, rho, rho2,
                     nb_try, device=0):
    # Set default type and GPU device
    torch.cuda.set_device(device)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # keep constant to normalize cost, uniform over folds by taking first batch
    # P, U, _ = utils.draw_p_u_dataset_scar(dataset_p, dataset_u, n_pos, n_unl,
    #                                       prior, 0)
    # U = torch.tensor(U.values,dtype=torch.float)  # Convert to torch
    # cst_norm = euclid_dist(U, U).max()

    # Draw cost for all seeds as batch
    Cx = torch.zeros([nb_try, n_pos, n_pos])
    Cy = torch.zeros([nb_try, n_unl, n_unl])
    for i in range(nb_try):
        P, U, y_u = utils.draw_p_u_dataset_scar(dataset_p, dataset_u, n_pos,
                                                n_unl, prior, seed_nb=i)
        P, U = torch.tensor(P.values, dtype=torch.float), \
               torch.tensor(U.values, dtype=torch.float)
        cx, cy = euclid_dist(P, P), euclid_dist(U, U)
        Cx[i], Cy[i] = cx, cy
        # Cx[i], Cy[i] = cx / cst_norm, cy / cst_norm
    del cx, cy

    # Compute init and weights
    mu = (torch.ones([n_pos]) / n_pos).expand(nb_try, -1)
    nu = (torch.ones([n_unl]) / n_unl).expand(nb_try, -1)
    if P.shape[1] == U.shape[1]:  # If domains are the same
        init_plan = prepare_initialisation(dataset_p, dataset_u, n_pos, n_unl,
                                           prior, nb_try)
    else:
        _, _, init_plan = compute_batch_flb_plan(
            mu, Cx, nu, Cy, eps=eps, rho=rho, rho2=rho2,
            nits_sinkhorn=50000, tol_sinkhorn=1e-5)

    # Compute the marginal of init and save as file
    pi_numpy = init_plan.sum(dim=1).cpu().data.numpy()
    fname = f'/ugw_init_{dataset_p}_{n_pos}_{dataset_u}_{n_unl}_' \
            f'prior{prior}_eps{eps}_rho{rho}_rho{rho2}_reps{nb_try}.npy'
    np.save(path + fname, pi_numpy)

    # Set params and start the grid wrt entropic param eps
    pi = log_batch_ugw_sinkhorn(mu, Cx, nu, Cy, init=init_plan,
                                eps=eps, rho=rho, rho2=rho2,
                                nits_plan=3000, tol_plan=1e-5,
                                nits_sinkhorn=3000, tol_sinkhorn=1e-6)
    if torch.any(torch.isnan(pi)):
        raise Exception(f"Solver got NaN plan with params (eps, rho) = "
                        f"{dataset_p, dataset_u, nb_try, eps, rho, rho2}")

    # Compute the marginal and save as file
    pi_numpy = pi.sum(dim=1).cpu().data.numpy()
    fname = f'/ugw_plan_{dataset_p}_{n_pos}_{dataset_u}_{n_unl}_' \
            f'prior{prior}_eps{eps}_rho{rho}_rho{rho2}_reps{nb_try}.npy'
    np.save(path + fname, pi_numpy)

    print(
        f"DONE = Dataset {dataset_p, dataset_u}, eps = {eps}, "
        f"rho = {rho, rho2}, reps = {nb_try}")
    return


if __name__ == '__main__':
    parallel_gpu = True

    # epsilon Set to 2**-9 but an be optimized via grid-search
    grid_eps = [2. ** k for k in range(-9, -8, 1)]
    grid_rho = [2. ** k for k in range(-10, -4, 1)]
    nb_try = 40

    # List all tasks for the Caltech datasets
    list_tasks = []
    # # Matching similar features - prior set to 10%
    n_pos, n_unl, prior = 100, 100, 0.1
    list_surf = ['surf_Caltech', 'surf_amazon', 'surf_webcam', 'surf_dslr']
    list_decaf = ['decaf_caltech', 'decaf_amazon', 'decaf_webcam',
                  'decaf_dslr']
    list_data = [('surf_Caltech', d) for d in list_surf] + [
        ('decaf_caltech', d) for d in list_decaf]
    list_tasks = list_tasks + [
        (data_pos, data_unl, n_pos, n_unl, prior, eps, rho, rho2, nb_try)
        for (data_pos, data_unl) in list_data for eps in grid_eps
        for rho in grid_rho for rho2 in grid_rho]

    # # Matching similar features - prior set to 20%
    n_pos, n_unl, prior = 100, 100, 0.2
    list_surf = ['surf_Caltech', 'surf_amazon', 'surf_webcam']
    list_decaf = ['decaf_caltech', 'decaf_amazon', 'decaf_webcam']
    list_data = [('surf_Caltech', d) for d in list_surf] + [
        ('decaf_caltech', d) for d in list_decaf]
    list_tasks = list_tasks + [
        (data_pos, data_unl, n_pos, n_unl, prior, eps, rho, rho2, nb_try)
        for (data_pos, data_unl) in list_data for eps in grid_eps
        for rho in grid_rho for rho2 in grid_rho]

    # Matching different features - prior set to 10%
    n_pos, n_unl, prior = 100, 100, 0.1
    list_surf = ['surf_Caltech', 'surf_amazon', 'surf_webcam', 'surf_dslr']
    list_decaf = ['decaf_caltech', 'decaf_amazon', 'decaf_webcam',
                  'decaf_dslr']
    list_data = [('surf_Caltech', d) for d in list_decaf] + [
        ('decaf_caltech', d) for d in list_surf]
    list_tasks = list_tasks + [
        (data_pos, data_unl, n_pos, n_unl, prior, eps, rho, rho2, nb_try)
        for (data_pos, data_unl) in list_data for eps in grid_eps
        for rho in grid_rho for rho2 in grid_rho]

    # # Matching different features - prior set to 20%
    n_pos, n_unl, prior = 100, 100, 0.2
    list_surf = ['surf_Caltech', 'surf_amazon', 'surf_webcam']
    list_decaf = ['decaf_caltech', 'decaf_amazon', 'decaf_webcam']
    list_data = [('surf_Caltech', d) for d in list_decaf] + [
        ('decaf_caltech', d) for d in list_surf]
    list_tasks = list_tasks + [
        (data_pos, data_unl, n_pos, n_unl, prior, eps, rho, rho2, nb_try)
        for (data_pos, data_unl) in list_data for eps in grid_eps
        for rho in grid_rho for rho2 in grid_rho]

    if parallel_gpu:
        assert torch.cuda.is_available()
        list_device = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        total_devices = torch.cuda.device_count()
        print(
            f"Parallel computation // Total GPUs available = {total_devices}")
        pll = Parallel(n_jobs=total_devices)
        iterator = (
            delayed(compute_plan_ugw)(data_pos, data_unl, n_pos, n_unl, prior,
                                      eps, rho, rho2, nb_try,
                                      device=list_device[k % total_devices])
            for
            k, (
                data_pos, data_unl, n_pos, n_unl, prior, eps, rho, rho2,
                nb_try) in
            enumerate(list_tasks))
        pll(iterator)

    else:
        print("Not Parallel")
        for (data_pos, data_unl, n_pos, n_unl, prior, eps, rho, rho2,
             nb_try) in list_tasks:
            compute_plan_ugw(data_pos, data_unl, n_pos, n_unl, prior, eps, rho,
                             rho2, nb_try)
            print(f'{data_pos, data_unl} done.')
