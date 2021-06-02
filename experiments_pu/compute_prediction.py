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

from solver.batch_stable_sinkhorn_solver import BatchStableSinkhornSolver
from solver.batch_lower_bound_solver import BatchLowerBoundSolver
import utils
from partial_gw import compute_cost_matrices

date = "marginals_without_rescaling"
path = os.getcwd() + "/saved_plans"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/" + date
if not os.path.isdir(path):
    os.mkdir(path)


def euclid_dist(x, y):
    return (x[:, None, :] - y[None, :, :]).norm(p=2, dim=2)


def prepare_initialisation(dataset_p, dataset_u, n_pos, n_unl, prior, nb_try):
    init_plan = torch.zeros([nb_try, n_pos, n_unl])
    for i in range(nb_try):
        # Draw dataset
        P, U, _ = utils.draw_p_u_dataset_scar(dataset_p, dataset_u, n_pos, n_unl, prior, seed_nb=i)
        Ctot, C1, C2, mu, nu = compute_cost_matrices(P, U, prior, nb_dummies=10)
        # Compute init
        init_plan[i] = torch.tensor(ot.emd(mu, nu, Ctot)[:n_pos, :])
    return init_plan


def compute_plan_ugw(dataset_p, dataset_u, n_pos, n_unl, prior, eps, rho, rho2, nb_try, solver, device=0):
    # Set default type and GPU device
    torch.cuda.set_device(device)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Draw cost for all seeds as batch
    Cx, Cy = torch.zeros([nb_try, n_pos, n_pos]), torch.zeros([nb_try, n_unl, n_unl])
    for i in range(nb_try):
        P, U, y_u = utils.draw_p_u_dataset_scar(dataset_p, dataset_u, n_pos, n_unl, prior, seed_nb=i)  # seed=i
        P, U = torch.tensor(P.values, dtype=torch.float), torch.tensor(U.values, dtype=torch.float)
        cx, cy = euclid_dist(P, P), euclid_dist(U, U)
        Cx[i], Cy[i] = cx, cy
    del cx, cy

    # Compute init and weights
    mu = (torch.ones([n_pos]) / n_pos).expand(nb_try, -1)
    nu = (torch.ones([n_unl]) / n_unl).expand(nb_try, -1)
    if P.shape[1] == U.shape[1]:
        init_plan = prepare_initialisation(dataset_p, dataset_u, n_pos, n_unl, prior, nb_try)
    else:
        solv_init = BatchLowerBoundSolver(nits_sinkhorn=50000, tol_sinkhorn=1e-5, eps=eps, rho=rho, rho2=rho2)
        _, _, init_plan = solv_init.compute_plan(mu, Cx, nu, Cy, exp_form=False)

    # Compute the marginal and save as file
    pi_numpy = init_plan.sum(dim=1).cpu().data.numpy()
    fname = f'/ugw_init_{dataset_p}_{n_pos}_{dataset_u}_{n_unl}_prior{prior}_eps{eps}_rho{rho}_rho{rho2}_reps{nb_try}.npy'
    np.save(path + fname, pi_numpy)

    # Set params and start the grid wrt entropic param eps
    solver.set_rho(rho, rho2)
    solver.set_eps(eps)
    pi = solver.ugw_sinkhorn(mu, Cx, nu, Cy, init=init_plan)
    if torch.any(torch.isnan(pi)):
        raise Exception(f"Solver got NaN plan with params (eps, rho) = "
                        f"{dataset_p, dataset_u, nb_try, solver.get_eps(), solver.get_rho()}")

    # Compute the marginal and save as file
    pi_numpy = pi.sum(dim=1).cpu().data.numpy()
    fname = f'/ugw_plan_{dataset_p}_{n_pos}_{dataset_u}_{n_unl}_prior{prior}_eps{eps}_rho{rho}_rho{rho2}_reps{nb_try}.npy'
    np.save(path + fname, pi_numpy)

    print(f"DONE = Dataset {dataset_p, dataset_u}, eps = {eps}, rho = {rho, rho2} , reps = {nb_try}")
    return


if __name__ == '__main__':
    parallel_gpu = True

    grid_eps = [2. ** k for k in range(-9, -8, 1)] # Set to 2**-9 but an be optimized via grid-search
    grid_rho = [2. ** k for k in range(-10, -4, 1)]
    nb_try = 40

    solver = BatchStableSinkhornSolver(nits_plan=3000, nits_sinkhorn=3000, tol_plan=1e-5, tol_sinkhorn=1e-6)

    # List all tasks for the Caltech datasets
    list_tasks = []
    # # Matching similar features - prior set to 10%
    n_pos, n_unl, prior = 100, 100, 0.1
    list_surf = ['surf_Caltech', 'surf_amazon', 'surf_webcam', 'surf_dslr']
    list_decaf = ['decaf_caltech', 'decaf_amazon', 'decaf_webcam', 'decaf_dslr']
    list_data = [('surf_Caltech', d) for d in list_surf] + [('decaf_caltech', d) for d in list_decaf]
    list_tasks = list_tasks + [(data_pos, data_unl, n_pos, n_unl, prior, eps, rho, rho2, nb_try)
                               for (data_pos, data_unl) in list_data for eps in grid_eps
                               for rho in grid_rho for rho2 in grid_rho]

    # # Matching similar features - prior set to 20%
    n_pos, n_unl, prior = 100, 100, 0.2
    list_surf = ['surf_Caltech', 'surf_amazon', 'surf_webcam']
    list_decaf = ['decaf_caltech', 'decaf_amazon', 'decaf_webcam']
    list_data = [('surf_Caltech', d) for d in list_surf] + [('decaf_caltech', d) for d in list_decaf]
    list_tasks = list_tasks + [(data_pos, data_unl, n_pos, n_unl, prior, eps, rho, rho2, nb_try)
                               for (data_pos, data_unl) in list_data for eps in grid_eps
                               for rho in grid_rho for rho2 in grid_rho]

    # Matching different features - prior set to 10%
    n_pos, n_unl, prior = 100, 100, 0.1
    list_surf = ['surf_Caltech', 'surf_amazon', 'surf_webcam', 'surf_dslr']
    list_decaf = ['decaf_caltech', 'decaf_amazon', 'decaf_webcam', 'decaf_dslr']
    list_data = [('surf_Caltech', d) for d in list_decaf] + [('decaf_caltech', d) for d in list_surf]
    list_tasks = list_tasks + [(data_pos, data_unl, n_pos, n_unl, prior, eps, rho, rho2, nb_try)
                               for (data_pos, data_unl) in list_data for eps in grid_eps
                               for rho in grid_rho for rho2 in grid_rho]

    # # Matching different features - prior set to 20%
    n_pos, n_unl, prior = 100, 100, 0.2
    list_surf = ['surf_Caltech', 'surf_amazon', 'surf_webcam']
    list_decaf = ['decaf_caltech', 'decaf_amazon', 'decaf_webcam']
    list_data = [('surf_Caltech', d) for d in list_decaf] + [('decaf_caltech', d) for d in list_surf]
    list_tasks = list_tasks + [(data_pos, data_unl, n_pos, n_unl, prior, eps, rho, rho2, nb_try)
                               for (data_pos, data_unl) in list_data for eps in grid_eps
                               for rho in grid_rho for rho2 in grid_rho]

    if parallel_gpu:
        assert torch.cuda.is_available()
        list_device = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        total_devices = torch.cuda.device_count()
        print(f"Parallel computation // Total GPUs available = {total_devices}")
        pll = Parallel(n_jobs=total_devices)
        iterator = (delayed(compute_plan_ugw)(data_pos, data_unl, n_pos, n_unl, prior,
                                              eps, rho, rho2, nb_try, solver, device=list_device[k % total_devices])
                    for k, (data_pos, data_unl, n_pos, n_unl, prior, eps, rho, rho2, nb_try) in enumerate(list_tasks))
        pll(iterator)

    else:
        print("Not Parallel")
        for (data_pos, data_unl, n_pos, n_unl, prior, eps, rho, rho2, nb_try) in list_tasks:
            compute_plan_ugw(data_pos, data_unl, n_pos, n_unl, prior, eps, rho, rho2, nb_try, solver)
            print(f'{data_pos, data_unl} done.')
