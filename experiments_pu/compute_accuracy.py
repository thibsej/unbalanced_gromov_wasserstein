import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score

from utils import draw_p_u_dataset_scar

path = os.getcwd() + "/saved_plans/marginals_without_rescaling"

if __name__ == '__main__':
    print('start')
    grid_eps = [2. ** k for k in range(-9, -8, 1)]
    grid_rho = [2. ** k for k in range(-10, -4, 1)]
    grid_params = [(eps, rho, rho2) for eps in grid_eps
                   for rho in grid_rho for rho2 in grid_rho]
    nb_try = 40

    list_tasks = []
    # # List tasks for Caltech datasets - prior set to 10%
    n_pos, n_unl, prior = 100, 100, 0.1
    list_surf = ['surf_Caltech', 'surf_amazon', 'surf_webcam', 'surf_dslr']
    list_decaf = ['decaf_caltech', 'decaf_amazon',
                  'decaf_webcam', 'decaf_dslr']
    list_data = [('surf_Caltech', d) for d in list_surf] + \
                [('decaf_caltech', d) for d in list_decaf]
    list_tasks = list_tasks + [
        (data_pos, data_unl, n_pos, n_unl, prior, nb_try)
        for (data_pos, data_unl) in list_data]

    # # List tasks for Caltech datasets - prior set to 20%
    n_pos, n_unl, prior = 100, 100, 0.2
    list_surf = ['surf_Caltech', 'surf_amazon', 'surf_webcam']
    list_decaf = ['decaf_caltech', 'decaf_amazon', 'decaf_webcam']
    list_data = [('surf_Caltech', d) for d in list_surf] + \
                [('decaf_caltech', d) for d in list_decaf]
    list_tasks = list_tasks + \
                 [(data_pos, data_unl, n_pos, n_unl, prior, nb_try)
                  for (data_pos, data_unl) in list_data]

    # List tasks for Caltech datasets - prior set to 10%
    n_pos, n_unl, prior = 100, 100, 0.1
    list_surf = ['surf_Caltech', 'surf_amazon', 'surf_webcam', 'surf_dslr']
    list_decaf = ['decaf_caltech', 'decaf_amazon', 'decaf_webcam',
                  'decaf_dslr']
    list_data = [('surf_Caltech', d) for d in list_decaf] + [
        ('decaf_caltech', d) for d in list_surf]
    list_tasks = list_tasks + [
        (data_pos, data_unl, n_pos, n_unl, prior, nb_try)
        for (data_pos, data_unl) in list_data]

    # # List tasks for Caltech datasets - prior set to 20%
    n_pos, n_unl, prior = 100, 100, 0.2
    list_surf = ['surf_Caltech', 'surf_amazon', 'surf_webcam']
    list_decaf = ['decaf_caltech', 'decaf_amazon', 'decaf_webcam']
    list_data = [('surf_Caltech', d) for d in list_decaf] + [
        ('decaf_caltech', d) for d in list_surf]
    list_tasks = list_tasks + [
        (data_pos, data_unl, n_pos, n_unl, prior, nb_try)
        for (data_pos, data_unl) in list_data]

    colnames = ['data_p', 'data_u', 'prior', 'eps', 'rho', 'rho2'] + \
               [str(i) for i in range(nb_try)]

    for (data_pos, data_unl, n_pos, n_unl, prior, nb_try) in list_tasks:
        df = pd.DataFrame(columns=colnames)
        for (eps, rho, rho2) in grid_params:
            # Load Data
            fname = f'/ugw_plan_{data_pos}_{n_pos}_{data_unl}_{n_unl}_' \
                    f'prior{prior}_eps{eps}_rho{rho}_rho{rho2}_' \
                    f'reps{nb_try}.npy'

            if not os.path.isfile(path + fname):
                print('skipped')
                continue
            print(f'treat {data_pos, data_unl, prior, eps, rho, rho2}')
            pi = np.load(path + fname)

            row = []
            row.append(data_pos)
            row.append(data_unl)
            row.append(prior)
            row.append(eps)
            row.append(rho)
            row.append(rho2)

            for i in range(nb_try):
                _, _, y_u = draw_p_u_dataset_scar(data_pos, data_unl, n_pos,
                                                  n_unl, prior, i)
                # Build prediction
                nu = pi[i]
                q = np.quantile(nu, 1 - prior)
                y_hat = nu > q
                row.append(accuracy_score(y_u, y_hat))

            df.loc[len(df)] = row

        # Save dataframe once processed
        df.to_csv(path + f'/perf_{data_pos}_{n_pos}_{data_unl}_{n_unl}_'
                         f'prior{prior}.csv')
        del df
    print('end')
