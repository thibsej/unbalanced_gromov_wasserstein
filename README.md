# Unbalanced Gromov-Wasserstein divergence
This repository is the official implementation of 
in the paper 
[The Unbalanced Gromov Wasserstein Distance: Conic Formulation and Relaxation](https://arxiv.org/abs/2009.04266). 
It computes the approximation of the UGW divergence based on entropic regularization and the 
Sinkhorn algorithm. 
It allows to compare weighted point clouds equipped with a cost matrix, or graphs with weights at the node 
and distances on the edges. The implementation of the Gromov-Wasserstein distance (GW) is also available with 
this package.

If you use this work for your research, please cite the paper:

```
@article{sejourne2020unbalanced,
  title={The Unbalanced Gromov Wasserstein Distance: Conic Formulation and Relaxation},
  author={S{\'e}journ{\'e}, Thibault and Vialard, Fran{\c{c}}ois-Xavier and Peyr{\'e}, Gabriel},
  journal={arXiv preprint arXiv:2009.04266},
  year={2020}
}
```
## Requirements

The package is installable via pip. It relies on the NumPy and PyTorch packages, and the examples 
use matplotlib. To install the dependencies and the package, run the following command on your terminal:

```setup
pip install -r requirements.txt
pip install unbalancedgw
```

## Run the solver
You can check the file 
[demo.py](https://github.com/thibsej/unbalanced_gromov_wasserstein/blob/master/examples/demo.py)
for a simple example using the package. The principle is the following: first import the method.
```python
import torch
from unbalancedgw.vanilla_ugw_solver import exp_ugw_sinkhorn
from unbalancedgw._vanilla_utils import ugw_cost
from unbalancedgw.utils import generate_measure
```
Then you can set the parameters of the method (entropic regularization and strength of marginal penalty), 
and generate the data.
```python
eps = 1.0
rho, rho2 = 1.0, 1.0

# Generate two mm-spaces with euclidean metrics
a, dx, _ = generate_measure(n_batch=1, n_sample=5, n_dim=3)
b, dy, _ = generate_measure(n_batch=1, n_sample=6, n_dim=2)
a, b, dx, dy = a[0], b[0], dx[0], dy[0]
```
Eventually you can compute the optimal UGW transport plan, and compute its associated UGW cost.
```python
pi, gamma = exp_ugw_sinkhorn(a, dx, b, dy, init=None, eps=eps,
                             rho=rho, rho2=rho2,
                             nits_plan=1000, tol_plan=1e-5,
                             nits_sinkhorn=1000, tol_sinkhorn=1e-5,
                             two_outputs=True)
cost = ugw_cost(pi, gamma, a, dx, b, dy, eps=eps, rho=rho, rho2=rho2)
```
If you want to switch to Balanced-GW, you can set the parameters as
```python
eps = 1.0
rho, rho2 = float("Inf"), float("Inf")
```

## PU learning experiments

We propose in the paper to apply UGW to domain adaptation data in a PU learning setting. 
The unbalanced plan perform a partial matching of the data, which allows to predict which samples
should be in the same class as the source dataset.

The code is available in the folder 
[/experiments_pu](https://github.com/thibsej/unbalanced_gromov_wasserstein/tree/master/experiments_pu).
The code is only available on the repo and uses extra packages. To reproduce the experiments, run the package,
install the dependencies and go into the folder.
```train
git clone https://github.com/thibsej/unbalanced_gromov_wasserstein
cd unbalanced_gromov_wasserstein/experiments_pu
pip install -r requirements.txt
```

### Download the data
The data is available [here](http://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code)

### Compute the prediction and accuracies
To compute the prediction and convert the accuracies in a pandas dataframe, run:

```train
python compute_prediction.py
python compute_accuracy.py
```

### Observe the accuracy results
Then you can run the notebook 
[display_performance.ipynb](https://github.com/thibsej/unbalanced_gromov_wasserstein/blob/master/experiments_pu/display_performance.ipynb)
which displays the accuracy for all tasks.
The reproduction of the results from 
[Chapel et al.](https://arxiv.org/abs/2002.08276)
is available in display_results_pgw.ipynb.

## Contributing
The code is available under a MIT license.
