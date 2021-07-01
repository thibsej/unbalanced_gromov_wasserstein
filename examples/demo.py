"""
==================================
Using a basic example of unbalancedgw
==================================

This example shows how to use the basic functions and solver of UGW.

"""

import torch

from unbalancedgw.vanilla_ugw_solver import exp_ugw_sinkhorn
from unbalancedgw._vanilla_utils import ugw_cost
from unbalancedgw.utils import generate_measure

torch.set_printoptions(8)

# Set up a solver for KL-(U)GW
# Set rho=float("Inf") to run balanced GW computation
eps = 1.0
rho, rho2 = 1.0, 1.0

# Generate two mm-spaces with euclidean metrics
a, dx, _ = generate_measure(n_batch=1, n_sample=5, n_dim=3)
b, dy, _ = generate_measure(n_batch=1, n_sample=6, n_dim=2)
# Reduce the first axis to keep one space (method generates batches)
a, b, dx, dy = a[0], b[0], dx[0], dy[0]

# Compute the bi-convex relaxation of the UGW problem
pi, gamma = exp_ugw_sinkhorn(a, dx, b, dy, init=None, eps=eps,
                             rho=rho, rho2=rho2,
                             nits_plan=1000, tol_plan=1e-5,
                             nits_sinkhorn=1000, tol_sinkhorn=1e-5,
                             two_outputs=True)

# Compute the loss
cost = ugw_cost(pi, gamma, a, dx, b, dy, eps=eps, rho=rho, rho2=rho2)
print("Cost of the biconvex relaxation: ", cost)

# Check the bi-convex relaxations is tight
# (otherwise the output is a local minima)
cost_pi = ugw_cost(pi, pi, a, dx, b, dy, eps=eps, rho=rho, rho2=rho2)
cost_gamma = ugw_cost(gamma, gamma, a, dx, b, dy, eps=eps, rho=rho, rho2=rho2)
print("UGW cost with twice the same inputs for pi / gamma: ",
      (cost_pi, cost_gamma))

# Switch to solving Balanced-GW
rho, rho2 = float("Inf"), float("Inf")

# Compute the loss and check the biconvex relaxation
pi, gamma = exp_ugw_sinkhorn(a, dx, b, dy, init=None, eps=eps,
                             rho=rho, rho2=rho2,
                             nits_plan=1000, tol_plan=1e-5,
                             nits_sinkhorn=1000, tol_sinkhorn=1e-5,
                             two_outputs=True)
cost = ugw_cost(pi, gamma, a, dx, b, dy, eps=eps, rho=rho, rho2=rho2)
print("GW cost of the biconvex relaxation: ", cost)
cost_pi = ugw_cost(pi, pi, a, dx, b, dy, eps=eps, rho=rho, rho2=rho2)
cost_gamma = ugw_cost(gamma, gamma, a, dx, b, dy, eps=eps, rho=rho, rho2=rho2)
print("GW cost with twice the same inputs for pi / gamma: ",
      (cost_pi, cost_gamma))
