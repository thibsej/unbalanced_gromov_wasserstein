"""
=================================
Using UGW to retrieve permutation
=================================

This example shows UGW retrieves permutation of pointcoulds, up to the
entropic approximation.

"""

import torch
import itertools
import pylab as pl

from unbalancedgw.vanilla_ugw_solver import log_ugw_sinkhorn
from unbalancedgw._vanilla_utils import l2_distortion, ugw_cost
from solver.utils import generate_measure

torch.set_printoptions(8)
torch.manual_seed(14)


# Helper that construct a plan from a permutation.
def create_plan(permutation):
    n = len(permutation)
    pi = torch.zeros(n, n)
    for i in range(n):
        pi[i, permutation[i]] = 1.
    return pi


n_sample = 8
# Generate two mm-spaces with euclidean metrics
a, dx, x_a = generate_measure(n_batch=1, n_sample=n_sample, n_dim=2,
                              equal=True)
b, dy, x_b = generate_measure(n_batch=1, n_sample=n_sample, n_dim=2,
                              equal=True)
a, b, dx, dy = a[0], b[0], dx[0], dy[0]
eps, rho, rho2 = 0.001, float("Inf"), float("Inf")

# Compute the loss and check the biconvex relaxation
pi, gamma = log_ugw_sinkhorn(a, dx, b, dy, init=None, eps=eps,
                             rho=rho, rho2=rho2,
                             nits_plan=1000, tol_plan=1e-5,
                             nits_sinkhorn=1000, tol_sinkhorn=1e-5,
                             two_outputs=True)
cost = ugw_cost(pi, gamma, a, dx, b, dy, eps=eps, rho=rho, rho2=rho2)
print("GW cost of the biconvex relaxation: ", cost)
print("sharp cost: ", l2_distortion(pi, pi, dx, dy))
temp = 1. * n_sample
plan = torch.eye(n_sample)
for perm in itertools.permutations(range(n_sample), n_sample):
    pi_ = create_plan(perm)
    cost = l2_distortion(pi_, pi_, dx, dy)
    if cost < temp:
        temp = cost
        plan = pi_

print("best cost on permutations: ", temp)

print(plan)
print(pi)
# pl.plot(x_a[0,:,0],x_a[0,:,1],"bo")
# pl.plot(x_b[0,:,0],x_b[0,:,1],"ro")
pl.imshow(plan)
pl.show()
pl.imshow(pi)
pl.show()
