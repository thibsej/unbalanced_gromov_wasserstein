import torch
from solver.vanilla_sinkhorn_solver import VanillaSinkhornSolver
from solver.utils import generate_measure
torch.set_printoptions(8)
import itertools
import pylab as pl
## Helper that construct a plan from a permutation.
def create_plan(permutation):
    n = len(permutation)
    pi = torch.zeros(n,n)
    for i in range(n):
        pi[i,permutation[i]] = 1.
    return pi




# Set up a solver for KL-(U)GW
# Set rho=None to run balanced GW computation
solv = VanillaSinkhornSolver(nits_plan=1000, nits_sinkhorn=1000, gradient=False, tol_plan=1e-5, tol_sinkhorn=1e-5,
                             eps=0.1, rho=None)

n_sample = 3
# Generate two mm-spaces with euclidean metrics
a, Cx, x_a = generate_measure(n_batch=1, n_sample=n_sample, n_dim=2,equal=True)
b, Cy, x_b = generate_measure(n_batch=1, n_sample=n_sample, n_dim=2,equal=True)
a, b, Cx, Cy = a[0], b[0], Cx[0], Cy[0] # Reduce the first axis to keep one space (method generates batches)

# Compute the bi-convex relaxation of the UGW problem





# Compute the loss and check the biconvex relaxation
pi, gamma = solv.alternate_sinkhorn(a, Cx, b, Cy)
cost = solv.ugw_cost(pi, gamma, a, Cx, b, Cy)
print("GW cost of the biconvex relaxation: ", cost)
cost_pi = solv.ugw_cost(pi, pi, a, Cx, b, Cy)
cost_gamma = solv.ugw_cost(gamma, gamma, a, Cx, b, Cy)
cost_gamma_sharp = solv.ugw_cost(gamma, gamma,a, Cx,b, Cy)
print("GW cost with twice the same inputs for pi / gamma: ", (cost_pi, cost_gamma))
print("sharp cost: ",solv.l2_distortion(pi, pi, Cx, Cy))
temp = 1000000
for perm in itertools.permutations(range(n_sample),n_sample):
    pi_ = create_plan(perm)
    cost = solv.l2_distortion(pi_, pi_, Cx, Cy)
    #print(cost)
    if cost < temp:
        temp = cost
        plan = pi_

print("best cost on permutations: ",temp)

print(plan)
print(pi)
pl.plot(x_a[0,:,0],x_a[0,:,1],"bo")
pl.plot(x_b[0,:,0],x_b[0,:,1],"ro")
pl.show()