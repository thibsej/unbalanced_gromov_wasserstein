import torch

from solver.vanilla_sinkhorn_solver import VanillaSinkhornSolver
from solver.utils import generate_measure
torch.set_printoptions(8)

# Set up a solver for (U)GW
# Set rho=None to run balanced GW computation
solv = VanillaSinkhornSolver(nits_plan=1000, nits_sinkhorn=1000, gradient=False, tol_plan=1e-5, tol_sinkhorn=1e-5,
                             eps=1.0, rho=None)

# Generate two mm-spaces with euclidean metrics
a, Cx, _ = generate_measure(n_batch=1, n_sample=5, n_dim=3)
b, Cy, _ = generate_measure(n_batch=1, n_sample=6, n_dim=2)
a, b, Cx, Cy = a[0], b[0], Cx[0], Cy[0] # Reduce the first axis to keep one space (method generates batches)

# Compute the bi-convex relaxation of the UGW problem
pi, gamma = solv.alternate_sinkhorn(a, Cx, b, Cy)

# Compute the loss
cost = solv.ugw_cost(pi, gamma, a, Cx, b, Cy)
print("Cost of the biconvex relaxation: ", cost)

# Check the bi-convex relaxations is tight (otherwise the output is a local minima)
cost_pi = solv.ugw_cost(pi, pi, a, Cx, b, Cy)
cost_gamma = solv.ugw_cost(gamma, gamma, a, Cx, b, Cy)
print("UGW cost with twice the same inputs for pi / gamma: ", (cost_pi, cost_gamma))
