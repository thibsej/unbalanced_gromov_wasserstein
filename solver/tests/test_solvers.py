import pytest

import torch
from solver.utils import generate_measure
from solver.vanilla_sinkhorn_solver import VanillaSinkhornSolver
from solver.batch_sinkhorn_solver import BatchSinkhornSolver

solvb, solvv = BatchSinkhornSolver(), VanillaSinkhornSolver()
a, Cx, _ = generate_measure(n_batch=1, n_sample=5, n_dim=3)
b, Cy, _ = generate_measure(n_batch=1, n_sample=6, n_dim=2)

@pytest.mark.parametrize('eps,rho', [(1.0, None), (1.0, 1.0)])
def test_solver_run_without_exception(eps, rho):
    print("test")
    solvb.eps, solvv.eps = eps, eps
    solvb.rho, solvv.rho = rho, rho
    solvb.alternate_sinkhorn(a, Cx, b, Cy)
    solvb.ugw_sinkhorn(a, Cx, b, Cy)
    solvv.ugw_sinkhorn(a[0], Cx[0], b[0], Cy[0])
    solvv.alternate_sinkhorn(a[0], Cx[0], b[0], Cy[0])

@pytest.mark.parametrize('eps,rho', [(1.0, None), (1.0, 1.0), (0.1, None), (0.1, 0.1)])
def test_consistency_both_solvers(eps, rho):
    print("test")
    solvb.eps, solvv.eps = eps, eps
    solvb.rho, solvv.rho = rho, rho
    pib, gab = solvb.alternate_sinkhorn(a, Cx, b, Cy)
    piv, gav = solvv.alternate_sinkhorn(a[0], Cx[0], b[0], Cy[0])
    assert torch.allclose(pib, piv, rtol=1e-5)
    assert torch.allclose(gab, gav, rtol=1e-5)
    pib = solvb.ugw_sinkhorn(a, Cx, b, Cy)
    piv = solvv.ugw_sinkhorn(a[0], Cx[0], b[0], Cy[0])
    assert torch.allclose(pib, piv, rtol=1e-5)

