import pytest

import torch
from solver.utils import generate_measure
from solver.vanilla_sinkhorn_solver import VanillaSinkhornSolver
from solver.batch_sinkhorn_solver import BatchSinkhornSolver

solvb, solvv = BatchSinkhornSolver(), VanillaSinkhornSolver()
a, Cx, _ = generate_measure(n_batch=1, n_sample=5, n_dim=3)
b, Cy, _ = generate_measure(n_batch=1, n_sample=6, n_dim=2)


@pytest.mark.parametrize('eps,rho,rho2', [(1.0, float('Inf'), None), (1.0, 1.0, None),
                                          (1.0, float('Inf'), 1.0), (1.0, 1.0, float('Inf')),
                                          (1.0, 0.1, 1.0)])
def test_solver_run_without_exception(eps, rho, rho2):
    print("test")
    solvv.set_eps(eps)
    solvv.set_rho(rho)
    solvv.set_rho2(rho2)
    solvb.set_eps(eps)
    solvb.set_rho(rho)
    solvb.set_rho2(rho2)
    solvb.alternate_sinkhorn(a, Cx, b, Cy)
    solvb.ugw_sinkhorn(a, Cx, b, Cy)
    solvv.ugw_sinkhorn(a[0], Cx[0], b[0], Cy[0])
    solvv.alternate_sinkhorn(a[0], Cx[0], b[0], Cy[0])


@pytest.mark.parametrize('eps,rho,rho2', [(1.0, float('Inf'), None), (1.0, 1.0, None),
                                          (1.0, float('Inf'), 1.0), (1.0, 1.0, float('Inf')),
                                          (1.0, 0.1, 1.0)])
def test_consistency_both_solvers(eps, rho, rho2):
    print("test")
    solvv.set_eps(eps)
    solvv.set_rho(rho)
    solvv.set_rho2(rho2)
    solvb.set_eps(eps)
    solvb.set_rho(rho)
    solvb.set_rho2(rho2)
    pib, gab = solvb.alternate_sinkhorn(a, Cx, b, Cy)
    piv, gav = solvv.alternate_sinkhorn(a[0], Cx[0], b[0], Cy[0])
    assert torch.allclose(pib, piv, rtol=1e-5)
    assert torch.allclose(gab, gav, rtol=1e-5)
    pib = solvb.ugw_sinkhorn(a, Cx, b, Cy)
    piv = solvv.ugw_sinkhorn(a[0], Cx[0], b[0], Cy[0])
    assert torch.allclose(pib, piv, rtol=1e-5)

# TODO: Make test with twice the same mm-space
# TODO test with high entropy
# TODO tests where the optimum is close to the identity
