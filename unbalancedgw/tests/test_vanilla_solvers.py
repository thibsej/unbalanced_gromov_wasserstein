import pytest

import torch
from unbalancedgw.utils import generate_measure
from unbalancedgw.vanilla_ugw_solver import log_ugw_sinkhorn, exp_ugw_sinkhorn

torch.manual_seed(42)
a, dx, _ = generate_measure(n_batch=1, n_sample=5, n_dim=3)
b, dy, _ = generate_measure(n_batch=1, n_sample=6, n_dim=2)
a, dx, b, dy = a.squeeze(), dx.squeeze(), b.squeeze(), dy.squeeze()


@pytest.mark.parametrize('eps,rho,rho2', [(1.0, float('Inf'), None),
                                          (1.0, 1.0, None),
                                          (1.0, float('Inf'), 1.0),
                                          (1.0, 1.0, float('Inf')),
                                          (1.0, 0.1, 1.0)])
def test_solver_run_without_exception(eps, rho, rho2):
    print("test")
    log_ugw_sinkhorn(a, dx, b, dy, init=None, eps=eps,
                     rho=rho, rho2=rho2,
                     nits_plan=5, tol_plan=1e-6,
                     nits_sinkhorn=5, tol_sinkhorn=1e-6,
                     two_outputs=False)
    log_ugw_sinkhorn(a, dx, b, dy, init=None, eps=eps,
                     rho=rho, rho2=rho2,
                     nits_plan=5, tol_plan=1e-6,
                     nits_sinkhorn=5, tol_sinkhorn=1e-6,
                     two_outputs=True)
    exp_ugw_sinkhorn(a, dx, b, dy, init=None, eps=eps,
                     rho=rho, rho2=rho2,
                     nits_plan=5, tol_plan=1e-6,
                     nits_sinkhorn=5, tol_sinkhorn=1e-6,
                     two_outputs=False)
    exp_ugw_sinkhorn(a, dx, b, dy, init=None, eps=eps,
                     rho=rho, rho2=rho2,
                     nits_plan=5, tol_plan=1e-6,
                     nits_sinkhorn=5, tol_sinkhorn=1e-6,
                     two_outputs=True)


@pytest.mark.parametrize('eps,rho,rho2', [(10.0, float('Inf'), None),
                                          (10.0, 1.0, None),
                                          (10.0, float('Inf'), 1.0),
                                          (10.0, 1.0, float('Inf')),
                                          (10.0, 0.1, 1.0)])
def test_consistency_both_solvers(eps, rho, rho2):
    print("test")
    # assert torch.allclose(pib, piv, rtol=1e-5)
    # assert torch.allclose(gab, gav, rtol=1e-5)
    # assert torch.allclose(pib, piv, rtol=1e-5)

# TODO: Make test with twice the same mm-space
# TODO test with high entropy
# TODO tests where the optimum is close to the identity
