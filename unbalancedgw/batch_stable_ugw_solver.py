import torch
from ._batch_utils import init_plan, compute_local_cost, \
    log_sinkhorn, exp_sinkhorn


def log_batch_ugw_sinkhorn(a, dx, b, dy, init=None, eps=1.0,
                           rho=float("Inf"), rho2=None,
                           nits_plan=3000, tol_plan=1e-6,
                           nits_sinkhorn=3000, tol_sinkhorn=1e-6,
                           two_outputs=False):
    """Solves the regularized UGW problem, keeps only one plan as output.
    the algorithm is run as much as possible in log-scale.

    Parameters
    ----------
    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    dx: torch.Tensor of size [Batch, size_X, size_X]
    Input metric of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    dy: torch.Tensor of size [Batch, size_Y, size_Y]
    Input metric of the second mm-space.

    init: torch.Tensor of size [Batch, size_X, size_Y]
    Transport plan at initialization. Defaults to None and initializes
    with tensor plan.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    nits_plan: int
    Maximum number of iterations to update the plan pi.

    tol_plan: float
    Tolerance on convergence of plan.

    nits_sinkhorn: int
    Maximum number of iterations to update Sinkhorn potentials in inner loop.

    tol_sinkhorn: float
    Tolerance on convergence of Sinkhorn potentials.

    two_outputs: bool
    If set to True, outputs the two plans of the alternate minimization of UGW.

    Returns
    ----------
    pi: torch.Tensor of size [Batch, size_X, size_Y]
    Transport plan
     which is a stationary point of UGW. The output is not
    in log-scale.
    """
    if rho2 is None:
        rho2 = rho

    # Initialize plan and local cost
    logpi = (init_plan(a, b, init=init) + 1e-30).log()
    logpi_prev = torch.zeros_like(logpi)
    up, vp = None, None
    for i in range(nits_plan):
        logpi_prev = logpi.clone()
        lcost = compute_local_cost(logpi.exp(), a, dx, b, dy, eps, rho, rho2)
        logmp = logpi.logsumexp(dim=(1, 2))
        up, vp, logpi = log_sinkhorn(
            lcost, up, vp, a, b, logmp.exp() + 1e-10, eps, rho, rho2,
            nits_sinkhorn, tol_sinkhorn
        )
        if torch.any(torch.isnan(logpi)):
            raise Exception(
                f"Solver got NaN plan with params (eps, rho, rho2) "
                f" = {eps, rho, rho2}. Try increasing argument eps."
            )
        logpi = (
                0.5 * (logmp - logpi.logsumexp(dim=(1, 2)))[:, None, None]
                + logpi
        )
        if (logpi - logpi_prev).abs().max().item() < tol_plan:
            break

    if two_outputs:
        return logpi.exp(), logpi_prev.exp()
    return logpi.exp()


def exp_batch_ugw_sinkhorn(a, dx, b, dy, init=None, eps=1.0,
                           rho=float("Inf"), rho2=None,
                           nits_plan=3000, tol_plan=1e-6,
                           nits_sinkhorn=3000, tol_sinkhorn=1e-6,
                           two_outputs=False):
    """Solves the regularized UGW problem, keeps only one plan as output.
    the algorithm is run as much as possible in log-scale.

    Parameters
    ----------
    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    dx: torch.Tensor of size [Batch, size_X, size_X]
    Input metric of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    dy: torch.Tensor of size [Batch, size_Y, size_Y]
    Input metric of the second mm-space.

    init: torch.Tensor of size [Batch, size_X, size_Y]
    Transport plan at initialization. Defaults to None and initializes
    with tensor plan.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    nits_plan: int
    Maximum number of iterations to update the plan pi.

    tol_plan: float
    Tolerance on convergence of plan.

    nits_sinkhorn: int
    Maximum number of iterations to update Sinkhorn potentials in inner loop.

    tol_sinkhorn: float
    Tolerance on convergence of Sinkhorn potentials.

    two_outputs: bool
    If set to True, outputs the two plans of the alternate minimization of UGW.

    Returns
    ----------
    pi: torch.Tensor of size [Batch, size_X, size_Y]
    Transport plan
     which is a stationary point of UGW. The output is not
    in log-scale.
    """
    if rho2 is None:
        rho2 = rho

    # Initialize plan and local cost
    pi = init_plan(a, b, init=init)
    pi_prev = torch.zeros_like(pi)
    up, vp = None, None
    for i in range(nits_plan):
        pi_prev = pi.clone()
        mp = pi.sum(dim=(1, 2))
        ecost = (-compute_local_cost(pi, a, dx, b, dy, eps, rho, rho2) /
                 (eps * mp[:, None, None])).exp()
        up, vp, pi = exp_sinkhorn(
            ecost, up, vp, a, b, mp, eps, rho, rho2,
            nits_sinkhorn, tol_sinkhorn
        )
        if torch.any(torch.isnan(pi)):
            raise Exception(
                f"Solver got NaN plan with params (eps, rho, rho2) "
                f" = {eps, rho, rho2}. Try increasing argument eps or switch"
                f" to log_ugw_sinkhorn method."
            )
        pi = (mp / pi.sum(dim=(1, 2))).sqrt() * pi
        if (pi - pi_prev).abs().max().item() < tol_plan:
            break

    if two_outputs:
        return pi, pi_prev
    return pi
