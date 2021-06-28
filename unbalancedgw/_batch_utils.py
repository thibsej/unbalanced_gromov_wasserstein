import torch


def quad_kl_div(pi, gamma, ref):
    """Compute the quadratic entropy (KL^otimes(pi otimes gamma | ref))
    with full plans

    Parameters
    ----------
    pi: first input, torch.Tensor of size [Batch, size_X, size_Y]

    gamma: second input torch.Tensor of size [Batch, size_X, size_Y]

    ref: Reference of the KL entropy to compare with (pi otimes gamma)

    Returns
    -------
    div: torch.Tensor of size [Batch]
    Quadratic Kl divergence between each batch.
    """
    massp, massg = pi.sum(dim=(1, 2)), gamma.sum(dim=(1, 2))
    div = (
        massg * torch.sum(pi * (pi / ref + 1e-10).log(), dim=(1, 2))
        + massp * torch.sum(gamma * (gamma / ref + 1e-10).log(), dim=(1, 2))
        - massp * massg
        + ref.sum(dim=(1, 2)) ** 2
    )
    return div


def quad_kl_div_marg(pi, gamma, ref):
    """Compute the quadratic entropy (KL^otimes(pi otimes gamma | ref))
    with plans marginals.

    Parameters
    ----------
    pi: torch.Tensor of size [Batch, size_X]
    First input of the tensorized KL.

    gamma: torch.Tensor of size [Batch, size_X]
    Second input of the tensorized KL.

    ref: torch.Tensor of size [Batch, size_X]
    Reference of the KL entropy to compare with (pi, gamma)

    Returns
    -------
    div: torch.Tensor of size [Batch]
    Quadratic Kl divergence between the marginals.

    """
    massp, massg = pi.sum(dim=1), gamma.sum(dim=1)
    div = (
        massg * torch.sum(pi * (pi / ref + 1e-10).log(), dim=1)
        + massp * torch.sum(gamma * (gamma / ref + 1e-10).log(), dim=1)
        - massp * massg
        + ref.sum(dim=1) ** 2
    )
    return div


def l2_distortion(pi, gamma, dx, dy):
    """Computes the L2 distortion (int |C_X - C_Y|^2 pi gamma)

    Parameters
    ----------
    pi: torch.Tensor of size [Batch, size_X, size_Y]
    First input plan to integrate against distortion.

    gamma: torch.Tensor of size [Batch, size_X, size_Y]
    Second input plan to integrate against distortion.

    dx: torch.Tensor of size [Batch, size_X, size_X]
    Metric of the first mm-space.

    dy: torch.Tensor of size [Batch, size_Y, size_Y]
    Metric of the second mm-space.

    Returns
    ----------
    distortion: torch.float of size [Batch]
    L2 distortion between the metric integrated against the plans
    """
    distxx = torch.einsum(
        "ijk,ij,ik->i", dx ** 2, pi.sum(dim=2), gamma.sum(dim=2)
    )
    distyy = torch.einsum(
        "ijk,ij,ik->i", dy ** 2, pi.sum(dim=1), gamma.sum(dim=1)
    )
    distxy = torch.sum(
        torch.einsum("kij,kjl->kil", dx, pi)
        * torch.einsum("kij,kjl->kil", gamma, dy),
        dim=(1, 2),
    )
    distortion = distxx + distyy - 2 * distxy
    return distortion


def ugw_cost(pi, gamma, a, dx, b, dy, eps, rho, rho2):
    """Computes the full (U)GW functional with entropic term and KL
    penalty of marginals.

    Parameters
    ----------
    pi: torch.Tensor of size [Batch, size_X, size_Y]
    First input transport plan.

    gamma: torch.Tensor of size [Batch, size_X, size_Y]
    Second input transport plan.

    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    dx: torch.Tensor of size [Batch, size_X, size_X]
    Input metric of the first mm-space.

    b: torch.Tensor of sier [Batch, size_Y]
    Input measure of the second mm-space.

    dy: torch.Tensor of size [Batch, size_Y, size_Y]
    Input metric of the second mm-space.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    Returns
    ----------
    cost: torch.float of size [Batch]
    Value of the UGW functionnal for each batch of input.
    """
    cost = l2_distortion(pi, gamma, dx, dy) + eps * quad_kl_div(
        pi, gamma, a[:, :, None] * b[:, None, :]
    )
    if rho < float("Inf"):
        cost = cost + rho * quad_kl_div_marg(
            torch.sum(pi, dim=2), torch.sum(gamma, dim=2), a
        )
    if rho2 < float("Inf"):
        cost = cost + rho2 * quad_kl_div_marg(
            torch.sum(pi, dim=1), torch.sum(gamma, dim=1), b
        )
    return cost


def init_plan(a, b, init=None):
    """Initialize the plan if None is given, otherwise use the input plan

    Parameters
    ----------
    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    init: torch.Tensor of size [Batch, size_X, size_Y]
    Initializes the plan. If None defaults to tensor plan.

    Returns
    ----------
    init: torch.Tensor of size [Batch, size_X, size_Y]
    Initialization of the plan to start running Sinkhorn-UGW.
    """
    if init is not None:
        return init
    else:
        return (
            a[:, :, None]
            * b[:, None, :]
            / (a.sum(dim=1) * b.sum(dim=1)).sqrt()[:, None, None]
        )


def compute_local_cost(pi, a, dx, b, dy, eps, rho, rho2, complete_cost=True):
    """Compute the local cost by averaging the distortion with the current
    transport plan.

    Parameters
    ----------
    pi: torch.Tensor of size [Batch, size_X, size_Y]
    transport plan used to compute local cost

    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    dx: torch.Tensor of size [Batch, size_X, size_X]
    Input metric of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    dy: torch.Tensor of size [Batch, size_Y, size_Y]
    Input metric of the second mm-space.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    complete_cost: bool
    If set to True, computes the full local cost, otherwise it computes the
    cross-part on (X,Y) to reduce computational complexity.

    Returns
    ----------
    lcost: torch.Tensor of size [Batch, size_X, size_Y]
    local cost depending on the current transport plan.
    """
    distxy = torch.einsum(
        "bij,bkj->bik", dx, torch.einsum("bkl,bjl->bkj", dy, pi)
    )
    kl_pi = torch.sum(
        pi * (pi / (a[:, :, None] * b[:, None, :]) + 1e-10).log(), dim=(1, 2),
    )
    if not complete_cost:
        return -2 * distxy + eps * kl_pi[:, None, None]

    mu, nu = torch.sum(pi, dim=2), torch.sum(pi, dim=1)
    distxx = torch.einsum("bij,bj->bi", dx ** 2, mu)
    distyy = torch.einsum("bkl,bl->bk", dy ** 2, nu)

    lcost = (
        distxx[:, :, None] + distyy[:, None, :] - 2 * distxy
    ) + eps * kl_pi[:, None, None]
    if rho < float("Inf"):
        lcost = (
            lcost
            + rho
            * torch.sum(mu * (mu / a + 1e-10).log(), dim=1)[:, None, None]
        )
    if rho2 < float("Inf"):
        lcost = (
            lcost
            + rho2
            * torch.sum(nu * (nu / b + 1e-10).log(), dim=1)[:, None, None]
        )
    return lcost


def kl_prox_softmin(ecost, a, b, eps, rho, rho2):
    """Prepares functions which perform updates of the Sikhorn algorithm
    in exponential scale.

    Parameters
    ----------
    ecost: torch.Tensor of size [Batch, size_X, size_Y]
    Exponential of the cost. Kernel of Sinkhorn operator.

    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    Returns
    ----------
    s_x: callable function
    Map outputing updates of potential from Y to X.

    s_y: callable function
    Map outputing updates of potential from X to Y.
    """

    tau = 1.0 / (1.0 + eps / rho)
    tau2 = 1.0 / (1.0 + eps / rho2)

    def s_y(v):
        return torch.einsum("bij,bj->bi", ecost, b * v) ** (-tau2)

    def s_x(u):
        return torch.einsum("bij,bi->bj", ecost, a * u) ** (-tau)

    return s_x, s_y


def aprox_softmin(cost, a, b, mass, eps, rho, rho2):
    """Prepares functions which perform updates of the Sikhorn algorithm
    in logarithmic scale.

    Parameters
    ----------
    cost: torch.Tensor of size [Batch, size_X, size_Y]
    cost used in Sinkhorn iterations.

    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    mass: torch.Tensor of size [Batch]
    Mass of the current plan.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    Returns
    ----------
    s_x: callable function
    Map outputing updates of potential from Y to X.

    s_y: callable function
    Map outputing updates of potential from X to Y.
    """

    tau = 1.0 / (1.0 + eps / rho)
    tau2 = 1.0 / (1.0 + eps / rho2)

    def s_y(g):
        return (
            -mass[:, None]
            * tau2
            * eps
            * (
                (g / (mass[:, None] * eps) + b.log())[:, None, :]
                - cost / (mass[:, None, None] * eps)
            ).logsumexp(dim=2)
        )

    def s_x(f):
        return (
            -mass[:, None]
            * tau
            * eps
            * (
                (f / (mass[:, None] * eps) + a.log())[:, :, None]
                - cost / (mass[:, None, None] * eps)
            ).logsumexp(dim=1)
        )

    return s_x, s_y


def optimize_mass(lcost, logpi, a, b, eps, rho, rho2):
    """
    Given a plan and its associated local cost, the method computes the
    optimal mass of the plan. It should be a more stable estimate of the
    mass than the mass of the plan it

    Parameters
    ----------
    lcost: torch.Tensor of size [Batch, size_X, size_Y]
    Local cost depending on the current plan.

    logpi: torch.Tensor of size [Batch, size_X, size_Y]
    Optimized plan in log-scale.

    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    Returns
    ----------
    const: torch.Tensor of size [Batch]
    Optimal constant to translate the plan in log scale.
    """
    ma, mb = a.sum(dim=1), b.sum(dim=1)
    logmu, lognu = logpi.logsumexp(dim=2), logpi.logsumexp(dim=1)
    mtot = rho * ma ** 2 + rho2 * mb ** 2 + eps * (ma * mb) ** 2
    const = (
        (lcost * logpi.exp()).sum(dim=(1, 2))
        + 2 * ma * rho * (a * (logmu - a.log())).sum(dim=1)
        + 2 * mb * rho2 * (b * (lognu - b.log())).sum(dim=1)
        + 2
        * ma
        * mb
        * eps
        * (
            a[:, :, None]
            * b[:, None, :]
            * (logpi - a.log()[:, :, None] - b.log()[:, None, :])
        ).sum(dim=(1, 2))
    )
    return -const / mtot


def log_translate_potential(u, v, lcost, a, b, mass, eps, rho, rho2):
    """Updates the dual potential by computing the optimal constant
    translation. It stabilizes and accelerates computations in sinkhorn
    loop.

    Parameters
    ----------
    u: torch.Tensor of size [Batch, size_X]
    First dual potential defined on X.

    v: torch.Tensor of size [Batch, size_Y]
    Second dual potential defined on Y.

    lcost: torch.Tensor of size [Batch, size_X, size_Y]
    Local cost depending on the current plan.

    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    mass: torch.Tensor of size [Batch]
    Mass of the current plan.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    Returns
    ----------
    u: torch.Tensor of size [Batch, size_X]
    First dual potential defined on X.

    v: torch.Tensor of size [Batch, size_Y]
    Second dual potential defined on Y.
    """
    c1 = (
        -torch.cat((u, v), 1) / (mass[:, None] * rho)
        + torch.cat((a, b), 1).log()
    ).logsumexp(dim=1) - torch.log(2 * torch.ones([1]))
    c2 = (
        (
            a.log()[:, :, None]
            + b.log()[:, None, :]
            + (
                (u[:, :, None] + v[:, None, :] - lcost)
                / (mass[:, None, None] * eps)
            )
        )
        .logsumexp(dim=2)
        .logsumexp(dim=1)
    )
    z = (0.5 * mass * eps) / (2.0 + 0.5 * (eps / rho) + 0.5 * (eps / rho2))
    k = z * (c1 - c2)
    return u + k[:, None], v + k[:, None]


def log_sinkhorn(
    lcost, f, g, a, b, mass, eps, rho, rho2, nits_sinkhorn, tol_sinkhorn
):
    """
    Parameters
    ----------
    lcost: torch.Tensor of size [Batch, size_X, size_Y]
    Local cost depending on the current plan.

    f: torch.Tensor of size [Batch, size_X]
    First dual potential defined on X.

    g: torch.Tensor of size [Batch, size_Y]
    Second dual potential defined on Y.

    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    mass: torch.Tensor of size [Batch]
    Mass of the current plan.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    nits_sinkhorn: int
    Maximum number of iterations to update Sinkhorn potentials in inner loop.

    tol_sinkhorn: float
    Tolerance on convergence of Sinkhorn potentials.

    Returns
    ----------
    u: torch.Tensor of size [Batch, size_X]
    First dual potential of Sinkhorn algorithm

    v: torch.Tensor of size [Batch, size_Y]
    Second dual potential of Sinkhorn algorithm

    logpi: torch.Tensor of size [Batch, size_X, size_Y]
    Optimal transport plan in log-space.
    """
    # Initialize potentials by finding best translation
    if f is None or g is None:
        f, g = torch.zeros_like(a), torch.zeros_like(b)
    f, g = log_translate_potential(f, g, lcost, a, b, mass, eps, rho, rho2)

    # perform Sinkhorn algorithm in LSE form
    s_x, s_y = aprox_softmin(lcost, a, b, mass, eps, rho, rho2)
    for j in range(nits_sinkhorn):
        f_prev = f.clone()
        g = s_x(f)
        f = s_y(g)
        if (f - f_prev).abs().max().item() < tol_sinkhorn:
            break
    logpi = (
        ((f[:, :, None] + g[:, None, :] - lcost) / (mass[:, None, None] * eps))
        + a.log()[:, :, None]
        + b.log()[:, None, :]
    )
    return f, g, logpi


def exp_translate_potential(u, v, ecost, a, b, mass, eps, rho, rho2):
    """Updates the dual potential by computing the optimal constant
    translation. It stabilizes and accelerates computations in sinkhorn
    loop.

    Parameters
    ----------
    u: torch.Tensor of size [Batch, size_X]
    First dual potential defined on X.

    v: torch.Tensor of size [Batch, size_Y]
    Second dual potential defined on Y.

    ecost: torch.Tensor of size [Batch, size_X, size_Y]
    Exponential kernel generated from the local cost.
    It depends on the current plan.

    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    mass: torch.Tensor of size [Batch]
    Mass of the current plan.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    Returns
    ----------
    u: torch.Tensor of size [Batch, size_X]
    First dual potential defined on X.

    v: torch.Tensor of size [Batch, size_Y]
    Second dual potential defined on Y.
    """
    k = (a * u ** (-eps / rho)).sum(dim=1)
    k = k + (b * v ** (-eps / rho)).sum(dim=1)
    k = k / (
        2
        * (
            u[:, :, None]
            * v[:, None, :]
            * ecost
            * a[:, :, None]
            * b[:, None, :]
        ).sum(dim=(1, 2))
    )
    z = (0.5 * mass * eps) / (2.0 + 0.5 * (eps / rho) + 0.5 * (eps / rho2))
    k = k ** z
    return u * k[:, None], v * k[:, None]


def exp_sinkhorn(
    ecost, u, v, a, b, mass, eps, rho, rho2, nits_sinkhorn, tol_sinkhorn
):
    """
    Parameters
    ----------
    ecost: torch.Tensor of size [Batch, size_X, size_Y]
    Exponential kernel generated from the local cost.
    It depends on the current plan.

    u: torch.Tensor of size [Batch, size_X]
    First dual potential defined on X.

    v: torch.Tensor of size [Batch, size_Y]
    Second dual potential defined on Y.

    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    mass: torch.Tensor of size [Batch]
    Mass of the current plan.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    nits_sinkhorn: int
    Maximum number of iterations to update Sinkhorn potentials in inner loop.

    tol_sinkhorn: float
    Tolerance on convergence of Sinkhorn potentials.

    Returns
    ----------
    u: torch.Tensor of size [Batch, size_X]
    First dual potential of Sinkhorn algorithm

    v: torch.Tensor of size [Batch, size_Y]
    Second dual potential of Sinkhorn algorithm

    logpi: torch.Tensor of size [Batch, size_X, size_Y]
    Optimal transport plan in log-space.
    """
    # Initialize potentials by finding best translation
    if u is None or v is None:
        u, v = torch.ones_like(a), torch.ones_like(b)
    u, v = exp_translate_potential(u, v, ecost, a, b, mass, eps, rho, rho2)

    # perform Sinkhorn algorithm in LSE form
    s_x, s_y = kl_prox_softmin(ecost, a, b, eps, rho, rho2)
    for j in range(nits_sinkhorn):
        u_prev = u.clone()
        v = s_x(u)
        u = s_y(v)
        if (u.log() - u_prev.log()).abs().max().item() * eps < tol_sinkhorn:
            break
    pi = u[:, :, None] * v[:, None, :] * ecost * a[:, :, None] * b[:, None, :]
    return u, v, pi


def compute_distance_histograms(a, dx, b, dy):
    """
    Computes the squared distance between histograms of distance of two
    mm-spaces. The histograms are obtained by normalising measures to be
    probabilities.

    Parameters
    ----------
    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    dx: torch.Tensor of size [Batch, size_X, size_X]
    Input metric of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    dy: torch.Tensor of size [Batch, size_X, size_X]
    Input metric of the first mm-space.

    Returns
    -------
    lcost: torch.Tensor of size [size_X, size_Y]
    distances between metric histograms
    """
    h_x = torch.einsum("bij, bj->bi", dx, a / a.sum())
    h_y = torch.einsum("bij, bj->bi", dy, b / b.sum())
    lcost = (h_x ** 2)[:, :, None] + (h_y ** 2)[:, None, :]
    lcost = lcost - 2 * h_x[:, :, None] * h_y[:, None, :]
    return lcost


def compute_batch_flb_plan(
    a, dx, b, dy, eps, rho, rho2, nits_sinkhorn, tol_sinkhorn
):
    """
    Computes the optimal plan associated to the First Lower Bound (FLB)
    defined in [Memoli 11'].It solved the Unbalanced OT problem between
    histograms of distance.

    Parameters
    ----------
    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    dx: torch.Tensor of size [Batch, size_X, size_X]
    Input metric of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    dy: torch.Tensor of size [Batch, size_X, size_X]
    Input metric of the first mm-space.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    nits_sinkhorn: int
    Maximum number of iterations to update Sinkhorn potentials in inner loop.

    tol_sinkhorn: float
    Tolerance on convergence of Sinkhorn potentials.

    Returns
    -------
    u: torch.Tensor of size [Batch, size_X]
    First dual potential of Sinkhorn algorithm

    v: torch.Tensor of size [Batch, size_Y]
    Second dual potential of Sinkhorn algorithm

    pi: torch.Tensor of size [Batch, size_X, size_Y]
    Optimal transport plan of the FLB problem.
    """
    lcost = compute_distance_histograms(a, dx, b, dy)
    u, v = torch.zeros_like(a), torch.zeros_like(b)
    u, v = log_translate_potential(u, v, lcost, a, b, 1.0, eps, rho, rho2)

    s_x, s_y = aprox_softmin(lcost, a, b, 1.0, eps, rho, rho2)
    for j in range(nits_sinkhorn):
        u_prev = u.clone()
        v = s_x(u)
        u = s_y(v)
        if (u - u_prev).abs().max().item() < tol_sinkhorn:
            break
    pi = ((u[:, :, None] + v[:, None, :] - lcost) / eps).exp()
    pi = pi * a[:, :, None] * b[:, None, :]
    return u, v, pi
