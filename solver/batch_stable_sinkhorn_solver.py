import torch


class BatchStableSinkhornSolver(object):
    def __init__(
        self,
        nits_plan=3000,
        nits_sinkhorn=3000,
        gradient=False,
        tol_plan=1e-7,
        tol_sinkhorn=1e-7,
        eps=1.0,
        rho=float("Inf"),
        rho2=None,
    ):
        """Initialization of the solver

        Parameters
        ----------
        nits_plan: Number of iterations to update the plans of (U)GW

        nits_sinkhorn: Number of iterations to perform Sinkhorn updates
        in inner loop

        gradient: Asks to save gradients if True for backpropagation

        tol_plan: Tolerance between updates of the plan to stop iterations

        tol_sinkhorn: Tolerance between updates of the Sinkhorn
        potentials to stop iterations

        eps: parameter of entropic regularization

        rho: float of tuple of float
        Parameter of relaxation of marginals. Set to float('Inf')
        to compute GW instead of UGW.
        """
        self.nits_plan = nits_plan
        self.nits_sinkhorn = nits_sinkhorn
        self.gradient = gradient
        self.tol_plan = tol_plan
        self.tol_sinkhorn = tol_sinkhorn
        self.set_eps(eps)
        self.set_rho(rho, rho2)

    def get_eps(self):
        return self.eps

    def set_eps(self, eps):
        self.eps = eps

    def get_rho(self):
        return self.rho, self.rho2

    def set_rho(self, rho, rho2=None):
        if rho is None:
            raise Exception("rho must be either finite or float(Inf)")
        else:
            self.rho = rho
            if rho2 is None:
                self.rho2 = self.rho
            else:
                self.rho2 = rho2

    @property
    def tau(self):
        return 1.0 / (1.0 + self.eps / self.rho)

    @property
    def tau2(self):
        return 1.0 / (1.0 + self.eps / self.rho2)

    #####################################################
    # Computation of GW costs
    #####################################################

    @staticmethod
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
            + massp
            * torch.sum(gamma * (gamma / ref + 1e-10).log(), dim=(1, 2))
            - massp * massg
            + ref.sum(dim=(1, 2)) ** 2
        )
        return div

    @staticmethod
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
        # TODO: Update the sums with "keepdim=true for axis 0"
        massp, massg = pi.sum(dim=1), gamma.sum(dim=1)
        div = (
            massg * torch.sum(pi * (pi / ref + 1e-10).log(), dim=1)
            + massp * torch.sum(gamma * (gamma / ref + 1e-10).log(), dim=1)
            - massp * massg
            + ref.sum(dim=1) ** 2
        )
        return div

    @staticmethod
    def l2_distortion(pi, gamma, Cx, Cy):
        """Computes the L2 distortion (int |C_X - C_Y|^2 pi gamma)

        Parameters
        ----------
        pi: torch.Tensor of size [Batch, size_X, size_Y]
        First input plan to integrate against distortion.

        gamma: torch.Tensor of size [Batch, size_X, size_Y]
        Second input plan to integrate against distortion.

        Cx: torch.Tensor of size [Batch, size_X, size_X]
        Metric of the first mm-space.

        Cy: torch.Tensor of size [Batch, size_Y, size_Y]
        Metric of the second mm-space.

        Returns
        ----------
        distortion: torch.float of size [Batch]
        L2 distortion between the metric intergated against the plans
        """
        A = torch.einsum(
            "ijk,ij,ik->i", Cx ** 2, pi.sum(dim=2), gamma.sum(dim=2)
        )
        B = torch.einsum(
            "ijk,ij,ik->i", Cy ** 2, pi.sum(dim=1), gamma.sum(dim=1)
        )
        C = torch.sum(
            torch.einsum("kij,kjl->kil", Cx, pi)
            * torch.einsum("kij,kjl->kil", gamma, Cy),
            dim=(1, 2),
        )
        distortion = A + B - 2 * C
        return distortion

    def ugw_cost(self, pi, gamma, a, Cx, b, Cy):
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

        Cx: torch.Tensor of size [Batch, size_X, size_X]
        Input metric of the first mm-space.

        b: torch.Tensor of sier [Batch, size_Y]
        Input measure of the second mm-space.

        Cy: torch.Tensor of size [Batch, size_Y, size_Y]
        Input metric of the second mm-space.

        Returns
        ----------
        cost: torch.float of size [Batch]
        Value of the UGW functionnal for each batch of input.
        """
        cost = self.l2_distortion(
            pi, gamma, Cx, Cy
        ) + self.eps * self.quad_kl_div(
            pi, gamma, a[:, :, None] * b[:, None, :]
        )
        if self.rho < float("Inf"):
            cost = cost + self.rho * self.quad_kl_div_marg(
                torch.sum(pi, dim=2), torch.sum(gamma, dim=2), a
            )
        if self.rho2 < float("Inf"):
            cost = cost + self.rho2 * self.quad_kl_div_marg(
                torch.sum(pi, dim=1), torch.sum(gamma, dim=1), b
            )
        return cost

    #####################################################
    # Methods for GW loops
    #####################################################

    @staticmethod
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

    def compute_local_cost(self, pi, a, Cx, b, Cy):
        """Compute the local cost by averaging the distortion with the current
        transport plan.

        Parameters
        ----------
        pi: torch.Tensor of size [Batch, size_X, size_Y]
        transport plan used to compute local cost

        a: torch.Tensor of size [Batch, size_X]
        Input measure of the first mm-space.

        Cx: torch.Tensor of size [Batch, size_X, size_X]
        Input metric of the first mm-space.

        b: torch.Tensor of size [Batch, size_Y]
        Input measure of the second mm-space.

        Cy: torch.Tensor of size [Batch, size_Y, size_Y]
        Input metric of the second mm-space.

        Returns
        ----------
        T: torch.Tensor of size [Batch, size_X, size_Y]
        local cost depending on the current transport plan.
        """
        mu, nu = torch.sum(pi, dim=2), torch.sum(pi, dim=1)
        A = torch.einsum("bij,bj->bi", Cx ** 2, mu)
        B = torch.einsum("bkl,bl->bk", Cy ** 2, nu)
        C = torch.einsum(
            "bij,bkj->bik", Cx, torch.einsum("bkl,bjl->bkj", Cy, pi)
        )
        kl_pi = torch.sum(
            pi * (pi / (a[:, :, None] * b[:, None, :]) + 1e-10).log(),
            dim=(1, 2),
        )
        T = (A[:, :, None] + B[:, None, :] - 2 * C) + self.eps * kl_pi[
            :, None, None
        ]
        if self.rho < float("Inf"):
            T = (
                T
                + self.rho
                * torch.sum(mu * (mu / a + 1e-10).log(), dim=1)[:, None, None]
            )
        if self.rho2 < float("Inf"):
            T = (
                T
                + self.rho2
                * torch.sum(nu * (nu / b + 1e-10).log(), dim=1)[:, None, None]
            )
        return T

    def ugw_sinkhorn(self, a, Cx, b, Cy, init=None):
        """Solves the regularized UGW problem, keeps only one plan as output.
        the algorithm is run as much as possible in log-scale.

        Parameters
        ----------
        a: torch.Tensor of size [Batch, size_X]
        Input measure of the first mm-space.

        Cx: torch.Tensor of size [Batch, size_X, size_X]
        Input metric of the first mm-space.

        b: torch.Tensor of size [Batch, size_Y]
        Input measure of the second mm-space.

        Cy: torch.Tensor of size [Batch, size_Y, size_Y]
        Input metric of the second mm-space.

        init: torch.Tensor of size [Batch, size_X, size_Y]
        Transport plan at initialization.

        Returns
        ----------
        pi: torch.Tensor of size [Batch, size_X, size_Y]
        Transport plan which is a stationnary point of UGW. The output is not
        in log-scale.
        """
        # Initialize plan and local cost
        logpi = (self.init_plan(a, b, init=init) + 1e-30).log()
        up, vp = None, None
        for i in range(self.nits_plan):
            logpi_prev = logpi.clone()
            Tp = self.compute_local_cost(logpi.exp(), a, Cx, b, Cy)
            # logmp = self.optimize_mass(Tp, logpi, a, b)
            logmp = logpi.logsumexp(dim=(1, 2))
            up, vp, logpi = self.sinkhorn_gw_procedure(
                Tp, up, vp, a, b, mass=logmp.exp() + 1e-10
            )
            if torch.any(torch.isnan(logpi)):
                raise Exception(
                    f"Solver got NaN plan with params (eps, rho) "
                    f" = {self.get_eps(), self.get_rho()}"
                )
            logpi = (
                0.5 * (logmp - logpi.logsumexp(dim=(1, 2)))[:, None, None]
                + logpi
            )
            if (logpi - logpi_prev).abs().max().item() < self.tol_plan:
                break
        return logpi.exp()

    #####################################################
    # Methods for Sinkhorn loops
    #####################################################

    def kl_prox_softmin(self, K, a, b):
        """Prepares functions which perform updates of the Sikhorn algorithm
        in exponential scale.

        Parameters
        ----------
        K: torch.Tensor of size [Batch, size_X, size_Y]
        Exponential of the local cost. Kernel of Sinkhorn operator.

        a: torch.Tensor of size [Batch, size_X]
        Input measure of the first mm-space.

        b: torch.Tensor of size [Batch, size_Y]
        Input measure of the second mm-space.

        Returns
        ----------
        s_x: callable function
        Map outputing updates of potential from Y to X.

        s_y: callable function
        Map outputing updates of potential from X to Y.
        """

        def s_y(v):
            return torch.einsum("bij,bj->bi", K, b * v) ** (-self.tau2)

        def s_x(u):
            return torch.einsum("bij,bi->bj", K, a * u) ** (-self.tau)

        return s_x, s_y

    def aprox_softmin(self, C, a, b, mass):
        """Prepares functions which perform updates of the Sikhorn algorithm
        in logarithmic scale.

        Parameters
        ----------
        C: torch.Tensor of size [Batch, size_X, size_Y]
        Local cost depending on the current plan.

        a: torch.Tensor of size [Batch, size_X]
        Input measure of the first mm-space.

        b: torch.Tensor of size [Batch, size_Y]
        Input measure of the second mm-space.

        mass: torch.Tensor of size [Batch]
        Mass of the current plan.

        Returns
        ----------
        s_x: callable function
        Map outputing updates of potential from Y to X.

        s_y: callable function
        Map outputing updates of potential from X to Y.
        """

        def s_y(g):
            return (
                -mass[:, None]
                * self.tau2
                * self.eps
                * (
                    (g / (mass[:, None] * self.eps) + b.log())[:, None, :]
                    - C / (mass[:, None, None] * self.eps)
                ).logsumexp(dim=2)
            )

        def s_x(f):
            return (
                -mass[:, None]
                * self.tau
                * self.eps
                * (
                    (f / (mass[:, None] * self.eps) + a.log())[:, :, None]
                    - C / (mass[:, None, None] * self.eps)
                ).logsumexp(dim=1)
            )

        return s_x, s_y

    def translate_potential(self, u, v, C, a, b, mass):
        """Updates the dual potential by computing the optimal constant
        translation. It stabilizes and accelerates computations in sinkhorn
        loop.

        Parameters
        ----------
        u: torch.Tensor of size [Batch, size_X]
        First dual potential defined on X.

        v: torch.Tensor of size [Batch, size_Y]
        Second dual potential defined on Y.

        C: torch.Tensor of size [Batch, size_X, size_Y]
        Local cost depending on the current plan.

        a: torch.Tensor of size [Batch, size_X]
        Input measure of the first mm-space.

        b: torch.Tensor of size [Batch, size_Y]
        Input measure of the second mm-space.

        mass: torch.Tensor of size [Batch]
        Mass of the current plan.

        Returns
        ----------
        u: torch.Tensor of size [Batch, size_X]
        First dual potential defined on X.

        v: torch.Tensor of size [Batch, size_Y]
        Second dual potential defined on Y.
        """
        c1 = (
            -torch.cat((u, v), 1) / (mass[:, None] * self.rho)
            + torch.cat((a, b), 1).log()
        ).logsumexp(dim=1) - torch.log(2 * torch.ones([1]))
        c2 = (
            (
                a.log()[:, :, None]
                + b.log()[:, None, :]
                + (
                    (u[:, :, None] + v[:, None, :] - C)
                    / (mass[:, None, None] * self.eps)
                )
            )
            .logsumexp(dim=2)
            .logsumexp(dim=1)
        )
        z = (0.5 * mass * self.eps) / (
            2.0 + 0.5 * (self.eps / self.rho) + 0.5 * (self.eps / self.rho2)
        )
        k = z * (c1 - c2)
        return u + k[:, None], v + k[:, None]

    def optimize_mass(self, Tp, logpi, a, b):
        """
        Given a plan and its associated local cost, the method computes the
        optimal mass of the plan. It should be a more stable estimate of the
        mass than the mass of the plan itself.

        Parameters
        ----------
        Tp: torch.Tensor of size [Batch, size_X, size_Y]
        Local cost depending on the current plan.

        logpi: torch.Tensor of size [Batch, size_X, size_Y]
        Optimized plan in log-scale.

        a: torch.Tensor of size [Batch, size_X]
        Input measure of the first mm-space.

        b: torch.Tensor of size [Batch, size_Y]
        Input measure of the second mm-space.

        Returns
        ----------
        const: torch.Tensor of size [Batch]
        Optimal constant to translate the plan in log scale.
        """
        ma, mb = a.sum(dim=1), b.sum(dim=1)
        logmu, lognu = logpi.logsumexp(dim=2), logpi.logsumexp(dim=1)
        mtot = (
            self.rho * ma ** 2
            + self.rho2 * mb ** 2
            + self.eps * (ma * mb) ** 2
        )
        const = (
            (Tp * logpi.exp()).sum(dim=(1, 2))
            + 2 * ma * self.rho * (a * (logmu - a.log())).sum(dim=1)
            + 2 * mb * self.rho2 * (b * (lognu - b.log())).sum(dim=1)
            + 2
            * ma
            * mb
            * self.eps
            * (
                a[:, :, None]
                * b[:, None, :]
                * (logpi - a.log()[:, :, None] - b.log()[:, None, :])
            ).sum(dim=(1, 2))
        )
        return -const / mtot

    def sinkhorn_gw_procedure(self, T, u, v, a, b, mass):
        """
        Parameters
        ----------
        T: torch.Tensor of size [Batch, size_X, size_Y]
        Local cost depending on the current plan.

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

        Returns
        ----------
        :return:
        """
        # TODO DEbug ad set to True
        # Initialize potentials by finding best translation
        if u is None or v is None:
            u, v = torch.zeros_like(a), torch.zeros_like(b)
        u, v = self.translate_potential(u, v, T, a, b, mass)

        # perform Sinkhorn algorithm in LSE form
        s_x, s_y = self.aprox_softmin(T, a, b, mass)
        for j in range(self.nits_sinkhorn):
            u_prev = u.clone()
            v = s_x(u)
            u = s_y(v)
            if (u - u_prev).abs().max().item() < self.tol_sinkhorn:
                break
        logpi = (
            (
                (u[:, :, None] + v[:, None, :] - T)
                / (mass[:, None, None] * self.eps)
            )
            + a.log()[:, :, None]
            + b.log()[:, None, :]
        )
        return u, v, logpi
