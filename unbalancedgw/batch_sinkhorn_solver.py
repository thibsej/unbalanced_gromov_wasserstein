import torch


class BatchSinkhornSolver(object):

    def __init__(self, nits_plan=3000, nits_sinkhorn=3000, gradient=False, tol_plan=1e-7, tol_sinkhorn=1e-7,
                 eps=1.0, rho=float('Inf'), rho2=None):
        """

        :param nits: Number of iterations to update the plans of (U)GW
        :param nits_sinkhorn: Number of iterations to perform Sinkhorn updates in inner loop
        :param gradient: Asks to save gradients if True for backpropagation
        :param tol: Tolerance between updates of the plan to stop iterations
        :param tol_sinkhorn: Tolerance between updates of the Sinkhorn potentials to stop iterations
        :param eps: parameter of entropic regularization
        :param rho: Parameter of relaxation of marginals. Set to float('Inf') to compute GW instead of UGW.
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
            raise Exception('rho must be either finite or float(Inf)')
        else:
            self.rho = rho
            if rho2 is None:
                self.rho2 = self.rho
            else:
                self.rho2 = rho2

    @property
    def tau(self):
        return 1. / (1. + self.eps / self.rho)

    @property
    def tau2(self):
        return 1. / (1. + self.eps / self.rho2)

    #####################################################
    # Computation of GW costs
    #####################################################
    # TODO: Update the sums with "keepdim=true for axis 0"
    @staticmethod
    def quad_kl_div(pi, gamma, ref):
        """
        Compute the quadratic entropy $KL^\otimes(\pi\otimes\gamma | ref)$ with full plans
        :param pi: first input, torch.Tensor of size [Batch, size_X, size_Y]
        :param gamma: second input torch.Tensor of size [Batch, size_X, size_Y]
        :param ref: Reference of the KL entropy to compare with $\pi\otimes\gamma$
        :return: torch.float
        """
        massp, massg = pi.sum(dim=(1, 2)), gamma.sum(dim=(1, 2))
        return massg * torch.sum(pi * (pi / ref + 1e-10).log(), dim=(1, 2)) \
               + massp * torch.sum(gamma * (gamma / ref + 1e-10).log(), dim=(1, 2)) \
               - massp * massg + ref.sum(dim=(1, 2)) ** 2

    @staticmethod
    def quad_kl_div_marg(pi, gamma, ref):
        """
        Compute the quadratic entropy $KL^\otimes(\pi\otimes\gamma | ref)$ with plans marginals
        :param pi: first input, torch.Tensor of size [Batch, size_X]
        :param gamma: second input torch.Tensor of size [Batch, size_X]
        :param ref: Reference of the KL entropy to compare
        :return: torch.float
        """
        massp, massg = pi.sum(dim=1), gamma.sum(dim=1)
        return massg * torch.sum(pi * (pi / ref + 1e-10).log(), dim=1) \
               + massp * torch.sum(gamma * (gamma / ref + 1e-10).log(), dim=1) \
               - massp * massg + ref.sum(dim=1) ** 2

    @staticmethod
    def l2_distortion(pi, gamma, Cx, Cy):
        """
        Computes the L2 distortion $\int |C_X - C_Y|^2 d\pi d\gamma$
        :param pi: torch.Tensor of size [Batch, size_X, size_Y]
        :param gamma: torch.Tensor of size [Batch, size_X, size_Y]
        :param Cx: torch.Tensor of size [Batch, size_X, size_X]
        :param Cy: torch.Tensor of size [Batch, size_Y, size_Y]
        :return: torch.float of size [Batch]
        """
        A = torch.einsum('ijk,ij,ik->i', Cx ** 2, pi.sum(dim=2), gamma.sum(dim=2))
        B = torch.einsum('ijk,ij,ik->i', Cy ** 2, pi.sum(dim=1), gamma.sum(dim=1))
        C = torch.sum(torch.einsum('kij,kjl->kil', Cx, pi) * torch.einsum('kij,kjl->kil', gamma, Cy), dim=(1, 2))
        return A + B - 2 * C

    def ugw_cost(self, pi, gamma, a, Cx, b, Cy):
        """
        Compute the full (U)GW functional with entropic term and KL penalty of marginals
        :param pi: torch.Tensor of size [Batch, size_X, size_Y]
        :param gamma: torch.Tensor of size [Batch, size_X, size_Y]
        :param a: torch.Tensor of size [Batch, size_X]
        :param Cx: torch.Tensor of size [Batch, size_X, size_X]
        :param b: torch.Tensor of sier [Batch, size_Y
        :param Cy: torch.Tensor of size [Batch, size_Y, size_Y]
        :return: torch.float of size [Batch]
        """
        cost = self.l2_distortion(pi, gamma, Cx, Cy) \
               + self.eps * self.quad_kl_div(pi, gamma, a[:, :, None] * b[:, None, :])
        if self.rho < float('Inf'):
            cost = cost + self.rho * self.quad_kl_div_marg(torch.sum(pi, dim=2), torch.sum(gamma, dim=2), a)
        if self.rho2 < float('Inf'):
            cost = cost + self.rho2 * self.quad_kl_div_marg(torch.sum(pi, dim=1), torch.sum(gamma, dim=1), b)
        return cost

    #####################################################
    # Methods for GW loops
    #####################################################

    @staticmethod
    def init_plan(a, b, init=None):
        """
        Initialize the plan if None is given, otherwise use the input plan
        :param a: torch.Tensor of size [Batch, size_X]
        :param b: torch.Tensor of size [Batch, size_Y]
        :param init: torch.Tensor of size [Batch, size_X, size_Y], defaults to None
        :return: torch.Tensor of size [Batch, size_X, size_Y]
        """
        if init is not None:
            return init
        else:
            return a[:, :, None] * b[:, None, :] / (a.sum(dim=1) * b.sum(dim=1)).sqrt()[:, None, None]

    def compute_local_cost(self, pi, a, Cx, b, Cy):
        mu, nu = torch.sum(pi, dim=2), torch.sum(pi, dim=1)
        A = torch.einsum('bij,bj->bi', Cx ** 2, mu)
        B = torch.einsum('bkl,bl->bk', Cy ** 2, nu)
        C = torch.einsum('bij,bkj->bik', Cx, torch.einsum('bkl,bjl->bkj', Cy, pi))
        kl_pi = torch.sum(pi * (pi / (a[:, :, None] * b[:, None, :]) + 1e-10).log(), dim=(1, 2))
        T = (A[:, :, None] + B[:, None, :] - 2 * C) + self.eps * kl_pi[:, None, None]
        if self.rho < float('Inf'):
            T = T + self.rho * torch.sum(mu * (mu / a + 1e-10).log(), dim=1)[:, None, None]
        if self.rho2 < float('Inf'):
            T = T + self.rho2 * torch.sum(nu * (nu / b + 1e-10).log(), dim=1)[:, None, None]
        return T

    def ugw_sinkhorn(self, a, Cx, b, Cy, init=None):
        """
        Solver the regularized UGW problem, keeps only one plan
        :param a: torch.Tensor of size [Batch, size_X]
        :param Cx: torch.Tensor of size [Batch, size_X, size_X]
        :param b: torch.Tensor of size [Batch, size_Y]
        :param Cy: torch.Tensor of size [Batch, size_Y, size_Y]
        :param init: transport plan at initialization, torch.Tensor of size [Batch, size_X, size_Y]
        :return: torch.Tensor of size [Batch, size_X, size_Y]
        """
        # Initialize plan and local cost
        pi = self.init_plan(a, b, init=init)
        up, vp = None, None
        for i in range(self.nits_plan):
            pi_prev = pi.clone()
            Tp = self.compute_local_cost(pi, a, Cx, b, Cy)
            mp = pi.sum(dim=(1, 2))
            up, vp, pi = self.sinkhorn_gw_procedure(Tp, up, vp, a, b, mass=mp)
            if torch.any(torch.isnan(pi)):
                raise Exception(f"Solver got NaN plan with params (eps, rho) = {self.get_eps(), self.get_rho()}")
            # assert ~torch.any(torch.isnan(pi)) # Raise error is there is a nan in matrix #TODO: Debug Nans
            pi = (mp / pi.sum(dim=(1, 2))).sqrt()[:, None, None] * pi
            if (pi - pi_prev).abs().max().item() < self.tol_plan:
                break
        return pi

    def alternate_sinkhorn(self, a, Cx, b, Cy, init=None):
        """
        Solver the regularized UGW problem, returs both plans $(\pi,\gamma)$
        :param a: torch.Tensor of size [Batch, size_X]
        :param Cx: torch.Tensor of size [Batch, size_X, size_X]
        :param b: torch.Tensor of size [Batch, size_Y]
        :param Cy: torch.Tensor of size [Batch, size_Y, size_Y]
        :param init: transport plan at initialization, torch.Tensor of size [Batch, size_X, size_Y]
        :return: two torch.Tensor of size [Batch, size_X, size_Y]
        """
        # Initialize plan and local cost
        pi = self.init_plan(a, b, init=init)
        ug, vg, up, vp = None, None, None, None
        for i in range(self.nits_plan):
            pi_prev = pi.clone()

            # Fix pi and optimize wrt gamma
            Tp = self.compute_local_cost(pi, a, Cx, b, Cy)
            mp = pi.sum(dim=(1, 2))
            ug, vg, gamma = self.sinkhorn_gw_procedure(Tp, ug, vg, a, b, mass=mp)
            gamma = (mp / gamma.sum(dim=(1, 2))).sqrt()[:, None, None] * gamma

            # Fix gamma and optimize wrt pi
            Tg = self.compute_local_cost(gamma, a, Cx, b, Cy)
            mg = gamma.sum(dim=(1, 2))
            up, vp, pi = self.sinkhorn_gw_procedure(Tg, up, vp, a, b, mass=mg)
            pi = (mg / pi.sum(dim=(1, 2))).sqrt()[:, None, None] * pi

            if (pi - pi_prev).abs().max().item() < self.tol_plan:
                break
        return pi, gamma

    #####################################################
    # Methods for Sinkhorn loops
    #####################################################

    def kl_prox_softmin(self, K, a, b):

        def s_y(v):
            return torch.einsum('bij,bj->bi', K, b * v) ** (-self.tau2)

        def s_x(u):
            return torch.einsum('bij,bi->bj', K, a * u) ** (-self.tau)

        return s_x, s_y

    def aprox_softmin(self, C, a, b, mass):

        def s_y(g):
            return - mass[:, None] * self.tau2 * self.eps \
                   * ((g / (mass[:, None] * self.eps) + b.log())[:, None, :] - C / (
                        mass[:, None, None] * self.eps)).logsumexp(dim=2)

        def s_x(f):
            return - mass[:, None] * self.tau * self.eps \
                   * ((f / (mass[:, None] * self.eps) + a.log())[:, :, None] - C / (
                        mass[:, None, None] * self.eps)).logsumexp(dim=1)

        return s_x, s_y

    def translate_potential(self, u, v, C, a, b, mass):
        c1 = (- torch.cat((u, v), 1) / (mass[:, None] * self.rho) + torch.cat((a, b), 1).log()).logsumexp(dim=1) \
             - torch.log(2 * torch.ones([1]))
        c2 = (a.log()[:, :, None] + b.log()[:, None, :]
              + ((u[:, :, None] + v[:, None, :] - C) / (mass[:, None, None] * self.eps))).logsumexp(dim=2).logsumexp(
            dim=1)
        z = (0.5 * mass * self.eps) / (2. + 0.5 * (self.eps / self.rho) + 0.5 * (self.eps / self.rho2))
        k = z * (c1 - c2)
        return u + k[:, None], v + k[:, None]

    def optimize_mass(self, Tp, pi, a, b):
        """
        Given a plan and its associated local cost, the method computes the optimal mass of the plan.
        It should be a more stable estimate of the mass than the mass of the plan itself.
        :param Tp:
        :param logpi:
        :param a:
        :param b:
        :return:
        """
        ma, mb = a.sum(dim=1), b.sum(dim=1)
        mu, nu = pi.sum(dim=2), pi.sum(dim=1)
        mtot = self.rho * ma ** 2 + self.rho2 * mb ** 2 + self.eps * (ma * mb) ** 2
        const = (Tp * pi).sum(dim=(1, 2)) + 2 * ma * self.rho * (a * (mu / a + 1e-10).log()).sum(dim=1) \
                + 2 * mb * self.rho * (b * (nu / b + 1e-10).log()).sum(dim=1) \
                + ma * mb * self.rho * (a[:, :, None] * b[:, None, :] *
                                        (pi / (a[:, :, None] * b[:, None, :]) + 1e-10).log()).sum(dim=(1, 2))
        return - const / mtot

    def sinkhorn_gw_procedure(self, T, u, v, a, b, mass, exp_form=False):  # TODO DEbug ad set to True
        if u is None or v is None:  # Initialize potentials by finding best translation
            u, v = torch.zeros_like(a), torch.zeros_like(b)
        u, v = self.translate_potential(u, v, T, a, b, mass)

        if exp_form:  # Check if acceleration via exp-sinkhorn has no underflow
            K = (-T / (mass[:, None, None] * self.eps)).exp()
            u_, v_ = (u / (mass[:, None] * self.eps)).exp(), (v / (mass[:, None] * self.eps)).exp()
            mask_K = K.gt((1e-5) * torch.ones_like(K)).all()
            mask_u = torch.gt(u_, (1e-5) * torch.ones_like(u_)).all() and \
                     torch.gt((1e5) * torch.ones_like(u_), u_).all() and ~torch.isinf(u_).any()
            mask_v = torch.gt(v_, (1e-5) * torch.ones_like(v_)).all() and \
                     torch.gt((1e5) * torch.ones_like(v_), v_).all() and ~torch.isinf(v_).any()
            exp_form = mask_K and mask_u and mask_v
            if ~exp_form:
                del K, u_, v_

        if exp_form:  # If so perform Sinkhorn
            u, v = u_, v_
            del u_, v_
            s_x, s_y = self.kl_prox_softmin(K, a, b)
            for j in range(self.nits_sinkhorn):
                u_prev = u.clone()
                v = s_x(u)
                u = s_y(v)
                if ((mass[:, None] * self.eps) * (u.log() - u_prev.log())).abs().max().item() < self.tol_sinkhorn:
                    break
            pi = u[:, :, None] * v[:, None, :] * K * a[:, :, None] * b[:, None, :]
            u, v = (mass[:, None] * self.eps) * u.log(), (mass[:, None] * self.eps) * v.log()

        if ~exp_form:  # Else perform Sinkhorn algorithm in LSE form
            s_x, s_y = self.aprox_softmin(T, a, b, mass)
            for j in range(self.nits_sinkhorn):
                u_prev = u.clone()
                v = s_x(u)
                u = s_y(v)
                if (u - u_prev).abs().max().item() < self.tol_sinkhorn:
                    break
            pi = ((u[:, :, None] + v[:, None, :] - T) / (mass[:, None, None] * self.eps)).exp() \
                 * a[:, :, None] * b[:, None, :]
        return u, v, pi