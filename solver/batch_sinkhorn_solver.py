import torch


class BatchSinkhornSolver(object):

    def __init__(self, nits_plan=3000, nits_sinkhorn=3000, gradient=False, tol=1e-7, tol_sinkhorn=1e-7,
                 eps=1.0, rho=None):
        """

        :param nits: Number of iterations to update the plans of (U)GW
        :param nits_sinkhorn: Number of iterations to perform Sinkhorn updates in inner loop
        :param gradient: Asks to save gradients if True for backpropagation
        :param tol: Tolerance between updates of the plan to stop iterations
        :param tol_sinkhorn: Tolerance between updates of the Sinkhorn potentials to stop iterations
        :param eps: parameter of entropic regularization
        :param rho: Parameter of relaxation of marginals. Set to None to compute GW instead of UGW.
        """
        self.nits_plan = nits_plan
        self.nits_sinkhorn = nits_sinkhorn
        self.gradient = gradient
        self.tol = tol
        self.tol_sinkhorn = tol_sinkhorn
        self.eps = eps
        self.rho = rho

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
        if self.rho is None:
            return self.l2_distortion(pi, gamma, Cx, Cy) \
                   + self.eps * self.quad_kl_div(pi, gamma, a[:, :, None] * b[:, None, :])
        return self.l2_distortion(pi, gamma, Cx, Cy) \
               + self.rho * self.quad_kl_div_marg(torch.sum(pi, dim=2), torch.sum(gamma, dim=2), a) \
               + self.rho * self.quad_kl_div_marg(torch.sum(pi, dim=1), torch.sum(gamma, dim=1), b) \
               + self.eps * self.quad_kl_div(pi, gamma, a[:, :, None] * b[:, None, :])

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
        if self.rho is None:
            A = torch.einsum('bij,bj->bi', Cx ** 2, a)
            B = torch.einsum('bkl,bl->bk', Cy ** 2, b)
            C = torch.einsum('bij,bkj->bik', Cx, torch.einsum('bkl,bjl->bkj', Cy, pi))
            kl_pi = torch.sum(pi * (pi / (a[:, :, None] * b[:, None, :]) + 1e-10).log(), dim=(1,2))
            return (A[:, :, None] + B[:, None, :] - 2 * C) + self.eps * kl_pi[:, None, None]
        else:
            mu, nu = torch.sum(pi, dim=2), torch.sum(pi, dim=1)
            A = torch.einsum('bij,bj->bi', Cx ** 2, mu)
            B = torch.einsum('bkl,bl->bk', Cy ** 2, nu)
            C = torch.einsum('bij,bkj->bik', Cx, torch.einsum('bkl,bjl->bkj', Cy, pi))
            kl_mu = torch.sum(mu * (mu / a + 1e-10).log(), dim=1)
            kl_nu = torch.sum(nu * (nu / b + 1e-10).log(), dim=1)
            kl_pi = torch.sum(pi * (pi / (a[:, :, None] * b[:, None, :]) + 1e-10).log(), dim=(1,2))
            return (A[:, :, None] + B[:, None, :] - 2 * C) \
                   + (self.rho * kl_mu + self.rho * kl_nu + self.eps * kl_pi)[:, None, None]

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
            pi = (mp / pi.sum(dim=(1, 2))).sqrt()[:, None, None] * pi
            if (pi - pi_prev).abs().max().item() < 1e-7:
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

            if (pi - pi_prev).abs().max().item() < 1e-7:
                break
        return pi, gamma

    #####################################################
    # Methods for Sinkhorn loops
    #####################################################

    def kl_prox_softmin(self, K, a, b):
        if self.rho is None:
            tau = 1.
        else:
            tau = 1 / (1 + (self.eps / self.rho))

        def s_y(v):
            return torch.einsum('bij,bj->bi', K, b * v) ** (-tau)

        def s_x(u):
            return torch.einsum('bij,bi->bj', K, a * u) ** (-tau)

        return s_x, s_y

    def aprox_softmin(self, C, a, b, mass):
        if self.rho is None:
            tau = 1.
        else:
            tau = 1 / (1 + (self.eps / self.rho))

        def s_y(g):
            return - mass[:, None] * tau * self.eps * ((g / self.eps + b.log())[:, None, :] - C / self.eps).logsumexp(dim=2)

        def s_x(f):
            return - mass[:, None] * tau * self.eps * ((f / self.eps + a.log())[:, :, None] - C / self.eps).logsumexp(dim=1)

        return s_x, s_y

    def translate_potential(self, u, v, C, a, b, mass):
        if self.rho is None:
            k = 0.5 * (a.sum(dim=1).log()
                       - ((u[:, :, None] + v[:, None, :] - C) / (mass[:, None, None] * self.eps)).logsumexp(dim=2).logsumexp(dim=1))
            return u + k[:, None], v + k[:, None]
        else:
            c1 = (torch.sum(a * (-u / (mass[:, None] * self.rho)).exp(), dim=1)
                  + torch.sum(b * (-v / (mass[:, None] * self.rho)).exp(), dim=1)).log()
            c2 = (a.log()[:, :, None] * b.log()[:, None, :]
                  + ((u[:, :, None] + v[:, None, :] - C) / (mass[:, None, None] * self.eps))).logsumexp(dim=2).logsumexp(dim=1)
            z = mass * (self.rho * self.eps) / (2 * self.rho + self.eps)
            k = z * (c1 - c2)
            return u + k[:, None], v + k[:, None]

    def sinkhorn_gw_procedure(self, T, u, v, a, b, mass, exp_form=True):
        if u is None or v is None:  # Initialize potentials by finding best translation
            u, v = torch.zeros_like(a), torch.zeros_like(b)
        u, v = self.translate_potential(u, v, T, a, b, mass)

        if exp_form:  # Check if acceleration via exp-sinkhorn has no underflow
            K = (-T / (mass[:, None, None] * self.eps)).exp()
            exp_form = K.gt(torch.zeros_like(K)).all()
            if ~exp_form:
                del K

        if exp_form:  # If so perform Sinkhorn
            u, v = (u / (mass[:, None] * self.eps)).exp(), (v / (mass[:, None] * self.eps)).exp()
            s_x, s_y = self.kl_prox_softmin(K, a, b)
            for j in range(self.nits_sinkhorn):
                u_prev = u.clone()
                v = s_x(u)
                u = s_y(v)
                if ((mass[:, None] * self.eps) * (u.log() - u_prev.log())).abs().max().item() < 1e-7:
                    break
            pi = u[:, :, None] * v[:, None, :] * K * a[:, :, None] * b[:, None, :]
            u, v = (mass[:, None] * self.eps) * u.log(), (mass[:, None] * self.eps) * v.log()

        if ~exp_form:  # Else perform Sinkhorn algorithm in LSE form
            s_x, s_y = self.aprox_softmin(T, a, b, mass)
            for j in range(self.nits_sinkhorn):
                u_prev = u.clone()
                v = s_x(u)
                u = s_y(v)
                if (u - u_prev).abs().max().item() < 1e-7:
                    break
            pi = ((u[:, :, None] + v[:, None, :] - T) / (mass[:, None, None] * self.eps)).exp() \
                 * a[:, :, None] * b[:, None, :]
        return u, v, pi
