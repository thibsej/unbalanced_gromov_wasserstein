import torch


class BatchLowerBoundSolver(object):

    def __init__(self, nits_sinkhorn=3000, gradient=False, tol_sinkhorn=1e-7, eps=1.0,
                 rho=float('Inf'), rho2=None):
        """
        :param nits: Number of iterations to update the plans of (U)GW
        :param nits_sinkhorn: Number of iterations to perform Sinkhorn updates in inner loop
        :param gradient: Asks to save gradients if True for backpropagation
        :param tol: Tolerance between updates of the plan to stop iterations
        :param tol_sinkhorn: Tolerance between updates of the Sinkhorn potentials to stop iterations
        :param eps: parameter of entropic regularization
        :param rho: Parameter of relaxation of marginals. Set to None to compute GW instead of UGW.
        """
        self.nits_sinkhorn = nits_sinkhorn
        self.gradient = gradient
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
    # Methods for GW loops
    #####################################################

    @staticmethod
    def compute_local_cost(a, Cx, b, Cy):
        h_x = torch.einsum('bij, bj->bi', Cx, a / a.sum(dim=1)[:,None])
        h_y = torch.einsum('bij, bj->bi', Cy, b / b.sum(dim=1)[:, None])
        return (h_x ** 2)[:, :, None] + (h_y ** 2)[:, None, :] - 2 * h_x[:, :, None] * h_y[:, None, :]

    #####################################################
    # Methods for Sinkhorn loops
    #####################################################

    def kl_prox_softmin(self, K, a, b):

        def s_y(v):
            return torch.einsum('bij,bj->bi', K, b * v) ** (-self.tau2)

        def s_x(u):
            return torch.einsum('bij,bi->bj', K, a * u) ** (-self.tau)

        return s_x, s_y

    def aprox_softmin(self, C, a, b):

        def s_y(g):
            return - self.tau2 * self.eps * ((g / (self.eps) + b.log())[:, None, :]
                                                    - C / (self.eps)).logsumexp(dim=2)

        def s_x(f):
            return - self.tau * self.eps * ((f / (self.eps) + a.log())[:, :, None]
                                                   - C / (self.eps)).logsumexp(dim=1)

        return s_x, s_y

    def translate_potential(self, u, v, C, a, b):
        c1 = (- torch.cat((u, v), 1) / self.rho + torch.cat((a, b), 1).log()).logsumexp(dim=1) \
             - torch.log(2 * torch.ones([1]))
        c2 = (a.log()[:, :, None] + b.log()[:, None, :]
              + ((u[:, :, None] + v[:, None, :] - C) / self.eps)).logsumexp(dim=2).logsumexp(
            dim=1)
        z = (0.5 * self.eps) / (2. + 0.5 * (self.eps / self.rho) + 0.5 * (self.eps / self.rho2))
        k = z * (c1 - c2)
        return u + k[:, None], v + k[:, None]

    def compute_plan(self, a, Cx, b, Cy, u=None, v=None, exp_form=True):
        T = self.compute_local_cost(a, Cx, b, Cy)

        if u is None or v is None:  # Initialize potentials by finding best translation
            u, v = torch.zeros_like(a), torch.zeros_like(b)
        u, v = self.translate_potential(u, v, T, a, b)

        if exp_form:  # Check if acceleration via exp-sinkhorn has no underflow
            K = (-T / (self.eps)).exp()
            u_, v_ = (u / (self.eps)).exp(), (v / (self.eps)).exp()
            mask_K = K.gt((1e-5) * torch.ones_like(K)).all()
            mask_u = torch.gt(u_, (1e-5) * torch.ones_like(u_)).all() and \
                     torch.gt((1e5) * torch.ones_like(u_), u_).all() and ~torch.isinf(u_).any()
            mask_v = torch.gt(v_, (1e-5) * torch.ones_like(v_)).all() and \
                     torch.gt((1e5) * torch.ones_like(v_), v_).all() and ~torch.isinf(v_).any()
            exp_form = mask_K and mask_u and mask_v
            if ~exp_form:
                del K, u_, v_

        if exp_form:  # If so perform Sinkhorn in EXP form
            u, v = (u / self.eps).exp(), (v / self.eps).exp()
            s_x, s_y = self.kl_prox_softmin(K, a, b)
            for j in range(self.nits_sinkhorn):
                u_prev = u.clone()
                v = s_x(u)
                u = s_y(v)
                if (self.eps * (u.log() - u_prev.log())).abs().max().item() < self.tol_sinkhorn:
                    break
            pi = u[:, :, None] * v[:, None, :] * K * a[:, :, None] * b[:, None, :]
            u, v = self.eps * u.log(), self.eps * v.log()

        if ~exp_form:  # Else perform Sinkhorn algorithm in LSE form
            s_x, s_y = self.aprox_softmin(T, a, b)
            for j in range(self.nits_sinkhorn):
                u_prev = u.clone()
                v = s_x(u)
                u = s_y(v)
                if (u - u_prev).abs().max().item() < self.tol_sinkhorn:
                    break
            pi = ((u[:, :, None] + v[:, None, :] - T) / self.eps).exp() * a[:, :, None] * b[:, None, :]
        return u, v, pi
