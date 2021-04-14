import torch


class LowerBoundSolver(object):

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
        self.eps = eps
        self.rho = rho
        self.rho2 = rho2
        @property
        def tau(self):
            return 1. / (1. + self.eps / self.rho)

        @property
        def tau2(self):
            if self.rho2 is None:
                return self.tau
            else:
                return 1. / (1. + self.eps / self.rho2)

    #####################################################
    # Methods for GW loops
    #####################################################

    @staticmethod
    def compute_local_cost(a, Cx, b, Cy):
        h_x, h_y = torch.einsum('ij, j->i', Cx, a), torch.einsum('ij, j->i', Cy, b)
        return (h_x ** 2)[:, None] + (h_y ** 2)[None, :] - 2 * h_x[:, None] * h_y[None, :]

    #####################################################
    # Methods for Sinkhorn loops
    #####################################################

    def kl_prox_softmin(self, K, a, b):
        if self.rho is None:
            tau = 1.
        else:
            tau = 1 / (1 + (self.eps / self.rho))

        def s_y(v):
            return torch.einsum('ij,j->i', K, b * v) ** (-tau)

        def s_x(u):
            return torch.einsum('ij,i->j', K, a * u) ** (-tau)

        return s_x, s_y

    def aprox_softmin(self, C, a, b):
        if self.rho is None:
            tau = 1.
        else:
            tau = 1 / (1 + (self.eps / self.rho))

        def s_y(g):
            return - tau * self.eps * ((g / self.eps + b.log())[None, :]
                                       - C / self.eps).logsumexp(dim=1)

        def s_x(f):
            return - tau * self.eps * ((f / self.eps + a.log())[:, None]
                                       - C / self.eps).logsumexp(dim=0)

        return s_x, s_y

    def translate_potential(self, u, v, C, a, b):
        """
        Initializes the potentials with the optimal constant translation for a warm start.
        Used in the inner Sinkhorn loop.
        :param u:
        :param v:
        :param C:
        :param a:
        :param b:
        :param mass:
        :return:
        """
        if self.rho is None:
            k = 0.5 * (a.sum().log()
                       - ((u[:, None] + v[None, :] - C) / self.eps).logsumexp(dim=1).logsumexp(dim=0))
            return u + k, v + k
        else:
            c1 = (torch.sum(a * (-u / self.rho).exp())
                  + torch.sum(b * (-v / self.rho).exp())).log()
            c2 = (a.log()[:, None] * b.log()[None, :]
                  + ((u[:, None] + v[None, :] - C) / self.eps)).logsumexp(dim=1).logsumexp(dim=0)
            z = (self.rho * self.eps) / (2 * self.rho + self.eps)
            k = z * (c1 - c2)
            return u + k, v + k

    def compute_plan(self, u, v, a, Cx, b, Cy, exp_form=True):
        T = self.compute_local_cost(a, Cx, b, Cy)

        if u is None or v is None:  # Initialize potentials by finding best translation
            u, v = torch.zeros_like(a), torch.zeros_like(b)
        u, v = self.translate_potential(u, v, T, a, b)

        if exp_form:  # Check if acceleration via exp-sinkhorn has no underflow
            K = (-T / self.eps).exp()
            exp_form = K.gt(torch.zeros_like(K)).all()
            if ~exp_form:
                del K

        if exp_form:  # If so perform Sinkhorn in EXP form
            u, v = (u / self.eps).exp(), (v / self.eps).exp()
            s_x, s_y = self.kl_prox_softmin(K, a, b)
            for j in range(self.nits_sinkhorn):
                u_prev = u.clone()
                v = s_x(u)
                u = s_y(v)
                if (self.eps * (u.log() - u_prev.log())).abs().max().item() < self.tol_sinkhorn:
                    break
            pi = u[:, None] * v[None, :] * K * a[:, None] * b[None, :]
            u, v = self.eps * u.log(), self.eps * v.log()

        if ~exp_form:  # Else perform Sinkhorn algorithm in LSE form
            s_x, s_y = self.aprox_softmin(T, a, b)
            for j in range(self.nits_sinkhorn):
                u_prev = u.clone()
                v = s_x(u)
                u = s_y(v)
                if (u - u_prev).abs().max().item() < self.tol_sinkhorn:
                    break
            pi = ((u[:, None] + v[None, :] - T) / self.eps).exp() * a[:, None] * b[None, :]
        return u, v, pi
