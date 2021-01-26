import torch
# from utils_pytorch import l2_distortion, grad_l2_distortion
# from solver.utils_pytorch import l2_distortion, grad_l2_distortion
import numpy as np


class BatchSinkhornSolver(object):

    def __init__(self, nits, nits_sinkhorn, gradient=False, tol=1e-7, tol_sinkhorn=1e-7, eps=1.0, rho=None):
        self.nits = nits
        self.nits_sinkhorn = nits_sinkhorn
        self.gradient = gradient
        self.tol = tol
        self.tol_sinkhorn = tol_sinkhorn
        self.eps = eps
        self.rho = rho
        if rho is None:
            self.tau = 1.
        else:
            self.tau = 1 / (1 + (self.eps / self.rho))

    #####################################################
    # Computation of GW costs
    #####################################################

    @staticmethod
    def quad_kl_div(pi, gamma, ref):
        massp, massg = pi.sum(dim=(1, 2)), gamma.sum(dim=(1, 2)),
        return massg * torch.sum(pi * (pi / ref + 1e-10).log(), dim=(1, 2)) \
               + massp * torch.sum(gamma * (gamma / ref + 1e-10).log(), dim=(1, 2)) \
               - massp * massg + ref.sum(dim=(1, 2)) ** 2

    @staticmethod
    def l2_distortion(pi, gamma, Cx, Cy):
        A = torch.einsum('ijk,ij,ik->i', Cx ** 2, pi.sum(dim=(1, 2)), gamma.sum(dim=(1, 2)))
        B = torch.einsum('ijk,ij,ik->i', Cy ** 2, pi.sum(dim=(1, 2)), gamma.sum(dim=(1, 2)))
        C = torch.sum(torch.einsum('kij,kjl->kil', Cx, pi) * torch.einsum('kij,kjl->kil', gamma, Cy), dim=(1, 2))
        return A + B - 2 * C

    def ugw_cost(self, pi, gamma, a, Cx, b, Cy):
        if self.rho is None:
            return self.l2_distortion(pi, Cx, Cy) \
                   + self.eps * self.quad_kl_div(pi, gamma, a[:, :, None] * b[:, None, :])
        return self.l2_distortion(pi, Cx, Cy) \
               + self.rho * self.quad_kl_div(torch.sum(pi, dim=2), torch.sum(gamma, dim=2), a) \
               + self.rho * self.quad_kl_div(torch.sum(pi, dim=1), torch.sum(gamma, dim=1), b) \
               + self.eps * self.quad_kl_div(pi, gamma, a[:, :, None] * b[:, None, :])

    #####################################################
    # Methods for GW loops
    #####################################################

    @staticmethod
    def init_plan(a, b, init):
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
        # Initialize plan and local cost
        pi = self.init_plan(a, b, init=init)
        up, vp = None, None
        for i in range(self.nits):
            pi_prev = pi.clone()
            Tp = self.compute_local_cost(pi, a, Cx, b, Cy)
            mp = pi.sum(dim=(1, 2))
            up, vp, pi = self.sinkhorn_gw_procedure(Tp, up, vp, a, b, mass=mp)
            pi = (mp / pi.sum(dim=(1, 2))).sqrt()[:, None, None] * pi
            if (pi - pi_prev).abs().max().item() < 1e-7:
                break
        return pi

    def alternate_sinkhorn(self, a, Cx, b, Cy, init=None):
        # Initialize plan and local cost
        pi = self.init_plan(a, b, init=init)
        ug, vg, up, vp = None, None, None, None
        for i in range(self.nits):
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

        def s_y(v):
            return torch.einsum('bij,bj->bi', K, b * v) ** (-self.tau)

        def s_x(u):
            return torch.einsum('bij,bi->bj', K, a * u) ** (-self.tau)

        return s_x, s_y

    def aprox_softmin(self, C, a, b, mass):

        def s_y(g):
            return - mass[:, None] * self.tau * self.eps * ((g / self.eps + b.log())[:, None, :] - C / self.eps).logsumexp(dim=2)

        def s_x(f):
            return - mass[:, None] * self.tau * self.eps * ((f / self.eps + a.log())[:, :, None] - C / self.eps).logsumexp(dim=1)

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
            u, v = self.translate_potential(torch.zeros_like(a), torch.zeros_like(b), T, a, b, mass)

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
            u, v = (mass[:, None, None] * self.eps) * u.log(), (mass[:, None, None] * self.eps) * v.log()

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
