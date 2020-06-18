import torch
from solver.utils_pytorch import l2_distortion, grad_l2_distortion, gw_cost, wfr_grad_distortion
import numpy as np


class TLBSinkhornSolver(object):

    def __init__(self, nits, nits_sinkhorn, gradient=False, tol=1e-7, tol_sinkhorn=1e-7):
        self.nits = nits
        self.nits_sinkhorn = nits_sinkhorn
        self.gradient = gradient
        self.tol = tol
        self.tol_sinkhorn = tol_sinkhorn

    @staticmethod
    def rescale_mass_plan(pi, gamma, a, Cx, b, Cy, rho, eps):
        """Same scaling to apply to pi and gamma"""
        mp, mg = pi.sum(), gamma.sum()
        mup, nup = torch.sum(pi, dim=1), torch.sum(pi, dim=0)
        mug, nug = torch.sum(gamma, dim=1), torch.sum(gamma, dim=0)
        s = mp * torch.sum(mug * (mug / a + 1e-10).log()) + mg * torch.sum(mup * (mup / a + 1e-10).log()) \
            + mp * torch.sum(nug * (nug / b + 1e-10).log()) + mg * torch.sum(nup * (nup / b + 1e-10).log())
        s = rho * s + eps * (mg * torch.sum(pi * (pi / (a[:, None] * b[None, :]) + 1e-10).log())
                             + mp * torch.sum(gamma * (gamma / (a[:, None] * b[None, :]) + 1e-10).log()))
        s = s + l2_distortion(pi, Cx, Cy)
        return (- s / (2 * (2 * rho + eps) * (mp * mg))).exp()

    @staticmethod
    def init_plan(a, b, init):
        if init is not None:
            return init
        else:
            return a[:, None] * b[None, :] / (a.sum() * b.sum()).sqrt()

    @staticmethod
    def translate_potential(u, v, C, a, b, rho, eps):
        c1 = (torch.sum(a * (-u / rho).exp()) + torch.sum(b * (-v / rho).exp())).log()
        c2 = (a.log()[:, None] * b.log()[None, :]
              + ((u[:, None] + v[None, :] - C) / eps)).logsumexp(dim=1).logsumexp(dim=0)
        z = (rho * eps) / (2 * rho + eps)
        k = z * (c1 - c2)
        return u + k, v + k

    @staticmethod
    def quad_kl_div(pi, gamma, ref):
        massp, massg = pi.sum(), gamma.sum()
        return massg * torch.sum(pi * (pi / ref + 1e-10).log()) \
               + massp * torch.sum(gamma * (gamma / ref + 1e-10).log()) - massp * massg + ref.sum() ** 2

    @staticmethod
    def l2_distortion(pi, gamma, Cx, Cy):
        A = torch.einsum('ij,i,j', Cx ** 2, torch.sum(pi, dim=1), torch.sum(gamma, dim=1))
        B = torch.einsum('ij,i,j', Cy ** 2, torch.sum(pi, dim=0), torch.sum(gamma, dim=0))
        C = torch.sum(torch.einsum('ij,jl->il', Cx, pi) * torch.einsum('ij,jl->il', gamma, Cy))
        return A + B - 2 * C

    def tlb_cost(self, pi, gamma, a, Cx, b, Cy, rho, eps):
        return l2_distortion(pi, Cx, Cy) + rho * self.quad_kl_div(torch.sum(pi, dim=1), torch.sum(gamma, dim=1), a) \
               + rho * self.quad_kl_div(torch.sum(pi, dim=0), torch.sum(gamma, dim=0), b) \
               + eps * self.quad_kl_div(pi, gamma, a[:, None] * b[None, :])

    @staticmethod
    def kl_prox_softmin(K, a, b, rho, eps):
        tau = rho / (rho + eps)

        def s_y(v):
            return torch.einsum('ij,j->i', K, b * v) ** (-tau)

        def s_x(u):
            return torch.einsum('ij,i->j', K, a * u) ** (-tau)

        return s_x, s_y

    @staticmethod
    def aprox_softmin(C, a, b, rho, eps):
        tau = rho / (rho + eps)

        def s_y(g):
            return - tau * eps * ((g / eps + b.log())[None, :] - C / eps).logsumexp(dim=1)

        def s_x(f):
            return - tau * eps * ((f / eps + a.log())[:, None] - C / eps).logsumexp(dim=0)

        return s_x, s_y

    def sinkhorn_procedure(self, T, u, v, a, b, rho, eps, exp_form=True):
        # TODO: sinkhorn must be stabilized for GW if the plan tends to zero, it produces inf values
        if u is None or v is None:
            u, v = self.translate_potential(torch.zeros_like(a), torch.zeros_like(b), T, a, b, rho, eps)
        if exp_form:
            K = (-T / eps).exp()
            exp_form = K.gt(torch.zeros_like(K)).all()
            if ~exp_form:
                del K
        if exp_form:
            u, v = (u / eps).exp(), (v / eps).exp()
            s_x, s_y = self.kl_prox_softmin(K, a, b, rho, eps)
            for j in range(self.nits_sinkhorn):
                u_prev = u.clone()
                v = s_x(u)
                u = s_y(v)
                if eps * (u.log() - u_prev.log()).abs().max().item() < 1e-7:
                    break
            pi = u[:, None] * v[None, :] * K * a[:, None] * b[None, :]
            u, v = eps * u.log(), eps * v.log()

        if ~exp_form:
            s_x, s_y = self.aprox_softmin(T, a, b, rho, eps)
            for j in range(self.nits_sinkhorn):
                u_prev = u.clone()
                v = s_x(u)
                u = s_y(v)
                if (u - u_prev).abs().max().item() < 1e-7:
                    break
            pi = ((u[:, None] + v[None, :] - T) / eps).exp() * a[:, None] * b[None, :]
        return u, v, pi

    def propagate_gradient(self, pi, gamma, up, vp, ug, vg, a, Cx, b, Cy, rho, eps):
        torch.set_grad_enabled(True)
        Tp = grad_l2_distortion(pi, Cx, Cy) / pi.sum()
        Tg = grad_l2_distortion(gamma, Cx, Cy) / gamma.sum()
        sp_x, sp_y = self.aprox_softmin(Tg, a, b, rho, eps)
        sg_x, sg_y = self.aprox_softmin(Tp, a, b, rho, eps)
        for i in range(3):
            vp, vg = sp_x(up), sg_x(ug)
            up, ug = sp_y(vp), sg_y(vg)
        pi = ((up[:, None] + vp[None, :] - Tg) / eps).exp() * a[:, None] * b[None, :]
        gamma = ((ug[:, None] + vg[None, :] - Tp) / eps).exp() * a[:, None] * b[None, :]
        return pi / pi.sum(), gamma / gamma.sum()

    def tlb_sinkhorn(self, a, Cx, b, Cy, rho, eps, init=None):
        torch.set_grad_enabled(False)
        # Initialize plan and local cost
        pi = self.init_plan(a, b, init=init)
        ug, vg, up, vp = None, None, None, None
        for i in range(self.nits):
            pi_prev = pi.clone()
            Tp = grad_l2_distortion(pi, Cx, Cy) / pi.sum()
            ug, vg, gamma = self.sinkhorn_procedure(Tp, ug, vg, a, b, rho, eps)
            Tg = grad_l2_distortion(gamma, Cx, Cy) / gamma.sum()
            up, vp, pi = self.sinkhorn_procedure(Tg, up, vp, a, b, rho, eps)
            pi = pi / pi.sum()
            if (pi - pi_prev).abs().max().item() < 1e-7:
                break
        gamma = gamma / gamma.sum()
        if self.gradient:
            pi, gamma = self.propagate_gradient(pi, gamma, up, vp, ug, vg, a, Cx, b, Cy, rho, eps)
        theta = self.rescale_mass_plan(pi, gamma, a, Cx, b, Cy, rho, eps)
        return theta * pi, theta * gamma

    def ugw_sinkhorn(self, a, Cx, b, Cy, rho, eps, init=None):
        torch.set_grad_enabled(False)
        # Initialize plan and local cost
        pi = self.init_plan(a, b, init=init)
        up, vp = None, None
        for i in range(self.nits):
            pi_prev = pi.clone()
            Tp = grad_l2_distortion(pi, Cx, Cy) / pi.sum()
            up, vp, pi = self.sinkhorn_procedure(Tp, up, vp, a, b, rho, eps, exp_form=False)
            pi = pi / pi.sum()
            if (pi - pi_prev).abs().max().item() < 1e-7:
                break
        theta = self.rescale_mass_plan(pi, pi, a, Cx, b, Cy, rho, eps)
        return theta * pi

    def tlb_wfr_sinkhorn(self, a, Cx, b, Cy, rho, eps, init=None):
        torch.set_grad_enabled(False)
        # Initialize plan and local cost
        pi = self.init_plan(a, b, init=init)
        ug, vg, up, vp = None, None, None, None
        for i in range(self.nits):
            pi_prev = pi.clone()
            Tp = (torch.from_numpy(wfr_grad_distortion(pi.cpu().data.numpy(), Cx.cpu().data.numpy(),
                                                      Cy.cpu().data.numpy())) / pi.sum()).cuda()
            assert ~torch.isnan(Tp).any()
            ug, vg, gamma = self.sinkhorn_procedure(Tp, ug, vg, a, b, rho, eps)
            Tg = (torch.from_numpy(wfr_grad_distortion(gamma.cpu().data.numpy(), Cx.cpu().data.numpy(),
                                                      Cy.cpu().data.numpy())) / gamma.sum()).cuda()
            up, vp, pi = self.sinkhorn_procedure(Tg, up, vp, a, b, rho, eps)
            pi = pi / pi.sum()
            if (pi - pi_prev).abs().max().item() < 1e-7:
                break
        gamma = gamma / gamma.sum()
        if self.gradient:
            pi, gamma = self.propagate_gradient(pi, gamma, up, vp, ug, vg, a, Cx, b, Cy, rho, eps)
        theta = self.rescale_mass_plan(pi, gamma, a, Cx, b, Cy, rho, eps)
        return theta * pi, theta * gamma
