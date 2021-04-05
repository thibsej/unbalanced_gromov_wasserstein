"""
All the code on functionals derived fron the theory of entropically regularized unbalanced optimal transport.
"""

__version__ = 0.1

from .utils_numpy import euclid_dist
from .utils_pytorch import l2_distortion, grad_l2_distortion, euclid_dist
from .tlb_kl_sinkhorn_solver import TLBSinkhornSolver
from .batch_sinkhorn_solver import BatchSinkhornSolver
