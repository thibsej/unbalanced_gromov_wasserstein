"""
All the code on functionals derived fron the theory of entropically regularized unbalanced optimal transport.
"""

__version__ = 0.1

from .utils import euclid_dist, generate_measure
from .batch_sinkhorn_solver import BatchSinkhornSolver
from .vanilla_sinkhorn_solver import VanillaSinkhornSolver
from .lower_bound_solver import LowerBoundSolver