"""
All the code on functionals derived fron the theory of entropically regularized unbalanced optimal transport.
"""

__version__ = 0.1

# from .utils import euclid_dist, generate_measure
# from .batch_sinkhorn_solver import BatchSinkhornSolver
# from .vanilla_sinkhorn_solver import VanillaSinkhornSolver
# from .lower_bound_solver import LowerBoundSolver
# from .batch_stable_sinkhorn_solver import BatchStableSinkhornSolver
from .batch_stable_ugw_solver import log_batch_ugw_sinkhorn, \
    exp_batch_ugw_sinkhorn
from .vanilla_ugw_solver import log_ugw_sinkhorn, exp_ugw_sinkhorn
from ._vanilla_utils import *
from ._batch_utils import *
