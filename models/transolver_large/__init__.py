"""
Transolver-Large Model for DrivAerNet Benchmark
Larger version of Transolver with increased capacity:
- n_hidden: 384 (vs 256)
- n_layers: 7 (vs 5)
- n_head: 12 (vs 8)
- mlp_ratio: 2 (vs 1)
- slice_num: 48 (vs 32)
"""

from .wrapper import LargeTransolverWrapper
from .model import LargeTransolverPressure

__all__ = ['LargeTransolverWrapper', 'LargeTransolverPressure']
