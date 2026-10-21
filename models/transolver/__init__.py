# models/transolver/__init__.py
"""
Transolver for 3D point cloud processing
Physics-informed transformer with original slice attention mechanism
"""

from .model import TransolverPressure, LightweightTransolverPressure
from .wrapper import TransolverWrapper

__all__ = ['TransolverPressure', 'LightweightTransolverPressure', 'TransolverWrapper']