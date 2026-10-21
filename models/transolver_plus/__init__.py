# models/transolver_plus/__init__.py
"""
Transolver++ for 3D point cloud processing
Physics-informed transformer with slice attention mechanism
"""

from .model import Transolver_plus_pressure, LightweightTransolver_plus_pressure
from .wrapper import TransolverPlusWrapper

__all__ = ['Transolver_plus_pressure', 'LightweightTransolver_plus_pressure', 'TransolverPlusWrapper']