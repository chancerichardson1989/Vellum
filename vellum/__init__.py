"""
Vellum — Experimental Neural Network Training Framework
========================================================
AWD: Adhesion-Weighted Descent optimizer
LBL: Liminal Boundary Layer constraint module
"""

from vellum.optimizer import AWD
from vellum.layers import LiminalBoundaryLayer, LBLSequential

__version__ = "0.1.0"
__all__ = ["AWD", "LiminalBoundaryLayer", "LBLSequential"]
