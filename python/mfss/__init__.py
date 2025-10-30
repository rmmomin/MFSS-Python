"""
Python port of the MFSS (Mixed Frequency State Space) toolbox.

The package is currently a work in progress.  Foundational abstractions are in
place together with Kalman filtering and smoothing (`StateSpace`) that support
diffuse initialisation and time-varying systems with diagonal measurement noise.
Additional components—accumulator augmentation, contribution decompositions, and
estimation tooling—will be ported in subsequent iterations.
"""

from .abstract_system import AbstractSystem
from .abstract_state_space import AbstractStateSpace
from .accumulator import Accumulator
from .state_space import StateSpace

__all__ = ["AbstractSystem", "AbstractStateSpace", "Accumulator", "StateSpace"]
