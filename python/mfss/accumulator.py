"""
Accumulator utilities for enforcing mixed-frequency aggregation.

The MATLAB implementation provides a rich set of augmentation routines that
modify state space systems to respect low frequency (sum/average) constraints.
The full functionality is still being ported.  The current Python version
covers the construction and basic validation of accumulator definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .abstract_system import AbstractSystem


class Accumulator(AbstractSystem):
    """
    State space accumulator definition.

    Parameters
    ----------
    index:
        Sequence of integer indices for the measurement variables that need
        aggregation.  Boolean masks are also accepted and converted to linear
        indices following MATLAB semantics.
    calendar:
        Array describing the alignment of the low-frequency observations.
        The shape must be ``(n + 1, len(index))`` where ``n`` is the number of
        high-frequency periods.
    horizon:
        Array describing the length (in high-frequency periods) of each
        low-frequency observation.  The expected shape matches ``calendar``.
    """

    index: np.ndarray
    calendar: np.ndarray
    horizon: np.ndarray
    accumulator_types: np.ndarray = field(init=False, repr=False)

    def __init__(
        self,
        index: Sequence[int] | np.ndarray,
        calendar: np.ndarray,
        horizon: np.ndarray,
    ) -> None:
        super().__init__()

        index_array = np.asarray(index)
        if index_array.dtype == bool:
            index_array = np.flatnonzero(index_array) + 1  # MATLAB is 1-indexed
        if index_array.ndim != 1:
            raise ValueError("index must be a one-dimensional sequence.")

        calendar = np.asarray(calendar, dtype=int)
        horizon = np.asarray(horizon, dtype=int)
        if calendar.shape != horizon.shape:
            raise ValueError("calendar and horizon must have identical shapes.")
        if calendar.shape[1] != index_array.size:
            raise ValueError("calendar/horizon width must equal len(index).")

        self.index = index_array.astype(int)
        self.calendar = calendar
        self.horizon = horizon

        # Average accumulators have 0 in the calendar array (mirroring MATLAB).
        self.accumulator_types = (self.calendar == 0).any(axis=0).astype(int)

        self.p = int(self.index.max())
        self.n = int(self.calendar.shape[0] - 1)
        self.time_invariant = False

    # ------------------------------------------------------------------
    # Behaviour still to be ported from MATLAB
    # ------------------------------------------------------------------
    def augment_state_space(self, system: "AbstractStateSpace") -> "AbstractStateSpace":
        """
        Augment the state space to enforce accumulation constraints.

        The MATLAB version adds additional states and adjusts the system
        matrices.  The port is not yet complete, so the method raises a
        ``NotImplementedError`` to signal the missing feature.
        """
        from .abstract_state_space import AbstractStateSpace  # local import

        if not isinstance(system, AbstractStateSpace):
            raise TypeError("system must be an AbstractStateSpace instance.")
        raise NotImplementedError("Accumulator augmentation has not been ported yet.")


__all__ = ["Accumulator"]
