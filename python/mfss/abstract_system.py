"""
Core abstractions describing a state space system.

This module contains a Python translation of the MATLAB ``AbstractSystem``
class.  The intent is that higher level state space objects inherit from
``AbstractSystem`` and re-use its validation helpers.  The translation keeps
the public API and the runtime checks organisationally similar to the original
code so future ports can build on the same semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AbstractSystem:
    """
    Base class for systems containing measurements and states.

    The MATLAB version stored most attributes as mutable public properties.
    Python keeps the same spirit but uses regular attributes.  Dimensions
    (`p`, `m`, `g`, `k`, `l`) are initialised to ``None`` and should be filled
    in by subclasses once system matrices are available.
    """

    # Dimensions of the system.  Subclasses determine the concrete values.
    p: Optional[int] = None  # Number of observed series
    m: Optional[int] = None  # Number of states
    g: Optional[int] = None  # Number of shocks
    k: Optional[int] = None  # Number of exogenous measurement series
    l: Optional[int] = None  # Number of exogenous state series

    # Time handling
    time_invariant: bool = True
    n: Optional[int] = None  # Number of observed time periods (used for TVP models)
    stationary_states: Optional[np.ndarray] = None

    # Static class level flags mimicking MATLAB persistent variables
    _use_mex: Optional[bool] = field(default=None, init=False, repr=False)
    _use_parallel: Optional[bool] = field(default=None, init=False, repr=False)

    @property
    def use_mex(self) -> bool:
        """
        Flag indicating whether compiled (MEX) routines should be used.

        The MATLAB implementation checked for the presence of MEX binaries
        every time.  The Python port keeps the lazily cached behaviour but
        defaults to ``False`` because no compiled extensions are available yet.
        """
        if AbstractSystem._use_mex is None:
            # Default is False until dedicated extensions are added.
            AbstractSystem._use_mex = False
        return AbstractSystem._use_mex

    @use_mex.setter
    def use_mex(self, use: bool) -> None:
        AbstractSystem._use_mex = bool(use)

    @property
    def use_parallel(self) -> bool:
        """
        Flag indicating whether parallel routines should be used.

        The default mirrors MATLAB's behaviour and opts out of parallelism
        because it often underperforms for the target workloads.
        """
        if AbstractSystem._use_parallel is None:
            AbstractSystem._use_parallel = False
        return AbstractSystem._use_parallel

    @use_parallel.setter
    def use_parallel(self, use: bool) -> None:
        AbstractSystem._use_parallel = bool(use)

    def check_conforming_system(self, system: "AbstractSystem") -> bool:
        """
        Check if the dimensions of another system match the current object.

        Parameters
        ----------
        system:
            Another instance inheriting from :class:`AbstractSystem`.

        Returns
        -------
        bool
            ``True`` when all dimension checks pass.  ``ValueError`` is raised
            upon the first mismatch.
        """

        if not isinstance(system, AbstractSystem):
            raise TypeError("System must inherit from AbstractSystem.")

        def _assert_equal(name: str, lhs: Optional[int], rhs: Optional[int]) -> None:
            if lhs is None or rhs is None:
                return
            if lhs != rhs:
                raise ValueError(f"{name} dimension mismatch: {lhs} != {rhs}")

        _assert_equal("p", self.p, system.p)
        _assert_equal("m", self.m, system.m)
        _assert_equal("g", self.g, system.g)

        if system.k is not None:
            _assert_equal("k", self.k, system.k)
        if system.l is not None:
            _assert_equal("l", self.l, system.l)

        if self.time_invariant != system.time_invariant:
            raise ValueError("Mismatch in time varying parameter usage.")

        if not self.time_invariant and self.n is not None and system.n is not None:
            if self.n != system.n:
                raise ValueError("Time dimension mismatch.")

        return True

    @staticmethod
    def enforce_symmetric(matrix: np.ndarray) -> np.ndarray:
        """
        Force a matrix to be symmetric.


        Parameters
        ----------
        matrix:
            Input matrix, typically a covariance matrix that accumulated
            numerical noise.

        Returns
        -------
        numpy.ndarray
            Symmetric part ``0.5 * (matrix + matrix.T)``.
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("matrix must be a numpy array.")
        return 0.5 * (matrix + matrix.T)


__all__ = ["AbstractSystem"]
