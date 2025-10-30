"""
Python translation of the MATLAB ``AbstractStateSpace`` class.

The original code is the backbone for both the analytical state space model
and its estimation routines.  This port focuses on replicating the parameter
book-keeping and validation logic while keeping the public API compatible with
the MATLAB toolbox.  Numerical routines (Kalman filter, smoothing, etc.) will
be ported in follow-up work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .abstract_system import AbstractSystem


ArrayLike = Union[np.ndarray, float, int]


def _as_column(vector: np.ndarray) -> np.ndarray:
    """Ensure ``vector`` is a 2D column array."""
    vector = np.asarray(vector)
    if vector.ndim == 1:
        return vector.reshape(-1, 1)
    return vector


@dataclass
class AbstractStateSpace(AbstractSystem):
    """
    Abstract representation of a state space model.

    Attributes mirror the MATLAB implementation and are documented there in
    detail.  Matrices follow the same naming convention:

    - ``Z, d, beta, H`` describe the observation equation
    - ``T, c, gamma, R, Q`` describe the state equation
    - ``tau`` stores the time-variation calendar for each matrix
    """

    # Measurement equation parameters
    Z: Optional[np.ndarray] = None
    d: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None

    # State equation parameters
    T: Optional[np.ndarray] = None
    c: Optional[np.ndarray] = None
    gamma: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None

    # Calendar for time-varying parameters
    tau: Dict[str, np.ndarray] = field(default_factory=dict)

    # Numerical differentiation controls
    numeric_grad_prec: int = 1
    delta: float = 1e-8

    # Internal storage for initial state
    _a0: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _A0: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _R0: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _Q0: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    # Lists of parameters used across multiple methods
    system_param: Tuple[str, ...] = (
        "Z",
        "d",
        "beta",
        "H",
        "T",
        "c",
        "gamma",
        "R",
        "Q",
    )
    symmetric_params: Tuple[str, ...] = ("H", "Q", "Q0")

    def __init__(  # type: ignore[override]
        self,
        Z: Optional[ArrayLike] = None,
        d: Optional[ArrayLike] = None,
        beta: Optional[ArrayLike] = None,
        H: Optional[ArrayLike] = None,
        T: Optional[ArrayLike] = None,
        c: Optional[ArrayLike] = None,
        gamma: Optional[ArrayLike] = None,
        R: Optional[ArrayLike] = None,
        Q: Optional[ArrayLike] = None,
    ) -> None:
        super().__init__()

        # dataclass default values do not run when overriding __init__, so set
        # them manually.
        self.Z = None
        self.d = None
        self.beta = None
        self.H = None
        self.T = None
        self.c = None
        self.gamma = None
        self.R = None
        self.Q = None
        self.tau = {}
        self.numeric_grad_prec = 1
        self.delta = 1e-8
        self._a0 = None
        self._A0 = None
        self._R0 = None
        self._Q0 = None

        if Z is None and d is None and beta is None and H is None:
            if any(item is not None for item in (T, c, gamma, R, Q)):
                raise ValueError(
                    "Observation equation parameters must be supplied together."
                )
            return

        parameters = self._interpret_constructor_args(
            Z, d, beta, H, T, c, gamma, R, Q
        )
        self.set_system_parameters(parameters)

    # ------------------------------------------------------------------
    # Dependent properties for initial state specification
    # ------------------------------------------------------------------
    @property
    def a0(self) -> Optional[np.ndarray]:
        return self._a0

    @a0.setter
    def a0(self, value: Optional[np.ndarray]) -> None:
        if value is None:
            self._a0 = None
            return

        value = _as_column(np.asarray(value))
        if self.m is not None and value.shape[0] != self.m:
            raise ValueError("a0 should be an m x 1 vector.")
        self._a0 = value

    @property
    def P0(self) -> Optional[np.ndarray]:
        if self._A0 is None or self._R0 is None or self._Q0 is None:
            return None
        diffuse = self._A0 @ self._A0.T
        diffuse[diffuse != 0] = np.inf
        return diffuse + self._R0 @ self._Q0 @ self._R0.T

    @P0.setter
    def P0(self, value: Optional[ArrayLike]) -> None:
        if value is None:
            self._A0 = None
            self._R0 = None
            self._Q0 = None
            return

        if self.m is None:
            raise ValueError("State dimension (m) must be set before assigning P0.")

        matrix = np.asarray(value, dtype=float)
        if matrix.ndim == 0 or matrix.shape == ():
            matrix = np.eye(self.m) * float(matrix)
        elif matrix.ndim == 1:
            if matrix.size != self.m:
                raise ValueError("kappa vector must be length m.")
            matrix = np.diag(matrix.astype(float))
        elif matrix.shape != (self.m, self.m):
            raise ValueError("P0 should be an m x m matrix.")

        diffuse = np.isinf(matrix).any(axis=1)
        select = np.eye(self.m)
        self._A0 = select[:, diffuse]
        self._R0 = select[:, ~diffuse]
        # Guard against the all-diffuse case where the submatrix can be empty.
        trimmed = matrix[~diffuse][:, ~diffuse]
        self._Q0 = trimmed if trimmed.size else np.zeros((0, 0))

    @property
    def Q0(self) -> Optional[np.ndarray]:
        return self._Q0

    @Q0.setter
    def Q0(self, value: Optional[np.ndarray]) -> None:
        if self.P0 is None:
            raise ValueError("Cannot set Q0 without first setting P0.")
        if value is None:
            self._Q0 = None
            return
        self._Q0 = np.asarray(value, dtype=float)

    @property
    def A0(self) -> Optional[np.ndarray]:
        if self._A0 is None or self._A0.size == 0:
            return None
        return self._A0

    @property
    def R0(self) -> Optional[np.ndarray]:
        if self._R0 is None or self._R0.size == 0:
            return None
        return self._R0

    # ------------------------------------------------------------------
    # Public utility methods
    # ------------------------------------------------------------------
    def parameters(self, index: Optional[int] = None) -> List[Optional[np.ndarray]]:
        """
        Return a list of structural parameters (optionally selecting an index).
        """
        params = [
            self.Z,
            self.d,
            self.beta,
            self.H,
            self.T,
            self.c,
            self.gamma,
            self.R,
            self.Q,
            self.a0,
            self.Q0,
        ]
        if index is None:
            return params
        return params[index]

    def lags_in_state(self, variable_position: int) -> Tuple[int, List[int]]:
        """
        Find the lags of a variable in the state space.
        """
        raise NotImplementedError("Lag discovery is not yet ported to Python.")

    def check_sample(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        w: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Validate the data sample against the system configuration.
        """
        if self.p is None:
            raise ValueError("Observation dimension 'p' is not initialised.")

        y = np.asarray(y, dtype=float)
        if y.shape[0] != self.p and y.shape[1] == self.p:
            y = y.T
        if y.shape[0] != self.p:
            raise ValueError(
                "Number of series in y does not match observation dimension."
            )

        # Measurement exogenous input
        if x is not None:
            x = np.asarray(x, dtype=float)
            if self.k is not None and x.shape[0] != self.k and x.shape[1] == self.k:
                x = x.T
        elif self.beta is not None and self.beta.size:
            raise ValueError("Model specified with beta but no x data provided.")
        else:
            x = np.zeros((self.k or 0, y.shape[1]))

        # State exogenous input
        if w is not None:
            w = np.asarray(w, dtype=float)
            if self.l is not None and w.shape[0] != self.l and w.shape[1] == self.l:
                w = w.T
        elif self.gamma is not None and self.gamma.size:
            raise ValueError("Model specified with gamma but no w data provided.")
        else:
            w = np.zeros((self.l or 0, y.shape[1] + 1))

        if not np.isfinite(x).all():
            raise ValueError("Input data x must not contain NaNs or infs.")
        if not np.isfinite(w).all():
            raise ValueError("Input data w must not contain NaNs or infs.")

        if not self.time_invariant:
            if self.n is None:
                raise ValueError("Time varying system requires n to be set.")
            if y.shape[1] != self.n:
                raise ValueError("Length of y does not match time varying calendar.")
            if x.size and x.shape[1] != self.n:
                raise ValueError("Length of x does not match time varying calendar.")
            if w.size and w.shape[1] != self.n + 1:
                raise ValueError("Length of w does not match time varying calendar.")
        else:
            self.n = y.shape[1]
            if x.size and x.shape[1] != self.n:
                raise ValueError("Length of x does not match deduced sample length.")
            if w.size and w.shape[1] != self.n + 1:
                raise ValueError("Length of w does not match deduced sample length.")
            self.set_invariant_tau()

        return y, x, w

    def validate_state_space(self) -> None:
        """
        Validate that all system matrices have consistent dimensions.
        """
        def _expect_size(matrix: np.ndarray, shape: Tuple[int, ...], name: str) -> None:
            if matrix is None:
                raise ValueError(f"{name} matrix is not set.")
            if matrix.shape[: len(shape)] != shape:
                raise ValueError(
                    f"{name} has invalid shape {matrix.shape}; expected {shape}."
                )

        if self.p is None or self.m is None or self.g is None:
            raise ValueError("System dimensions (p, m, g) must be initialised.")

        if self.time_invariant:
            z_shape = (self.p, self.m)
            d_shape = (self.p, 1)
            beta_shape = (self.p, self.k or 0)
            h_shape = (self.p, self.p)
            t_shape = (self.m, self.m)
            c_shape = (self.m, 1)
            gamma_shape = (self.m, self.l or 0)
            r_shape = (self.m, self.g)
            q_shape = (self.g, self.g)

            _expect_size(self.Z, z_shape, "Z")
            _expect_size(_as_column(self.d), d_shape, "d")
            _expect_size(self.beta, beta_shape, "beta")
            _expect_size(self.H, h_shape, "H")
            _expect_size(self.T, t_shape, "T")
            _expect_size(_as_column(self.c), c_shape, "c")
            _expect_size(self.gamma, gamma_shape, "gamma")
            _expect_size(self.R, r_shape, "R")
            _expect_size(self.Q, q_shape, "Q")
            return

        # Time varying dimensions: determine expected calendar length
        max_taus = np.array(
            [np.max(self.tau[name]) for name in self.system_param], dtype=int
        )

        _expect_size(self.Z, (self.p, self.m, max_taus[0]), "Z")
        _expect_size(_as_column(self.d), (self.p, max_taus[1]), "d")
        _expect_size(self.beta, (self.p, self.k or 0, max_taus[2]), "beta")
        _expect_size(self.H, (self.p, self.p, max_taus[3]), "H")
        _expect_size(self.T, (self.m, self.m, max_taus[4]), "T")
        _expect_size(_as_column(self.c), (self.m, max_taus[5]), "c")
        _expect_size(self.gamma, (self.m, self.l or 0, max_taus[6]), "gamma")
        _expect_size(self.R, (self.m, self.g, max_taus[7]), "R")
        _expect_size(self.Q, (self.g, self.g, max_taus[8]), "Q")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _interpret_constructor_args(
        self,
        Z: Optional[ArrayLike],
        d: Optional[ArrayLike],
        beta: Optional[ArrayLike],
        H: Optional[ArrayLike],
        T: Optional[ArrayLike],
        c: Optional[ArrayLike],
        gamma: Optional[ArrayLike],
        R: Optional[ArrayLike],
        Q: Optional[ArrayLike],
    ) -> Dict[str, ArrayLike]:
        """
        Handle the different constructor signatures available in MATLAB.
        """
        if isinstance(Z, AbstractStateSpace):
            params = {
                key: getattr(Z, key)
                for key in ("Z", "d", "beta", "H", "T", "c", "gamma", "R", "Q")
            }
            return params

        if isinstance(Z, dict) and d is None and beta is None and H is None:
            missing = [name for name in self.system_param if name not in Z]
            if missing:
                raise ValueError(f"Missing parameters: {', '.join(missing)}")
            return Z  # type: ignore[return-value]

        provided = [Z, d, beta, H, T, c, gamma, R, Q]
        if any(item is None for item in provided):
            raise ValueError("All 9 system matrices must be provided.")
        return {
            "Z": Z,
            "d": d,
            "beta": beta,
            "H": H,
            "T": T,
            "c": c,
            "gamma": gamma,
            "R": R,
            "Q": Q,
        }

    def set_system_parameters(self, parameters: Dict[str, ArrayLike]) -> None:
        """
        Parse and assign system matrices from a MATLAB-style parameter structure.
        """
        struct_flags = {
            name: isinstance(parameters.get(name), dict) for name in self.system_param
        }

        if not any(struct_flags.values()):
            self.time_invariant = True
        else:
            self.time_invariant = False
            tau_lengths = np.zeros(len(self.system_param), dtype=int)
            tau_lengths[
                [i for i, name in enumerate(self.system_param) if struct_flags[name]]
            ] = np.array(
                [
                    len(parameters[name][f"tau{name}"])
                    for name in self.system_param
                    if struct_flags[name]
                ],
                dtype=int,
            )

            subtract = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
            n_candidates = tau_lengths - subtract
            valid_candidates = n_candidates[[i for i, v in enumerate(struct_flags.values()) if v]]
            if valid_candidates.size and not np.all(valid_candidates == valid_candidates.max()):
                raise ValueError("Bad tau specification in system parameters.")
            self.n = int(valid_candidates.max()) if valid_candidates.size else None

        def _set_time_varying(length: int) -> None:
            if self.time_invariant:
                self.time_invariant = False
                self.n = length
            elif self.n != length:
                raise ValueError("Time-varying calendar length mismatch.")

        # Measurement equation
        self._assign_measurement_parameters(parameters, struct_flags, _set_time_varying)
        # State equation
        self._assign_state_parameters(parameters, struct_flags, _set_time_varying)

        if not self.time_invariant:
            tau_dims = np.array(
                [
                    len(self.tau["Z"]),
                    len(self.tau["d"]),
                    len(self.tau["beta"]),
                    len(self.tau["H"]),
                    len(self.tau["T"]) - 1,
                    len(self.tau["c"]) - 1,
                    len(self.tau["gamma"]) - 1,
                    len(self.tau["R"]) - 1,
                    len(self.tau["Q"]) - 1,
                ],
                dtype=int,
            )
            if not np.all(tau_dims == self.n):
                raise ValueError("Inconsistent tau dimensions after assignment.")

    # Separate chunks of the massive MATLAB method to keep things readable.
    def _assign_measurement_parameters(
        self,
        parameters: Dict[str, ArrayLike],
        struct_flags: Dict[str, bool],
        set_time_varying,
    ) -> None:
        Z_param = parameters["Z"]
        if struct_flags.get("Z", False):
            set_time_varying(len(Z_param["tauZ"]))
            if len(Z_param["tauZ"]) != self.n:
                raise ValueError("tauZ length must equal n.")
            self.tau["Z"] = np.asarray(Z_param["tauZ"], dtype=int)
            self.Z = np.asarray(Z_param["Zt"], dtype=float)
        else:
            self.Z = np.asarray(Z_param, dtype=float)
            if not self.time_invariant:
                self.tau["Z"] = np.ones(self.n, dtype=int)
        self.p = self.Z.shape[0]
        self.m = self.Z.shape[1]

        d_param = parameters["d"]
        if d_param is None or (isinstance(d_param, (list, tuple)) and len(d_param) == 0):
            self.d = np.zeros((self.p, 1))
            if not self.time_invariant and self.n is not None:
                self.tau["d"] = np.ones(self.n, dtype=int)
        elif struct_flags.get("d", False):
            set_time_varying(len(d_param["taud"]))
            if len(d_param["taud"]) != self.n:
                raise ValueError("taud length must equal n.")
            self.tau["d"] = np.asarray(d_param["taud"], dtype=int)
            self.d = np.asarray(d_param["dt"], dtype=float)
        else:
            d_array = np.asarray(d_param, dtype=float)
            if d_array.ndim == 2 and d_array.shape[1] > 1:
                set_time_varying(d_array.shape[1])
                self.tau["d"] = np.arange(1, self.n + 1, dtype=int)
            else:
                if d_array.ndim == 1:
                    d_array = d_array.reshape(-1, 1)
                if not self.time_invariant and self.n is not None:
                    self.tau["d"] = np.ones(self.n, dtype=int)
            self.d = d_array

        beta_param = parameters["beta"]
        if beta_param is None or (
            isinstance(beta_param, (list, tuple)) and len(beta_param) == 0
        ):
            self.beta = np.zeros((self.p, 0))
            if not self.time_invariant and self.n is not None:
                self.tau["beta"] = np.ones(self.n, dtype=int)
        elif struct_flags.get("beta", False):
            set_time_varying(len(beta_param["taubeta"]))
            if len(beta_param["taubeta"]) != self.n:
                raise ValueError("taubeta length must equal n.")
            self.tau["beta"] = np.asarray(beta_param["taubeta"], dtype=int)
            self.beta = np.asarray(beta_param["betat"], dtype=float)
        else:
            beta_array = np.asarray(beta_param, dtype=float)
            if beta_array.ndim == 3 and beta_array.shape[2] > 1:
                set_time_varying(beta_array.shape[2])
                self.tau["beta"] = np.arange(1, self.n + 1, dtype=int)
            else:
                if not self.time_invariant and self.n is not None:
                    self.tau["beta"] = np.ones(self.n, dtype=int)
            self.beta = beta_array
        self.k = self.beta.shape[1] if self.beta.size else 0

        H_param = parameters["H"]
        if struct_flags.get("H", False):
            set_time_varying(len(H_param["tauH"]))
            if len(H_param["tauH"]) != self.n:
                raise ValueError("tauH length must equal n.")
            self.tau["H"] = np.asarray(H_param["tauH"], dtype=int)
            self.H = np.asarray(H_param["Ht"], dtype=float)
        else:
            self.H = np.asarray(H_param, dtype=float)
            if not self.time_invariant and self.n is not None:
                self.tau["H"] = np.ones(self.n, dtype=int)

    def _assign_state_parameters(
        self,
        parameters: Dict[str, ArrayLike],
        struct_flags: Dict[str, bool],
        set_time_varying,
    ) -> None:
        T_param = parameters["T"]
        if struct_flags.get("T", False):
            set_time_varying(len(T_param["tauT"]) - 1)
            if len(T_param["tauT"]) != (self.n or 0) + 1:
                raise ValueError("tauT length must equal n + 1.")
            self.tau["T"] = np.asarray(T_param["tauT"], dtype=int)
            self.T = np.asarray(T_param["Tt"], dtype=float)
        else:
            self.T = np.asarray(T_param, dtype=float)
            if not self.time_invariant:
                self.tau["T"] = np.ones((self.n or 0) + 1, dtype=int)

        c_param = parameters["c"]
        if c_param is None or (
            isinstance(c_param, (list, tuple)) and len(c_param) == 0
        ):
            self.c = np.zeros((self.m, 1))
            if not self.time_invariant and self.n is not None:
                self.tau["c"] = np.ones(self.n + 1, dtype=int)
        elif struct_flags.get("c", False):
            set_time_varying(len(c_param["tauc"]) - 1)
            if len(c_param["tauc"]) != (self.n or 0) + 1:
                raise ValueError("tauc length must equal n + 1.")
            self.tau["c"] = np.asarray(c_param["tauc"], dtype=int)
            self.c = np.asarray(c_param["ct"], dtype=float)
        else:
            c_array = np.asarray(c_param, dtype=float)
            if c_array.ndim == 2 and c_array.shape[1] > 1:
                set_time_varying(c_array.shape[1] - 1)
                self.tau["c"] = np.concatenate(
                    [np.arange(1, self.n + 1, dtype=int), np.array([self.n], dtype=int)]
                )
            else:
                if c_array.ndim == 1:
                    c_array = c_array.reshape(-1, 1)
                if not self.time_invariant and self.n is not None:
                    self.tau["c"] = np.ones(self.n + 1, dtype=int)
            self.c = c_array

        gamma_param = parameters["gamma"]
        if gamma_param is None or (
            isinstance(gamma_param, (list, tuple)) and len(gamma_param) == 0
        ):
            self.gamma = np.zeros((self.m, 0))
            if not self.time_invariant and self.n is not None:
                self.tau["gamma"] = np.ones(self.n + 1, dtype=int)
        elif struct_flags.get("gamma", False):
            set_time_varying(len(gamma_param["taugamma"]) - 1)
            if len(gamma_param["taugamma"]) != (self.n or 0) + 1:
                raise ValueError("taugamma length must equal n + 1.")
            self.tau["gamma"] = np.asarray(gamma_param["taugamma"], dtype=int)
            self.gamma = np.asarray(gamma_param["gammat"], dtype=float)
        else:
            gamma_array = np.asarray(gamma_param, dtype=float)
            if gamma_array.ndim == 3 and gamma_array.shape[2] > 1:
                set_time_varying(gamma_array.shape[2] - 1)
                self.tau["gamma"] = np.concatenate(
                    [np.arange(1, self.n + 1, dtype=int), np.array([self.n], dtype=int)]
                )
            else:
                if not self.time_invariant and self.n is not None:
                    self.tau["gamma"] = np.ones(self.n + 1, dtype=int)
            self.gamma = gamma_array
        self.l = self.gamma.shape[1] if self.gamma.size else 0

        Q_param = parameters["Q"]
        if struct_flags.get("Q", False):
            set_time_varying(len(Q_param["tauQ"]) - 1)
            if len(Q_param["tauQ"]) != (self.n or 0) + 1:
                raise ValueError("tauQ length must equal n + 1.")
            self.tau["Q"] = np.asarray(Q_param["tauQ"], dtype=int)
            self.Q = np.asarray(Q_param["Qt"], dtype=float)
        else:
            self.Q = np.asarray(Q_param, dtype=float)
            if not self.time_invariant:
                self.tau["Q"] = np.ones((self.n or 0) + 1, dtype=int)
        self.g = self.Q.shape[0]

        R_param = parameters["R"]
        if R_param is None or (
            isinstance(R_param, (list, tuple)) and len(R_param) == 0
        ):
            if self.m != self.g:
                raise ValueError(
                    "Shock dimension does not match state dimension with no R specified."
                )
            self.R = np.eye(self.m)
            if not self.time_invariant and self.n is not None:
                self.tau["R"] = np.ones(self.n + 1, dtype=int)
        elif struct_flags.get("R", False):
            set_time_varying(len(R_param["tauR"]) - 1)
            if len(R_param["tauR"]) != (self.n or 0) + 1:
                raise ValueError("tauR length must equal n + 1.")
            self.tau["R"] = np.asarray(R_param["tauR"], dtype=int)
            self.R = np.asarray(R_param["Rt"], dtype=float)
        else:
            self.R = np.asarray(R_param, dtype=float)
            if not self.time_invariant and self.n is not None:
                self.tau["R"] = np.ones(self.n + 1, dtype=int)

    def set_time_varying(self, n: int) -> None:
        """
        Set the time dimension for a time-varying system.
        """
        if self.time_invariant:
            self.time_invariant = False
            self.n = n
        else:
            if self.n != n:
                raise ValueError("Time varying calendar length mismatch.")

    def set_invariant_tau(self) -> None:
        """
        Build a calendar of ones for time-invariant systems.
        """
        if self.n is None:
            raise ValueError("Sample size n must be set before building tau.")

        taus: List[np.ndarray] = [np.ones(self.n, dtype=int) for _ in range(4)]
        taus.extend(np.ones(self.n + 1, dtype=int) for _ in range(5))

        for name, tau in zip(self.system_param, taus, strict=False):
            self.tau[name] = tau


__all__ = ["AbstractStateSpace"]
