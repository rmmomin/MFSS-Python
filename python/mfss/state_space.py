"""
State space model with Kalman filtering support.

This module ports the MATLAB ``StateSpace`` class, providing Kalman filtering
and Rauch–Tung–Striebel smoothing with diffuse initialisation and time-varying
system calendars. Measurement errors must remain diagonal for now; contribution
decompositions will follow in future iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .abstract_state_space import AbstractStateSpace


@dataclass
class StateSpace(AbstractStateSpace):
    """
    State space model with known parameters.

    Parameters mirror the MATLAB constructor.  The implementation supports
    diffuse initial conditions, time-varying calendars, and smoothing while
    assuming diagonal measurement error covariance matrices.
    """

    def __init__(
        self,
        Z: np.ndarray,
        H: np.ndarray,
        T: np.ndarray,
        Q: np.ndarray,
        *,
        d: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        a0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(Z, d, beta, H, T, c, gamma, R, Q)

        if a0 is not None:
            self.a0 = a0
        if P0 is not None:
            self.P0 = P0

        self.validate_state_space()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    # Helper slicing utilities -------------------------------------------------
    def _slice_measure_matrix(self, arr: np.ndarray, tau_vec: np.ndarray, idx: int) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 3:
            return arr[:, :, int(tau_vec[idx]) - 1]
        return arr

    def _slice_measure_vector(
        self,
        arr: Optional[np.ndarray],
        tau_vec: np.ndarray,
        idx: int,
        length: int,
    ) -> np.ndarray:
        if arr is None:
            return np.zeros(length, dtype=float)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            if arr.shape[1] == 1:
                return arr[:, 0]
            return arr[:, int(tau_vec[idx]) - 1]
        raise ValueError("Unsupported measurement vector shape.")

    def _slice_state_matrix(self, arr: np.ndarray, tau_vec: np.ndarray, idx: int) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 3:
            return arr[:, :, int(tau_vec[idx]) - 1]
        return arr

    def _slice_state_vector(
        self,
        arr: Optional[np.ndarray],
        tau_vec: np.ndarray,
        idx: int,
        length: int,
    ) -> np.ndarray:
        if arr is None:
            return np.zeros(length, dtype=float)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            if arr.shape[1] == 1:
                return arr[:, 0]
            return arr[:, int(tau_vec[idx]) - 1]
        raise ValueError("Unsupported state vector shape.")

    def _measurement_cholesky(self, H_arr: np.ndarray, tauH: np.ndarray, n: int) -> np.ndarray:
        chol = np.zeros((n, self.p), dtype=float)
        for t in range(n):
            H_t = self._slice_measure_matrix(H_arr, tauH, t)
            diag = np.diag(H_t).astype(float, copy=False)
            diag = np.maximum(diag, 0.0)
            chol[t] = np.sqrt(diag)
        return chol

    def filter(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        w: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, Dict[str, np.ndarray]]:
        """
        Run the Kalman filter (including diffuse initialisation) on possibly
        time-varying systems.

        Returns
        -------
        a : ndarray
            Predicted state means with shape ``(m, T + 1)``; column ``t`` holds
            the mean prior to observing ``y_t`` and the final column is the
            one-step-ahead prediction after the last observation.
        logli : float
            Log-likelihood of the observed data given the model.
        filter_out : dict
            Additional arrays mirroring the MATLAB toolbox output.
        """

        self.validate_state_space()
        y, x, w = self.check_sample(y, x, w)

        if self.a0 is None or self.P0 is None:
            raise ValueError("Initial state (a0) and covariance (P0) must be specified.")

        if not self.tau:
            self.set_invariant_tau()

        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)

        m = self.m or 0
        p = self.p or 0
        n = y.shape[1]
        k = x.shape[0]
        l = w.shape[0]

        tau = self.tau
        tauZ = tau["Z"]
        tauH = tau["H"]
        tau_d = tau["d"]
        tau_beta = tau["beta"]
        tauT = tau["T"]
        tau_c = tau["c"]
        tau_gamma = tau["gamma"]
        tau_R = tau["R"]
        tau_Q = tau["Q"]

        Z_arr = np.asarray(self.Z, dtype=float)
        d_arr = None if self.d is None else np.asarray(self.d, dtype=float)
        beta_arr = np.asarray(self.beta if self.beta is not None else np.zeros((p, 0)), dtype=float)
        H_arr = np.asarray(self.H, dtype=float)
        T_arr = np.asarray(self.T, dtype=float)
        c_arr = None if self.c is None else np.asarray(self.c, dtype=float)
        gamma_arr = np.asarray(self.gamma if self.gamma is not None else np.zeros((m, 0)), dtype=float)
        R_arr = np.asarray(self.R if self.R is not None else np.eye(m), dtype=float)
        Q_arr = np.asarray(self.Q, dtype=float)

        def slice_measure_matrix(arr: np.ndarray, tau_vec: np.ndarray, idx: int) -> np.ndarray:
            if arr.ndim == 3:
                return arr[:, :, int(tau_vec[idx]) - 1]
            return arr

        def slice_measure_vector(arr: Optional[np.ndarray], tau_vec: np.ndarray, idx: int, length: int) -> np.ndarray:
            if arr is None:
                return np.zeros(length, dtype=float)
            if arr.ndim == 1:
                return arr
            if arr.ndim == 2:
                if arr.shape[1] == 1:
                    return arr[:, 0]
                return arr[:, int(tau_vec[idx]) - 1]
            raise ValueError("Unsupported measurement vector shape.")

        def slice_transition_matrix(arr: np.ndarray, tau_vec: np.ndarray, idx: int) -> np.ndarray:
            if arr.ndim == 3:
                return arr[:, :, int(tau_vec[idx]) - 1]
            return arr

        def slice_state_vector(arr: Optional[np.ndarray], tau_vec: np.ndarray, idx: int, length: int) -> np.ndarray:
            if arr is None:
                return np.zeros(length, dtype=float)
            if arr.ndim == 1:
                return arr
            if arr.ndim == 2:
                if arr.shape[1] == 1:
                    return arr[:, 0]
                return arr[:, int(tau_vec[idx]) - 1]
            raise ValueError("Unsupported state vector shape.")

        def slice_state_matrix(arr: np.ndarray, tau_vec: np.ndarray, idx: int) -> np.ndarray:
            if arr.ndim == 3:
                return arr[:, :, int(tau_vec[idx]) - 1]
            return arr

        a = np.zeros((m, n + 1))
        v = np.full((p, n), np.nan)
        LogL = np.zeros((p, n))
        Fd = np.zeros((p, n))
        F = np.zeros((p, n))
        Kd = np.zeros((m, p, n))
        K = np.zeros((m, p, n))
        Pd = np.zeros((m, m, n + 1))
        P = np.zeros((m, m, n + 1))

        a0_vec = np.asarray(self.a0, dtype=float).reshape(-1)
        A0 = self.A0
        R0_init = self.R0
        Q0_init = self.Q0

        if A0 is None:
            Pd0 = np.zeros((m, m))
        else:
            Pd0 = A0 @ A0.T

        if R0_init is None:
            R0_init = np.zeros((m, 0))
        else:
            R0_init = np.asarray(R0_init, dtype=float)

        if Q0_init is None or Q0_init.size == 0:
            Q0_init = np.zeros((R0_init.shape[1], R0_init.shape[1]))
        else:
            Q0_init = np.asarray(Q0_init, dtype=float)
        P0_finite = R0_init @ Q0_init @ R0_init.T

        diffuse_tol = 1e-10

        T0 = slice_transition_matrix(T_arr, tauT, 0)
        c0 = slice_state_vector(c_arr, tau_c, 0, m)
        gamma0 = slice_state_matrix(gamma_arr, tau_gamma, 0)
        R_process0 = slice_state_matrix(R_arr, tau_R, 0)
        Q_process0 = slice_state_matrix(Q_arr, tau_Q, 0)
        w0 = w[:, 0] if l else np.zeros(0, dtype=float)

        a[:, 0] = T0 @ a0_vec + c0 + (gamma0 @ w0 if l else 0.0)
        Pd[:, :, 0] = AbstractStateSpace.enforce_symmetric(T0 @ Pd0 @ T0.T)
        P[:, :, 0] = AbstractStateSpace.enforce_symmetric(
            T0 @ P0_finite @ T0.T + R_process0 @ Q_process0 @ R_process0.T
        )

        diffuse_active = np.any(np.abs(Pd[:, :, 0]) > diffuse_tol)
        iT = 0

        while diffuse_active:
            if iT >= n:
                raise RuntimeError(
                    "StateSpace filter unable to transition from diffuse to standard recursion."
                )
            iT += 1
            col = iT - 1

            Z_t = slice_measure_matrix(Z_arr, tauZ, col)
            H_t = slice_measure_matrix(H_arr, tauH, col)
            if not np.allclose(H_t, np.diag(np.diag(H_t))):
                raise NotImplementedError("Current filter implementation requires diagonal H.")
            d_t = slice_measure_vector(d_arr, tau_d, col, p)
            beta_t = slice_measure_matrix(beta_arr, tau_beta, col)
            x_t = x[:, col] if k else np.zeros(0, dtype=float)

            ati = a[:, col].copy()
            Pstarti = P[:, :, col].copy()
            Pdti = Pd[:, :, col].copy()

            valid = np.nonzero(np.isfinite(y[:, col]))[0]
            for idx in valid:
                z = Z_t[idx, :]
                h_val = H_t[idx, idx]
                beta_row = beta_t[idx, :] if k else np.zeros(0, dtype=float)
                predicted = float(z @ ati + d_t[idx])
                if k:
                    predicted += float(beta_row @ x_t)

                resid = float(y[idx, col] - predicted)
                Fd_val = float(z @ Pdti @ z)
                F_val = float(z @ Pstarti @ z + h_val)
                if F_val <= 0:
                    raise np.linalg.LinAlgError("Innovation variance became non-positive.")

                if abs(Fd_val) > diffuse_tol:
                    Kd_vec = Pdti @ z / Fd_val
                    K_vec = Pstarti @ z / F_val
                    ati = ati + Kd_vec * resid
                    correction = (np.outer(K_vec, Kd_vec) + np.outer(Kd_vec, K_vec) - np.outer(Kd_vec, Kd_vec)) * F_val
                    Pstarti = Pstarti - correction
                    Pdti = Pdti - np.outer(Kd_vec, Kd_vec) * Fd_val
                    LogL[idx, col] = np.log(Fd_val)
                    Kd[:, idx, col] = Kd_vec
                    K[:, idx, col] = K_vec
                else:
                    K_vec = Pstarti @ z / F_val
                    ati = ati + K_vec * resid
                    Pstarti = Pstarti - np.outer(K_vec, K_vec) * F_val
                    LogL[idx, col] = np.log(F_val) + (resid * resid) / F_val
                    K[:, idx, col] = K_vec

                v[idx, col] = resid
                Fd[idx, col] = Fd_val
                F[idx, col] = F_val

            if col + 1 <= n:
                T_next = slice_transition_matrix(T_arr, tauT, col + 1)
                c_next = slice_state_vector(c_arr, tau_c, col + 1, m)
                gamma_next = slice_state_matrix(gamma_arr, tau_gamma, col + 1)
                R_next = slice_state_matrix(R_arr, tau_R, col + 1)
                Q_next = slice_state_matrix(Q_arr, tau_Q, col + 1)
                w_next = w[:, col + 1] if l else np.zeros(0, dtype=float)

                a[:, col + 1] = T_next @ ati + c_next + (gamma_next @ w_next if l else 0.0)
                Pd[:, :, col + 1] = AbstractStateSpace.enforce_symmetric(T_next @ Pdti @ T_next.T)
                P[:, :, col + 1] = AbstractStateSpace.enforce_symmetric(
                    T_next @ Pstarti @ T_next.T + R_next @ Q_next @ R_next.T
                )

            diffuse_active = np.any(np.abs(Pd[:, :, col + 1]) > diffuse_tol) if col + 1 <= n else False

        dt = iT

        for t in range(dt, n):
            Z_t = slice_measure_matrix(Z_arr, tauZ, t)
            H_t = slice_measure_matrix(H_arr, tauH, t)
            if not np.allclose(H_t, np.diag(np.diag(H_t))):
                raise NotImplementedError("Current filter implementation requires diagonal H.")
            d_t = slice_measure_vector(d_arr, tau_d, t, p)
            beta_t = slice_measure_matrix(beta_arr, tau_beta, t)
            x_t = x[:, t] if k else np.zeros(0, dtype=float)

            ati = a[:, t].copy()
            Pti = P[:, :, t].copy()

            valid = np.nonzero(np.isfinite(y[:, t]))[0]
            for idx in valid:
                z = Z_t[idx, :]
                h_val = H_t[idx, idx]
                beta_row = beta_t[idx, :] if k else np.zeros(0, dtype=float)
                predicted = float(z @ ati + d_t[idx])
                if k:
                    predicted += float(beta_row @ x_t)

                resid = float(y[idx, t] - predicted)
                F_val = float(z @ Pti @ z + h_val)
                if F_val <= 0:
                    raise np.linalg.LinAlgError("Innovation variance became non-positive.")

                K_vec = Pti @ z / F_val
                ati = ati + K_vec * resid
                Pti = Pti - np.outer(K_vec, K_vec) * F_val

                v[idx, t] = resid
                F[idx, t] = F_val
                LogL[idx, t] = np.log(F_val) + (resid * resid) / F_val
                K[:, idx, t] = K_vec

            if t + 1 <= n:
                T_next = slice_transition_matrix(T_arr, tauT, t + 1)
                c_next = slice_state_vector(c_arr, tau_c, t + 1, m)
                gamma_next = slice_state_matrix(gamma_arr, tau_gamma, t + 1)
                R_next = slice_state_matrix(R_arr, tau_R, t + 1)
                Q_next = slice_state_matrix(Q_arr, tau_Q, t + 1)
                w_next = w[:, t + 1] if l else np.zeros(0, dtype=float)

                a[:, t + 1] = T_next @ ati + c_next + (gamma_next @ w_next if l else 0.0)
                P[:, :, t + 1] = AbstractStateSpace.enforce_symmetric(
                    T_next @ Pti @ T_next.T + R_next @ Q_next @ R_next.T
                )

        mask = np.isfinite(y)
        observed = int(mask.sum())
        logli = -0.5 * observed * np.log(2.0 * np.pi) - 0.5 * float(np.sum(LogL[mask]))

        filter_out = {
            "a": a,
            "P": P,
            "Pd": Pd,
            "v": v,
            "F": F,
            "Fd": Fd,
            "K": K,
            "Kd": Kd,
            "dt": dt,
        }

        return a, logli, filter_out

    def smooth(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        w: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Run the Rauch–Tung–Striebel smoother with diffuse initialisation.

        Returns
        -------
        alpha : ndarray
            Smoothed state means with shape ``(m, T)``.
        smoother_out : dict
            Additional smoother diagnostics (state covariances, adjoint
            variables, innovations).
        filter_out : dict
            Filter output reused by the smoother (identical to the filter call).
        """

        y, x, w = self.check_sample(y, x, w)
        _, _, filter_out = self.filter(y, x, w)

        a = filter_out["a"]
        P = filter_out["P"]
        Pd = filter_out["Pd"]
        v = filter_out["v"]
        F = filter_out["F"]
        Fd = filter_out["Fd"]
        K = filter_out["K"]
        Kd = filter_out["Kd"]
        dt = int(filter_out["dt"])

        m = self.m or 0
        p = self.p or 0
        g = self.g or 0
        n = y.shape[1]

        tau = self.tau
        tauZ = tau["Z"]
        tauT = tau["T"]
        tauQ = tau["Q"]
        tauR = tau["R"]

        Z_arr = np.asarray(self.Z, dtype=float)
        T_arr = np.asarray(self.T, dtype=float)
        Q_arr = np.asarray(self.Q, dtype=float)
        R_arr = np.asarray(self.R if self.R is not None else np.eye(m), dtype=float)

        def slice_measure_matrix(arr: np.ndarray, tau_vec: np.ndarray, idx: int) -> np.ndarray:
            if arr.ndim == 3:
                return arr[:, :, int(tau_vec[idx]) - 1]
            return arr

        def slice_transition_matrix(arr: np.ndarray, tau_vec: np.ndarray, idx: int) -> np.ndarray:
            if arr.ndim == 3:
                return arr[:, :, int(tau_vec[idx]) - 1]
            return arr

        def slice_state_matrix(arr: np.ndarray, tau_vec: np.ndarray, idx: int) -> np.ndarray:
            if arr.ndim == 3:
                return arr[:, :, int(tau_vec[idx]) - 1]
            return arr

        alpha = np.zeros((m, n))
        V = np.zeros((m, m, n))
        eta = np.zeros((g, n))
        r = np.zeros((m, n))
        N = np.zeros((m, m, n))
        r1_mat = np.zeros((m, n))

        Im = np.eye(m)
        r_t = np.zeros(m)
        N_t = np.zeros((m, m))

        for t in range(n - 1, dt - 1, -1):
            z_slice = slice_measure_matrix(Z_arr, tauZ, t)

            valid = np.nonzero(np.isfinite(y[:, t]))[0][::-1]
            for idx in valid:
                z_row = z_slice[idx, :]
                L_t = Im - np.outer(K[:, idx, t], z_row)
                resid = v[idx, t]
                F_val = F[idx, t]
                if not np.isfinite(F_val) or abs(F_val) < 1e-12:
                    continue
                r_t = z_row * (resid / F_val) + L_t.T @ r_t
                N_t = np.outer(z_row, z_row) / F_val + L_t.T @ N_t @ L_t

            r[:, t] = r_t
            N[:, :, t] = N_t
            alpha[:, t] = a[:, t] + P[:, :, t] @ r[:, t]
            V[:, :, t] = P[:, :, t] - P[:, :, t] @ N[:, :, t] @ P[:, :, t]

            Q_next = slice_state_matrix(Q_arr, tauQ, t + 1)
            R_next = slice_state_matrix(R_arr, tauR, t + 1)
            eta[:, t] = Q_next @ (R_next.T @ r[:, t])

            T_curr = slice_transition_matrix(T_arr, tauT, t)
            r_t = T_curr.T @ r_t
            N_t = AbstractStateSpace.enforce_symmetric(T_curr.T @ N_t @ T_curr)

        r0_t = r_t
        r1_t = np.zeros(m)
        N0_t = N_t
        N1_t = np.zeros((m, m))
        N2_t = np.zeros((m, m))

        for t in range(dt - 1, -1, -1):
            z_slice = slice_measure_matrix(Z_arr, tauZ, t)

            valid = np.nonzero(np.isfinite(y[:, t]))[0][::-1]
            for idx in valid:
                z_row = z_slice[idx, :]
                Fd_val = Fd[idx, t]
                F_val = F[idx, t]

                if Fd_val != 0:
                    Kd_vec = Kd[:, idx, t]
                    K_vec = K[:, idx, t]
                    Ld_t = Im - np.outer(Kd_vec, z_row)
                    L0_t = np.outer(Kd_vec - K_vec, z_row) * (F_val / Fd_val)

                    r1_t = z_row * (v[idx, t] / Fd_val) + L0_t.T @ r0_t + Ld_t.T @ r1_t
                    r0_t = Ld_t.T @ r0_t

                    N0_t = Ld_t.T @ N0_t @ Ld_t
                    N1_prev = N1_t.copy()
                    N2_prev = N2_t.copy()
                    N1_t = (
                        np.outer(z_row, z_row) / Fd_val
                        + Ld_t.T @ N0_t @ L0_t
                        + Ld_t.T @ N1_prev @ Ld_t
                    )
                    N2_t = (
                        np.outer(z_row, z_row) * (F_val / (Fd_val**2))
                        + L0_t.T @ N1_prev @ L0_t
                        + Ld_t.T @ N1_prev @ L0_t
                        + L0_t.T @ N1_prev @ Ld_t
                        + Ld_t.T @ N2_prev @ Ld_t
                    )
                else:
                    K_vec = K[:, idx, t]
                    Lstart = Im - np.outer(K_vec, z_row)
                    r0_t = z_row * (v[idx, t] / F_val) + Lstart.T @ r0_t
                    N0_t = (
                        np.outer(z_row, z_row) / F_val + Lstart.T @ N0_t @ Lstart
                    )

            r[:, t] = r0_t
            r1_mat[:, t] = r1_t
            N[:, :, t] = N0_t

            alpha[:, t] = (
                a[:, t] + P[:, :, t] @ r[:, t] + Pd[:, :, t] @ r1_mat[:, t]
            )
            V[:, :, t] = (
                P[:, :, t]
                - P[:, :, t] @ N0_t @ P[:, :, t]
                - (Pd[:, :, t] @ N1_t @ P[:, :, t]).T
                - P[:, :, t] @ N1_t @ Pd[:, :, t]
                - Pd[:, :, t] @ N2_t @ Pd[:, :, t]
            )

            Q_curr = slice_state_matrix(Q_arr, tauQ, t)
            R_curr = slice_state_matrix(R_arr, tauR, t)
            eta[:, t] = Q_curr @ (R_curr.T @ r[:, t])

            T_curr = slice_transition_matrix(T_arr, tauT, t)
            r0_t = T_curr.T @ r0_t
            r1_t = T_curr.T @ r1_t
            N0_t = T_curr.T @ N0_t @ T_curr
            N1_t = T_curr.T @ N1_t @ T_curr
            N2_t = T_curr.T @ N2_t @ T_curr

        a0_vec = np.asarray(self.a0, dtype=float).reshape(-1)
        if self.Q0 is not None and self.R0 is not None and self.Q0.size and self.R0.size:
            Pstar0 = self.R0 @ self.Q0 @ self.R0.T
        else:
            Pstar0 = np.zeros((m, m))
        if dt > 0 and self.A0 is not None:
            Pd0 = self.A0 @ self.A0.T
            a0tilde = a0_vec + Pstar0 @ r0_t + Pd0 @ r1_t
        else:
            a0tilde = a0_vec + Pstar0 @ r_t

        smoother_out = {
            "alpha": alpha,
            "V": V,
            "eta": eta,
            "r": r,
            "N": N,
            "r1": r1_mat,
            "a0tilde": a0tilde,
        }

        return alpha, smoother_out, filter_out

    def _filter_weights(
        self,
        y: np.ndarray,
        x: np.ndarray,
        w: np.ndarray,
        f_out: Dict[str, np.ndarray],
    ) -> Dict[str, list]:
        """
        Build filter contribution weights mirroring the MATLAB implementation.
        """

        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)

        m = self.m or 0
        p = self.p or 0
        k = self.k or 0
        l = self.l or 0
        n = y.shape[1]

        tau = self.tau
        tauZ = tau["Z"]
        tauH = tau["H"]
        tau_d = tau["d"]
        tau_beta = tau["beta"]
        tauT = tau["T"]
        tau_c = tau["c"]
        tau_gamma = tau["gamma"]

        Z_arr = np.asarray(self.Z, dtype=float)
        H_arr = np.asarray(self.H, dtype=float)
        T_arr = np.asarray(self.T, dtype=float)
        beta_arr = np.asarray(self.beta if self.beta is not None else np.zeros((p, 0)), dtype=float)
        gamma_arr = np.asarray(self.gamma if self.gamma is not None else np.zeros((m, 0)), dtype=float)

        C_diag = self._measurement_cholesky(H_arr, tauH, n)

        weights_y = [[None for _ in range(n)] for _ in range(n + 1)]
        weights_d = [[None for _ in range(n)] for _ in range(n + 1)]
        weights_x = [[None for _ in range(n)] for _ in range(n + 1)]
        weights_c = [[None for _ in range(n + 1)] for _ in range(n + 1)]
        weights_w = [[None for _ in range(n + 1)] for _ in range(n + 1)]
        weights_a0 = [None for _ in range(n + 1)]

        Im = np.eye(m)
        EPS = np.finfo(float).eps ** 2

        K = np.asarray(f_out["K"], dtype=float)
        Kd = np.asarray(f_out["Kd"], dtype=float)
        Fd = np.asarray(f_out["Fd"], dtype=float)
        dt = int(f_out["dt"])

        Kstar = np.zeros((n, m, p))
        Lstar = np.zeros((n + 1, m, m))
        Lstar[0] = self._slice_state_matrix(T_arr, tauT, 0)

        for j in range(n):
            T_next = self._slice_state_matrix(T_arr, tauT, j + 1)
            L_product = Im.copy()
            Kstar_temp = np.zeros((m, p))
            Z_t = self._slice_measure_matrix(Z_arr, tauZ, j)

            for p_idx in range(p - 1, -1, -1):
                if np.isnan(y[p_idx, j]):
                    continue
                if j >= dt or Fd[p_idx, j] == 0:
                    K_vec = K[:, p_idx, j]
                else:
                    K_vec = Kd[:, p_idx, j]
                Kstar_temp[:, p_idx] = L_product @ K_vec
                L_product = L_product @ (Im - np.outer(K_vec, Z_t[p_idx, :]))

            Kstar[j] = T_next @ Kstar_temp
            Lstar[j + 1] = T_next @ L_product

            observed = np.isfinite(y[:, j])
            if not np.any(observed):
                continue

            KCinv = Kstar[j][:, observed].copy()
            denom = C_diag[j, observed]
            with np.errstate(divide="ignore", invalid="ignore"):
                KCinv[:, denom != 0] = KCinv[:, denom != 0] / denom[denom != 0]
            KCinv[:, denom == 0] = 0.0

            # Data contribution
            omega = np.zeros((m, p))
            omega[:, observed] = KCinv * y[observed, j]
            if np.any(np.abs(omega) > EPS):
                weights_y[j + 1][j] = omega

            # Parameter d contribution
            d_vec = self._slice_measure_vector(self.d, tau_d, j, p)
            mat_d = np.zeros((m, p))
            mat_d[:, observed] = -KCinv * d_vec[observed]
            if np.any(np.abs(mat_d) > EPS):
                weights_d[j + 1][j] = mat_d

            # Exogenous measurement contribution
            if k:
                beta_t = self._slice_measure_matrix(beta_arr, tau_beta, j)
                beta_sub = beta_t[observed, :]
                if beta_sub.size:
                    mat_x = -KCinv @ (beta_sub * x[:, j])
                    if np.any(np.abs(mat_x) > EPS):
                        weights_x[j + 1][j] = mat_x

            # Parameter c contribution
            c_vec = self._slice_state_vector(self.c, tau_c, j, m)
            mat_c = np.diag(c_vec)
            if np.any(np.abs(mat_c) > EPS):
                weights_c[j][j] = mat_c

            # Exogenous state contribution
            if l:
                gamma_t = self._slice_state_matrix(gamma_arr, tau_gamma, j)
                mat_w = gamma_t * w[:, j]
                if np.any(np.abs(mat_w) > EPS):
                    weights_w[j][j] = mat_w

        # Final-step contributions
        c_final = np.diag(self._slice_state_vector(self.c, tau_c, n, m))
        if np.any(np.abs(c_final) > EPS):
            weights_c[n][n] = c_final

        if l:
            gamma_final = self._slice_state_matrix(gamma_arr, tau_gamma, n)
            mat_w_final = gamma_final * w[:, n]
            if np.any(np.abs(mat_w_final) > EPS):
                weights_w[n][n] = mat_w_final

        # Propagate effects through the transition
        for j in range(n):
            if weights_c[j][j] is not None:
                mat = Lstar[j + 1] @ weights_c[j][j]
                if np.any(np.abs(mat) > EPS):
                    weights_c[j + 1][j] = mat
            if l and weights_w[j][j] is not None:
                mat = Lstar[j + 1] @ weights_w[j][j]
                if np.any(np.abs(mat) > EPS):
                    weights_w[j + 1][j] = mat

            for t in range(j + 1, n):
                mat = weights_y[t][j]
                if mat is not None:
                    propagated = Lstar[t + 1] @ mat
                    if np.all(np.abs(propagated) < EPS):
                        break
                    weights_y[t + 1][j] = propagated

            for t in range(j + 1, n):
                mat = weights_d[t][j]
                if mat is not None:
                    propagated = Lstar[t + 1] @ mat
                    if np.all(np.abs(propagated) < EPS):
                        break
                    weights_d[t + 1][j] = propagated

            if k:
                for t in range(j + 1, n):
                    mat = weights_x[t][j]
                    if mat is not None:
                        propagated = Lstar[t + 1] @ mat
                        if np.all(np.abs(propagated) < EPS):
                            break
                        weights_x[t + 1][j] = propagated

            for t in range(j + 1, n):
                mat = weights_c[t][j]
                if mat is not None:
                    propagated = Lstar[t + 1] @ mat
                    if np.all(np.abs(propagated) < EPS):
                        break
                    weights_c[t + 1][j] = propagated

            if l:
                for t in range(j + 1, n):
                    mat = weights_w[t][j]
                    if mat is not None:
                        propagated = Lstar[t + 1] @ mat
                        if np.all(np.abs(propagated) < EPS):
                            break
                        weights_w[t + 1][j] = propagated

        # Initial state contribution
        a0_vec = np.asarray(self.a0, dtype=float).reshape(-1)
        a0_mat = self._slice_state_matrix(T_arr, tauT, 0) @ np.diag(a0_vec)
        if np.any(np.abs(a0_mat) > EPS):
            weights_a0[0] = a0_mat

        for t in range(1, n + 1):
            prev = weights_a0[t - 1]
            if prev is None:
                break
            propagated = Lstar[t] @ prev
            if np.all(np.abs(propagated) < EPS):
                break
            weights_a0[t] = propagated

        return {
            "y": weights_y,
            "d": weights_d,
            "x": weights_x,
            "c": weights_c,
            "w": weights_w,
            "a0": weights_a0,
        }

    def decompose_filtered(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        w: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, list]]:
        """
        Decompose filtered states into contributions from data, parameters, and exogenous series.
        """

        y, x, w = self.check_sample(y, x, w)
        _, _, filter_out = self.filter(y, x, w)

        weights = self._filter_weights(y, x, w, filter_out)

        m = self.m or 0
        p = self.p or 0
        k = self.k or 0
        l = self.l or 0
        n = y.shape[1]

        data_contr = np.zeros((m, p, n + 1))
        param_contr = np.zeros((m, n + 1))
        exogM_contr = np.zeros((m, k, n + 1))
        exogS_contr = np.zeros((m, l, n + 1))

        for t in range(n + 1):
            for j in range(n):
                mat = weights["y"][t][j]
                if mat is not None:
                    data_contr[:, :, t] += mat

                mat_d = weights["d"][t][j]
                if mat_d is not None:
                    param_contr[:, t] += np.sum(mat_d, axis=1)

                mat_x = weights["x"][t][j]
                if mat_x is not None and k:
                    exogM_contr[:, :, t] += mat_x

                mat_c = weights["c"][t][j]
                if mat_c is not None:
                    param_contr[:, t] += np.sum(mat_c, axis=1)

                mat_w = weights["w"][t][j]
                if mat_w is not None and l:
                    exogS_contr[:, :, t] += mat_w

            a0_mat = weights["a0"][t]
            if a0_mat is not None:
                param_contr[:, t] += np.sum(a0_mat, axis=1)

        return data_contr, param_contr, exogM_contr, exogS_contr, weights

    def _build_Ldagger(self, y: np.ndarray, f_out: Dict[str, np.ndarray]) -> np.ndarray:
        y = np.asarray(y, dtype=float)

        m = self.m or 0
        p = self.p or 0
        n = y.shape[1]

        tau = self.tau
        tauZ = tau["Z"]
        tauT = tau["T"]

        Z_arr = np.asarray(self.Z, dtype=float)
        T_arr = np.asarray(self.T, dtype=float)
        K = np.asarray(f_out["K"], dtype=float)
        Kd = np.asarray(f_out["Kd"], dtype=float)
        Fd = np.asarray(f_out["Fd"], dtype=float)
        dt = int(f_out["dt"])

        Im = np.eye(m)
        Ldagger = np.zeros((m, m, n))

        for t in range(dt):
            Sstar_prod = Im.copy()
            Z_t = self._slice_measure_matrix(Z_arr, tauZ, t)
            for j in range(p):
                if np.isnan(y[j, t]):
                    continue
                if Fd[j, t] != 0:
                    Sinfty = Im - np.outer(Kd[:, j, t], Z_t[j, :])
                    Sstar = Sinfty
                else:
                    Sstar = Im - np.outer(K[:, j, t], Z_t[j, :])
                Sstar_prod = Sstar @ Sstar_prod
            Ldagger[:, :, t] = self._slice_state_matrix(T_arr, tauT, t + 1) @ Sstar_prod

        for t in range(dt, n):
            Lprod = Im.copy()
            Z_t = self._slice_measure_matrix(Z_arr, tauZ, t)
            for j in range(p):
                if np.isnan(y[j, t]):
                    continue
                K_vec = K[:, j, t]
                Lprod = (Im - np.outer(K_vec, Z_t[j, :])) @ Lprod
            Ldagger[:, :, t] = self._slice_state_matrix(T_arr, tauT, t + 1) @ Lprod

        return Ldagger

    def _build_M0ti(
        self,
        y: np.ndarray,
        f_out: Dict[str, np.ndarray],
        t: int,
        obs_idx: int,
    ) -> np.ndarray:
        y = np.asarray(y, dtype=float)

        m = self.m or 0
        p = self.p or 0

        tauZ = self.tau["Z"]
        Z_arr = np.asarray(self.Z, dtype=float)
        K = np.asarray(f_out["K"], dtype=float)
        Kd = np.asarray(f_out["Kd"], dtype=float)
        F = np.asarray(f_out["F"], dtype=float)
        Fd = np.asarray(f_out["Fd"], dtype=float)

        M0ti = np.zeros((m, p))
        Im = np.eye(m)
        Sstar_prod = Im.copy()
        Z_t = self._slice_measure_matrix(Z_arr, tauZ, t)

        for j in range(obs_idx + 1, p):
            if np.isnan(y[j, t]):
                continue
            Zj = Z_t[j, :]
            if Fd[j, t] != 0:
                Sstar = Im - np.outer(Kd[:, j, t], Zj)
                Mstart = np.zeros(m)
            else:
                Sstar = Im - np.outer(K[:, j, t], Zj)
                Mstart = Zj / F[j, t]
            M0ti[:, j] = Sstar_prod @ Mstart
            Sstar_prod = Sstar_prod @ Sstar.T

        return M0ti

    def _build_smoother_weight_parts(
        self,
        y: np.ndarray,
        f_out: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        y = np.asarray(y, dtype=float)

        m = self.m or 0
        p = self.p or 0
        n = y.shape[1]

        tau = self.tau
        tauZ = tau["Z"]
        tauT = tau["T"]

        Z_arr = np.asarray(self.Z, dtype=float)
        T_arr = np.asarray(self.T, dtype=float)
        K = np.asarray(f_out["K"], dtype=float)
        Kd = np.asarray(f_out["Kd"], dtype=float)
        F = np.asarray(f_out["F"], dtype=float)
        Fd = np.asarray(f_out["Fd"], dtype=float)
        dt = int(f_out["dt"])

        Im = np.eye(m)
        Ip = np.eye(p)

        Aa = np.zeros((p, m, n))
        Mdagger = np.zeros((m, p, n))
        Aainfty = np.zeros((p, m, dt))
        Linfty = np.zeros((m, m, dt))
        Minfty_tilde = np.zeros((m, p, dt))

        Ldagger = self._build_Ldagger(y, f_out)

        for t in range(min(dt, n)):
            Sstar_prod = Im.copy()
            Sinfty_prod = Im.copy()
            Z_t = self._slice_measure_matrix(Z_arr, tauZ, t)
            for j in range(p):
                if np.isnan(y[j, t]):
                    continue
                Z_row = Z_t[j, :]
                Aa[j, :, t] = Z_row @ Sstar_prod
                Aainfty[j, :, t] = Z_row @ Sinfty_prod

                if Fd[j, t] != 0:
                    Sinfty = Im - np.outer(Kd[:, j, t], Z_row)
                    Sstar = Sinfty
                    Minfty_tilde[:, j, t] = Sstar_prod.T @ (Z_row / Fd[j, t])
                else:
                    Sstar = Im - np.outer(K[:, j, t], Z_row)
                    Sinfty = Im
                    Mdagger[:, j, t] = Sstar_prod.T @ (Z_row / F[j, t])

                Sstar_prod = Sstar @ Sstar_prod
                Sinfty_prod = Sinfty @ Sinfty_prod

            if t < dt:
                Linfty[:, :, t] = self._slice_state_matrix(T_arr, tauT, t + 1) @ Sinfty_prod

        for t in range(dt, n):
            Lprod = Im.copy()
            Z_t = self._slice_measure_matrix(Z_arr, tauZ, t)
            for j in range(p):
                if np.isnan(y[j, t]):
                    continue
                Z_row = Z_t[j, :]
                K_vec = K[:, j, t]
                Aa[j, :, t] = Z_row @ Lprod
                Lprod = (Im - np.outer(K_vec, Z_row)) @ Lprod

            mask = np.isfinite(y[:, t])
            Finv = np.zeros((p, p))
            if np.any(mask):
                Finv[np.ix_(mask, mask)] = np.diag(1.0 / F[mask, t])
            Mdagger[:, :, t] = Aa[:, :, t].T @ Finv

        Lzero = np.zeros((m, m, dt))
        Minfty = np.zeros((m, p, dt))
        for t in range(min(dt, n)):
            Mzero = np.zeros((m, p))
            Lzerosum = np.zeros((m, m))
            Z_t = self._slice_measure_matrix(Z_arr, tauZ, t)

            Sstar_prod_stack = np.zeros((m, m, p + 1))
            Sstar_prod_stack[:, :, p] = Im
            for i in range(p - 1, -1, -1):
                Zi = Z_t[i, :]
                if Fd[i, t] != 0:
                    Sstar = Im - np.outer(Kd[:, i, t], Zi)
                else:
                    Sstar = Im - np.outer(K[:, i, t], Zi)
                Sstar_prod_stack[:, :, i] = Sstar.T @ Sstar_prod_stack[:, :, i + 1]

            Sinfty_prod = Im.copy()
            for i in range(p):
                if Fd[i, t] != 0:
                    Zi = Z_t[i, :]
                    S0ti = np.outer(Kd[:, i, t] - K[:, i, t], Zi) * (F[i, t] / Fd[i, t])
                    Sinfty = Im - np.outer(Kd[:, i, t], Zi)
                else:
                    S0ti = np.zeros((m, m))
                    Sinfty = Im

                Mzero += Sinfty_prod @ S0ti.T @ self._build_M0ti(y, f_out, t, i)
                Lzerosum += Sinfty_prod @ S0ti.T @ Sstar_prod_stack[:, :, i + 1]
                Sinfty_prod = Sinfty_prod @ Sinfty

            Minfty[:, :, t] = Minfty_tilde[:, :, t] + Mzero
            Lzero[:, :, t] = self._slice_state_matrix(T_arr, tauT, t + 1) @ Lzerosum.T

        Ay = np.zeros((p, p, n))
        for t in range(n):
            Ay_tilde = np.zeros((p, p))
            Z_t = self._slice_measure_matrix(Z_arr, tauZ, t)
            for j in range(p - 1):
                if np.isnan(y[j, t]):
                    continue
                Lprod = Im.copy()
                for i in range(j + 1, p):
                    if np.isnan(y[i, t]):
                        continue
                    Zi = Z_t[i, :]
                    if t < dt and Fd[j, t] != 0:
                        Ay_tilde[i, j] = Zi @ Lprod @ Kd[:, j, t]
                    else:
                        Ay_tilde[i, j] = Zi @ Lprod @ K[:, j, t]

                    if t < dt and Fd[i, t] != 0:
                        Sk = Im - np.outer(Kd[:, i, t], Zi)
                    else:
                        Sk = Im - np.outer(K[:, i, t], Zi)
                    Lprod = Sk @ Lprod
            Ay[:, :, t] = Ip - Ay_tilde

        return {
            "Ay": Ay,
            "Aa": Aa,
            "Aainfty": Aainfty,
            "Ldagger": Ldagger,
            "Mdagger": Mdagger,
            "Linfty": Linfty,
            "Minfty": Minfty,
            "Lzero": Lzero,
        }

    def _r_weight_recursion(
        self,
        y: np.ndarray,
        x: np.ndarray,
        f_weights: Dict[str, list],
        components: Dict[str, np.ndarray],
        C_diag: np.ndarray,
        T_limit: int,
        otherOmega: Optional[Dict[str, list]],
    ) -> Dict[str, list]:
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)

        m = self.m or 0
        p = self.p or 0
        k = self.k or 0
        l = self.l or 0
        n = y.shape[1]

        beta_arr = np.asarray(self.beta if self.beta is not None else np.zeros((p, 0)), dtype=float)
        gamma_arr = np.asarray(self.gamma if self.gamma is not None else np.zeros((m, 0)), dtype=float)

        zeroMP = np.zeros((m, p))
        zeroMK = np.zeros((m, k))
        zeroMM = np.zeros((m, m))
        zeroML = np.zeros((m, l))

        EPS = np.finfo(float).eps ** 2

        def get_slice(arr: np.ndarray, idx: int) -> Optional[np.ndarray]:
            if arr.ndim < 3 or arr.size == 0 or idx >= arr.shape[2]:
                return None
            return arr[:, :, idx]

        omegar = [[None for _ in range(n)] for _ in range(T_limit)]
        omegard = [[None for _ in range(n)] for _ in range(T_limit)]
        omegarx = [[None for _ in range(n)] for _ in range(T_limit)]
        omegarc = [[None for _ in range(n + 1)] for _ in range(T_limit)]
        omegarw = [[None for _ in range(n + 1)] for _ in range(T_limit)]
        omegara0 = [None for _ in range(T_limit)]

        for iT in range(T_limit - 1, -1, -1):
            Lown_slice = get_slice(components["Lown"], iT)
            Lother_slice = get_slice(components["Lother"], iT) if "Lother" in components else None
            M_slice = get_slice(components["M"], iT)
            Aa_slice = get_slice(components["Aa"], iT)
            Ay_slice = get_slice(components["Ay"], iT)

            for iJ in range(n - 1, -1, -1):
                # Forward propagated effects
                if iT == T_limit - 1 or Lown_slice is None:
                    forward_y = zeroMP
                    forward_d = zeroMP
                    forward_x = zeroMK
                    forward_c = zeroMM
                    forward_w = zeroML
                else:
                    forward_y = (
                        Lown_slice.T @ omegar[iT + 1][iJ]
                        if omegar[iT + 1][iJ] is not None
                        else zeroMP
                    )
                    forward_d = (
                        Lown_slice.T @ omegard[iT + 1][iJ]
                        if omegard[iT + 1][iJ] is not None
                        else zeroMP
                    )
                    forward_x = (
                        Lown_slice.T @ omegarx[iT + 1][iJ]
                        if omegarx[iT + 1][iJ] is not None
                        else zeroMK
                    )
                    forward_c = (
                        Lown_slice.T @ omegarc[iT + 1][iJ]
                        if omegarc[iT + 1][iJ] is not None
                        else zeroMM
                    )
                    forward_w = (
                        Lown_slice.T @ omegarw[iT + 1][iJ]
                        if omegarw[iT + 1][iJ] is not None
                        else zeroML
                    )

                if otherOmega is not None and Lother_slice is not None and iT != n - 1:
                    forward_other_y = (
                        Lother_slice.T @ otherOmega["y"][iT + 1][iJ]
                        if otherOmega["y"][iT + 1][iJ] is not None
                        else zeroMP
                    )
                    forward_other_d = (
                        Lother_slice.T @ otherOmega["d"][iT + 1][iJ]
                        if otherOmega["d"][iT + 1][iJ] is not None
                        else zeroMP
                    )
                    forward_other_x = (
                        Lother_slice.T @ otherOmega["x"][iT + 1][iJ]
                        if otherOmega["x"][iT + 1][iJ] is not None
                        else zeroMK
                    )
                    forward_other_c = (
                        Lother_slice.T @ otherOmega["c"][iT + 1][iJ]
                        if otherOmega["c"][iT + 1][iJ] is not None
                        else zeroMM
                    )
                    forward_other_w = (
                        Lother_slice.T @ otherOmega["w"][iT + 1][iJ]
                        if otherOmega["w"][iT + 1][iJ] is not None
                        else zeroML
                    )
                else:
                    forward_other_y = zeroMP
                    forward_other_d = zeroMP
                    forward_other_x = zeroMK
                    forward_other_c = zeroMM
                    forward_other_w = zeroML

                # Filter effect via a_t
                if M_slice is not None and Aa_slice is not None:
                    filter_y = (
                        -M_slice @ Aa_slice @ f_weights["y"][iT][iJ]
                        if f_weights["y"][iT][iJ] is not None
                        else zeroMP
                    )
                    filter_d = (
                        -M_slice @ Aa_slice @ f_weights["d"][iT][iJ]
                        if f_weights["d"][iT][iJ] is not None
                        else zeroMP
                    )
                    filter_x = (
                        -M_slice @ Aa_slice @ f_weights["x"][iT][iJ]
                        if f_weights["x"][iT][iJ] is not None
                        else zeroMK
                    )
                    filter_c = (
                        -M_slice @ Aa_slice @ f_weights["c"][iT][iJ]
                        if f_weights["c"][iT][iJ] is not None
                        else zeroMM
                    )
                    filter_w = (
                        -M_slice @ Aa_slice @ f_weights["w"][iT][iJ]
                        if f_weights["w"][iT][iJ] is not None
                        else zeroML
                    )
                else:
                    filter_y = zeroMP
                    filter_d = zeroMP
                    filter_x = zeroMK
                    filter_c = zeroMM
                    filter_w = zeroML

                # Contemporaneous effect
                contemp_y = zeroMP
                contemp_d = zeroMP
                contemp_x = zeroMK
                if iT == iJ and Ay_slice is not None and M_slice is not None:
                    valid_y = np.isfinite(y[:, iJ])
                    if np.any(valid_y):
                        Ay_cols = Ay_slice[:, valid_y]
                        denom = C_diag[iT, valid_y]
                        with np.errstate(divide="ignore", invalid="ignore"):
                            scaled = Ay_cols / denom
                        base = M_slice @ scaled
                        contemp_y[:, valid_y] = base * y[valid_y, iJ]
                        d_vec = self._slice_measure_vector(self.d, self.tau["d"], iJ, p)
                        contemp_d[:, valid_y] = -base * d_vec[valid_y]
                        if k:
                            beta_sub = self._slice_measure_matrix(beta_arr, self.tau["beta"], iJ)[valid_y, :]
                            contemp_x = -base @ (beta_sub * x[:, iJ])

                omegar_temp = forward_y + forward_other_y + filter_y + contemp_y
                if np.any(np.abs(omegar_temp) > EPS):
                    omegar[iT][iJ] = omegar_temp

                omegard_temp = forward_d + forward_other_d + filter_d + contemp_d
                if np.any(np.abs(omegard_temp) > EPS):
                    omegard[iT][iJ] = omegard_temp

                omegarx_temp = forward_x + forward_other_x + filter_x + contemp_x
                if np.any(np.abs(omegarx_temp) > EPS):
                    omegarx[iT][iJ] = omegarx_temp

                omegarc_temp = forward_c + forward_other_c + filter_c
                if np.any(np.abs(omegarc_temp) > EPS):
                    omegarc[iT][iJ] = omegarc_temp

                omegarw_temp = forward_w + forward_other_w + filter_w
                if np.any(np.abs(omegarw_temp) > EPS):
                    omegarw[iT][iJ] = omegarw_temp

            # a0 contribution
            if M_slice is not None and Aa_slice is not None:
                if iT != T_limit - 1 and Lown_slice is not None and omegara0[iT + 1] is not None:
                    forward_a0 = Lown_slice.T @ omegara0[iT + 1]
                else:
                    forward_a0 = zeroMM
                if otherOmega is not None and Lother_slice is not None and iT != n - 1 and otherOmega["a0"][iT + 1] is not None:
                    forward_other_a0 = Lother_slice.T @ otherOmega["a0"][iT + 1]
                else:
                    forward_other_a0 = zeroMM
                base_a0 = f_weights["a0"][iT]
                if base_a0 is not None:
                    filter_a0 = -M_slice @ Aa_slice @ base_a0
                else:
                    filter_a0 = zeroMM
                a0_temp = forward_a0 + forward_other_a0 + filter_a0
                if np.any(np.abs(a0_temp) > EPS):
                    omegara0[iT] = a0_temp

        return {
            "y": omegar,
            "d": omegard,
            "x": omegarx,
            "c": omegarc,
            "w": omegarw,
            "a0": omegara0,
        }

    def _r_weights(
        self,
        y: np.ndarray,
        x: np.ndarray,
        f_out: Dict[str, np.ndarray],
        f_weights: Dict[str, list],
        C_diag: np.ndarray,
    ) -> Tuple[Dict[str, list], Dict[str, list]]:
        components = self._build_smoother_weight_parts(y, f_out)

        m = self.m or 0
        empty = np.zeros((m, m, 0))

        r_comp = {
            "Ay": components["Ay"],
            "Aa": components["Aa"],
            "M": components["Mdagger"],
            "Lown": components["Ldagger"],
            "Lother": empty,
        }
        r = self._r_weight_recursion(y, x, f_weights, r_comp, C_diag, y.shape[1], None)

        r1_comp = {
            "Ay": components["Ay"],
            "Aa": components["Aa"],
            "M": components["Minfty"],
            "Lown": components["Linfty"],
            "Lother": components["Lzero"],
        }
        r1 = self._r_weight_recursion(
            y,
            x,
            f_weights,
            r1_comp,
            C_diag,
            int(f_out["dt"]),
            r,
        )

        return r, r1

    def _smoother_weights(
        self,
        y: np.ndarray,
        x: np.ndarray,
        w: np.ndarray,
        f_out: Dict[str, np.ndarray],
    ) -> Dict[str, list]:
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)

        m = self.m or 0
        p = self.p or 0
        k = self.k or 0
        l = self.l or 0
        n = y.shape[1]

        H_arr = np.asarray(self.H, dtype=float)
        tauH = self.tau["H"]
        C_diag = self._measurement_cholesky(H_arr, tauH, n)

        f_weights = self._filter_weights(y, x, w, f_out)
        r_weight, r1_weight = self._r_weights(y, x, f_out, f_weights, C_diag)

        P = np.asarray(f_out["P"], dtype=float)
        Pd = np.asarray(f_out["Pd"], dtype=float)
        dt = int(f_out["dt"])

        omega = [[None for _ in range(n)] for _ in range(n + 1)]
        omegad = [[None for _ in range(n)] for _ in range(n + 1)]
        omegax = [[None for _ in range(n)] for _ in range(n + 1)]
        omegac = [[None for _ in range(n + 1)] for _ in range(n + 1)]
        omegaw = [[None for _ in range(n + 1)] for _ in range(n + 1)]
        omegaa0 = [None for _ in range(n + 1)]

        zeroMP = np.zeros((m, p))
        zeroMK = np.zeros((m, k))
        zeroMM = np.zeros((m, m))
        zeroML = np.zeros((m, l))

        EPS = np.finfo(float).eps ** 2

        # Diffuse portion
        for iT in range(min(dt, n + 1)):
            for iJ in range(n):
                fw_y = f_weights["y"][iT][iJ] if iT < len(f_weights["y"]) else None
                fw_d = f_weights["d"][iT][iJ] if iT < len(f_weights["d"]) else None
                fw_x = f_weights["x"][iT][iJ] if iT < len(f_weights["x"]) else None
                fw_c = f_weights["c"][iT][iJ] if iT < len(f_weights["c"]) else None
                fw_w = f_weights["w"][iT][iJ] if iT < len(f_weights["w"]) else None

                rw_y = r_weight["y"][iT][iJ] if iT < len(r_weight["y"]) else None
                rw_d = r_weight["d"][iT][iJ] if iT < len(r_weight["d"]) else None
                rw_x = r_weight["x"][iT][iJ] if iT < len(r_weight["x"]) else None
                rw_c = r_weight["c"][iT][iJ] if iT < len(r_weight["c"]) else None
                rw_w = r_weight["w"][iT][iJ] if iT < len(r_weight["w"]) else None

                r1w_y = r1_weight["y"][iT][iJ] if iT < len(r1_weight["y"]) else None
                r1w_d = r1_weight["d"][iT][iJ] if iT < len(r1_weight["d"]) else None
                r1w_x = r1_weight["x"][iT][iJ] if iT < len(r1_weight["x"]) else None
                r1w_c = r1_weight["c"][iT][iJ] if iT < len(r1_weight["c"]) else None
                r1w_w = r1_weight["w"][iT][iJ] if iT < len(r1_weight["w"]) else None

                mat_y = zeroMP.copy()
                if fw_y is not None:
                    mat_y += fw_y
                if rw_y is not None:
                    mat_y += P[:, :, iT] @ rw_y
                if r1w_y is not None:
                    mat_y += Pd[:, :, iT] @ r1w_y
                if np.any(np.abs(mat_y) > EPS):
                    omega[iT][iJ] = mat_y

                mat_d = zeroMP.copy()
                if fw_d is not None:
                    mat_d += fw_d
                if rw_d is not None:
                    mat_d += P[:, :, iT] @ rw_d
                if r1w_d is not None:
                    mat_d += Pd[:, :, iT] @ r1w_d
                if np.any(np.abs(mat_d) > EPS):
                    omegad[iT][iJ] = mat_d

                if k:
                    mat_x = zeroMK.copy()
                    if fw_x is not None:
                        mat_x += fw_x
                    if rw_x is not None:
                        mat_x += P[:, :, iT] @ rw_x
                    if r1w_x is not None:
                        mat_x += Pd[:, :, iT] @ r1w_x
                    if np.any(np.abs(mat_x) > EPS):
                        omegax[iT][iJ] = mat_x

                mat_c = zeroMM.copy()
                if fw_c is not None:
                    mat_c += fw_c
                if rw_c is not None:
                    mat_c += P[:, :, iT] @ rw_c
                if r1w_c is not None:
                    mat_c += Pd[:, :, iT] @ r1w_c
                if np.any(np.abs(mat_c) > EPS):
                    omegac[iT][iJ] = mat_c

                if l:
                    mat_w = zeroML.copy()
                    if fw_w is not None:
                        mat_w += fw_w
                    if rw_w is not None:
                        mat_w += P[:, :, iT] @ rw_w
                    if r1w_w is not None:
                        mat_w += Pd[:, :, iT] @ r1w_w
                    if np.any(np.abs(mat_w) > EPS):
                        omegaw[iT][iJ] = mat_w

            fw_a0 = f_weights["a0"][iT] if iT < len(f_weights["a0"]) else None
            rw_a0 = r_weight["a0"][iT] if iT < len(r_weight["a0"]) else None
            r1w_a0 = r1_weight["a0"][iT] if iT < len(r1_weight["a0"]) else None

            mat_a0 = (fw_a0 if fw_a0 is not None else zeroMM)
            if rw_a0 is not None:
                mat_a0 = mat_a0 + P[:, :, iT] @ rw_a0
            if r1w_a0 is not None:
                mat_a0 = mat_a0 + Pd[:, :, iT] @ r1w_a0
            if np.any(np.abs(mat_a0) > EPS):
                omegaa0[iT] = mat_a0

        # Standard portion
        for iT in range(dt, n):
            for iJ in range(n):
                fw_y = f_weights["y"][iT][iJ]
                rw_y = r_weight["y"][iT][iJ]
                if fw_y is not None or rw_y is not None:
                    mat = (fw_y if fw_y is not None else zeroMP) + (P[:, :, iT] @ rw_y if rw_y is not None else 0)
                    if np.any(np.abs(mat) > EPS):
                        omega[iT][iJ] = mat

                fw_d = f_weights["d"][iT][iJ]
                rw_d = r_weight["d"][iT][iJ]
                if fw_d is not None or rw_d is not None:
                    mat = (fw_d if fw_d is not None else zeroMP) + (P[:, :, iT] @ rw_d if rw_d is not None else 0)
                    if np.any(np.abs(mat) > EPS):
                        omegad[iT][iJ] = mat

                if k:
                    fw_x = f_weights["x"][iT][iJ]
                    rw_x = r_weight["x"][iT][iJ]
                    if fw_x is not None or rw_x is not None:
                        mat = (fw_x if fw_x is not None else zeroMK) + (P[:, :, iT] @ rw_x if rw_x is not None else 0)
                        if np.any(np.abs(mat) > EPS):
                            omegax[iT][iJ] = mat

                fw_c = f_weights["c"][iT][iJ]
                rw_c = r_weight["c"][iT][iJ]
                if fw_c is not None or rw_c is not None:
                    mat = (fw_c if fw_c is not None else zeroMM) + (P[:, :, iT] @ rw_c if rw_c is not None else 0)
                    if np.any(np.abs(mat) > EPS):
                        omegac[iT][iJ] = mat

                if l:
                    fw_w = f_weights["w"][iT][iJ]
                    rw_w = r_weight["w"][iT][iJ]
                    if fw_w is not None or rw_w is not None:
                        mat = (fw_w if fw_w is not None else zeroML) + (P[:, :, iT] @ rw_w if rw_w is not None else 0)
                        if np.any(np.abs(mat) > EPS):
                            omegaw[iT][iJ] = mat

            fw_a0 = f_weights["a0"][iT]
            rw_a0 = r_weight["a0"][iT]
            if fw_a0 is not None or rw_a0 is not None:
                mat = (fw_a0 if fw_a0 is not None else zeroMM) + (P[:, :, iT] @ rw_a0 if rw_a0 is not None else 0)
                if np.any(np.abs(mat) > EPS):
                    omegaa0[iT] = mat

        return {
            "y": omega,
            "d": omegad,
            "x": omegax,
            "c": omegac,
            "w": omegaw,
            "a0": omegaa0,
        }

    def decompose_smoothed(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        w: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, list]]:
        """
        Decompose smoothed states into contributions mirroring the MATLAB toolbox.
        """

        y, x, w = self.check_sample(y, x, w)
        _, smoother_out, filter_out = self.smooth(y, x, w)

        weights = self._smoother_weights(y, x, w, filter_out)

        m = self.m or 0
        p = self.p or 0
        k = self.k or 0
        l = self.l or 0
        n = y.shape[1]

        data_contr = np.zeros((m, p, n))
        param_contr = np.zeros((m, n))
        exogM_contr = np.zeros((m, k, n))
        exogS_contr = np.zeros((m, l, n))

        for t in range(n):
            for j in range(n):
                mat = weights["y"][t][j]
                if mat is not None:
                    data_contr[:, :, t] += mat

                mat_d = weights["d"][t][j]
                if mat_d is not None:
                    param_contr[:, t] += np.sum(mat_d, axis=1)

                mat_x = weights["x"][t][j]
                if mat_x is not None and k:
                    exogM_contr[:, :, t] += mat_x

                mat_c = weights["c"][t][j]
                if mat_c is not None:
                    param_contr[:, t] += np.sum(mat_c, axis=1)

                mat_w = weights["w"][t][j]
                if mat_w is not None and l:
                    exogS_contr[:, :, t] += mat_w

            a0_mat = weights["a0"][t]
            if a0_mat is not None:
                param_contr[:, t] += np.sum(a0_mat, axis=1)

        return data_contr, param_contr, exogM_contr, exogS_contr, weights

__all__ = ["StateSpace"]
