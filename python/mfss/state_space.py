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


__all__ = ["StateSpace"]
