"""Phase-adaptive derivative estimation (Section 3.5.3, Eq. 23).

Velocities and accelerations are obtained by weighted local quadratic
regression. Each sample is weighted by a tricube kernel of phase-dependent
bandwidth, by its membership of the phase of the target instant and by its
inverse posterior variance, so the estimator never borrows samples across
touchdown or toe-off. Because the weights do not depend on the positions,
the estimator is linear and is returned as operator matrices.
"""

from __future__ import annotations

import numpy as np

from .events import phase_memberships


def tricube(u: np.ndarray) -> np.ndarray:
    a = np.clip(1.0 - np.abs(u) ** 3, 0.0, None)
    return a ** 3


def local_poly_rows(targets, n_src: int, bandwidth, src_weight, ridge: float = 1e-8) -> np.ndarray:
    """Rows mapping samples to local quadratic coefficients.

    targets : (M,) target instants in frames (may be fractional).
    bandwidth : scalar or (M,) half-width in frames.
    src_weight : (n_src,) or (M, n_src) non-negative sample weights.
    Returns (M, 3, n_src) rows for a0, a1, a2 with ``y(t'+t) ~ a0 + a1 t' + a2 t'^2``.
    """
    targets = np.atleast_1d(np.asarray(targets, dtype=np.float64))
    M = targets.size
    src = np.arange(n_src, dtype=np.float64)
    dt = src[None, :] - targets[:, None]
    h = np.broadcast_to(np.asarray(bandwidth, dtype=np.float64), (M,))[:, None]
    w = tricube(dt / (h + 1.0)) * np.broadcast_to(src_weight, (M, n_src))
    X = np.stack([np.ones_like(dt), dt, dt ** 2], axis=-1)  # (M, n, 3)
    A = np.einsum("mnd,mn,mne->mde", X, w, X) + ridge * np.eye(3)
    Xw = X * w[..., None]
    return np.linalg.solve(A, np.swapaxes(Xw, 1, 2))


def phase_adaptive_operators(
    n: int,
    tau_td: float,
    tau_to: float,
    fps: float,
    h_contact: float = 5.0,
    h_flight: float = 15.0,
    beta: float = 1.0,
    sample_weight: np.ndarray | None = None,
    adaptive: bool = True,
    sample_beta: float = 0.25,
):
    """Operators (D0, D1, D2), each (n, n), giving position, velocity (per s) and acceleration (per s^2).

    The estimates of the three phases are blended with the memberships of
    Eq. (18) at the target instant; the samples entering each phase are
    weighted with memberships of sharpness ``sample_beta`` so that samples on
    the far side of an event carry negligible weight. With ``adaptive=False``
    a single bandwidth and no phase gating is used, which corresponds to the
    ablation without phase-adaptive derivatives.
    """
    t = np.arange(n, dtype=np.float64)
    sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if not adaptive:
        rows = local_poly_rows(t, n, h_contact, sw)
        return rows[:, 0], rows[:, 1] * fps, 2.0 * rows[:, 2] * fps ** 2
    m = phase_memberships(t, tau_td, tau_to, beta)
    ms = phase_memberships(t, tau_td, tau_to, sample_beta)
    D = np.zeros((3, n, n))
    for phase, h in (("pre", h_flight), ("contact", h_contact), ("post", h_flight)):
        rows = local_poly_rows(t, n, h, sw * ms[phase])
        D += m[phase][None, :, None] * np.transpose(rows, (1, 0, 2))
    return D[0], D[1] * fps, 2.0 * D[2] * fps ** 2


def rows_at(times, n: int, phase: str, tau_td: float, tau_to: float, fps: float,
            bandwidth: float, beta: float = 0.25, sample_weight=None):
    """Position and velocity rows evaluated at fractional ``times`` within one phase."""
    sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    m = phase_memberships(np.arange(n), tau_td, tau_to, beta)[phase]
    rows = local_poly_rows(times, n, bandwidth, sw * m)
    return rows[:, 0], rows[:, 1] * fps, 2.0 * rows[:, 2] * fps ** 2


def apply(D: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Apply an (n, n) operator along the first axis of y (n, ...); NaNs are treated as zero."""
    return np.tensordot(D, np.nan_to_num(y), axes=(1, 0))


def central_differences(y: np.ndarray, fps: float):
    """First and second central differences along axis 0 (reference derivatives)."""
    v = np.gradient(y, 1.0 / fps, axis=0)
    a = np.gradient(v, 1.0 / fps, axis=0)
    return v, a
