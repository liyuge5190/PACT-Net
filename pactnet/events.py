"""Differentiable contact events and soft phase membership (Section 3.5.1).

This module holds the NumPy versions used after the temporal network; the
PyTorch versions used during training live in ``pactnet.model``.
"""

from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


def soft_argmax(logits: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    """Sub-frame event instant from per-frame logits (Eq. 17); logits (..., T)."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return (p * np.arange(z.shape[-1])).sum(-1)


def phase_memberships(t, tau_td: float, tau_to: float, beta: float = 1.0, impact_frames: float = 1.6):
    """Soft memberships of the approach, contact and flight phases (Eq. 18).

    The three memberships sum to one by construction. ``impact`` marks the
    impact window following touchdown (8 ms, i.e. 1.6 frames at 200 Hz).
    """
    t = np.asarray(t, dtype=np.float64)
    a = sigmoid((t - tau_td) / beta)
    b = sigmoid((t - tau_to) / beta)
    c = sigmoid((t - tau_td - impact_frames) / beta)
    return {
        "pre": 1.0 - a,
        "contact": a * (1.0 - b),
        "post": a * b,
        "impact": a * (1.0 - c),
    }


def events_from_contacts(contact_mask: np.ndarray):
    """Touchdown and toe-off (frames) of the last contact run of a boolean mask."""
    from .gravity import find_runs

    runs = find_runs(contact_mask)
    if not runs:
        raise ValueError("no contact in mask")
    td, to = runs[-1]
    return float(td) - 0.5, float(to) - 0.5


def kinematic_events(joints: np.ndarray, td_guess: float, to_guess: float, fps: float,
                     sex: str = "M", search: int = 8, g0: float = 9.81):
    """Touchdown and toe-off from the centre of mass vertical acceleration.

    Used when no trained temporal network is available. The contact is the
    interval around the foot-contact estimate in which the smoothed vertical
    CoM acceleration exceeds -g/2; crossings are interpolated linearly.
    """
    from .derivatives import local_poly_rows
    from .skeleton import center_of_mass

    c = center_of_mass(np.nan_to_num(joints), sex)
    valid = np.isfinite(joints).all(axis=(1, 2)).astype(np.float64)
    T = c.shape[0]
    lo = int(max(0, np.floor(td_guess) - search))
    hi = int(min(T - 1, np.ceil(to_guess) + search))
    t = np.arange(lo, hi + 1)
    rows = local_poly_rows(t, T, 2.0, valid)
    az = 2.0 * (rows[:, 2] @ c[:, 2]) * fps ** 2
    thr = -0.5 * g0
    above = az > thr
    if not above.any():
        return float(td_guess), float(to_guess)
    k_peak = int(np.argmax(az))
    k0 = k_peak
    while k0 > 0 and above[k0 - 1]:
        k0 -= 1
    k1 = k_peak
    while k1 < above.size - 1 and above[k1 + 1]:
        k1 += 1

    def cross(i, j):
        a, b = az[i], az[j]
        return t[i] + (thr - a) / (b - a) * (t[j] - t[i]) if b != a else float(t[i])

    td = cross(k0 - 1, k0) if k0 > 0 else float(t[k0])
    to = cross(k1, k1 + 1) if k1 < above.size - 1 else float(t[k1])
    return float(td), float(to)
