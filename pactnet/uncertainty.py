"""Unscented propagation of measurement uncertainty (Section 3.8, Eq. 34).

The state propagated through the descriptor map consists of the joint
coordinates of the frames that enter the descriptors, the metric scale and
the two event instants. Its covariance is block diagonal per coordinate.
Following Julier and Uhlmann (2004) the sigma points are spread by
``sqrt(n + kappa)`` with ``n + kappa = 3``, and the covariance is evaluated
about the central sigma point (the modified form for negative kappa).
"""

from __future__ import annotations

import numpy as np

from .descriptors import DESCRIPTORS, DescriptorConfig, TakeoffDescriptors


def unscented_moments(f, n: int, perturb, spread: float = np.sqrt(3.0), chunk: int = 256):
    """Mean and covariance of f under a Gaussian with axis-aligned square-root directions.

    f : callable mapping a batch of perturbation specifications to (N, m) outputs.
    perturb : callable (indices, signs) -> batch specification passed to f;
        index -1 denotes the central point.
    """
    y0 = f(perturb(np.array([-1]), np.array([0.0])))[0]
    m = y0.size
    W = 1.0 / (2.0 * spread ** 2)
    S = np.zeros((m, m))
    mean_shift = np.zeros(m)
    idx_all = np.arange(n)
    for start in range(0, n, chunk):
        idx = idx_all[start : start + chunk]
        for sign in (1.0, -1.0):
            y = f(perturb(idx, np.full(idx.size, sign * spread)))
            with np.errstate(invalid="ignore", over="ignore"):
                d = y - y0
            d = np.where(np.isfinite(d), d, 0.0)
            mean_shift += W * d.sum(0)
            S += W * d.T @ d
    return y0 + mean_shift, S


def descriptor_intervals(
    Y: np.ndarray,
    sd: np.ndarray,
    tau_td: float,
    tau_to: float,
    fps: float,
    sex: str,
    stance_side: str,
    scale_rel_se: float = 0.0,
    event_sd: tuple = (0.5, 0.5),
    valid: np.ndarray | None = None,
    cfg: DescriptorConfig | None = None,
    frames: tuple | None = None,
    chunk: int = 256,
    z: float = 1.96,
):
    """Descriptors, their covariance and 95 percent predictive half-widths.

    Y : (T, J, 3) refined joint trajectory, sd : (T, J, 3) standard deviations.
    scale_rel_se : relative standard error of the metric scale (Eq. 15).
    event_sd : standard deviations (frames) of the touchdown and toe-off instants.
    """
    T = Y.shape[0]
    if frames is None:
        lo = int(max(0, np.floor(tau_td - (cfg or DescriptorConfig()).approach_window * fps) - 16))
        hi = int(min(T, np.ceil(tau_to) + 17))
    else:
        lo, hi = frames
    Y = np.nan_to_num(np.asarray(Y, dtype=np.float64))
    sdw = np.nan_to_num(np.asarray(sd, dtype=np.float64)[lo:hi], nan=0.0)
    base_desc = TakeoffDescriptors(T, tau_td, tau_to, fps, sex, stance_side, valid, cfg)
    n_coord = sdw.size
    n = n_coord + 3
    std_extra = np.array([scale_rel_se, event_sd[0], event_sd[1]])
    cache = {}

    def model(td, to):
        key = (round(td, 6), round(to, 6))
        if key not in cache:
            cache[key] = TakeoffDescriptors(T, td, to, fps, sex, stance_side, valid, cfg)
        return cache[key]

    def perturb(idx, step):
        return idx, step

    def f(spec):
        idx, step = spec
        out = np.empty((idx.size, len(DESCRIPTORS)))
        coord = idx < n_coord
        if (idx < 0).any():
            return base_desc(Y)[None]
        if coord.any():
            ci = idx[coord]
            batch = np.repeat(Y[None], ci.size, axis=0)
            flat = batch[:, lo:hi].reshape(ci.size, -1)
            flat[np.arange(ci.size), ci] += step[coord] * sdw.ravel()[ci]
            batch[:, lo:hi] = flat.reshape((ci.size,) + Y[lo:hi].shape)
            out[coord] = base_desc(batch)
        for k in np.flatnonzero(~coord):
            e = idx[k] - n_coord
            d = step[k] * std_extra[e]
            if e == 0:
                out[k] = base_desc(Y * (1.0 + d))
            elif e == 1:
                out[k] = model(tau_td + d, tau_to)(Y)
            else:
                out[k] = model(tau_td, tau_to + d)(Y)
        return out

    mean, cov = unscented_moments(f, n, perturb, chunk=chunk)
    point = base_desc(Y)
    half = z * np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return point, mean, cov, half
