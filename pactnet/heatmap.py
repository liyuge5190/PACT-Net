"""Heteroscedastic two-dimensional representation (Section 3.3.1, Eq. 3-7)."""

from __future__ import annotations

import numpy as np


def heatmap_moments(logits: np.ndarray, kappa: float = 0.05, affine: np.ndarray | None = None):
    """Location, anisotropic covariance and visibility from heatmap logits.

    Parameters
    ----------
    logits : (..., H, W) heatmap logits produced by the 2D backbone.
    kappa : softmax temperature (Eq. 3).
    affine : optional (..., 2, 3) map from heatmap coordinates (x = column,
        y = row, pixel centres at integer positions) to image pixels.

    Returns
    -------
    mu : (..., 2) first moment (Eq. 4)
    cov : (..., 2, 2) second central moment (Eq. 5)
    vis : (...,) one minus normalized entropy (Eq. 6)
    """
    logits = np.asarray(logits, dtype=np.float64)
    H, W = logits.shape[-2:]
    flat = logits.reshape(logits.shape[:-2] + (H * W,)) / kappa
    flat = flat - flat.max(axis=-1, keepdims=True)
    p = np.exp(flat)
    p /= p.sum(axis=-1, keepdims=True)

    ys, xs = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    grid = np.stack([xs.ravel(), ys.ravel()], axis=-1)  # (HW, 2)

    mu = p @ grid
    d = grid - mu[..., None, :]
    cov = np.einsum("...n,...ni,...nj->...ij", p, d, d)

    entropy = -(p * np.log(np.clip(p, 1e-300, None))).sum(axis=-1)
    vis = 1.0 - entropy / np.log(H * W)

    if affine is not None:
        A = np.asarray(affine, dtype=np.float64)
        L = A[..., :2, :2]
        mu = np.einsum("...ij,...j->...i", L, mu) + A[..., :2, 2]
        cov = L @ cov @ np.swapaxes(L, -1, -2)
    return mu, cov, vis


def information_matrix(cov: np.ndarray, vis: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Information matrix ``s (Sigma + eps I)^-1`` (Eq. 7)."""
    cov = np.asarray(cov, dtype=np.float64)
    reg = cov + eps * np.eye(2)
    return np.asarray(vis, dtype=np.float64)[..., None, None] * np.linalg.inv(reg)


def isotropic_information(cov: np.ndarray, vis: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Scalar-weight variant used by the ablation without heteroscedastic fusion."""
    tr = np.trace(np.asarray(cov, dtype=np.float64), axis1=-2, axis2=-1) / 2.0 + eps
    return (np.asarray(vis, dtype=np.float64) / tr)[..., None, None] * np.eye(2)
