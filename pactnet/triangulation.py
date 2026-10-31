"""Robust information-form triangulation (Section 3.3.2, Eq. 8-12)."""

from __future__ import annotations

import numpy as np

from .camera import Camera


def _ridge(H: np.ndarray) -> np.ndarray:
    scale = np.trace(H, axis1=-2, axis2=-1)[..., None, None] / 3.0
    return (1e-9 * scale + 1e-12) * np.eye(3)


def triangulate_dlt(cameras: list[Camera], mu: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Weighted linear triangulation on undistorted normalized coordinates.

    mu : (V, N, 2) pixel detections, weight : (V, N) with zero for missing views.
    Returns (N, 3); points seen by fewer than two views are NaN.
    """
    V, N, _ = mu.shape
    A = np.zeros((N, 2 * V, 4))
    for v, cam in enumerate(cameras):
        xy = cam.undistort_normalized(np.nan_to_num(mu[v]))
        P = np.hstack([cam.R, cam.t[:, None]])
        w = weight[v][:, None]
        A[:, 2 * v] = w * (xy[:, :1] * P[2] - P[0])
        A[:, 2 * v + 1] = w * (xy[:, 1:] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        X = Xh[:, :3] / Xh[:, 3:]
    X[(weight > 0).sum(0) < 2] = np.nan
    return X


def triangulate_robust(
    cameras: list[Camera],
    mu: np.ndarray,
    info: np.ndarray,
    valid: np.ndarray | None = None,
    robust_scale: float = 12.0,
    iterations: int = 3,
    return_weights: bool = False,
):
    """Heteroscedastic triangulation in information form.

    Parameters
    ----------
    cameras : list of V cameras.
    mu : (V, N, 2) detections in pixels.
    info : (V, N, 2, 2) information matrices from Eq. (7).
    valid : (V, N) boolean mask of usable detections.
    robust_scale : c in Eq. (9) and (12).
    iterations : number of reweighted Gauss-Newton updates.

    Returns
    -------
    X : (N, 3) positions.
    cov : (N, 3, 3) posterior covariance (Eq. 11).
    """
    mu = np.asarray(mu, dtype=np.float64)
    V, N, _ = mu.shape
    if valid is None:
        valid = np.isfinite(mu).all(-1)
    valid = valid & np.isfinite(mu).all(-1)
    info = np.where(valid[..., None, None], np.nan_to_num(info), 0.0)
    mu0 = np.nan_to_num(mu)

    scalar_w = np.trace(info, axis1=-2, axis2=-1)
    X = triangulate_dlt(cameras, mu, np.sqrt(scalar_w) * valid)
    ok = np.isfinite(X).all(-1)
    X = np.where(ok[:, None], X, 0.0)

    c2 = robust_scale ** 2

    def linearize(Xc, wc):
        H = np.zeros((N, 3, 3))
        b = np.zeros((N, 3))
        e = np.zeros((V, N))
        for v, cam in enumerate(cameras):
            uv, Jv = cam.project_with_jacobian(Xc)
            r = mu0[v] - uv
            e[v] = np.einsum("ni,nij,nj->n", r, info[v], r)
            JtL = np.einsum("nij,nik->njk", Jv, info[v])
            H += wc[v][:, None, None] * (JtL @ Jv)
            rhs = r + np.einsum("nij,nj->ni", Jv, Xc)
            b += wc[v][:, None] * np.einsum("njk,nk->nj", JtL, rhs)
        return H, b, e

    w = valid.astype(np.float64)
    for _ in range(iterations):
        H, b, _ = linearize(X, w)
        solvable = ok & ((w > 1e-6).sum(0) >= 2)
        Hs = H[solvable] + _ridge(H[solvable])
        X[solvable] = np.linalg.solve(Hs, b[solvable][..., None])[..., 0]
        e = np.zeros((V, N))
        for v, cam in enumerate(cameras):
            r = mu0[v] - cam.project(X)
            e[v] = np.einsum("ni,nij,nj->n", r, info[v], r)
        w = np.exp(-np.clip(e, 0.0, None) / c2) * valid
    H, _, _ = linearize(X, w)

    cov = np.full((N, 3, 3), np.nan)
    good = ok & ((w > 1e-6).sum(0) >= 2)
    cov[good] = np.linalg.inv(H[good] + _ridge(H[good]))
    X[~good] = np.nan
    if return_weights:
        return X, cov, w
    return X, cov


def triangulate_sequence(cameras, mu, info, valid=None, **kwargs):
    """Triangulate (V, T, J, 2) detections into (T, J, 3) positions and covariances."""
    V, T, Jn, _ = mu.shape
    flat_valid = None if valid is None else valid.reshape(V, T * Jn)
    X, cov = triangulate_robust(
        cameras, mu.reshape(V, T * Jn, 2), info.reshape(V, T * Jn, 2, 2), flat_valid, **kwargs
    )
    return X.reshape(T, Jn, 3), cov.reshape(T, Jn, 3, 3)
