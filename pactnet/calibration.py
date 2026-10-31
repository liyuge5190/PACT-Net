"""Camera calibration (Appendix A).

Intrinsics are estimated from a planar checkerboard (Zhang, 2000) through
OpenCV. Extrinsics are refined by bundle adjustment over the athlete's own
keypoints, which fixes the relative camera geometry up to a similarity
transform; the absolute scale and the vertical are recovered afterwards by
gravity anchoring (``pactnet.gravity``).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .camera import Camera, rodrigues, rotation_to_rvec
from .triangulation import triangulate_robust


def calibrate_intrinsics_checkerboard(images, pattern_size=(9, 6), square_size=1.0):
    """Zhang calibration from grayscale checkerboard images using OpenCV.

    Returns (K, dist, rms). The square size only sets the units of the board
    poses and does not influence K or the distortion coefficients.
    """
    import cv2

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2) * square_size
    obj_points, img_points, shape = [], [], None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    for img in images:
        shape = img.shape[::-1]
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if not found:
            continue
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
    if len(obj_points) < 3:
        raise ValueError("at least three checkerboard views are required")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, shape, None, None)
    return K, dist.ravel()[:5], rms


def bundle_adjust_extrinsics(
    cameras: list[Camera],
    mu: np.ndarray,
    info: np.ndarray,
    valid: np.ndarray | None = None,
    robust_scale: float = 12.0,
    max_points: int = 4000,
    seed: int = 0,
):
    """Refine extrinsics from keypoint correspondences.

    Parameters
    ----------
    cameras : initial cameras (approximate extrinsics, e.g. tape-measured).
    mu : (V, N, 2) keypoint detections pooled over warm-up frames.
    info : (V, N, 2, 2) information matrices (Eq. 7).

    The first camera is held fixed and the result is rescaled so that the
    mean distance of the camera centres from their centroid matches the
    initial configuration; the scale is therefore arbitrary until gravity
    anchoring is applied. Returns (cameras, reprojection_rms_px).
    """
    V, N, _ = mu.shape
    if valid is None:
        valid = np.isfinite(mu).all(-1)
    seen = valid.sum(0) >= 2
    idx = np.flatnonzero(seen)
    if idx.size > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_points, replace=False))
    mu, info, valid = mu[:, idx], info[:, idx], valid[:, idx]
    N = idx.size

    X0, _ = triangulate_robust(cameras, mu, info, valid, robust_scale=robust_scale)
    keep = np.isfinite(X0).all(-1)
    mu, info, valid, X0 = mu[:, keep], info[:, keep], valid[:, keep], X0[keep]
    N = X0.shape[0]

    L = np.linalg.cholesky(np.nan_to_num(info) + 1e-9 * np.eye(2))
    obs_v, obs_n = np.nonzero(valid)
    n_obs = obs_v.size

    def unpack(x):
        cams = [cameras[0]]
        for v in range(1, V):
            p = x[6 * (v - 1) : 6 * v]
            cams.append(Camera(cameras[v].K, rodrigues(p[:3]), p[3:], cameras[v].dist, cameras[v].name))
        pts = x[6 * (V - 1) :].reshape(N, 3)
        return cams, pts

    def residuals(x):
        cams, pts = unpack(x)
        res = np.empty((n_obs, 2))
        for v in range(V):
            sel = obs_v == v
            n = obs_n[sel]
            r = mu[v, n] - cams[v].project(pts[n])
            res[sel] = np.einsum("nji,nj->ni", L[v, n], r)
        return res.ravel()

    x0 = [np.r_[rotation_to_rvec(c.R), c.t] for c in cameras[1:]]
    x0 = np.concatenate(x0 + [X0.ravel()])

    sparsity = lil_matrix((2 * n_obs, x0.size), dtype=np.int8)
    rows = np.arange(n_obs)
    for k in range(6):
        cam_cols = 6 * (obs_v - 1) + k
        m = obs_v > 0
        sparsity[2 * rows[m], cam_cols[m]] = 1
        sparsity[2 * rows[m] + 1, cam_cols[m]] = 1
    for k in range(3):
        cols = 6 * (V - 1) + 3 * obs_n + k
        sparsity[2 * rows, cols] = 1
        sparsity[2 * rows + 1, cols] = 1

    sol = least_squares(
        residuals, x0, jac_sparsity=sparsity, loss="cauchy", f_scale=robust_scale,
        x_scale="jac", method="trf", max_nfev=200,
    )
    cams, _ = unpack(sol.x)

    def spread(cs):
        C = np.stack([c.center for c in cs])
        return np.linalg.norm(C - C.mean(0), axis=1).mean()

    s = spread(cameras) / spread(cams)
    c0 = cams[0].center
    out = [cams[0]] + [c.transformed(s, np.eye(3), (1.0 - s) * c0) for c in cams[1:]]

    Xn, _ = triangulate_robust(out, mu, info, valid, robust_scale=robust_scale)
    res = [mu[v, valid[v]] - out[v].project(Xn[valid[v]]) for v in range(V)]
    res = np.concatenate(res)
    res = res[np.isfinite(res).all(-1)]
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=-1)))) if res.size else float("nan")
    return out, rms
