"""Gravity-anchored metric self-calibration (Section 3.4, Eq. 14-16).

During any interval in which no segment touches the ground the centre of
mass is in free fall. Fitting ``c(t) = c0 + v0 t + a t^2 / 2`` to the
reconstructed CoM in each such window identifies the scale factor
``s = g0 / |a|`` and the vertical direction ``g = a / |a|`` of a
reconstruction that is otherwise defined only up to a similarity transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter

from .camera import Camera
from .skeleton import FOOT_POINTS, J, center_of_mass, com_covariance

G0 = 9.81


def find_runs(mask: np.ndarray):
    """Start and end (exclusive) indices of runs of True in a boolean vector."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _fill_nan(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    t = np.arange(x.shape[0])
    flat = out.reshape(x.shape[0], -1)
    for k in range(flat.shape[1]):
        good = np.isfinite(flat[:, k])
        if good.sum() >= 2 and not good.all():
            flat[:, k] = np.interp(t, t[good], flat[good, k])
    return flat.reshape(x.shape)


def _speed(p: np.ndarray, window: int = 7) -> np.ndarray:
    filled = _fill_nan(p)
    w = min(window, (p.shape[0] // 2) * 2 - 1)
    v = savgol_filter(filled, w, 2, deriv=1, axis=0)
    return np.linalg.norm(v, axis=-1)


def detect_contacts(joints: np.ndarray, ratio: float = 0.25, min_contact: int = 8, min_gap: int = 4):
    """Per-foot ground contact from foot-point speed relative to pelvis speed.

    The criterion is invariant to the unknown scale and orientation of the
    reconstruction. Returns a dict ``{"l": mask, "r": mask}`` of (T,) arrays.
    """
    pelvis_speed = _speed(joints[:, J["pelvis"]])
    valid = np.isfinite(joints[:, J["pelvis"]]).all(-1)
    out = {}
    for side, (heel, toe, _) in FOOT_POINTS.items():
        foot_speed = np.minimum(_speed(joints[:, heel]), _speed(joints[:, toe]))
        mask = (foot_speed < ratio * np.maximum(pelvis_speed, 1e-9)) & valid
        for s, e in find_runs(~mask):
            if e - s < min_gap and s > 0 and e < mask.size:
                mask[s:e] = True
        for s, e in find_runs(mask):
            if e - s < min_contact:
                mask[s:e] = False
        out[side] = mask
    return out


@dataclass
class ContactSegmentation:
    contacts: dict
    flight_windows: list
    td: int
    to: int
    stance_side: str


def segment_trial(joints: np.ndarray, min_flight: int = 12, trim: int = 2, **kwargs) -> ContactSegmentation:
    """Locate the takeoff contact and the flight windows of a trial.

    The takeoff contact is the last ground contact in the sequence; the
    windows are the two flight phases preceding it (F1, F2) and the flight
    phase following it (F3), each trimmed by ``trim`` frames at both ends.
    """
    contacts = detect_contacts(joints, **kwargs)
    any_contact = contacts["l"] | contacts["r"]
    runs = find_runs(any_contact)
    if not runs:
        raise ValueError("no ground contact detected")
    td, to = runs[-1]
    stance_side = "l" if contacts["l"][td:to].sum() >= contacts["r"][td:to].sum() else "r"
    valid = np.isfinite(joints).all(axis=(1, 2))
    guarded = any_contact.copy()
    for s, e in runs:
        guarded[max(0, s - 1) : e + 1] = True
    flights = [(s, e) for s, e in find_runs(~guarded & valid) if e - s >= min_flight]
    before = [w for w in flights if w[1] <= td][-2:]
    after = [w for w in flights if w[0] >= to][:1]
    windows = [(s + trim, e - trim) for s, e in before + after if e - s - 2 * trim >= 6]
    return ContactSegmentation(contacts, windows, int(td), int(to), stance_side)


def fit_parabola(c: np.ndarray, t: np.ndarray, var: np.ndarray | None = None):
    """Weighted least-squares fit of Eq. (14), independently per axis.

    Returns (coefficients (3, 3) as rows c0, v0, a; covariance of a (3, 3);
    residual RMS).
    """
    n = t.size
    X = np.stack([np.ones(n), t, 0.5 * t ** 2], axis=1)
    coef = np.zeros((3, 3))
    var_a = np.zeros(3)
    rss = 0.0
    for k in range(3):
        w = np.ones(n) if var is None else 1.0 / np.maximum(var[:, k], 1e-12)
        XtW = X.T * w
        A = XtW @ X
        beta = np.linalg.solve(A, XtW @ c[:, k])
        r = c[:, k] - X @ beta
        sigma2 = float((w * r ** 2).sum() / max(n - 3, 1))
        cov = np.linalg.inv(A) * sigma2
        coef[:, k] = beta
        var_a[k] = cov[2, 2]
        rss += float((r ** 2).sum())
    return coef, np.diag(var_a), np.sqrt(rss / (3 * n))


def fit_shared_acceleration(blocks):
    """Joint fit of Eq. (14) over several flight windows sharing one acceleration vector.

    blocks : list of (t, c, var) with t (n,), c (n, 3), var (n, 3) or None.
    Each window keeps its own position and velocity; the acceleration is common
    to all windows, which is the inverse-variance combination of the windows.
    Returns (a (3,), covariance of a (3, 3)).
    """
    K = len(blocks)
    a = np.zeros(3)
    var_a = np.zeros(3)
    for k in range(3):
        rows, ys, ws = [], [], []
        for i, (t, c, var) in enumerate(blocks):
            X = np.zeros((t.size, 2 * K + 1))
            X[:, 2 * i] = 1.0
            X[:, 2 * i + 1] = t
            X[:, -1] = 0.5 * t ** 2
            rows.append(X)
            ys.append(c[:, k])
            ws.append(np.ones(t.size) if var is None else 1.0 / np.maximum(var[:, k], 1e-12))
        X, y, w = np.vstack(rows), np.concatenate(ys), np.concatenate(ws)
        A = (X.T * w) @ X
        beta = np.linalg.solve(A, (X.T * w) @ y)
        r = y - X @ beta
        sigma2 = float((w * r ** 2).sum() / max(y.size - X.shape[1], 1))
        a[k] = beta[-1]
        var_a[k] = np.linalg.inv(A)[-1, -1] * sigma2
    return a, np.diag(var_a)


def rotation_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector ``u`` onto unit vector ``v``."""
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    axis = np.cross(u, v)
    s = np.linalg.norm(axis)
    c = float(np.dot(u, v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        perp = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(u, [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        return 2.0 * np.outer(perp, perp) - np.eye(3)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]) / s
    return np.eye(3) + s * Kx + (1.0 - c) * Kx @ Kx


@dataclass
class GravityCalibration:
    scale: float
    scale_se: float
    gravity_dir: np.ndarray
    R: np.ndarray
    d: np.ndarray
    windows: list
    window_scales: list
    window_residual_rms: list
    segmentation: ContactSegmentation | None = None
    extra: dict = field(default_factory=dict)

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Eq. (16) followed by the yaw and origin alignment."""
        return self.scale * np.einsum("ij,...j->...i", self.R, X) + self.d

    def apply_cov(self, cov: np.ndarray) -> np.ndarray:
        return self.scale ** 2 * np.einsum("ij,...jk,lk->...il", self.R, cov, self.R)

    def apply_cameras(self, cameras: list[Camera]) -> list[Camera]:
        return [c.transformed(self.scale, self.R, self.d) for c in cameras]

    @property
    def relative_scale_mad(self) -> float:
        s = np.asarray(self.window_scales)
        if s.size < 2:
            return float("nan")
        return float(np.mean(np.abs(s - s.mean())) / s.mean())


def _finish_alignment(Y, R_g, scale, seg):
    """Yaw so that the CoM heading is +x, ground at z = 0, takeoff toe at x = y = 0."""
    Z = scale * np.einsum("ij,...j->...i", R_g, Y)
    com = center_of_mass(Z)
    good = np.flatnonzero(np.isfinite(com).all(-1))
    heading = com[good[-1], :2] - com[good[0], :2]
    psi = np.arctan2(heading[1], heading[0])
    Rz = np.array([[np.cos(-psi), -np.sin(-psi), 0], [np.sin(-psi), np.cos(-psi), 0], [0, 0, 1.0]])
    R = Rz @ R_g
    Z = scale * np.einsum("ij,...j->...i", R, Y)
    heel, toe, _ = FOOT_POINTS[seg.stance_side]
    sole = np.minimum(Z[seg.td : seg.to, heel, 2], Z[seg.td : seg.to, toe, 2])
    ground = np.nanmedian(sole)
    toe_td = Z[seg.td, toe]
    d = np.array([-toe_td[0], -toe_td[1], -ground])
    return R, d


def gravity_anchor(
    joints: np.ndarray,
    cov: np.ndarray | None,
    fps: float,
    sex: str = "M",
    g0: float = G0,
    segmentation: ContactSegmentation | None = None,
    windows: list | None = None,
) -> GravityCalibration:
    """Recover scale, vertical and a runway-aligned frame from a raw trial.

    joints : (T, J, 3) triangulated positions, defined up to a similarity.
    cov : (T, J, 3, 3) posterior covariances from Eq. (11), or None.
    """
    seg = segmentation if segmentation is not None else segment_trial(joints)
    wins = windows if windows is not None else seg.flight_windows
    if not wins:
        raise ValueError("no flight window available for gravity anchoring")
    com = center_of_mass(joints, sex)
    com_var = None
    if cov is not None:
        com_var = np.diagonal(com_covariance(np.nan_to_num(cov), sex), axis1=-2, axis2=-1)

    scales, rms, blocks = [], [], []
    for s, e in wins:
        t = (np.arange(s, e) - 0.5 * (s + e - 1)) / fps
        c = com[s:e]
        ok = np.isfinite(c).all(-1)
        var = None if com_var is None else com_var[s:e][ok]
        coef, _, res = fit_parabola(c[ok], t[ok], var)
        s_k = g0 / np.linalg.norm(coef[2])
        scales.append(s_k)
        rms.append(res * s_k)
        blocks.append((t[ok], c[ok], var))
    a, cov_a = fit_shared_acceleration(blocks)
    na = np.linalg.norm(a)
    gdir = a / na
    var_na = float(gdir @ cov_a @ gdir)
    scale = float(g0 / na)
    scale_se = float(g0 / na ** 2 * np.sqrt(max(var_na, 0.0)))
    R_g = rotation_between(gdir, np.array([0.0, 0.0, -1.0]))
    R, d = _finish_alignment(joints, R_g, scale, seg)
    return GravityCalibration(scale, scale_se, gdir, R, d, list(wins), scales, rms, seg)


def stature_anchor(joints: np.ndarray, cameras: list[Camera], stature: float,
                   segmentation: ContactSegmentation | None = None) -> GravityCalibration:
    """Alternative scale and vertical used when gravity anchoring is disabled.

    The vertical is taken as the mean image-down direction of the cameras and
    the scale from the athlete's stature through the summed lengths of the
    head, trunk, thigh and shank (ratio to stature 0.92).
    """
    seg = segmentation if segmentation is not None else segment_trial(joints)
    down = np.mean([c.R[1] for c in cameras], axis=0)
    down /= np.linalg.norm(down)
    R_g = rotation_between(down, np.array([0.0, 0.0, -1.0]))

    def seglen(a, b):
        return np.nanmedian(np.linalg.norm(joints[:, J[a]] - joints[:, J[b]], axis=-1))

    chain = (seglen("head", "neck") + seglen("neck", "pelvis")
             + 0.5 * (seglen("l_hip", "l_knee") + seglen("r_hip", "r_knee"))
             + 0.5 * (seglen("l_knee", "l_ankle") + seglen("r_knee", "r_ankle")))
    scale = float(0.92 * stature / chain)
    R, d = _finish_alignment(joints, R_g, scale, seg)
    return GravityCalibration(scale, float("nan"), down, R, d, [], [], [], seg)
