"""Trial input/output and preprocessing (stages 1-3 of Figure 1).

The raw per-trial layout is documented in ``docs/data_format.md``. The
preprocessing step runs heteroscedastic fusion, robust triangulation and
gravity-anchored self-calibration, resamples the reference kinematics to the
markerless frame rate and derives the reference events.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .camera import cameras_from_json, cameras_to_json
from .forceplate import contact_events
from .gravity import gravity_anchor, segment_trial, stature_anchor
from .heatmap import information_matrix, isotropic_information
from .triangulation import triangulate_sequence

STRING_KEYS = ("cameras", "sex", "athlete", "venue")


def save_trial(path, trial: dict) -> None:
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in trial.items()})


def _unwrap(v):
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v.item()
    return v


def load_trial(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: _unwrap(z[k]) for k in z.files}


def trial_id(trial: dict) -> str:
    return f"{trial['athlete']}_s{int(trial['session'])}_t{int(trial['trial']):02d}"


def list_trials(root) -> list[Path]:
    return sorted(Path(root).glob("*.npz"))


def information_from_detections(trial: dict, heteroscedastic: bool = True, eps: float = 1.0):
    cov = np.asarray(trial["kp2d_cov"], dtype=np.float64)
    vis = np.asarray(trial["kp2d_vis"], dtype=np.float64)
    if heteroscedastic:
        return information_matrix(cov, vis, eps)
    return isotropic_information(cov, vis, eps)


def fuse_and_triangulate(trial: dict, cfg: dict, view_mask=None, jitter_px: float = 0.0, rng=None):
    """Stages 1-2: information matrices and robust information-form triangulation."""
    cams = cameras_from_json(trial["cameras"])
    mu = np.asarray(trial["kp2d_mu"], dtype=np.float64)
    if jitter_px > 0 and rng is not None:
        mu = mu + rng.normal(0.0, jitter_px, mu.shape)
    info = information_from_detections(trial, cfg["fusion"]["heteroscedastic"], cfg["fusion"]["information_eps"])
    valid = np.isfinite(mu).all(-1)
    if view_mask is not None:
        valid &= np.asarray(view_mask, bool)[:, None, None]
    Y, C = triangulate_sequence(cams, mu, info, valid,
                                robust_scale=cfg["fusion"]["robust_scale"],
                                iterations=cfg["fusion"]["iterations"])
    return Y, C, cams


def reference_at_rate(trial: dict) -> np.ndarray | None:
    """Reference joint centres resampled to the markerless frame times by cubic spline."""
    if "ref_joints" not in trial or np.asarray(trial["ref_joints"]).size == 0:
        return None
    ref = np.asarray(trial["ref_joints"], dtype=np.float64)
    t_ref = np.arange(ref.shape[0]) / float(trial["ref_fps"])
    T = np.asarray(trial["kp2d_mu"]).shape[1]
    t = np.arange(T) / float(trial["fps"])
    out = np.full((T,) + ref.shape[1:], np.nan)
    flat = ref.reshape(ref.shape[0], -1)
    res = out.reshape(T, -1)
    for k in range(flat.shape[1]):
        good = np.isfinite(flat[:, k])
        if good.sum() < 4:
            continue
        cs = CubicSpline(t_ref[good], flat[good, k])
        inside = (t >= t_ref[good][0]) & (t <= t_ref[good][-1])
        res[inside, k] = cs(t[inside])
    return out


def reference_events(trial: dict, ref_joints: np.ndarray | None):
    """Reference touchdown and toe-off in markerless frame units, and their source."""
    fps = float(trial["fps"])
    if bool(trial.get("has_force", False)) and np.asarray(trial["grf"]).size:
        td, to = contact_events(np.asarray(trial["grf"], dtype=np.float64)[:, 2], float(trial["force_fps"]))
        return td * fps, to * fps, "force_platform"
    if "ref_td" in trial and "ref_to" in trial:
        return float(trial["ref_td"]) * fps, float(trial["ref_to"]) * fps, "reference_kinematics"
    if ref_joints is not None:
        seg = segment_trial(ref_joints)
        return seg.td - 0.5, seg.to - 0.5, "reference_kinematics"
    return float("nan"), float("nan"), "none"


def register_yaw_translation(Y: np.ndarray, ref: np.ndarray, frames: np.ndarray):
    """Rotation about the vertical and horizontal translation mapping Y onto ref (heights untouched)."""
    P = Y[frames].reshape(-1, 3)
    Q = ref[frames].reshape(-1, 3)
    m = np.isfinite(P).all(-1) & np.isfinite(Q).all(-1)
    P, Q = P[m, :2], Q[m, :2]
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R2 = (U @ D @ Vt).T
    R = np.eye(3)
    R[:2, :2] = R2
    t = np.zeros(3)
    t[:2] = Q.mean(0) - R2 @ P.mean(0)
    return R, t


def preprocess_trial(trial: dict, cfg: dict) -> dict:
    """Run stages 1-3 and return the cached representation of one trial."""
    fps = float(trial["fps"])
    Y_raw, C_raw, cams = fuse_and_triangulate(trial, cfg)
    seg = segment_trial(Y_raw, trim=cfg["calibration"]["flight_trim"])
    sex = str(trial.get("sex", "M"))
    if cfg["calibration"]["gravity_anchoring"]:
        gc = gravity_anchor(Y_raw, C_raw, fps, sex, g0=cfg["calibration"]["g0"], segmentation=seg)
    else:
        gc = stature_anchor(Y_raw, cams, float(trial["stature"]), segmentation=seg)
    Y = gc.apply(Y_raw)
    C = gc.apply_cov(C_raw)
    R_all, d_all = gc.R * gc.scale, gc.d.copy()

    ref = reference_at_rate(trial)
    registered = False
    if ref is not None:
        approach = np.arange(0, max(seg.td - 5, 10))
        R_y, t_y = register_yaw_translation(Y, ref, approach)
        Y = Y @ R_y.T + t_y
        C = np.einsum("ij,...jk,lk->...il", R_y, C, R_y)
        R_all, d_all = R_y @ R_all, R_y @ d_all + t_y
        registered = True
    td_ref, to_ref, ev_src = reference_events(trial, ref)

    scale = float(np.cbrt(np.linalg.det(R_all)))
    metric_cams = [c.transformed(scale, R_all / scale, d_all) for c in cams]

    grf = np.asarray(trial.get("grf", np.zeros((0, 3))), dtype=np.float64)
    return {
        "id": trial_id(trial),
        "joints": Y.astype(np.float32),
        "cov": C.astype(np.float32),
        "td_init": float(seg.td) - 0.5,
        "to_init": float(seg.to) - 0.5,
        "stance_side": seg.stance_side,
        "scale": float(gc.scale),
        "scale_se": float(gc.scale_se),
        "scale_windows": json.dumps([float(s) for s in gc.window_scales]),
        "scale_mad": float(gc.relative_scale_mad),
        "flight_windows": json.dumps([[int(a), int(b)] for a, b in gc.windows]),
        "ref_joints": (ref if ref is not None else np.zeros((0,))).astype(np.float32),
        "ref_td": float(td_ref),
        "ref_to": float(to_ref),
        "ref_event_source": ev_src,
        "registered": registered,
        "cameras_metric": cameras_to_json(metric_cams),
        "kp2d_mu": np.asarray(trial["kp2d_mu"], dtype=np.float32),
        "kp2d_cov": np.asarray(trial["kp2d_cov"], dtype=np.float32),
        "kp2d_vis": np.asarray(trial["kp2d_vis"], dtype=np.float32),
        "grf": grf.astype(np.float32),
        "force_fps": float(trial.get("force_fps", 0.0)),
        "has_force": bool(trial.get("has_force", False)),
        "fps": fps,
        "mass": float(trial["mass"]),
        "stature": float(trial.get("stature", np.nan)),
        "sex": sex,
        "athlete": str(trial["athlete"]),
        "session": int(trial["session"]),
        "trial": int(trial["trial"]),
        "venue": str(trial.get("venue", "indoor")),
        "level": int(trial.get("level", 0)),
        "truth_scale": float(trial.get("truth_scale", np.nan)),
    }


def loso_folds(athletes) -> list[dict]:
    """Leave-one-subject-out folds with a rotating validation athlete (Appendix E)."""
    ids = sorted(set(athletes))
    folds = []
    for i, test in enumerate(ids):
        val = ids[(i + 1) % len(ids)]
        train = [a for a in ids if a not in (test, val)]
        folds.append({"test": test, "val": val, "train": train})
    return folds
