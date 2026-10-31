"""Evaluation metrics (Table 2) and reliability statistics (Sections 4.10-4.11).

All agreement statistics take trial-level vectors; no athlete- or
session-level averaging is applied unless a function says so.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .skeleton import BONE_INDEX, J


def _valid(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


# ----------------------------------------------------------------------------- positions

def mpjpe(pred: np.ndarray, ref: np.ndarray, root: int = J["pelvis"]) -> float:
    """Mean per-joint position error after root alignment, in the input units."""
    p = pred - pred[..., root : root + 1, :]
    r = ref - ref[..., root : root + 1, :]
    e = np.linalg.norm(p - r, axis=-1)
    return float(np.nanmean(e))


def p_mpjpe(pred: np.ndarray, ref: np.ndarray) -> float:
    """MPJPE after per-frame rigid Procrustes alignment (rotation and translation)."""
    errs = []
    for P, R in zip(pred.reshape(-1, *pred.shape[-2:]), ref.reshape(-1, *ref.shape[-2:])):
        m = np.isfinite(P).all(-1) & np.isfinite(R).all(-1)
        if m.sum() < 3:
            continue
        Pm, Rm = P[m] - P[m].mean(0), R[m] - R[m].mean(0)
        U, _, Vt = np.linalg.svd(Pm.T @ Rm)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
        rot = U @ D @ Vt
        errs.append(np.linalg.norm(Pm @ rot - Rm, axis=-1).mean())
    return float(np.mean(errs)) if errs else float("nan")


def lower_limb_mpjpe(pred, ref) -> float:
    idx = [J[n] for n in ("l_hip", "l_knee", "l_ankle", "l_heel", "l_toe",
                          "r_hip", "r_knee", "r_ankle", "r_heel", "r_toe")]
    p = pred - pred[..., J["pelvis"] : J["pelvis"] + 1, :]
    r = ref - ref[..., J["pelvis"] : J["pelvis"] + 1, :]
    return float(np.nanmean(np.linalg.norm(p[..., idx, :] - r[..., idx, :], axis=-1)))


def bone_length_sd(pred: np.ndarray) -> float:
    d = pred[..., BONE_INDEX[:, 1], :] - pred[..., BONE_INDEX[:, 0], :]
    L = np.linalg.norm(d, axis=-1)
    return float(np.nanmean(np.nanstd(L, axis=0)))


# ----------------------------------------------------------------------------- derivatives

def mpjve(pred, ref, fps) -> float:
    vp = np.gradient(pred, 1.0 / fps, axis=0)
    vr = np.gradient(ref, 1.0 / fps, axis=0)
    return float(np.nanmean(np.linalg.norm(vp - vr, axis=-1)))


def acceleration_rmse(pred, ref, fps) -> float:
    ap = np.gradient(np.gradient(pred, 1.0 / fps, axis=0), 1.0 / fps, axis=0)
    ar = np.gradient(np.gradient(ref, 1.0 / fps, axis=0), 1.0 / fps, axis=0)
    return float(np.sqrt(np.nanmean(np.sum((ap - ar) ** 2, axis=-1))))


def relative_error(est, ref) -> float:
    return float(abs(est - ref) / abs(ref) * 100.0)


# ----------------------------------------------------------------------------- agreement

def icc_2_1(x: np.ndarray, y: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement."""
    x, y = _valid(np.asarray(x, float), np.asarray(y, float))
    data = np.stack([x, y], axis=1)
    n, k = data.shape
    grand = data.mean()
    ms_r = k * np.sum((data.mean(1) - grand) ** 2) / (n - 1)
    ms_c = n * np.sum((data.mean(0) - grand) ** 2) / (k - 1)
    resid = data - data.mean(1, keepdims=True) - data.mean(0, keepdims=True) + grand
    ms_e = np.sum(resid ** 2) / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


def bland_altman(est, ref) -> dict:
    est, ref = _valid(np.asarray(est, float), np.asarray(ref, float))
    d = est - ref
    bias, sd = d.mean(), d.std(ddof=1)
    return {"bias": float(bias), "loa_low": float(bias - 1.96 * sd), "loa_high": float(bias + 1.96 * sd)}


def agreement(est, ref) -> dict:
    est, ref = _valid(np.asarray(est, float), np.asarray(ref, float))
    d = est - ref
    out = {
        "n": int(d.size),
        "ref_mean": float(ref.mean()), "ref_sd": float(ref.std(ddof=1)),
        "est_mean": float(est.mean()), "est_sd": float(est.std(ddof=1)),
        "mae": float(np.abs(d).mean()), "rmse": float(np.sqrt((d ** 2).mean())),
        "r": float(np.corrcoef(est, ref)[0, 1]) if d.size > 2 else float("nan"),
        "icc": icc_2_1(est, ref) if d.size > 2 else float("nan"),
    }
    out.update(bland_altman(est, ref))
    return out


# ----------------------------------------------------------------------------- calibration

def coverage(est, half_width, ref) -> float:
    est, half_width, ref = map(lambda a: np.asarray(a, float), (est, half_width, ref))
    m = np.isfinite(est) & np.isfinite(ref) & np.isfinite(half_width)
    return float(np.mean(np.abs(est[m] - ref[m]) <= half_width[m]) * 100.0)


def regression_ece(est, sigma, ref, n_bins: int = 10) -> float:
    """Expected calibration error of Gaussian predictive intervals over ``n_bins`` levels."""
    est, sigma, ref = map(lambda a: np.asarray(a, float), (est, sigma, ref))
    m = np.isfinite(est) & np.isfinite(ref) & (sigma > 0)
    zabs = np.abs(est[m] - ref[m]) / sigma[m]
    levels = (np.arange(n_bins) + 0.5) / n_bins
    freq = np.array([(zabs <= norm.ppf(0.5 + p / 2.0)).mean() for p in levels])
    return float(np.mean(np.abs(freq - levels)))


def gaussian_nll(est, sigma, ref) -> float:
    est, sigma, ref = map(lambda a: np.asarray(a, float), (est, sigma, ref))
    m = np.isfinite(est) & np.isfinite(ref) & (sigma > 0)
    return float(np.mean(0.5 * ((est[m] - ref[m]) / sigma[m]) ** 2 + np.log(sigma[m]) + 0.5 * np.log(2 * np.pi)))


# ----------------------------------------------------------------------------- reliability

def within_athlete_r(est, ref, athlete) -> float:
    est, ref, athlete = np.asarray(est, float), np.asarray(ref, float), np.asarray(athlete)
    e, r = est.copy(), ref.copy()
    for a in np.unique(athlete):
        m = athlete == a
        e[m] -= np.nanmean(est[m])
        r[m] -= np.nanmean(ref[m])
    e, r = _valid(e, r)
    return float(np.corrcoef(e, r)[0, 1])


def clustered_bootstrap_mae(est, ref, athlete, n_boot: int = 2000, seed: int = 0):
    """95 percent interval of the MAE from resampling athletes with replacement."""
    est, ref, athlete = np.asarray(est, float), np.asarray(ref, float), np.asarray(athlete)
    ids = np.unique(athlete)
    groups = [np.abs(est[athlete == a] - ref[athlete == a]) for a in ids]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, ids.size, ids.size)
        err = np.concatenate([groups[i] for i in pick])
        stats.append(np.nanmean(err))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def per_athlete_mae(est, ref, athlete) -> dict:
    est, ref, athlete = np.asarray(est, float), np.asarray(ref, float), np.asarray(athlete)
    vals = np.array([np.nanmean(np.abs(est[athlete == a] - ref[athlete == a])) for a in np.unique(athlete)])
    return {"median": float(np.median(vals)), "q1": float(np.percentile(vals, 25)), "q3": float(np.percentile(vals, 75))}


def sem_and_mdc(est, ref) -> tuple:
    """SEM = RMSE / sqrt(2) and MDC95 = 1.96 * sqrt(2) * SEM (Weir, 2005)."""
    est, ref = _valid(np.asarray(est, float), np.asarray(ref, float))
    rmse = np.sqrt(np.mean((est - ref) ** 2))
    sem = rmse / np.sqrt(2.0)
    return float(sem), float(1.96 * np.sqrt(2.0) * sem)


def between_trial_sd(values, athlete, session) -> float:
    """Pooled within-athlete, within-session standard deviation."""
    values = np.asarray(values, float)
    keys = np.array([f"{a}|{s}" for a, s in zip(athlete, session)])
    ss, dof = 0.0, 0
    for k in np.unique(keys):
        v = values[(keys == k) & np.isfinite(values)]
        if v.size > 1:
            ss += np.sum((v - v.mean()) ** 2)
            dof += v.size - 1
    return float(np.sqrt(ss / dof)) if dof else float("nan")


def attempts_required(half_width: float, between_sd: float) -> tuple:
    """Ratio rho of interval half-width to biological SD and the attempts needed, ceil(rho^2)."""
    rho = half_width / between_sd
    return float(rho), int(max(1, np.ceil(rho ** 2 - 1e-9)))


def between_session_icc(values, athlete, session) -> float:
    """ICC(2,1) between the athlete means of consecutive sessions."""
    values = np.asarray(values, float)
    athlete, session = np.asarray(athlete), np.asarray(session)
    x, y = [], []
    for a in np.unique(athlete):
        s = np.unique(session[athlete == a])
        for s0, s1 in zip(s[:-1], s[1:]):
            x.append(np.nanmean(values[(athlete == a) & (session == s0)]))
            y.append(np.nanmean(values[(athlete == a) & (session == s1)]))
    return icc_2_1(np.array(x), np.array(y)) if len(x) > 2 else float("nan")
