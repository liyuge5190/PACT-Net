"""Synthetic long jump trials in the raw data layout of ``docs/data_format.md``.

The generator produces a centre of mass trajectory that is in free fall
during every flight phase and is driven by prescribed ground reaction force
profiles during each support phase, an articulated body whose whole-body
centre of mass matches that trajectory exactly (Eq. 13), surface keypoints
displaced from the functional joint centres, heteroscedastic multi-view
detections and a force-platform signal. It is used to exercise the pipeline
end to end and to test individual stages against known ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erf

from .camera import Camera, cameras_to_json, look_at, rodrigues
from .skeleton import J, NUM_JOINTS, center_of_mass

G0 = 9.81

INDOOR_CAMERAS = (
    (-5.60, 4.20, 1.35), (-2.80, 4.60, 1.70), (0.40, 4.20, 1.25),
    (-5.60, -4.20, 1.40), (-2.80, -4.60, 1.75), (0.40, -4.20, 1.30),
)
OUTDOOR_CAMERAS = (
    (-5.20, 3.60, 1.50), (-2.60, 3.80, 1.50), (0.40, 3.60, 1.50),
    (-5.20, -3.60, 1.50), (-2.60, -3.80, 1.50), (0.40, -3.60, 1.50),
)
CAMERA_TARGETS = ((-4.0, 0.0, 1.0), (-1.6, 0.0, 1.0), (0.6, 0.0, 1.0)) * 2


def make_cameras(venue: str = "indoor", focal: float = 1250.0, size=(2048, 1536), rng=None) -> list[Camera]:
    positions = INDOOR_CAMERAS if venue == "indoor" else OUTDOOR_CAMERAS
    K = np.array([[focal, 0.0, size[0] / 2.0], [0.0, focal, size[1] / 2.0], [0.0, 0.0, 1.0]])
    cams = []
    for i, (p, tgt) in enumerate(zip(positions, CAMERA_TARGETS)):
        dist = np.zeros(5)
        if rng is not None:
            dist[:2] = rng.normal([-0.06, 0.01], [0.01, 0.003])
        R = look_at(p, tgt)
        cams.append(Camera(K, R, -R @ np.asarray(p), dist, name=f"C{i + 1}"))
    return cams


@dataclass
class Athlete:
    athlete_id: str
    sex: str
    level: int
    stature: float
    mass: float
    takeoff_side: str
    contact_time: float
    vx_td: float
    dvx: float
    vz_td: float
    vz_to: float
    braking_impulse: float
    impact_peak: float


def sample_athlete(rng: np.random.Generator, index: int) -> Athlete:
    sex = "M" if index % 13 < 7 else "F"
    level = 1 if index % 2 == 0 else 2
    male = sex == "M"
    fast = level == 1
    return Athlete(
        athlete_id=f"A{index + 1:02d}",
        sex=sex,
        level=level,
        stature=float(rng.normal(1.84 if male else 1.74, 0.04)),
        mass=float(rng.normal(74.0 if male else 60.0, 4.0)),
        takeoff_side="l" if rng.random() < 0.7 else "r",
        contact_time=float(rng.normal(0.114 if fast else 0.124, 0.005)),
        vx_td=float(rng.normal(9.25 if male else 8.65, 0.25) + (0.15 if fast else -0.15)),
        dvx=float(rng.normal(1.33, 0.12)),
        vz_td=float(rng.normal(-0.55, 0.10)),
        vz_to=float(rng.normal(3.06, 0.18)),
        braking_impulse=float(rng.normal(0.198, 0.02)),
        impact_peak=float(rng.normal(4.2, 0.4)),
    )


@dataclass
class _Phase:
    kind: str
    foot: str | None
    t0: float
    t1: float
    vx: float = 0.0
    vz_end: float = 0.0


def _build_phases(rng, ath: Athlete, trial: dict, t_lo: float):
    other = "r" if ath.takeoff_side == "l" else "l"
    Tc = trial["contact_time"]
    phases = [_Phase("contact", ath.takeoff_side, 0.0, Tc)]
    t = 0.0
    T_f2 = float(rng.normal(0.112, 0.006))
    phases.insert(0, _Phase("flight", None, -T_f2, 0.0, trial["vx_td"], trial["vz_td"]))
    t = -T_f2
    foot = other
    k = 0
    approach = trial["vx_td"] + 0.08
    while t > t_lo:
        T_s = float(rng.normal(0.105 if k == 0 else 0.100, 0.005))
        phases.insert(0, _Phase("support", foot, t - T_s, t))
        t -= T_s
        T_f = float(rng.normal(0.108 if k == 0 else 0.110, 0.006))
        phases.insert(0, _Phase("flight", None, t - T_f, t, approach + 0.02 * k + rng.normal(0, 0.03),
                                float(rng.normal(-0.45, 0.06))))
        t -= T_f
        foot = ath.takeoff_side if foot == other else other
        k += 1
    phases.append(_Phase("flight", None, Tc, np.inf, trial["vx_td"] - trial["dvx"], 0.0))
    return phases


def _acceleration(phases, tt, trial, rng):
    """Piecewise CoM acceleration (x, z) on the fine grid and the support labels."""
    ax = np.zeros_like(tt)
    az = np.full_like(tt, -G0)
    for i, ph in enumerate(phases):
        if ph.kind == "flight":
            continue
        m = (tt >= ph.t0) & (tt < ph.t1)
        T = ph.t1 - ph.t0
        s = tt[m] - ph.t0
        if ph.kind == "support":
            prev, nxt = phases[i - 1], phases[i + 1]
            Tn = nxt.t1 - nxt.t0
            vz_start_next = nxt.vz_end + G0 * Tn
            dvz = vz_start_next - prev.vz_end
            dvx = nxt.vx - prev.vx
            A = (dvz + G0 * T) * np.pi / (2.0 * T)
            az[m] = -G0 + A * np.sin(np.pi * s / T)
            B = G0 * rng.uniform(0.8, 1.2)
            ax[m] = dvx / T - B * np.sin(2.0 * np.pi * s / T)
        else:
            Tb = 0.62 * T
            Ib = trial["braking_impulse"]
            Ip = Ib - trial["dvx"] / G0
            Ab = Ib * np.pi / (2.0 * Tb)
            Ap = Ip * np.pi / (2.0 * (T - Tb))
            fx = np.where(s < Tb, -Ab * np.sin(np.pi * s / Tb),
                          Ap * np.sin(np.pi * np.clip(s - Tb, 0, None) / (T - Tb)))
            ti, sig, Ai = trial["impact_time"], 0.008, trial["impact_peak"]
            bump_int = Ai * sig * np.sqrt(np.pi / 2.0) * (erf((T - ti) / (sig * np.sqrt(2))) + erf(ti / (sig * np.sqrt(2))))
            Jz = (trial["vz_to"] - trial["vz_td"]) / G0 + T
            As = (Jz - bump_int) / (2.0 * T / np.pi)
            fz = As * np.sin(np.pi * s / T) + Ai * np.exp(-0.5 * ((s - ti) / sig) ** 2)
            ax[m] = G0 * fx
            az[m] = G0 * (fz - 1.0)
    return ax, az


def _cumtrapz(y, dt):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dt)
    return out


def _pitch(vx, vz, psi):
    """Rotate sagittal offsets so that positive ``psi`` raises the heel."""
    c, s = np.cos(psi), np.sin(psi)
    return vx * c + vz * s, -vx * s + vz * c


def _two_link(H, A, L1, L2, bend):
    dvec = A - H
    dn = np.linalg.norm(dvec, axis=-1, keepdims=True)
    u = dvec / np.maximum(dn, 1e-9)
    d = np.clip(dn, abs(L1 - L2) + 1e-6, L1 + L2 - 1e-6)
    a = (L1 ** 2 - L2 ** 2 + d ** 2) / (2.0 * d)
    h = np.sqrt(np.clip(L1 ** 2 - a ** 2, 0.0, None))
    w = bend - np.sum(bend * u, axis=-1, keepdims=True) * u
    w /= np.maximum(np.linalg.norm(w, axis=-1, keepdims=True), 1e-9)
    return H + a * u + h * w


class _Body:
    def __init__(self, stature: float):
        k = stature / 1.80
        self.thigh, self.shank = 0.47 * k, 0.46 * k
        self.hip_w, self.pelvis_thorax, self.thorax_neck, self.neck_head = 0.10 * k, 0.26 * k, 0.24 * k, 0.24 * k
        self.shoulder_w, self.upper_arm, self.forearm = 0.19 * k, 0.30 * k, 0.27 * k
        self.foot_len, self.ankle_fwd, self.ankle_h = 0.26 * k, 0.19 * k, 0.08 * k
        self.reach = 0.995 * (self.thigh + self.shank)

    def foot_points(self, toe, psi):
        ax, az = _pitch(-self.ankle_fwd, self.ankle_h, psi)
        hx, hz = _pitch(-self.foot_len, 0.0, psi)
        ankle = toe + np.stack([ax, np.zeros_like(ax), az], -1)
        heel = toe + np.stack([hx, np.zeros_like(hx), hz], -1)
        return ankle, heel


def _foot_schedule(phases, side):
    return [(ph.t0, ph.t1) for ph in phases if ph.kind != "flight" and ph.foot == side]


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _feet(times, com_fn, phases, body, toe_pos, psi_off, ath, Tc):
    """Toe position and pitch of both feet, plus stance flags (pitch is a lower bound in stance)."""
    out = {}
    for side in ("l", "r"):
        sched = _foot_schedule(phases, side)
        toe = np.zeros((times.size, 3))
        psi = np.zeros(times.size)
        stance = np.zeros(times.size, bool)
        lat = 0.09 if side == "l" else -0.09
        for k, t in enumerate(times):
            idx = [i for i, (a, b) in enumerate(sched) if a <= t < b]
            if idx:
                i = idx[0]
                a, b = sched[i]
                stance[k] = True
                toe[k] = toe_pos[(side, i)]
                frac = (t - a) / (b - a)
                psi_end = 0.35
                psi[k] = psi_end * _smoothstep((frac - 0.4) / 0.6)
                continue
            prev = [i for i, (a, b) in enumerate(sched) if b <= t]
            nxt = [i for i, (a, b) in enumerate(sched) if a > t]
            c_t = com_fn(t)
            if prev:
                ip = prev[-1]
                t_off = sched[ip][1]
                start = toe_pos[(side, ip)]
                p_off = psi_off[(side, ip)]
            else:
                ip, t_off, start, p_off = None, times[0] - 0.2, None, 0.6
            if nxt:
                t_on = sched[nxt[0]][0]
                target = toe_pos[(side, nxt[0])]
                if start is None:
                    start = target - np.array([4.2, 0.0, 0.0])
                s = (t - t_off) / (t_on - t_off)
                f = 2.7 * s ** 2 - 1.7 * s ** 3
                toe[k] = start + (target - start) * f
                toe[k, 2] = 0.30 * np.sin(np.pi * s)
                psi[k] = p_off * (1.0 - s) ** 2 - 0.6 * s * (1.0 - s)
            else:
                if side == ath.takeoff_side:
                    rel, dur = np.array([-0.40, lat, -0.92]), 0.22
                else:
                    rel, dur = np.array([0.55, lat, -0.55]), max(Tc + 0.05 - t_off, 0.1)
                s = np.clip((t - t_off) / dur, 0.0, 1.0)
                f = 2.7 * s ** 2 - 1.7 * s ** 3
                tgt = c_t + rel
                toe[k] = start + (tgt - start) * f
                toe[k, 2] += 0.15 * np.sin(np.pi * s) * (side != ath.takeoff_side)
                psi[k] = p_off * (1.0 - s) + 0.35 * s
        out[side] = (toe, psi, stance)
    return out


def _pose(times, com, feet, body, ath, sex, lean, arm, iterations=8):
    N = times.size
    p = com - np.array([0.0, 0.0, 0.08])
    Y = np.zeros((N, NUM_JOINTS, 3))
    fwd = np.tile(np.array([1.0, 0.0, 0.0]), (N, 1))
    psi_used = {}
    for _ in range(iterations):
        Y[:, J["pelvis"]] = p
        for side in ("l", "r"):
            lat = body.hip_w if side == "l" else -body.hip_w
            hip = p + np.array([0.0, lat, 0.0])
            toe, psi0, stance = feet[side]
            psi = psi0.copy()
            ankle, heel = body.foot_points(toe, psi)
            need = stance & (np.linalg.norm(hip - ankle, axis=-1) > body.reach)
            if need.any():
                lo, hi = psi[need].copy(), np.full(need.sum(), 1.8)
                for _b in range(40):
                    mid = 0.5 * (lo + hi)
                    a_mid, _ = body.foot_points(toe[need], mid)
                    too_far = np.linalg.norm(hip[need] - a_mid, axis=-1) > body.reach
                    lo = np.where(too_far, mid, lo)
                    hi = np.where(too_far, hi, mid)
                psi[need] = hi
                ankle, heel = body.foot_points(toe, psi)
            dvec = ankle - hip
            dn = np.linalg.norm(dvec, axis=-1, keepdims=True)
            pull = (~stance)[:, None] & (dn > body.reach)
            shift = np.where(pull, dvec / np.maximum(dn, 1e-9) * (body.reach - dn), 0.0)
            ankle, heel, toe = ankle + shift, heel + shift, toe + shift
            knee = _two_link(hip, ankle, body.thigh, body.shank, fwd)
            Y[:, J[f"{side}_hip"]] = hip
            Y[:, J[f"{side}_knee"]] = knee
            Y[:, J[f"{side}_ankle"]] = ankle
            Y[:, J[f"{side}_heel"]] = heel
            Y[:, J[f"{side}_toe"]] = toe
            psi_used[side] = psi
        up = np.stack([np.sin(lean), np.zeros(N), np.cos(lean)], -1)
        Y[:, J["thorax"]] = p + body.pelvis_thorax * up
        Y[:, J["neck"]] = Y[:, J["thorax"]] + body.thorax_neck * up
        Y[:, J["head"]] = Y[:, J["neck"]] + body.neck_head * up
        for side, sign in (("l", 1.0), ("r", -1.0)):
            sh = Y[:, J["neck"]] + np.array([0.0, sign * body.shoulder_w, -0.03])
            th = arm[side]
            el = sh + body.upper_arm * np.stack([np.sin(th), np.zeros(N), -np.cos(th)], -1)
            wr = el + body.forearm * np.stack([np.sin(th + 1.4), np.zeros(N), -np.cos(th + 1.4)], -1)
            Y[:, J[f"{side}_shoulder"]] = sh
            Y[:, J[f"{side}_elbow"]] = el
            Y[:, J[f"{side}_wrist"]] = wr
        p = p + (com - center_of_mass(Y, sex))
    return Y, psi_used


SURFACE_OFFSETS = {
    "hip": np.array([0.03, 0.07, 0.02]),
    "knee": np.array([0.015, 0.045, 0.0]),
    "ankle": np.array([0.0, 0.035, 0.0]),
    "shoulder": np.array([0.0, 0.02, 0.03]),
}


def surface_keypoints(Y: np.ndarray) -> np.ndarray:
    S = Y.copy()
    for name, off in SURFACE_OFFSETS.items():
        for side, sign in (("l", 1.0), ("r", -1.0)):
            S[..., J[f"{side}_{name}"], :] += off * np.array([1.0, sign, 1.0])
    return S


def simulate_trial(
    rng: np.random.Generator,
    athlete: Athlete,
    venue: str = "indoor",
    session: int = 1,
    trial_index: int = 1,
    fps: float = 200.0,
    ref_fps: float = 400.0,
    force_fps: float = 2000.0,
    t_before: float = 0.605,
    t_after: float = 0.595,
    leave_after_to: float = 0.147,
    pixel_noise: float = 0.5,
    raw_similarity: bool = True,
) -> dict:
    """Generate one trial; times are relative to touchdown unless stated otherwise."""
    dt = 1.0 / force_fps
    trial = {
        "contact_time": float(np.clip(rng.normal(athlete.contact_time, 0.006), 0.095, 0.15)),
        "vx_td": float(rng.normal(athlete.vx_td, 0.12)),
        "dvx": float(rng.normal(athlete.dvx, 0.10)),
        "vz_td": float(rng.normal(athlete.vz_td, 0.08)),
        "vz_to": float(rng.normal(athlete.vz_to, 0.15)),
        "braking_impulse": float(rng.normal(athlete.braking_impulse, 0.015)),
        "impact_peak": float(rng.normal(athlete.impact_peak, 0.3)),
        "impact_time": float(rng.normal(0.028, 0.002)),
    }
    Tc = trial["contact_time"]
    phases = _build_phases(rng, athlete, trial, -t_before - 0.05)

    step = int(round(force_fps / fps))
    offset = int(rng.integers(0, step))
    t_first = -int(round(t_before * force_fps)) * dt - offset * dt
    t0 = phases[0].t0
    k0 = int(np.floor((t0 - t_first) / dt))
    n_fine = int(np.ceil((t_after + 0.05 - (t_first + k0 * dt)) / dt))
    tt = t_first + (k0 + np.arange(n_fine)) * dt

    ax, az = _acceleration(phases, tt, trial, rng)
    v0 = np.array([phases[0].vx, 0.0, phases[0].vz_end + G0 * (phases[0].t1 - phases[0].t0)])
    if phases[0].kind == "support":
        v0 = np.array([phases[1].vx, 0.0, phases[1].vz_end + G0 * (phases[1].t1 - phases[1].t0)])
    vx = v0[0] + _cumtrapz(ax, dt)
    vz = v0[2] + _cumtrapz(az, dt)
    cx = _cumtrapz(vx, dt)
    cz = _cumtrapz(vz, dt)
    i_td = int(np.argmin(np.abs(tt)))
    cz_td = float(rng.normal(0.98, 0.02)) * athlete.stature / 1.80
    toe_td_x = float(rng.uniform(-0.15, -0.03))
    cx += (toe_td_x - float(rng.normal(0.60, 0.03))) - cx[i_td]
    cz += cz_td - cz[i_td]
    com_fine = np.stack([cx, np.zeros_like(cx), cz], -1)

    def com_fn(t):
        return np.array([np.interp(t, tt, cx), 0.0, np.interp(t, tt, cz)])

    body = _Body(athlete.stature)
    toe_pos = {}
    for side in ("l", "r"):
        lat = 0.09 if side == "l" else -0.09
        for i, (a, b) in enumerate(_foot_schedule(phases, side)):
            if a == 0.0:
                x = toe_td_x
            else:
                x = com_fn(a)[0] + float(rng.normal(0.33, 0.03))
            toe_pos[(side, i)] = np.array([x, lat + rng.normal(0, 0.01), 0.0])

    ref_step = int(round(force_fps / ref_fps))
    n_ref = int(round((t_before + t_after) * ref_fps)) + 1
    t_ref = t_first + np.arange(n_ref) * ref_step * dt
    com_ref = np.stack([np.interp(t_ref, tt, cx), np.zeros(n_ref), np.interp(t_ref, tt, cz)], -1)

    lean = 0.14 - 0.22 * np.exp(-(((t_ref - 0.02) / 0.09) ** 2))
    stride = 0.45
    base = 2.0 * np.pi * (t_ref - Tc) / stride
    sgn = 1.0 if athlete.takeoff_side == "l" else -1.0
    arm = {"l": -sgn * 0.9 * np.cos(base), "r": sgn * 0.9 * np.cos(base)}
    psi_off = {}
    for side in ("l", "r"):
        for i, _ in enumerate(_foot_schedule(phases, side)):
            psi_off[(side, i)] = 0.75
    for _pass in range(3):
        feet = _feet(t_ref, com_fn, phases, body, toe_pos, psi_off, athlete, Tc)
        Y, psi_used = _pose(t_ref, com_ref, feet, body, athlete, athlete.sex, lean, arm)
        for side in ("l", "r"):
            for i, (a, b) in enumerate(_foot_schedule(phases, side)):
                k = int(np.searchsorted(t_ref, b) - 1)
                if 0 <= k < t_ref.size:
                    psi_off[(side, i)] = float(psi_used[side][k])

    T = int(round((t_before + t_after) * fps)) + 1
    Y_mk = Y[:: int(round(ref_fps / fps))][:T]
    t_mk = t_ref[:: int(round(ref_fps / fps))][:T]

    true_cams = make_cameras(venue, rng=rng)
    S = surface_keypoints(Y_mk)
    V = len(true_cams)
    mu = np.zeros((V, T, NUM_JOINTS, 2))
    cov = np.zeros((V, T, NUM_JOINTS, 2, 2))
    vis = np.zeros((V, T, NUM_JOINTS))
    noise = pixel_noise * (1.25 if venue == "outdoor" else 1.0)
    legs = [J[n] for n in JOINTS_LOWER]
    knee_gap = np.abs(Y_mk[:, J["l_knee"], 0] - Y_mk[:, J["r_knee"], 0])
    overlap = knee_gap < 0.18
    for v, cam in enumerate(true_cams):
        uv = cam.project(S)
        cam_side = 1.0 if cam.center[1] > 0 else -1.0
        sig_a = noise * rng.uniform(0.7, 1.3, (T, NUM_JOINTS))
        sig_b = sig_a * rng.uniform(0.5, 0.9, (T, NUM_JOINTS))
        ang = rng.uniform(0, np.pi, (T, NUM_JOINTS))
        occl = np.zeros((T, NUM_JOINTS), bool)
        for j in legs:
            far = (np.sign(Y_mk[:, j, 1]) != cam_side) & (np.abs(Y_mk[:, j, 1]) > 0.02)
            occl[:, j] = far & overlap
        sig_a = np.where(occl, 4.0 * noise, sig_a)
        sig_b = np.where(occl, 1.0 * noise, sig_b)
        c, s = np.cos(ang), np.sin(ang)
        Rm = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
        D = np.zeros((T, NUM_JOINTS, 2, 2))
        D[..., 0, 0] = sig_a ** 2
        D[..., 1, 1] = sig_b ** 2
        C = Rm @ D @ np.swapaxes(Rm, -1, -2)
        Lc = np.linalg.cholesky(C)
        e = np.einsum("tjab,tjb->tja", Lc, rng.normal(size=(T, NUM_JOINTS, 2)))
        outlier = rng.random((T, NUM_JOINTS)) < 0.005
        e[outlier] += rng.normal(0, 30.0, (outlier.sum(), 2))
        mu[v] = uv + e
        cov[v] = C * rng.uniform(0.8, 1.25, (T, NUM_JOINTS))[..., None, None]
        vis[v] = np.where(occl, rng.uniform(0.4, 0.6, (T, NUM_JOINTS)), rng.uniform(0.8, 0.95, (T, NUM_JOINTS)))
        inside = (uv[..., 0] > 0) & (uv[..., 0] < 2048) & (uv[..., 1] > 0) & (uv[..., 1] < 1536)
        mu[v][~inside] = np.nan
    gone = t_mk > Tc + leave_after_to
    mu[:, gone] = np.nan

    if raw_similarity:
        s_raw = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
        R_raw = rodrigues(rng.normal(size=3) * 0.8)
        d_raw = rng.normal(size=3) * 2.0
    else:
        s_raw, R_raw, d_raw = 1.0, np.eye(3), np.zeros(3)
    raw_cams = [c.transformed(s_raw, R_raw, d_raw) for c in true_cams]

    i0 = int(np.argmin(np.abs(tt - t_first)))
    n_force = int(round((t_before + t_after) * force_fps)) + 1
    t_force = tt[i0 : i0 + n_force]
    grf = athlete.mass * np.stack([np.interp(t_force, tt, ax), np.zeros(n_force), np.interp(t_force, tt, az) + G0], -1)
    on_plate = (t_force >= 0.0) & (t_force < Tc)
    grf[~on_plate] = 0.0
    grf += rng.normal(0.0, 4.0, grf.shape)
    has_force = venue == "indoor"

    ref = Y + rng.normal(0.0, 0.0008, Y.shape)
    return {
        "kp2d_mu": mu.astype(np.float32),
        "kp2d_cov": cov.astype(np.float32),
        "kp2d_vis": vis.astype(np.float32),
        "fps": float(fps),
        "cameras": cameras_to_json(raw_cams),
        "ref_joints": ref.astype(np.float32),
        "ref_fps": float(ref_fps),
        "grf": (grf if has_force else np.zeros((0, 3))).astype(np.float32),
        "force_fps": float(force_fps),
        "has_force": bool(has_force),
        "ref_td": float(-t_first),
        "ref_to": float(Tc - t_first),
        "mass": float(athlete.mass),
        "sex": athlete.sex,
        "stature": float(athlete.stature),
        "level": int(athlete.level),
        "athlete": athlete.athlete_id,
        "session": int(session),
        "trial": int(trial_index),
        "venue": venue,
        "truth_scale": float(1.0 / s_raw),
        "truth_contact_time": float(Tc),
    }


JOINTS_LOWER = ("l_knee", "l_ankle", "l_heel", "l_toe", "r_knee", "r_ankle", "r_heel", "r_toe")


def make_corpus(seed: int = 0, n_athletes: int = 6, sessions: int = 2, trials_per_session: int = 4,
                outdoor_sessions: tuple = (2,)):
    """Yield synthetic trials for a small corpus."""
    rng = np.random.default_rng(seed)
    athletes = [sample_athlete(rng, i) for i in range(n_athletes)]
    for ath in athletes:
        for s in range(1, sessions + 1):
            venue = "outdoor" if s in outdoor_sessions else "indoor"
            for k in range(1, trials_per_session + 1):
                yield simulate_trial(rng, ath, venue=venue, session=s, trial_index=k)
