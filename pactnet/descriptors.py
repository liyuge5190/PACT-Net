"""Takeoff descriptors and external limb loading (Section 3.8, Eq. 33).

All descriptors are computed from joint trajectories through linear
derivative operators (Eq. 23), so a whole batch of trajectories, such as the
sigma points of the unscented transform, is evaluated at once. Forces are in
body weights (BW), impulses in BW s.

Determined by centroidal momentum balance and reported here: the resultant
contact force and its components, the linear impulse and its braking and
propulsive partitions, and the centre of pressure. Joint loads and internal
forces are not determined and are not reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .angles import knee_extension_velocity, knee_flexion
from .derivatives import local_poly_rows, phase_adaptive_operators
from .events import phase_memberships
from .skeleton import FOOT_POINTS, J, com_weights, segment_com_matrix, segment_parameters

G0 = 9.81

DESCRIPTORS = (
    "approach_velocity",
    "td_horizontal_velocity",
    "to_horizontal_velocity",
    "horizontal_velocity_loss",
    "td_vertical_velocity",
    "to_vertical_velocity",
    "resultant_takeoff_velocity",
    "takeoff_angle",
    "com_height_td",
    "com_height_to",
    "contact_time",
    "velocity_conversion_ratio",
    "peak_vertical_force",
    "time_to_peak_vertical_force",
    "peak_braking_force",
    "vertical_impulse",
    "braking_impulse",
    "propulsive_impulse",
    "peak_leg_compression",
    "leg_stiffness",
    "peak_knee_extension_velocity",
    "peak_knee_velocity_timing",
    "knee_flexion_td",
)

UNITS = {
    "approach_velocity": "m/s", "td_horizontal_velocity": "m/s", "to_horizontal_velocity": "m/s",
    "horizontal_velocity_loss": "m/s", "td_vertical_velocity": "m/s", "to_vertical_velocity": "m/s",
    "resultant_takeoff_velocity": "m/s", "takeoff_angle": "deg", "com_height_td": "m",
    "com_height_to": "m", "contact_time": "ms", "velocity_conversion_ratio": "-",
    "peak_vertical_force": "BW", "time_to_peak_vertical_force": "ms", "peak_braking_force": "BW",
    "vertical_impulse": "BW s", "braking_impulse": "BW s", "propulsive_impulse": "BW s",
    "peak_leg_compression": "m", "leg_stiffness": "BW/m", "peak_knee_extension_velocity": "deg/s",
    "peak_knee_velocity_timing": "ms", "knee_flexion_td": "deg",
}


@dataclass
class DescriptorConfig:
    h_contact: float = 5.0
    h_flight: float = 15.0
    beta: float = 1.0
    sample_beta: float = 0.25        # sharpness of the sample memberships in Eq. (23)
    approach_window: float = 0.23   # s before touchdown for the final-stride approach velocity
    adaptive: bool = True


class TakeoffDescriptors:
    """Descriptor map Phi = f(Y) for fixed events, sampling rate and athlete sex."""

    def __init__(self, n_frames, tau_td, tau_to, fps, sex="M", stance_side="l",
                 valid=None, cfg: DescriptorConfig | None = None):
        self.cfg = cfg = cfg or DescriptorConfig()
        self.n, self.fps = n_frames, fps
        self.tau_td, self.tau_to = float(tau_td), float(tau_to)
        self.side = stance_side
        sw = np.ones(n_frames) if valid is None else np.asarray(valid, dtype=np.float64)
        self.w = com_weights(sex)
        self.A = segment_com_matrix(sex)
        self.m = segment_parameters(sex)[2]
        h_pre = cfg.h_flight if cfg.adaptive else cfg.h_contact
        mem = phase_memberships(np.arange(n_frames), tau_td, tau_to, cfg.sample_beta)
        if not cfg.adaptive:
            mem = {k: np.ones(n_frames) for k in mem}
        n_back = cfg.approach_window * fps
        rows = local_poly_rows([tau_td, tau_td - n_back], n_frames, h_pre, sw * mem["pre"])
        self.pos_td, self.vel_td = rows[0, 0], rows[0, 1] * fps
        self.pos_back = rows[1, 0]
        rows = local_poly_rows([tau_to], n_frames, h_pre, sw * mem["post"])
        self.pos_to, self.vel_to = rows[0, 0], rows[0, 1] * fps
        k0, k1 = int(np.ceil(tau_td)), int(np.floor(tau_to))
        self.contact_frames = np.arange(k0, k1 + 1)
        h_con = cfg.h_contact
        rows = local_poly_rows(self.contact_frames, n_frames, h_con, sw * mem["contact"])
        self.pos_con = rows[:, 0]
        self.acc_con = 2.0 * rows[:, 2] * fps ** 2
        _, self.D1, _ = phase_adaptive_operators(n_frames, tau_td, tau_to, fps, cfg.h_contact,
                                                 cfg.h_flight, cfg.beta, sw, adaptive=cfg.adaptive,
                                                 sample_beta=cfg.sample_beta)
        self.td_frame = int(np.clip(np.round(tau_td), 0, n_frames - 1))

    def __call__(self, Y: np.ndarray) -> np.ndarray:
        """Y (..., T, J, 3) -> descriptors (..., len(DESCRIPTORS))."""
        Y = np.nan_to_num(np.asarray(Y, dtype=np.float64))
        fps = self.fps
        c = np.einsum("j,...tjk->...tk", self.w, Y)
        v_td = np.einsum("t,...tk->...k", self.vel_td, c)
        v_to = np.einsum("t,...tk->...k", self.vel_to, c)
        c_td = np.einsum("t,...tk->...k", self.pos_td, c)
        c_back = np.einsum("t,...tk->...k", self.pos_back, c)
        c_to = np.einsum("t,...tk->...k", self.pos_to, c)
        approach = (c_td[..., 0] - c_back[..., 0]) / self.cfg.approach_window
        dvx = v_td[..., 0] - v_to[..., 0]
        tc = (self.tau_to - self.tau_td) / fps
        theta = np.rad2deg(np.arctan2(v_to[..., 2], v_to[..., 0]))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (v_to[..., 2] - v_td[..., 2]) / dvx
        Jz = (v_to[..., 2] - v_td[..., 2]) / G0 + tc

        acc = np.einsum("kt,...tj->...kj", self.acc_con, c)
        F = (acc + np.array([0.0, 0.0, G0])) / G0
        Fz, Fx = F[..., 2], F[..., 0]
        kpk = np.argmax(Fz, axis=-1)
        peak = np.take_along_axis(Fz, kpk[..., None], -1)[..., 0]
        lo = np.take_along_axis(Fz, np.clip(kpk - 1, 0, Fz.shape[-1] - 1)[..., None], -1)[..., 0]
        hi = np.take_along_axis(Fz, np.clip(kpk + 1, 0, Fz.shape[-1] - 1)[..., None], -1)[..., 0]
        den = lo - 2.0 * peak + hi
        frac = np.where(np.abs(den) > 1e-9, 0.5 * (lo - hi) / np.where(np.abs(den) > 1e-9, den, 1.0), 0.0)
        frac = np.clip(frac, -0.5, 0.5)
        t_peak = (self.contact_frames[0] + kpk + frac - self.tau_td) / fps * 1000.0
        brake = np.clip(-Fx, 0.0, None)
        dt = 1.0 / fps
        J_brake = brake.sum(-1) * dt
        J_prop = np.clip(Fx, 0.0, None).sum(-1) * dt

        cop = self._centre_of_pressure(Y, c, F)
        hip = np.einsum("kt,...tj->...kj", self.pos_con, Y[..., J[f"{self.side}_hip"], :])
        leg = np.linalg.norm(hip - cop, axis=-1)
        compression = leg[..., 0] - leg.min(-1)
        stiffness = peak / np.maximum(compression, 1e-3)

        w_ext = knee_extension_velocity(Y, self.side, self.D1)[..., self.contact_frames]
        kw = np.argmax(w_ext, axis=-1)
        w_peak = np.take_along_axis(w_ext, kw[..., None], -1)[..., 0]
        w_time = (self.contact_frames[0] + kw - self.tau_td) / fps * 1000.0
        knee_td = knee_flexion(Y[..., self.td_frame, :, :], self.side)

        cols = [
            approach, v_td[..., 0], v_to[..., 0], dvx, v_td[..., 2], v_to[..., 2],
            np.linalg.norm(v_to, axis=-1), theta, c_td[..., 2], c_to[..., 2],
            np.full(approach.shape, tc * 1000.0), ratio, peak, t_peak, brake.max(-1), Jz,
            J_brake, J_prop, compression, stiffness, w_peak, w_time, knee_td,
        ]
        return np.stack(cols, axis=-1)

    def _centre_of_pressure(self, Y, c, F):
        """Ground point with no horizontal net moment, clamped to the stance foot."""
        r = np.einsum("sj,...tjk->...tsk", self.A, Y)
        r_k = np.einsum("kt,...tsj->...ksj", self.pos_con, r)
        r_acc = np.einsum("kt,...tsj->...ksj", self.acc_con, r)
        c_k = np.einsum("kt,...tj->...kj", self.pos_con, c)
        c_acc = (F * G0) - np.array([0.0, 0.0, G0])
        Ldot = np.einsum("s,...ksj->...kj", self.m, np.cross(r_k - c_k[..., None, :], r_acc - c_acc[..., None, :])) / G0
        Fz = np.where(F[..., 2] > 1.0, F[..., 2], 1.0)
        cz = c_k[..., 2]
        dx = (-cz * F[..., 0] - Ldot[..., 1]) / Fz
        dy = (Ldot[..., 0] - cz * F[..., 1]) / Fz
        heel_i, toe_i, _ = FOOT_POINTS[self.side]
        heel = np.einsum("kt,...tj->...kj", self.pos_con, Y[..., heel_i, :])
        toe = np.einsum("kt,...tj->...kj", self.pos_con, Y[..., toe_i, :])
        ball = toe + 0.25 * (heel - toe)
        p = np.where((F[..., 2] > 1.0)[..., None], c_k[..., :2] + np.stack([dx, dy], -1), ball[..., :2])
        ab = toe[..., :2] - heel[..., :2]
        t = np.clip(np.sum((p - heel[..., :2]) * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-12), 0.0, 1.0)
        on_foot = heel[..., :2] + t[..., None] * ab
        return np.concatenate([on_foot, np.zeros(on_foot.shape[:-1] + (1,))], axis=-1)

    def force_waveform(self, Y: np.ndarray) -> np.ndarray:
        """Contact force (K, 3) in BW at the contact frames."""
        c = np.einsum("j,tjk->tk", self.w, np.nan_to_num(Y))
        acc = self.acc_con @ c
        return (acc + np.array([0.0, 0.0, G0])) / G0


def descriptor_dict(values: np.ndarray) -> dict:
    return {k: float(v) for k, v in zip(DESCRIPTORS, values)}
