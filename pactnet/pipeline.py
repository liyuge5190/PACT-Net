"""End-to-end inference on a preprocessed trial (Figure 1, stages 4-5).

Input is the cache produced by :func:`pactnet.data.preprocess_trial`
(stages 1-3). The contact-gated temporal network is optional: without a
checkpoint the triangulated trajectory and its posterior covariance are
passed on directly and the events are estimated kinematically.
"""

from __future__ import annotations

import time

import numpy as np

from .anatomy import apply_joint_centre_offsets, fit_joint_centre_offsets, project_bone_lengths, reference_bone_lengths
from .descriptors import DESCRIPTORS, DescriptorConfig, TakeoffDescriptors
from .events import kinematic_events
from .physics import ContactRefiner, PhysicsConfig
from .uncertainty import descriptor_intervals


def network_window(td_init: float, T: int, length: int = 121):
    half = length // 2
    start = int(round(td_init)) - half
    return start, start + length


def crop(a: np.ndarray, start: int, stop: int, fill=np.nan) -> np.ndarray:
    """Crop along axis 0 with padding outside the available range."""
    T = a.shape[0]
    out = np.full((stop - start,) + a.shape[1:], fill, dtype=np.float64)
    s, e = max(start, 0), min(stop, T)
    if e > s:
        out[s - start : e - start] = a[s:e]
    return out


class Pipeline:
    def __init__(self, cfg: dict, model=None, device: str = "cpu"):
        self.cfg = cfg
        self.model = model
        self.device = device
        p = cfg["physics"]
        self.physics_cfg = PhysicsConfig(
            lambda_smooth=p["lambda_smooth"], lambda_physics=p["lambda_physics"],
            lambda_anatomy=p["lambda_anatomy"], lambda_limits=p["lambda_limits"],
            friction=p["friction"], margin=p["margin"], beta=cfg["model"]["event_beta"],
        )
        d = cfg["derivatives"]
        self.desc_cfg = DescriptorConfig(h_contact=d["h_contact"], h_flight=d["h_flight"],
                                         beta=cfg["model"]["event_beta"], adaptive=d["phase_adaptive"])

    def temporal_stage(self, cache: dict) -> dict:
        joints = np.asarray(cache["joints"], dtype=np.float64)
        cov = np.asarray(cache["cov"], dtype=np.float64)
        start, stop = network_window(cache["td_init"], joints.shape[0], self.cfg["data"]["network_window"])
        Yw = crop(joints, start, stop)
        Cw = crop(cov, start, stop)
        sd = np.sqrt(np.clip(np.diagonal(Cw, axis1=-2, axis2=-1), 1e-10, None))
        if self.model is not None:
            from .model import run_network

            out = run_network(self.model, Yw, Cw, self.device)
            return {"start": start, "joints": out["joints"], "sd": out["sd"],
                    "tau_td": out["tau"][0], "tau_to": out["tau"][1], "event_sd": out["event_sd"]}
        td, to = cache["td_init"] - start, cache["to_init"] - start
        _, to_k = kinematic_events(Yw, td, to, cache["fps"], cache["sex"])
        return {"start": start, "joints": Yw, "sd": sd, "tau_td": float(td), "tau_to": float(to_k),
                "event_sd": np.array([0.5, 0.5])}

    def run(self, cache: dict, offsets: dict | None = None, intervals: bool = True) -> dict:
        timings = {}
        t0 = time.time()
        st = self.temporal_stage(cache)
        timings["temporal"] = time.time() - t0
        Y, sd = st["joints"], st["sd"]
        td, to = st["tau_td"], st["tau_to"]
        fps, sex, side = float(cache["fps"]), str(cache["sex"]), str(cache["stance_side"])
        valid = np.isfinite(Y).all(axis=(1, 2))

        t0 = time.time()
        approach = Y[: max(int(np.floor(td)) - 4, 1)]
        if self.cfg["anatomy"]["joint_centre_correction"]:
            if offsets is None:
                offsets = fit_joint_centre_offsets(approach, self.cfg["anatomy"]["offset_regularizer"])
            Y = apply_joint_centre_offsets(Y, offsets)
        lengths = reference_bone_lengths(Y[: max(int(np.floor(td)), 1)])
        Y = np.where(valid[:, None, None], project_bone_lengths(np.nan_to_num(Y), lengths), np.nan)
        timings["anatomy"] = time.time() - t0

        t0 = time.time()
        physics = None
        if self.cfg["physics"]["enabled"]:
            physics = ContactRefiner(self.physics_cfg, sex).refine(Y, sd, td, to, fps, side, lengths)
            Y = np.where(valid[:, None, None], physics["joints"], np.nan)
        timings["physics"] = time.time() - t0

        t0 = time.time()
        desc = TakeoffDescriptors(Y.shape[0], td, to, fps, sex, side, valid, self.desc_cfg)
        point = desc(Y)
        result = {
            "id": cache["id"],
            "start": st["start"],
            "joints": Y,
            "sd": sd,
            "tau_td": td,
            "tau_to": to,
            "event_sd": np.asarray(st["event_sd"], dtype=np.float64),
            "stance_side": side,
            "descriptors": point,
            "force": desc.force_waveform(Y),
            "contact_frames": desc.contact_frames,
            "physics_history": physics["history"] if physics else [],
        }
        if intervals:
            rel_se = cache["scale_se"] / cache["scale"] if np.isfinite(cache["scale_se"]) else 0.0
            _, mean, cov, half = descriptor_intervals(
                Y, sd, td, to, fps, sex, side, scale_rel_se=rel_se, event_sd=tuple(st["event_sd"]),
                valid=valid, cfg=self.desc_cfg, chunk=self.cfg["uncertainty"]["chunk"],
                z=self.cfg["uncertainty"]["z"],
            )
            result["half_width"] = half
            result["descriptor_cov"] = cov
        timings["descriptors"] = time.time() - t0
        result["timings"] = timings
        return result


def reference_descriptors(cache: dict, start: int, length: int, cfg: dict) -> np.ndarray:
    """Descriptors of the reference system for the same window: kinematics from the
    optoelectronic reference, kinetics from the force platform where available."""
    from .forceplate import kinetic_descriptors

    ref = crop(np.asarray(cache["ref_joints"], dtype=np.float64), start, start + length)
    td, to = cache["ref_td"] - start, cache["ref_to"] - start
    d = cfg["derivatives"]
    dcfg = DescriptorConfig(h_contact=d["h_contact"], h_flight=d["h_flight"], beta=cfg["model"]["event_beta"])
    valid = np.isfinite(ref).all(axis=(1, 2))
    values = TakeoffDescriptors(length, td, to, cache["fps"], cache["sex"], cache["stance_side"], valid, dcfg)(ref)
    out = dict(zip(DESCRIPTORS, values))
    if cache["has_force"]:
        kin = kinetic_descriptors(np.asarray(cache["grf"], dtype=np.float64), cache["force_fps"], cache["mass"])
        for k in ("contact_time", "peak_vertical_force", "time_to_peak_vertical_force", "peak_braking_force",
                  "vertical_impulse", "braking_impulse", "propulsive_impulse"):
            out[k] = kin[k]
        out["leg_stiffness"] = kin["peak_vertical_force"] / out["peak_leg_compression"]
    else:
        for k in ("peak_vertical_force", "time_to_peak_vertical_force", "peak_braking_force",
                  "vertical_impulse", "braking_impulse", "propulsive_impulse", "leg_stiffness"):
            out[k] = np.nan
    return np.array([out[k] for k in DESCRIPTORS])
