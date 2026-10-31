"""Segment frames, joint angles and joint angular velocity (Eq. 31-32).

Segment axes follow the convention X anterior, Y superior (along the long
axis, distal to proximal) and Z to the athlete's right for both sides.
Relative rotations are decomposed in the Z-X-Y order of the joint coordinate
system (Grood and Suntay, 1983): flexion about Z, ab/adduction about X and
axial rotation about Y.
"""

from __future__ import annotations

import numpy as np

from .skeleton import J

SEGMENTS = (
    "pelvis", "trunk",
    "l_thigh", "r_thigh", "l_shank", "r_shank", "l_foot", "r_foot",
    "l_upper_arm", "r_upper_arm", "l_forearm", "r_forearm",
)

JOINT_SEGMENTS = {
    "l_hip": ("pelvis", "l_thigh"), "r_hip": ("pelvis", "r_thigh"),
    "l_knee": ("l_thigh", "l_shank"), "r_knee": ("r_thigh", "r_shank"),
    "l_ankle": ("l_shank", "l_foot"), "r_ankle": ("r_shank", "r_foot"),
    "l_shoulder": ("trunk", "l_upper_arm"), "r_shoulder": ("trunk", "r_upper_arm"),
    "l_elbow": ("l_upper_arm", "l_forearm"), "r_elbow": ("r_upper_arm", "r_forearm"),
}

FLEXION_SIGN = {"knee": -1.0}


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _frame_from_y_and_z(y_axis, z_ref):
    Y = _unit(y_axis)
    X = _unit(np.cross(Y, z_ref))
    Z = np.cross(X, Y)
    return np.stack([X, Y, Z], axis=-1)


def _blend_plane_normal(n_raw, la, lb, fallback, min_sin=np.sin(np.deg2rad(15.0))):
    s = np.linalg.norm(n_raw, axis=-1, keepdims=True) / np.maximum(la * lb, 1e-12)
    w = np.clip(s / min_sin, 0.0, 1.0)
    return _unit(w * _unit(n_raw) + (1.0 - w) * fallback)


def segment_frames(Y: np.ndarray) -> dict:
    """Origins (..., 3) and rotations (..., 3, 3) whose columns are the X, Y, Z axes."""
    p = lambda name: Y[..., J[name], :]
    frames = {}
    right = _unit(p("r_hip") - p("l_hip"))
    frames["pelvis"] = (p("pelvis"), _frame_from_y_and_z(p("thorax") - p("pelvis"), right))
    shoulder_right = _unit(p("r_shoulder") - p("l_shoulder"))
    frames["trunk"] = (p("neck"), _frame_from_y_and_z(p("neck") - p("pelvis"), shoulder_right))
    pelvis_z = frames["pelvis"][1][..., 2]
    trunk_z = frames["trunk"][1][..., 2]
    for s in ("l", "r"):
        hip, knee, ankle = p(f"{s}_hip"), p(f"{s}_knee"), p(f"{s}_ankle")
        a, b = hip - knee, ankle - knee
        z_leg = _blend_plane_normal(np.cross(a, b), np.linalg.norm(a, axis=-1, keepdims=True),
                                    np.linalg.norm(b, axis=-1, keepdims=True), pelvis_z)
        frames[f"{s}_thigh"] = (knee, _frame_from_y_and_z(hip - knee, z_leg))
        frames[f"{s}_shank"] = (ankle, _frame_from_y_and_z(knee - ankle, z_leg))
        heel, toe = p(f"{s}_heel"), p(f"{s}_toe")
        Xf = _unit(toe - heel)
        Zf = _unit(np.cross(Xf, ankle - heel))
        Yf = np.cross(Zf, Xf)
        frames[f"{s}_foot"] = (toe, np.stack([Xf, Yf, Zf], axis=-1))
        sh, el, wr = p(f"{s}_shoulder"), p(f"{s}_elbow"), p(f"{s}_wrist")
        a, b = sh - el, wr - el
        z_arm = _blend_plane_normal(-np.cross(a, b), np.linalg.norm(a, axis=-1, keepdims=True),
                                    np.linalg.norm(b, axis=-1, keepdims=True), trunk_z)
        frames[f"{s}_upper_arm"] = (el, _frame_from_y_and_z(sh - el, z_arm))
        frames[f"{s}_forearm"] = (wr, _frame_from_y_and_z(el - wr, z_arm))
    return frames


def decompose_zxy(R: np.ndarray) -> np.ndarray:
    """Angles (alpha, beta, gamma) in radians with R = Rz(alpha) Rx(beta) Ry(gamma)."""
    beta = np.arcsin(np.clip(R[..., 2, 1], -1.0, 1.0))
    alpha = np.arctan2(-R[..., 0, 1], R[..., 1, 1])
    gamma = np.arctan2(-R[..., 2, 0], R[..., 2, 2])
    return np.stack([alpha, beta, gamma], axis=-1)


def relative_rotations(Y: np.ndarray, frames: dict | None = None) -> dict:
    frames = segment_frames(Y) if frames is None else frames
    return {
        joint: np.swapaxes(frames[prox][1], -1, -2) @ frames[dist][1]
        for joint, (prox, dist) in JOINT_SEGMENTS.items()
    }


def joint_angles(Y: np.ndarray) -> dict:
    """Joint angles in degrees: (..., 3) as (flexion, adduction, rotation) per joint (Eq. 31)."""
    out = {}
    for joint, R in relative_rotations(Y).items():
        ang = np.rad2deg(decompose_zxy(R))
        ang[..., 0] *= FLEXION_SIGN.get(joint.split("_", 1)[1], 1.0)
        out[joint] = ang
    frames = segment_frames(Y)
    trunk_y = frames["trunk"][1][..., 1]
    out["trunk_lean"] = np.rad2deg(np.arctan2(trunk_y[..., 0], trunk_y[..., 2]))[..., None]
    pelvis_y = frames["pelvis"][1][..., 1]
    out["pelvic_tilt"] = np.rad2deg(np.arctan2(pelvis_y[..., 0], pelvis_y[..., 2]))[..., None]
    return out


def vee(S: np.ndarray) -> np.ndarray:
    return np.stack([S[..., 2, 1] - S[..., 1, 2], S[..., 0, 2] - S[..., 2, 0], S[..., 1, 0] - S[..., 0, 1]], -1) / 2.0


def angular_velocity(R_seq: np.ndarray, D1: np.ndarray) -> np.ndarray:
    """Angular velocity (..., T, 3) in rad/s of rotations (..., T, 3, 3) (Eq. 32).

    ``D1`` is a (T, T) first-derivative operator acting on the time axis.
    """
    R_dot = np.einsum("st,...tij->...sij", D1, np.nan_to_num(R_seq))
    return vee(R_dot @ np.swapaxes(R_seq, -1, -2))


def knee_rotation(Y: np.ndarray, side: str) -> np.ndarray:
    """Relative rotation of the shank with respect to the thigh, built from the leg only."""
    p = lambda name: Y[..., J[name], :]
    right = _unit(p("r_hip") - p("l_hip"))
    pelvis_z = _frame_from_y_and_z(p("thorax") - p("pelvis"), right)[..., 2]
    hip, knee, ankle = p(f"{side}_hip"), p(f"{side}_knee"), p(f"{side}_ankle")
    a, b = hip - knee, ankle - knee
    z_leg = _blend_plane_normal(np.cross(a, b), np.linalg.norm(a, axis=-1, keepdims=True),
                                np.linalg.norm(b, axis=-1, keepdims=True), pelvis_z)
    R_thigh = _frame_from_y_and_z(hip - knee, z_leg)
    R_shank = _frame_from_y_and_z(knee - ankle, z_leg)
    return np.swapaxes(R_thigh, -1, -2) @ R_shank


def knee_flexion(Y: np.ndarray, side: str) -> np.ndarray:
    """Knee flexion angle in degrees."""
    return -np.rad2deg(decompose_zxy(knee_rotation(Y, side))[..., 0])


def knee_extension_velocity(Y: np.ndarray, side: str, D1: np.ndarray) -> np.ndarray:
    """Knee extension angular velocity (deg/s) of ``side`` over time, positive for extension."""
    omega = angular_velocity(knee_rotation(Y, side), D1)
    return np.rad2deg(omega[..., 2])
