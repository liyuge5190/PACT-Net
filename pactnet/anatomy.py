"""Anatomical joint centre correction and skeletal consistency (Section 3.6).

Surface keypoints are mapped to functional joint centres by a constant
offset in the proximal segment frame (Eq. 24). For each joint the offset is
identified from the approach strides with a symmetric centre-of-rotation
criterion regularized towards the detected keypoint (Eq. 25), which is a
linear least-squares problem solved in closed form. Segment lengths are then
enforced by projection along the kinematic tree (Eq. 26).
"""

from __future__ import annotations

import numpy as np

from .angles import JOINT_SEGMENTS, segment_frames
from .skeleton import BONE_INDEX, J

CORRECTED_JOINTS = ("l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
                    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow")


def fit_joint_centre_offsets(sequences, lam: float = 0.01, joints=CORRECTED_JOINTS) -> dict:
    """Identify segment-local offsets from one or more (T, J, 3) approach sequences.

    Returns ``{joint: delta}`` with delta (3,) expressed in the proximal segment frame.
    """
    if isinstance(sequences, np.ndarray):
        sequences = [sequences]
    rows = {j: [] for j in joints}
    for Y in sequences:
        Y = np.asarray(Y, dtype=np.float64)
        ok = np.isfinite(Y).all(axis=(1, 2))
        Y = Y[ok]
        if Y.shape[0] == 0:
            continue
        fr = segment_frames(Y)
        for j in joints:
            prox, dist = JOINT_SEGMENTS[j]
            (op, Rp), (od, Rd) = fr[prox], fr[dist]
            y = Y[:, J[j]]
            ap_obs = np.einsum("tki,tk->ti", Rp, y - op)
            ad_obs = np.einsum("tki,tk->ti", Rd, y - od)
            rows[j].append((Rp, Rd, op, od, ap_obs, ad_obs))
    offsets = {}
    for j in joints:
        if not rows[j]:
            offsets[j] = np.zeros(3)
            continue
        Rp = np.concatenate([r[0] for r in rows[j]])
        Rd = np.concatenate([r[1] for r in rows[j]])
        op = np.concatenate([r[2] for r in rows[j]])
        od = np.concatenate([r[3] for r in rows[j]])
        ap_obs = np.concatenate([r[4] for r in rows[j]]).mean(0)
        ad_obs = np.concatenate([r[5] for r in rows[j]]).mean(0)
        T = Rp.shape[0]
        A = np.concatenate([Rp, -Rd], axis=2).reshape(3 * T, 6)
        b = (od - op).reshape(3 * T)
        reg = np.sqrt(lam * T)
        A_full = np.vstack([A, reg * np.eye(6)])
        b_full = np.concatenate([b, reg * ap_obs, reg * ad_obs])
        sol, *_ = np.linalg.lstsq(A_full, b_full, rcond=None)
        offsets[j] = sol[:3] - ap_obs
    return offsets


def apply_joint_centre_offsets(Y: np.ndarray, offsets: dict) -> np.ndarray:
    """Eq. (24): ``c_j = y_j + R_s(j) delta_j`` with frames from the uncorrected keypoints."""
    out = np.array(Y, dtype=np.float64, copy=True)
    fr = segment_frames(Y)
    for j, delta in offsets.items():
        prox, _ = JOINT_SEGMENTS[j]
        out[..., J[j], :] = Y[..., J[j], :] + np.einsum("...ki,i->...k", fr[prox][1], delta)
    return out


def reference_bone_lengths(Y: np.ndarray) -> np.ndarray:
    d = Y[..., BONE_INDEX[:, 1], :] - Y[..., BONE_INDEX[:, 0], :]
    L = np.linalg.norm(d, axis=-1)
    return np.nanmedian(L.reshape(-1, L.shape[-1]), axis=0)


def project_bone_lengths(Y: np.ndarray, lengths: np.ndarray | None = None) -> np.ndarray:
    """Eq. (26): enforce bone lengths exactly, parent to child along the tree.

    Each subtree is translated with its root so that descendants keep their
    own bone directions.
    """
    Y = np.array(Y, dtype=np.float64, copy=True)
    L = reference_bone_lengths(Y) if lengths is None else lengths
    children = {}
    for b, (p, c) in enumerate(BONE_INDEX):
        children.setdefault(int(p), []).append((b, int(c)))

    def descendants(j):
        out = []
        for _, c in children.get(j, []):
            out.append(c)
            out.extend(descendants(c))
        return out

    for b, (p, c) in enumerate(BONE_INDEX):
        d = Y[..., c, :] - Y[..., p, :]
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        new_c = Y[..., p, :] + L[b] * d / np.maximum(n, 1e-12)
        shift = new_c - Y[..., c, :]
        Y[..., c, :] = new_c
        for k in descendants(int(c)):
            Y[..., k, :] += shift
    return Y
