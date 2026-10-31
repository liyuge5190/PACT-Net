"""Skeleton definition, segmental inertial model and centre of mass (Eq. 13).

Joint centres are indexed by ``JOINTS``. Segment inertial parameters follow
de Leva (1996), Table 4. Because no hand keypoint is detected, the hand is
placed on the forearm axis beyond the wrist (``eta > 1``).
"""

from __future__ import annotations

import numpy as np

JOINTS = (
    "pelvis",
    "r_hip", "r_knee", "r_ankle", "r_heel", "r_toe",
    "l_hip", "l_knee", "l_ankle", "l_heel", "l_toe",
    "thorax", "neck", "head",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
)
NUM_JOINTS = len(JOINTS)
J = {name: i for i, name in enumerate(JOINTS)}

# Kinematic tree in topological order (parent, child).
BONES = (
    ("pelvis", "r_hip"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
    ("r_ankle", "r_heel"), ("r_ankle", "r_toe"),
    ("pelvis", "l_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ("l_ankle", "l_heel"), ("l_ankle", "l_toe"),
    ("pelvis", "thorax"), ("thorax", "neck"), ("neck", "head"),
    ("neck", "l_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("neck", "r_shoulder"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
)
BONE_INDEX = np.array([(J[p], J[c]) for p, c in BONES], dtype=np.int64)

# Index permutation that exchanges left and right (used for mirroring).
FLIP_PERM = np.array(
    [J[n.replace("l_", "#").replace("r_", "l_").replace("#", "r_")] for n in JOINTS],
    dtype=np.int64,
)

FOOT_POINTS = {
    "l": (J["l_heel"], J["l_toe"], J["l_ankle"]),
    "r": (J["r_heel"], J["r_toe"], J["r_ankle"]),
}

# name, proximal, distal, (mass %, CoM % from proximal) male, female
_DE_LEVA = (
    ("head", "head", "neck", (6.94, 59.76), (6.68, 58.94)),
    ("trunk", "neck", "pelvis", (43.46, 44.86), (42.57, 41.51)),
    ("l_upper_arm", "l_shoulder", "l_elbow", (2.71, 57.72), (2.55, 57.54)),
    ("r_upper_arm", "r_shoulder", "r_elbow", (2.71, 57.72), (2.55, 57.54)),
    ("l_forearm", "l_elbow", "l_wrist", (1.62, 45.74), (1.38, 45.59)),
    ("r_forearm", "r_elbow", "r_wrist", (1.62, 45.74), (1.38, 45.59)),
    ("l_hand", "l_elbow", "l_wrist", (0.61, 130.0), (0.56, 128.0)),
    ("r_hand", "r_elbow", "r_wrist", (0.61, 130.0), (0.56, 128.0)),
    ("l_thigh", "l_hip", "l_knee", (14.16, 40.95), (14.78, 36.12)),
    ("r_thigh", "r_hip", "r_knee", (14.16, 40.95), (14.78, 36.12)),
    ("l_shank", "l_knee", "l_ankle", (4.33, 44.59), (4.81, 44.16)),
    ("r_shank", "r_knee", "r_ankle", (4.33, 44.59), (4.81, 44.16)),
    ("l_foot", "l_heel", "l_toe", (1.37, 44.15), (1.29, 40.14)),
    ("r_foot", "r_heel", "r_toe", (1.37, 44.15), (1.29, 40.14)),
)
SEGMENT_NAMES = tuple(s[0] for s in _DE_LEVA)
NUM_SEGMENTS = len(_DE_LEVA)


def segment_parameters(sex: str = "M"):
    """Return (prox_idx, dist_idx, mass_fraction, eta) arrays for ``sex``."""
    col = 3 if str(sex).upper().startswith("M") else 4
    prox = np.array([J[s[1]] for s in _DE_LEVA], dtype=np.int64)
    dist = np.array([J[s[2]] for s in _DE_LEVA], dtype=np.int64)
    mass = np.array([s[col][0] for s in _DE_LEVA], dtype=np.float64)
    eta = np.array([s[col][1] for s in _DE_LEVA], dtype=np.float64) / 100.0
    return prox, dist, mass / mass.sum(), eta


def segment_com_matrix(sex: str = "M") -> np.ndarray:
    """Matrix A (S, J) with segment centres of mass ``r = A @ y``."""
    prox, dist, _, eta = segment_parameters(sex)
    A = np.zeros((NUM_SEGMENTS, NUM_JOINTS))
    for i in range(NUM_SEGMENTS):
        A[i, prox[i]] += 1.0 - eta[i]
        A[i, dist[i]] += eta[i]
    return A


def com_weights(sex: str = "M") -> np.ndarray:
    """Joint weights w (J,) such that the whole-body CoM is ``sum_j w_j y_j`` (Eq. 13)."""
    _, _, mass, _ = segment_parameters(sex)
    return mass @ segment_com_matrix(sex)


def center_of_mass(joints: np.ndarray, sex: str = "M") -> np.ndarray:
    """Whole-body centre of mass of ``joints`` (..., J, 3) -> (..., 3)."""
    return np.einsum("j,...jk->...k", com_weights(sex), joints)


def com_covariance(cov: np.ndarray, sex: str = "M") -> np.ndarray:
    """Covariance of the CoM from per-joint covariances (..., J, 3, 3), joints independent."""
    w = com_weights(sex)
    return np.einsum("j,...jkl->...kl", w ** 2, cov)


def bone_lengths(joints: np.ndarray) -> np.ndarray:
    """Bone lengths (..., B) of ``joints`` (..., J, 3)."""
    d = joints[..., BONE_INDEX[:, 1], :] - joints[..., BONE_INDEX[:, 0], :]
    return np.linalg.norm(d, axis=-1)


def mirror(joints: np.ndarray, lateral_axis: int = 1) -> np.ndarray:
    """Mirror ``joints`` (..., J, 3) about the sagittal plane and exchange sides."""
    out = joints[..., FLIP_PERM, :].copy()
    out[..., lateral_axis] *= -1.0
    return out
