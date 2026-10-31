"""Pinhole camera with radial and tangential distortion (Eq. 2).

Distortion coefficients follow the OpenCV order ``(k1, k2, p1, p2, k3)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Rotation matrix from a rotation vector."""
    rvec = np.asarray(rvec, dtype=np.float64)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * Kx + (1.0 - np.cos(theta)) * Kx @ Kx


def rotation_to_rvec(R: np.ndarray) -> np.ndarray:
    """Rotation vector from a rotation matrix."""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-12:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        w, v = np.linalg.eigh((R + R.T) / 2.0)
        axis = v[:, np.argmax(w)]
        return axis * theta
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2.0 * np.sin(theta)) * theta


def look_at(position, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World-to-camera rotation for a camera at ``position`` looking at ``target``."""
    position = np.asarray(position, float)
    z = np.asarray(target, float) - position
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


@dataclass
class Camera:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    name: str = ""

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)
        d = np.zeros(5)
        src = np.asarray(self.dist, dtype=np.float64).ravel()
        d[: min(5, src.size)] = src[:5]
        self.dist = d

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def to_camera(self, X: np.ndarray) -> np.ndarray:
        return X @ self.R.T + self.t

    def _distort(self, xn: np.ndarray, yn: np.ndarray):
        k1, k2, p1, p2, k3 = self.dist
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        return xd, yd, radial, r2

    def project(self, X: np.ndarray) -> np.ndarray:
        """Project world points (..., 3) to pixels (..., 2)."""
        P = self.to_camera(np.asarray(X, dtype=np.float64))
        z = np.where(np.abs(P[..., 2]) < 1e-9, 1e-9, P[..., 2])
        xd, yd, _, _ = self._distort(P[..., 0] / z, P[..., 1] / z)
        u = self.K[0, 0] * xd + self.K[0, 1] * yd + self.K[0, 2]
        v = self.K[1, 1] * yd + self.K[1, 2]
        return np.stack([u, v], axis=-1)

    def project_with_jacobian(self, X: np.ndarray):
        """Return pixels (..., 2) and the Jacobian d(pixel)/d(X) (..., 2, 3)."""
        X = np.asarray(X, dtype=np.float64)
        P = self.to_camera(X)
        z = np.where(np.abs(P[..., 2]) < 1e-9, 1e-9, P[..., 2])
        xn, yn = P[..., 0] / z, P[..., 1] / z
        k1, k2, p1, p2, k3 = self.dist
        xd, yd, radial, r2 = self._distort(xn, yn)
        dr = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r2 ** 2
        dxd_dx = radial + 2.0 * xn * xn * dr + 2.0 * p1 * yn + 6.0 * p2 * xn
        dxd_dy = 2.0 * xn * yn * dr + 2.0 * p1 * xn + 2.0 * p2 * yn
        dyd_dx = 2.0 * xn * yn * dr + 2.0 * p1 * xn + 2.0 * p2 * yn
        dyd_dy = radial + 2.0 * yn * yn * dr + 6.0 * p1 * yn + 2.0 * p2 * xn
        Dd = np.stack([np.stack([dxd_dx, dxd_dy], -1), np.stack([dyd_dx, dyd_dy], -1)], -2)
        Kp = self.K[:2, :2]
        dn_dP = np.zeros(X.shape[:-1] + (2, 3))
        dn_dP[..., 0, 0] = 1.0 / z
        dn_dP[..., 0, 2] = -xn / z
        dn_dP[..., 1, 1] = 1.0 / z
        dn_dP[..., 1, 2] = -yn / z
        Jac = Kp @ Dd @ dn_dP @ self.R
        u = self.K[0, 0] * xd + self.K[0, 1] * yd + self.K[0, 2]
        v = self.K[1, 1] * yd + self.K[1, 2]
        return np.stack([u, v], axis=-1), Jac

    def undistort_normalized(self, uv: np.ndarray, iterations: int = 8) -> np.ndarray:
        """Normalized undistorted coordinates (..., 2) of pixel coordinates ``uv``."""
        uv = np.asarray(uv, dtype=np.float64)
        Kinv = np.linalg.inv(self.K)
        xd = Kinv[0, 0] * uv[..., 0] + Kinv[0, 1] * uv[..., 1] + Kinv[0, 2]
        yd = Kinv[1, 1] * uv[..., 1] + Kinv[1, 2]
        xn, yn = xd.copy(), yd.copy()
        k1, k2, p1, p2, k3 = self.dist
        for _ in range(iterations):
            r2 = xn * xn + yn * yn
            radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            dx = 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
            dy = p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
            xn = (xd - dx) / radial
            yn = (yd - dy) / radial
        return np.stack([xn, yn], axis=-1)

    def transformed(self, scale: float, R_w: np.ndarray, d: np.ndarray) -> "Camera":
        """Camera expressed in a new world frame ``X' = scale * R_w @ X + d``."""
        R_new = self.R @ R_w.T
        t_new = scale * self.t - R_new @ np.asarray(d, float)
        return Camera(self.K.copy(), R_new, t_new, self.dist.copy(), self.name)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "K": self.K.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "dist": self.dist.tolist(),
        }

    @staticmethod
    def from_dict(d: dict) -> "Camera":
        return Camera(d["K"], d["R"], d["t"], d.get("dist", np.zeros(5)), d.get("name", ""))


def cameras_to_json(cameras) -> str:
    return json.dumps([c.to_dict() for c in cameras])


def cameras_from_json(text) -> list:
    if isinstance(text, np.ndarray):
        text = text.item()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return [Camera.from_dict(d) for d in json.loads(str(text))]
