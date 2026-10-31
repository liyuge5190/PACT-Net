"""Contact-phase physics-consistent refinement (Section 3.7, Eq. 27-30).

The joint trajectory around the takeoff contact is refined so that the
centroidal momentum balance (Eq. 27) holds under the unilateral, friction,
non-penetration, complementarity and centre-of-pressure conditions (Eq. 28).
The contact force is expressed in body weights and follows from the
centre-of-mass acceleration; the centre of pressure is the ground point at
which the net moment has no horizontal component. The objective (Eq. 29) is
minimized by an augmented Lagrangian scheme whose inner problem is solved by
Levenberg-damped Gauss-Newton steps (Eq. 30) with a sparse finite-difference
Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .events import phase_memberships
from .skeleton import BONE_INDEX, FOOT_POINTS, J, com_weights, segment_com_matrix, segment_parameters

G0 = 9.81


@dataclass
class PhysicsConfig:
    lambda_smooth: float = 0.5      # lambda_1
    lambda_physics: float = 5.0     # lambda_2, initial penalty of the augmented Lagrangian
    lambda_anatomy: float = 1.0     # lambda_3
    lambda_limits: float = 2.0      # lambda_4
    friction: float = 0.9           # mu
    margin: int = 10                # frames refined on either side of the contact
    outer_iterations: int = 3
    inner_iterations: int = 5
    penalty_growth: float = 4.0
    beta: float = 1.0
    impact_frames: float = 1.6
    contact_tolerance: float = 0.01  # m, sole vertices within this height support load
    foot_half_width: float = 0.045   # m
    force_floor: float = 1.0         # BW, below which the centre of pressure is not constrained
    cop_scale: float = 0.25          # m
    bone_scale: float = 0.005        # m
    limit_scale: float = 0.05
    min_flexion_sin: float = np.sin(np.deg2rad(5.0))


def _cross(a, b):
    return np.cross(a, b)


def _point_segment_distance(p, a, b):
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


class ContactRefiner:
    def __init__(self, cfg: PhysicsConfig | None = None, sex: str = "M"):
        self.cfg = cfg or PhysicsConfig()
        self.w = com_weights(sex)
        self.A = segment_com_matrix(sex)
        self.m = segment_parameters(sex)[2]

    def _setup(self, Yhat, sd, tau_td, tau_to, fps, stance_side, lengths):
        cfg = self.cfg
        T = Yhat.shape[0]
        s = max(0, int(np.floor(tau_td)) - cfg.margin)
        e = min(T, int(np.ceil(tau_to)) + cfg.margin + 1)
        self.window = (s, e)
        W = e - s
        self.W, self.fps = W, fps
        self.Yhat = np.nan_to_num(Yhat[s:e])
        self.sd = np.where(np.isfinite(sd[s:e]), np.maximum(sd[s:e], 1e-4), 1.0)
        t = np.arange(s, e, dtype=np.float64)
        mem = phase_memberships(t, tau_td, tau_to, cfg.beta, cfg.impact_frames)
        self.m_con = mem["contact"][1:-1]
        self.m_imp = mem["impact"][1:-1]
        heel, toe, _ = FOOT_POINTS[stance_side]
        self.stance = (heel, toe)
        self.soles = np.array([FOOT_POINTS["l"][0], FOOT_POINTS["l"][1], FOOT_POINTS["r"][0], FOOT_POINTS["r"][1]])
        self.lengths = lengths
        self.legs = [(J[f"{k}_hip"], J[f"{k}_knee"], J[f"{k}_ankle"], 1.0) for k in ("l", "r")]
        self.arms = [(J[f"{k}_shoulder"], J[f"{k}_elbow"], J[f"{k}_wrist"], -1.0) for k in ("l", "r")]

        Jn = Yhat.shape[1]
        frames = [np.repeat(np.arange(W), Jn * 3)]
        frames.append(np.repeat(np.arange(1, W - 1), Jn * 3))
        n_phys = {"free": 3, "uni": 1, "fric": 1, "cop": 1, "comp": 1}
        for k, n in n_phys.items():
            frames.append(np.repeat(np.arange(1, W - 1), n))
        frames.append(np.repeat(np.arange(W), 4))                     # non-penetration
        frames.append(np.repeat(np.arange(W), BONE_INDEX.shape[0]))    # anatomy
        frames.append(np.repeat(np.arange(W), 4))                     # limits
        self.row_frame = np.concatenate(frames)

    def constraints(self, Y):
        """Equality and inequality constraint values (Eq. 28) of a window trajectory."""
        cfg, fps = self.cfg, self.fps
        c = np.einsum("j,wjk->wk", self.w, Y)
        acc = (c[2:] - 2.0 * c[1:-1] + c[:-2]) * fps ** 2
        F = (acc + np.array([0.0, 0.0, G0])) / G0
        mcon = self.m_con
        Fn, Ft = F[:, 2], F[:, :2]
        free = np.clip(1.0 - mcon, 0.0, 1.0)[:, None] * F
        uni = -Fn * mcon
        fric = (np.linalg.norm(Ft, axis=-1) - cfg.friction * Fn) * mcon

        r_seg = np.einsum("sj,wjk->wsk", self.A, Y)
        rdd = (r_seg[2:] - 2.0 * r_seg[1:-1] + r_seg[:-2]) * fps ** 2
        rel = r_seg[1:-1] - c[1:-1, None, :]
        Ldot = np.einsum("s,wsk->wk", self.m, _cross(rel, rdd - acc[:, None, :])) / G0
        cz = c[1:-1, 2]
        loaded = Fn > cfg.force_floor
        Fz = np.where(loaded, Fn, 1.0)
        dx = (-cz * F[:, 0] - Ldot[:, 1]) / Fz
        dy = (Ldot[:, 0] - cz * F[:, 1]) / Fz
        cop = c[1:-1, :2] + np.stack([dx, dy], -1)
        heel_i, toe_i = self.stance
        heel, toe = Y[1:-1, heel_i], Y[1:-1, toe_i]
        a = np.where((heel[:, 2] < cfg.contact_tolerance)[:, None], heel, toe + 0.25 * (heel - toe))
        dist = _point_segment_distance(cop, a[:, :2], toe[:, :2]) - cfg.foot_half_width
        cop_g = np.where(loaded, np.clip(dist, 0.0, None) / cfg.cop_scale, 0.0) * mcon
        dmin = np.minimum(heel[:, 2], toe[:, 2])
        comp = np.clip(Fn, 0.0, None) * np.clip(dmin - cfg.contact_tolerance, 0.0, None) / 0.01 * mcon
        pen = -Y[:, self.soles, 2] / 0.01
        return {"free": free, "uni": uni, "fric": fric, "cop": cop_g, "comp": comp, "pen": pen}, F, cop

    def residuals(self, x, zeta, rho):
        cfg = self.cfg
        Y = x.reshape(self.Yhat.shape)
        r_data = ((Y - self.Yhat) / self.sd).ravel()
        d2 = Y[2:] - 2.0 * Y[1:-1] + Y[:-2]
        ws = np.sqrt(cfg.lambda_smooth * np.clip(1.0 - self.m_imp, 0.0, 1.0))
        r_smooth = (ws[:, None, None] * d2 / self.sd[1:-1]).ravel()
        g, _, _ = self.constraints(Y)
        sr = np.sqrt(rho)
        parts = [r_data, r_smooth, (sr * (g["free"] + zeta["free"] / rho)).ravel()]
        for k in ("uni", "fric", "cop", "comp"):
            parts.append(sr * np.clip(g[k] + zeta[k] / rho, 0.0, None))
        parts.append((sr * np.clip(g["pen"] + zeta["pen"] / rho, 0.0, None)).ravel())
        d = Y[:, BONE_INDEX[:, 1]] - Y[:, BONE_INDEX[:, 0]]
        bl = np.linalg.norm(d, axis=-1)
        parts.append((np.sqrt(cfg.lambda_anatomy) * (bl - self.lengths) / cfg.bone_scale).ravel())
        parts.append((np.sqrt(cfg.lambda_limits) * self._limits(Y) / cfg.limit_scale).ravel())
        return np.concatenate(parts)

    def _limits(self, Y):
        cfg = self.cfg
        right_p = Y[:, J["r_hip"]] - Y[:, J["l_hip"]]
        right_t = Y[:, J["r_shoulder"]] - Y[:, J["l_shoulder"]]
        right_p /= np.maximum(np.linalg.norm(right_p, axis=-1, keepdims=True), 1e-9)
        right_t /= np.maximum(np.linalg.norm(right_t, axis=-1, keepdims=True), 1e-9)
        out = []
        for chains, ref in ((self.legs, right_p), (self.arms, right_t)):
            for p, mid, d, sign in chains:
                a = Y[:, p] - Y[:, mid]
                b = Y[:, d] - Y[:, mid]
                n = sign * np.cross(a, b)
                den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
                g = -np.sum(n * ref, -1) / np.maximum(den, 1e-9) - cfg.min_flexion_sin
                out.append(np.clip(g, 0.0, None))
        return np.stack(out, -1)

    def _jacobian(self, fun, x, r0, h=1e-6):
        W = self.W
        nv = x.size // W
        var_frame = np.repeat(np.arange(W), nv)
        var_local = np.tile(np.arange(nv), W)
        rows, cols, vals = [], [], []
        for m in range(3):
            for q in range(nv):
                sel = (var_frame % 3 == m) & (var_local == q)
                if not sel.any():
                    continue
                xp = x.copy()
                xp[sel] += h
                d = (fun(xp) - r0) / h
                nz = np.flatnonzero(d)
                t = self.row_frame[nz]
                f = (t - 1) + ((m - (t - 1)) % 3)
                ok = (f >= 0) & (f < W)
                rows.append(nz[ok])
                cols.append(f[ok] * nv + q)
                vals.append(d[nz[ok]])
        return sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r0.size, x.size)
        )

    def refine(self, Yhat, sd, tau_td, tau_to, fps, stance_side, lengths=None):
        """Refine ``Yhat`` (T, J, 3) with per-coordinate standard deviations ``sd`` (T, J, 3)."""
        cfg = self.cfg
        if lengths is None:
            d = Yhat[:, BONE_INDEX[:, 1]] - Yhat[:, BONE_INDEX[:, 0]]
            lengths = np.nanmedian(np.linalg.norm(d, axis=-1), axis=0)
        self._setup(Yhat, sd, tau_td, tau_to, fps, stance_side, lengths)
        x = self.Yhat.ravel().copy()
        g0, _, _ = self.constraints(self.Yhat)
        zeta = {k: np.zeros_like(v) for k, v in g0.items()}
        rho = cfg.lambda_physics
        history = []
        prev_violation = np.inf
        for _outer in range(cfg.outer_iterations):
            fun = lambda z: self.residuals(z, zeta, rho)
            r = fun(x)
            cost = 0.5 * r @ r
            nu = None
            for _inner in range(cfg.inner_iterations):
                Jm = self._jacobian(fun, x, r)
                JtJ = (Jm.T @ Jm).tocsc()
                grad = Jm.T @ r
                diag = JtJ.diagonal()
                damp = sparse.diags(np.maximum(diag, 1e-6 * diag.max()), format="csc")
                if nu is None:
                    nu = 1e-3
                accepted = False
                for _try in range(8):
                    step = spsolve(JtJ + nu * damp, -grad)
                    r_new = fun(x + step)
                    cost_new = 0.5 * r_new @ r_new
                    if cost_new < cost:
                        x, r, cost = x + step, r_new, cost_new
                        nu = max(nu / 3.0, 1e-9)
                        accepted = True
                        break
                    nu *= 4.0
                if not accepted or np.linalg.norm(step) < 1e-7:
                    break
            g, _, _ = self.constraints(x.reshape(self.Yhat.shape))
            zeta["free"] = zeta["free"] + rho * g["free"]
            for k in ("uni", "fric", "cop", "comp", "pen"):
                zeta[k] = np.clip(zeta[k] + rho * g[k], 0.0, None)
            violation = max(
                float(np.abs(g["free"]).max()),
                *[float(np.clip(g[k], 0.0, None).max()) for k in ("uni", "fric", "cop", "comp", "pen")],
            )
            history.append({"cost": float(cost), "violation": violation, "rho": rho})
            if violation > 0.25 * prev_violation:
                rho *= cfg.penalty_growth
            prev_violation = violation

        Yw = x.reshape(self.Yhat.shape)
        _, F, cop = self.constraints(Yw)
        s, e = self.window
        Y_out = np.array(Yhat, dtype=np.float64, copy=True)
        Y_out[s:e] = Yw
        T = Yhat.shape[0]
        force = np.full((T, 3), np.nan)
        force[s + 1 : e - 1] = F
        cop_full = np.full((T, 2), np.nan)
        cop_full[s + 1 : e - 1] = cop
        return {"joints": Y_out, "force": force, "cop": cop_full, "window": (s, e), "history": history}
