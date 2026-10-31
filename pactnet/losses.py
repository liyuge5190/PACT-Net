"""Composite training objective (Section 3.9, Eq. 35).

L = L_3D + lambda_u L_NLL + lambda_e L_event + lambda_p L_phy + lambda_b L_bone + lambda_d L_deriv

Positions enter L_3D and L_bone in centimetres, velocities in m/s and
accelerations in units of g, so that the weights of Table 3 act on terms of
comparable magnitude.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import phase_memberships
from .skeleton import BONE_INDEX, com_weights

G0 = 9.81


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(x.dtype)
    while mask.dim() < x.dim():
        mask = mask[..., None]
    mask = mask.expand_as(x)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


class PACTLoss(torch.nn.Module):
    def __init__(self, cfg: dict, fps: float = 200.0, sex_weights=None):
        super().__init__()
        self.w = cfg["loss"]
        self.fps = fps
        self.beta = cfg["model"]["event_beta"]
        cw = com_weights("M") if sex_weights is None else sex_weights
        self.register_buffer("com_w", torch.as_tensor(cw, dtype=torch.float32))
        self.register_buffer("bones", torch.as_tensor(BONE_INDEX, dtype=torch.long))

    def forward(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        pred, logvar = out["joints"], out["logvar"]
        ref, m = batch["ref"], batch["ref_valid"] & batch["valid"]
        fps = self.fps

        l3d = _masked_mean(torch.linalg.norm(pred - ref, dim=-1) * 100.0, m)

        err2 = (ref - pred.detach()) ** 2
        nll = _masked_mean(0.5 * (err2 * torch.exp(-logvar) + logvar), m)

        lev = F.smooth_l1_loss(out["tau"], batch["tau_ref"])

        c = torch.einsum("j,btjk->btk", self.com_w, pred)
        acc = (c[:, 2:] - 2.0 * c[:, 1:-1] + c[:, :-2]) * fps ** 2
        force = acc / G0
        force = force + torch.tensor([0.0, 0.0, 1.0], device=force.device, dtype=force.dtype)
        mem = phase_memberships(pred.shape[1], out["tau"].detach(), self.beta)
        mcon = mem["contact"][:, 1:-1]
        m3 = m[:, 2:] & m[:, 1:-1] & m[:, :-2]
        free = (1.0 - mcon) * (force ** 2).sum(-1)
        uni = mcon * F.relu(-force[..., 2]) ** 2
        lphy = _masked_mean(free + uni, m3)

        d = pred[:, :, self.bones[:, 1]] - pred[:, :, self.bones[:, 0]]
        L = torch.linalg.norm(d, dim=-1) * 100.0
        mf = m[..., None].to(L.dtype)
        n = mf.sum(1).clamp_min(2.0)
        mean = (L * mf).sum(1) / n
        var = (((L - mean[:, None]) ** 2) * mf).sum(1) / (n - 1.0)
        lbone = torch.sqrt(var + 1e-8).mean()

        v_p = (pred[:, 2:] - pred[:, :-2]) * (fps / 2.0)
        v_r = (ref[:, 2:] - ref[:, :-2]) * (fps / 2.0)
        a_p = (pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]) * fps ** 2
        a_r = (ref[:, 2:] - 2.0 * ref[:, 1:-1] + ref[:, :-2]) * fps ** 2
        lder = _masked_mean(torch.linalg.norm(v_p - v_r, dim=-1), m3) + _masked_mean(
            torch.linalg.norm(a_p - a_r, dim=-1) / G0, m3
        )

        total = (l3d + self.w["lambda_u"] * nll + self.w["lambda_e"] * lev + self.w["lambda_p"] * lphy
                 + self.w["lambda_b"] * lbone + self.w["lambda_d"] * lder)
        parts = {"l3d": l3d, "nll": nll, "event": lev, "phy": lphy, "bone": lbone, "deriv": lder}
        return total, {k: float(v.detach()) for k, v in parts.items()}
