"""Contact-event-gated temporal network (Section 3.5, Eq. 17-22).

The network refines the metrically resolved triangulated sequence. A
lightweight temporal head yields touchdown and toe-off logits whose soft
arg-max gives sub-frame event instants (Eq. 17). The events parameterize an
additive attention bias that penalizes attention between frames on opposite
sides of an event with a learnable, non-negative strength per head (Eq. 20),
and a relative position encoding on a time axis normalized by the estimated
contact duration (Eq. 21). Local and global features are fused by a learned
gate (Eq. 22). The output is a residual correction of the joint positions and
a per-coordinate log-variance.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_argmax(logits: torch.Tensor, temperature: float):
    """Event instants (B, E) and their spread from logits (B, T, E) (Eq. 17)."""
    T = logits.shape[1]
    p = torch.softmax(logits / temperature, dim=1)
    t = torch.arange(T, device=logits.device, dtype=logits.dtype)[None, :, None]
    tau = (p * t).sum(1)
    sd = torch.sqrt(((t - tau[:, None, :]) ** 2 * p).sum(1).clamp_min(1e-6))
    return tau, sd


def event_crossing(T: int, tau: torch.Tensor, beta: float) -> torch.Tensor:
    """Sum over events of |sigma((t - tau)/beta) - sigma((t' - tau)/beta)|, shape (B, T, T) (Eq. 20)."""
    t = torch.arange(T, device=tau.device, dtype=tau.dtype)[None, :, None]
    s = torch.sigmoid((t - tau[:, None, :]) / beta)
    return (s[:, :, None, :] - s[:, None, :, :]).abs().sum(-1)


def phase_memberships(T: int, tau: torch.Tensor, beta: float, impact_frames: float = 1.6):
    """Soft memberships (B, T) of the approach, contact and flight phases (Eq. 18)."""
    t = torch.arange(T, device=tau.device, dtype=tau.dtype)[None, :]
    a = torch.sigmoid((t - tau[:, :1]) / beta)
    b = torch.sigmoid((t - tau[:, 1:2]) / beta)
    c = torch.sigmoid((t - tau[:, :1] - impact_frames) / beta)
    return {"pre": 1.0 - a, "contact": a * (1.0 - b), "post": a * b, "impact": a * (1.0 - c)}


class PhaseRelativePosition(nn.Module):
    """P_{t,t'} = omega_h^T phi(t~ - t~') with t~ normalized by the contact duration (Eq. 21)."""

    def __init__(self, heads: int, bands: int = 6, normalized: bool = True, nominal_contact: float = 24.0):
        super().__init__()
        self.normalized = normalized
        self.nominal = nominal_contact
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(bands, dtype=torch.float32) / 8.0)
        self.omega = nn.Linear(2 * bands + 1, heads, bias=False)
        nn.init.zeros_(self.omega.weight)

    def forward(self, T: int, tau: torch.Tensor) -> torch.Tensor:
        t = torch.arange(T, device=tau.device, dtype=tau.dtype)
        diff = t[:, None] - t[None, :]
        if self.normalized:
            duration = (tau[:, 1] - tau[:, 0]).clamp_min(4.0)
            delta = diff[None] / duration[:, None, None]
        else:
            delta = (diff / self.nominal)[None].expand(tau.shape[0], T, T)
        ang = delta[..., None] * self.freqs
        phi = torch.cat([torch.tanh(delta / 2.0)[..., None], torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.omega(phi).permute(0, 3, 1, 2)


class GatedAttention(nn.Module):
    """Multi-head self-attention with the discontinuity-preserving bias of Eq. (19)-(20)."""

    def __init__(self, dim: int, heads: int, gated_heads: int, dropout: float, gating: bool = True):
        super().__init__()
        self.heads, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.gating = gating
        self.gamma_raw = nn.Parameter(torch.zeros(heads))
        mask = torch.zeros(heads)
        mask[:gated_heads] = 1.0
        self.register_buffer("gamma_mask", mask)

    def gamma(self) -> torch.Tensor:
        return F.softplus(self.gamma_raw) * self.gamma_mask

    def forward(self, x, crossing, rel, key_valid):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        if rel is not None:
            logits = logits + rel
        if self.gating and crossing is not None:
            logits = logits - self.gamma()[None, :, None, None] * crossing[:, None]
        logits = logits.masked_fill(~key_valid[:, None, None, :], float("-inf"))
        attn = self.drop(torch.softmax(logits, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim, heads, gated_heads, dropout, gating):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = GatedAttention(dim, heads, gated_heads, dropout, gating)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, crossing, rel, key_valid):
        x = x + self.drop(self.attn(self.n1(x), crossing, rel, key_valid))
        return x + self.drop(self.mlp(self.n2(x)))


class PACTNet(nn.Module):
    def __init__(self, num_joints: int = 20, hidden: int = 256, blocks: int = 8, heads: int = 8,
                 gated_heads: int = 4, event_beta: float = 1.0, softargmax_temperature: float = 0.5,
                 phase_normalized_encoding: bool = True, discontinuity_attention: bool = True,
                 dropout: float = 0.1, **unused):
        super().__init__()
        self.J = num_joints
        self.beta = event_beta
        self.temperature = softargmax_temperature
        in_dim = num_joints * 3 + 3 + num_joints + 1
        self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.event_head = nn.Sequential(
            nn.Conv1d(hidden, 128, 5, padding=2), nn.GELU(),
            nn.Conv1d(128, 128, 5, padding=2), nn.GELU(),
            nn.Conv1d(128, 2, 1),
        )
        self.local = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2, groups=hidden), nn.GELU(), nn.Conv1d(hidden, hidden, 1)
        )
        self.relpos = PhaseRelativePosition(heads, normalized=phase_normalized_encoding)
        self.blocks = nn.ModuleList(
            [Block(hidden, heads, gated_heads, dropout, discontinuity_attention) for _ in range(blocks)]
        )
        self.gate = nn.Linear(2 * hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.out_pos = nn.Linear(hidden, num_joints * 3)
        self.out_logvar = nn.Linear(hidden, num_joints * 3)
        nn.init.zeros_(self.out_pos.weight)
        nn.init.zeros_(self.out_pos.bias)
        nn.init.zeros_(self.out_logvar.weight)
        nn.init.constant_(self.out_logvar.bias, math.log(0.01 ** 2))

    @staticmethod
    def features(joints, logvar_in, valid):
        """Root-relative pose, root trajectory, input log-variance and validity per frame."""
        B, T, Jn, _ = joints.shape
        root = joints[:, :, 0]
        rel = (joints - root[:, :, None]).reshape(B, T, Jn * 3)
        centre = root[:, T // 2 : T // 2 + 1]
        traj = (root - centre) / 5.0
        lv = (logvar_in - math.log(0.01 ** 2)) / 4.0
        x = torch.cat([rel, traj, lv, valid[..., None].to(joints.dtype)], dim=-1)
        return x * valid[..., None].to(joints.dtype)

    def forward(self, joints, logvar_in, valid):
        """joints (B, T, J, 3) metric with zeros at invalid frames, logvar_in (B, T, J), valid (B, T) bool."""
        B, T, Jn, _ = joints.shape
        h = self.embed(self.features(joints, logvar_in, valid))
        z = self.event_head(h.transpose(1, 2)).transpose(1, 2)
        z = z.masked_fill(~valid[..., None], -1e4)
        tau, tau_sd = soft_argmax(z, self.temperature)
        crossing = event_crossing(T, tau, self.beta)
        rel = self.relpos(T, tau)
        g = h
        for blk in self.blocks:
            g = blk(g, crossing, rel, valid)
        loc = self.local(h.transpose(1, 2)).transpose(1, 2)
        G = torch.sigmoid(self.gate(torch.cat([loc, g], dim=-1)))
        Z = self.norm(loc * G + g * (1.0 - G))
        delta = self.out_pos(Z).view(B, T, Jn, 3) * 0.05
        logvar = self.out_logvar(Z).view(B, T, Jn, 3).clamp(-16.0, 2.0)
        return {"joints": joints + delta, "logvar": logvar, "tau": tau, "tau_sd": tau_sd, "event_logits": z}


def network_inputs(Yw: np.ndarray, Cw: np.ndarray):
    """Arrays (T, J, 3), (T, J, 3, 3) -> tensors joints, logvar, valid with a batch axis."""
    valid = np.isfinite(Yw).all(axis=(1, 2))
    var = np.trace(np.nan_to_num(Cw, nan=1.0), axis1=-2, axis2=-1) / 3.0
    logvar = np.log(np.clip(var, 1e-10, 1.0))
    joints = torch.as_tensor(np.nan_to_num(Yw), dtype=torch.float32)[None]
    return joints, torch.as_tensor(logvar, dtype=torch.float32)[None], torch.as_tensor(valid)[None]


@torch.no_grad()
def run_network(model: PACTNet, Yw: np.ndarray, Cw: np.ndarray, device: str = "cpu") -> dict:
    model.eval()
    joints, logvar, valid = network_inputs(Yw, Cw)
    out = model(joints.to(device), logvar.to(device), valid.to(device))
    v = valid[0].numpy()
    Y = out["joints"][0].cpu().numpy().astype(np.float64)
    sd = np.exp(0.5 * out["logvar"][0].cpu().numpy().astype(np.float64))
    Y[~v] = np.nan
    return {"joints": Y, "sd": sd, "tau": out["tau"][0].cpu().numpy().astype(np.float64),
            "event_sd": out["tau_sd"][0].cpu().numpy().astype(np.float64)}


def build_model(cfg: dict) -> PACTNet:
    return PACTNet(**cfg["model"])


def load_model(path: str, cfg: dict, device: str = "cpu") -> PACTNet:
    model = build_model(cfg)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device).eval()
