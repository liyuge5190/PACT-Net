"""Configuration loading with dotted overrides."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Load the default configuration, merge ``path`` on top and apply ``key.sub=value`` overrides."""
    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path, "r", encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


ABLATIONS = {
    "full": [],
    "no_heteroscedastic_fusion": ["fusion.heteroscedastic=false"],
    "no_gravity_anchoring": ["calibration.gravity_anchoring=false"],
    "no_discontinuity_attention": ["model.discontinuity_attention=false"],
    "no_phase_normalized_encoding": ["model.phase_normalized_encoding=false"],
    "no_phase_adaptive_derivatives": ["derivatives.phase_adaptive=false"],
    "no_joint_centre_correction": ["anatomy.joint_centre_correction=false"],
    "no_physics_refinement": ["physics.enabled=false"],
    "no_uncertainty_weighting": ["loss.lambda_u=0.0"],
}
