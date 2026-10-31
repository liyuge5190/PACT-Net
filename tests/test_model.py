import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pactnet.losses import PACTLoss  # noqa: E402
from pactnet.model import PACTNet, event_crossing, soft_argmax  # noqa: E402


def test_soft_argmax_subframe():
    T = 50
    t = torch.arange(T, dtype=torch.float32)
    logits = (-(t - 20.3) ** 2)[None, :, None]
    tau, sd = soft_argmax(logits, 0.5)
    assert abs(float(tau[0, 0]) - 20.3) < 0.1
    assert float(sd[0, 0]) < 1.0


def test_event_crossing_is_zero_within_phase():
    tau = torch.tensor([[20.0, 40.0]])
    C = event_crossing(60, tau, 1.0)
    assert float(C[0, 5, 10]) < 1e-3
    assert float(C[0, 5, 30]) > 0.99


def test_forward_and_backward(cfg):
    small = dict(cfg)
    small["model"] = dict(cfg["model"], hidden=32, blocks=2, heads=4, gated_heads=2)
    model = PACTNet(**small["model"])
    B, T, J = 2, 121, 20
    joints = torch.randn(B, T, J, 3) * 0.3
    valid = torch.ones(B, T, dtype=torch.bool)
    valid[:, 100:] = False
    out = model(joints, torch.full((B, T, J), -9.0), valid)
    assert out["joints"].shape == (B, T, J, 3)
    assert out["tau"].shape == (B, 2)
    batch = {"ref": joints + 0.01, "ref_valid": valid.clone(), "valid": valid,
             "tau_ref": torch.tensor([[60.0, 84.0], [61.0, 85.0]])}
    loss, parts = PACTLoss(small)(out, batch)
    loss.backward()
    assert np.isfinite(float(loss))
    assert all(np.isfinite(v) for v in parts.values())
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
