import numpy as np

from pactnet.camera import rodrigues
from pactnet.gravity import fit_shared_acceleration


def test_shared_acceleration_recovers_scale_and_vertical():
    rng = np.random.default_rng(1)
    s_true = 0.37
    R = rodrigues(np.array([0.4, -0.7, 0.2]))
    g_raw = R @ np.array([0.0, 0.0, -9.81]) / s_true
    blocks = []
    for n in (22, 22, 29):
        t = (np.arange(n) - (n - 1) / 2) / 200.0
        c = rng.normal(size=3) + np.outer(t, rng.normal(size=3)) + 0.5 * np.outer(t ** 2, g_raw)
        blocks.append((t, c + rng.normal(0, 1e-5, c.shape), None))
    a, cov = fit_shared_acceleration(blocks)
    assert abs(9.81 / np.linalg.norm(a) / s_true - 1.0) < 1e-3
    assert np.degrees(np.arccos(np.dot(a / np.linalg.norm(a), g_raw / np.linalg.norm(g_raw)))) < 0.1


def test_gravity_anchoring_on_synthetic_trial(cache):
    assert abs(cache["scale"] / cache["truth_scale"] - 1.0) < 0.03
    assert len(eval(cache["flight_windows"])) == 3
    assert abs(cache["td_init"] - cache["ref_td"]) < 2.0
