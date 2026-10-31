import numpy as np

from pactnet.derivatives import phase_adaptive_operators


def test_operators_exact_for_quadratic_within_phase():
    n, fps = 121, 200.0
    D0, D1, D2 = phase_adaptive_operators(n, 60.0, 84.0, fps)
    t = np.arange(n) / fps
    y = 1.0 + 3.0 * t - 4.905 * t ** 2
    for k in (20, 30, 100, 110):
        assert abs(D1[k] @ y - (3.0 - 9.81 * t[k])) < 1e-4
        assert abs(D2[k] @ y + 9.81) < 1e-3


def test_velocity_step_is_not_smeared_across_touchdown():
    n, fps, td = 121, 200.0, 60.0
    t = np.arange(n) - td
    y = np.where(t < 0, 9.0 * t, 8.0 * t) / fps
    _, D1, _ = phase_adaptive_operators(n, td, 84.0, fps)
    _, D1_single, _ = phase_adaptive_operators(n, td, 84.0, fps, adaptive=False)
    assert abs(D1[55] @ y - 9.0) < 0.02
    assert abs(D1[64] @ y - 8.0) < 0.05
    for k in (57, 58):
        assert abs(D1[k] @ y - 9.0) < abs(D1_single[k] @ y - 9.0)
