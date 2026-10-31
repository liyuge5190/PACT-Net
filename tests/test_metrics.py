import numpy as np

from pactnet import metrics as M


def test_icc_perfect_agreement():
    x = np.linspace(0, 1, 20)
    assert abs(M.icc_2_1(x, x) - 1.0) < 1e-12


def test_attempts_required_matches_square_rule():
    for rho, n in ((0.76, 1), (1.20, 2), (1.31, 2), (1.53, 3), (2.80, 8), (3.00, 9)):
        assert M.attempts_required(rho, 1.0)[1] == n


def test_mdc_equals_196_rmse():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=500)
    est = ref + rng.normal(0, 0.2, 500)
    sem, mdc = M.sem_and_mdc(est, ref)
    rmse = np.sqrt(np.mean((est - ref) ** 2))
    assert abs(mdc - 1.96 * rmse) < 1e-12
