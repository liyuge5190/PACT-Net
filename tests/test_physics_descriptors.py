import numpy as np

from pactnet.descriptors import DESCRIPTORS, TakeoffDescriptors
from pactnet.forceplate import kinetic_descriptors
from pactnet.physics import ContactRefiner
from pactnet.uncertainty import descriptor_intervals


def _stance_side(Y, td, to):
    k = slice(int(np.ceil(td)), int(np.floor(to)))
    return "l" if np.nanmedian(Y[k, 10, 2]) < np.nanmedian(Y[k, 5, 2]) else "r"


def _reference(trial):
    Y = np.asarray(trial["ref_joints"], dtype=np.float64)[::2]
    kin = kinetic_descriptors(np.asarray(trial["grf"], dtype=np.float64), trial["force_fps"], trial["mass"])
    return Y, kin["td"] * trial["fps"], kin["to"] * trial["fps"], kin


def test_vertical_impulse_from_kinematics_matches_platform(trial):
    Y, td, to, kin = _reference(trial)
    D = TakeoffDescriptors(Y.shape[0], td, to, trial["fps"], trial["sex"], _stance_side(Y, td, to))
    values = dict(zip(DESCRIPTORS, D(Y)))
    assert abs(values["vertical_impulse"] / kin["vertical_impulse"] - 1.0) < 0.03
    assert abs(values["contact_time"] - kin["contact_time"]) < 1e-6


def test_refinement_reduces_error_and_enforces_flight(trial):
    rng = np.random.default_rng(3)
    Y, td, to, _ = _reference(trial)
    noisy = Y + rng.normal(0.0, 0.004, Y.shape)
    side = _stance_side(Y, td, to)
    out = ContactRefiner(sex=trial["sex"]).refine(noisy, np.full(Y.shape, 0.004), td, to, trial["fps"], side)
    s, e = out["window"]
    before = np.linalg.norm(noisy[s:e] - Y[s:e], axis=-1).mean()
    after = np.linalg.norm(out["joints"][s:e] - Y[s:e], axis=-1).mean()
    assert after < before
    flight = np.r_[s + 1 : int(td) - 2, int(to) + 3 : e - 1]
    assert np.nanmax(np.abs(out["force"][flight])) < 0.5


def test_unscented_interval_matches_monte_carlo(trial):
    rng = np.random.default_rng(5)
    Y, td, to, _ = _reference(trial)
    Y = Y[60:181]
    td, to = td - 60, to - 60
    side = _stance_side(Y, td, to)
    sd = np.full(Y.shape, 0.003)
    point, _, _, half = descriptor_intervals(Y, sd, td, to, trial["fps"], trial["sex"], side,
                                             event_sd=(1e-9, 1e-9))
    D = TakeoffDescriptors(Y.shape[0], td, to, trial["fps"], trial["sex"], side)
    mc = np.array([D(Y + rng.normal(0.0, 0.003, Y.shape)) for _ in range(300)])
    k = DESCRIPTORS.index("to_horizontal_velocity")
    assert abs(half[k] / (1.96 * mc[:, k].std()) - 1.0) < 0.2
