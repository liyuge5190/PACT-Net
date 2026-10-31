import numpy as np

from pactnet.descriptors import DESCRIPTORS
from pactnet.pipeline import Pipeline, reference_descriptors


def test_pipeline_without_network(cache, cfg):
    res = Pipeline(cfg).run(cache, intervals=False)
    est = dict(zip(DESCRIPTORS, res["descriptors"]))
    ref = dict(zip(DESCRIPTORS, reference_descriptors(cache, res["start"], res["joints"].shape[0], cfg)))
    assert all(np.isfinite(v) for v in est.values() if v is not None)
    assert abs(est["to_horizontal_velocity"] - ref["to_horizontal_velocity"]) < 0.3
    assert abs(est["takeoff_angle"] - ref["takeoff_angle"]) < 3.0
    assert abs(est["vertical_impulse"] / ref["vertical_impulse"] - 1.0) < 0.1
