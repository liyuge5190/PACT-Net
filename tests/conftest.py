import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pactnet.config import load_config  # noqa: E402
from pactnet import synthetic  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def trial():
    rng = np.random.default_rng(2024)
    athlete = synthetic.sample_athlete(rng, 0)
    return synthetic.simulate_trial(rng, athlete)


@pytest.fixture(scope="session")
def cache(trial, cfg):
    from pactnet.data import preprocess_trial

    return preprocess_trial(trial, cfg)
