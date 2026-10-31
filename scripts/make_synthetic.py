"""Generate a synthetic corpus in the raw data layout of docs/data_format.md."""

import argparse
from pathlib import Path

import numpy as np

from pactnet.data import save_trial, trial_id
from pactnet.synthetic import sample_athlete, simulate_trial


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--athletes", type=int, default=6)
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--trials", type=int, default=4, help="trials per session")
    ap.add_argument("--outdoor-sessions", type=int, nargs="*", default=[2])
    ap.add_argument("--pixel-noise", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    athletes = [sample_athlete(rng, i) for i in range(args.athletes)]
    n = 0
    for ath in athletes:
        for s in range(1, args.sessions + 1):
            venue = "outdoor" if s in args.outdoor_sessions else "indoor"
            for k in range(1, args.trials + 1):
                trial = simulate_trial(rng, ath, venue=venue, session=s, trial_index=k,
                                       pixel_noise=args.pixel_noise)
                save_trial(out / f"{trial_id(trial)}.npz", trial)
                n += 1
    print(f"wrote {n} trials to {out}")


if __name__ == "__main__":
    main()
