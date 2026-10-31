"""Run stages 4-5 and uncertainty propagation on preprocessed trials."""

import argparse
import json
from pathlib import Path

import numpy as np

from pactnet.config import ABLATIONS, load_config
from pactnet.data import list_trials, load_trial
from pactnet.descriptors import DESCRIPTORS
from pactnet.pipeline import Pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="results/full")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ablation", default="full", choices=sorted(ABLATIONS))
    ap.add_argument("--set", nargs="*", default=[], help="overrides key.sub=value")
    ap.add_argument("--checkpoints", default=None,
                    help="directory with fold_<athlete>/seed_<seed>/best.pt; omitted: no temporal network")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-intervals", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config, ABLATIONS[args.ablation] + args.set)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = {}
    for path in list_trials(args.cache):
        cache = load_trial(path)
        model = None
        if args.checkpoints:
            from pactnet.model import load_model

            athlete = str(cache["athlete"])
            ckpt = Path(args.checkpoints) / f"fold_{athlete}" / f"seed_{args.seed}" / "best.pt"
            if athlete not in models:
                models[athlete] = load_model(str(ckpt), cfg, args.device)
            model = models[athlete]
        res = Pipeline(cfg, model, args.device).run(cache, intervals=not args.no_intervals)
        np.savez_compressed(
            out / f"{cache['id']}.npz",
            joints=res["joints"], sd=res["sd"], start=res["start"], tau_td=res["tau_td"], tau_to=res["tau_to"],
            event_sd=res["event_sd"], descriptors=res["descriptors"],
            half_width=res.get("half_width", np.full(len(DESCRIPTORS), np.nan)),
            force=res["force"], contact_frames=res["contact_frames"],
            timings=json.dumps(res["timings"]),
        )
        print(f"{cache['id']}: " + ", ".join(f"{k} {v:.3f}" for k, v in
                                             zip(DESCRIPTORS[:8], res["descriptors"][:8])))


if __name__ == "__main__":
    main()
