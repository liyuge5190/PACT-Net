"""Stages 1-3: heteroscedastic fusion, robust triangulation and gravity anchoring."""

import argparse
import json
from pathlib import Path

from pactnet.config import ABLATIONS, load_config
from pactnet.data import list_trials, load_trial, preprocess_trial, save_trial


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ablation", default="full", choices=sorted(ABLATIONS))
    ap.add_argument("--set", nargs="*", default=[], help="overrides key.sub=value")
    args = ap.parse_args()

    cfg = load_config(args.config, ABLATIONS[args.ablation] + args.set)
    out = Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in list_trials(args.raw):
        trial = load_trial(path)
        try:
            cache = preprocess_trial(trial, cfg)
        except ValueError as err:
            print(f"{path.name}: skipped ({err})")
            continue
        save_trial(out / f"{cache['id']}.npz", cache)
        summary.append({"id": cache["id"], "scale": cache["scale"], "scale_se": cache["scale_se"],
                        "scale_window_mad": cache["scale_mad"], "td_init": cache["td_init"],
                        "to_init": cache["to_init"], "stance_side": cache["stance_side"]})
        print(f"{cache['id']}: scale {cache['scale']:.4f} (se {cache['scale_se']:.4f}), "
              f"touchdown {cache['td_init']:.1f}, toe-off {cache['to_init']:.1f}")
    with open(out / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
