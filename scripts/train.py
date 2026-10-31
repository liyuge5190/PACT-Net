"""Train the contact-gated temporal network with leave-one-subject-out folds."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pactnet.config import ABLATIONS, load_config
from pactnet.data import list_trials, loso_folds
from pactnet.dataset import TakeoffDataset
from pactnet.losses import PACTLoss
from pactnet.model import build_model


def athlete_of(path: Path) -> str:
    return path.stem.split("_")[0]


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    """Checkpoint criterion of Appendix E: validation MPJPE plus the derivative loss."""
    model.eval()
    scores = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["joints"], batch["logvar"], batch["valid"])
        _, parts = loss_fn(out, batch)
        scores.append(parts["l3d"] * 10.0 + parts["deriv"])
    return float(np.mean(scores)) if scores else float("inf")


def train_fold(cfg, train_paths, val_paths, out_dir: Path, seed: int, device: str, epochs: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    tc = cfg["train"]
    model = build_model(cfg).to(device)
    loss_fn = PACTLoss(cfg, fps=float(cfg["data"]["fps"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    train_dl = DataLoader(TakeoffDataset(train_paths, cfg, train=True, seed=seed),
                          batch_size=tc["batch_size"], shuffle=True)
    val_dl = DataLoader(TakeoffDataset(val_paths, cfg, train=False), batch_size=tc["batch_size"])
    best, log = float("inf"), []
    for epoch in range(epochs):
        model.train()
        t0, running = time.time(), []
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["joints"], batch["logvar"], batch["valid"])
            loss, parts = loss_fn(out, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running.append(parts)
        sched.step()
        entry = {"epoch": epoch + 1, "time": time.time() - t0,
                 **{k: float(np.mean([p[k] for p in running])) for k in running[0]}}
        if (epoch + 1) % tc["eval_every"] == 0 or epoch + 1 == epochs:
            score = validate(model, val_dl, loss_fn, device)
            entry["val_score"] = score
            if score < best:
                best = score
                torch.save({"model": model.state_dict(), "epoch": epoch + 1, "config": cfg}, out_dir / "best.pt")
        log.append(entry)
        print(json.dumps(entry))
    with open(out_dir / "log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="runs/full")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ablation", default="full", choices=sorted(ABLATIONS))
    ap.add_argument("--set", nargs="*", default=[], help="overrides key.sub=value")
    ap.add_argument("--fold", default="all", help="'all' or the test athlete id")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config, ABLATIONS[args.ablation] + args.set)
    paths = list_trials(args.cache)
    by_athlete = {}
    for p in paths:
        by_athlete.setdefault(athlete_of(p), []).append(p)
    folds = loso_folds(by_athlete.keys())
    if args.fold != "all":
        folds = [f for f in folds if f["test"] == args.fold]
    seeds = args.seeds if args.seeds is not None else cfg["train"]["seeds"]
    epochs = args.epochs if args.epochs is not None else cfg["train"]["epochs"]
    for fold in folds:
        train_paths = [p for a in fold["train"] for p in by_athlete[a]]
        val_paths = by_athlete[fold["val"]]
        for seed in seeds:
            out_dir = Path(args.out) / f"fold_{fold['test']}" / f"seed_{seed}"
            best = train_fold(cfg, train_paths, val_paths, out_dir, seed, args.device, epochs)
            print(f"fold {fold['test']} seed {seed}: best validation score {best:.4f}")


if __name__ == "__main__":
    main()
