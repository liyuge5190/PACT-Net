"""Agreement, calibration and reliability statistics against the reference system.

Every statistic uses the trial as the unit of analysis. Kinetic descriptors
are evaluated on trials with a force platform only.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pactnet import metrics as M
from pactnet.config import load_config
from pactnet.data import list_trials, load_trial
from pactnet.descriptors import DESCRIPTORS, UNITS
from pactnet.forceplate import resample_waveform
from pactnet.pipeline import crop, reference_descriptors
from pactnet.skeleton import BONE_INDEX

KINETIC = {"peak_vertical_force", "time_to_peak_vertical_force", "peak_braking_force", "vertical_impulse",
           "braking_impulse", "propulsive_impulse", "leg_stiffness"}
TABLE13 = ("approach_velocity", "to_horizontal_velocity", "takeoff_angle", "vertical_impulse",
           "contact_time", "leg_stiffness", "velocity_conversion_ratio")
TABLE12 = ("to_horizontal_velocity", "takeoff_angle", "contact_time", "vertical_impulse",
           "peak_vertical_force", "velocity_conversion_ratio")


def trial_row(res, cache, cfg):
    fps = float(cache["fps"])
    est = np.asarray(res["joints"], dtype=np.float64)
    T = est.shape[0]
    start = int(res["start"])
    ref = crop(np.asarray(cache["ref_joints"], dtype=np.float64), start, start + T)
    row = {"id": str(cache["id"]), "athlete": str(cache["athlete"]), "session": int(cache["session"]),
           "venue": str(cache["venue"]), "has_force": bool(cache["has_force"])}
    row["mpjpe_mm"] = 1000.0 * M.mpjpe(est, ref)
    row["p_mpjpe_mm"] = 1000.0 * M.p_mpjpe(est, ref)
    row["lower_limb_mpjpe_mm"] = 1000.0 * M.lower_limb_mpjpe(est, ref)
    row["bone_length_sd_mm"] = 1000.0 * M.bone_length_sd(est[np.isfinite(est).all(axis=(1, 2))])

    td_ref, to_ref = float(cache["ref_td"]) - start, float(cache["ref_to"]) - start
    k = np.arange(int(np.ceil(td_ref)), int(np.floor(to_ref)) + 1)
    vp, vr = np.gradient(est, 1.0 / fps, axis=0), np.gradient(ref, 1.0 / fps, axis=0)
    ap, ar = np.gradient(vp, 1.0 / fps, axis=0), np.gradient(vr, 1.0 / fps, axis=0)
    row["mpjve_ms"] = float(np.nanmean(np.linalg.norm(vp[k] - vr[k], axis=-1)))
    row["acc_rmse_ms2"] = float(np.sqrt(np.nanmean(np.sum((ap[k] - ar[k]) ** 2, axis=-1))))
    row["td_error_ms"] = abs(float(res["tau_td"]) - td_ref) / fps * 1000.0
    row["to_error_ms"] = abs(float(res["tau_to"]) - to_ref) / fps * 1000.0

    d_est = np.asarray(res["descriptors"], dtype=np.float64)
    half = np.asarray(res["half_width"], dtype=np.float64)
    d_ref = reference_descriptors(cache, start, T, cfg)
    i_w, i_t = DESCRIPTORS.index("peak_knee_extension_velocity"), DESCRIPTORS.index("peak_knee_velocity_timing")
    row["peak_omega_error_pct"] = M.relative_error(d_est[i_w], d_ref[i_w])
    t_est = float(res["tau_td"]) + d_est[i_t] * fps / 1000.0
    t_ref = td_ref + d_ref[i_t] * fps / 1000.0
    row["peak_timing_offset_ms"] = abs(t_est - t_ref) / fps * 1000.0

    if np.isfinite(cache.get("truth_scale", np.nan)):
        row["scale_error_pct"] = 100.0 * abs(float(cache["scale"]) / float(cache["truth_scale"]) - 1.0)
    else:
        raw = np.asarray(cache["joints"], dtype=np.float64)
        refj = np.asarray(cache["ref_joints"], dtype=np.float64)
        le = np.linalg.norm(raw[:, BONE_INDEX[:, 1]] - raw[:, BONE_INDEX[:, 0]], axis=-1)
        lr = np.linalg.norm(refj[:, BONE_INDEX[:, 1]] - refj[:, BONE_INDEX[:, 0]], axis=-1)
        row["scale_error_pct"] = 100.0 * abs(np.nanmedian(np.nanmedian(le, 0) / np.nanmedian(lr, 0)) - 1.0)

    if cache["has_force"]:
        frames = np.asarray(res["contact_frames"])
        times = (start + frames) / fps
        w_ref = resample_waveform(np.asarray(cache["grf"], dtype=np.float64), float(cache["force_fps"]),
                                  times, float(cache["mass"]), cutoff=100.0)[:, 2]
        w_est = np.asarray(res["force"])[:, 2]
        row["waveform_rrmse_pct"] = float(np.sqrt(np.mean((w_est - w_ref) ** 2)) / np.ptp(w_ref) * 100.0)
    else:
        row["waveform_rrmse_pct"] = np.nan

    for name, e, h, r in zip(DESCRIPTORS, d_est, half, d_ref):
        if name in KINETIC and not cache["has_force"]:
            r = np.nan
        row[f"est:{name}"], row[f"half:{name}"], row[f"ref:{name}"] = float(e), float(h), float(r)
    return row


def column(rows, key, mask=None):
    v = np.array([r[key] for r in rows], dtype=np.float64)
    return v if mask is None else v[mask]


def summarize(rows):
    athlete = np.array([r["athlete"] for r in rows])
    session = np.array([r["session"] for r in rows])
    venue = np.array([r["venue"] for r in rows])
    out = {"n_trials": len(rows)}
    trial_keys = ("mpjpe_mm", "p_mpjpe_mm", "lower_limb_mpjpe_mm", "bone_length_sd_mm", "mpjve_ms",
                  "acc_rmse_ms2", "peak_omega_error_pct", "peak_timing_offset_ms", "td_error_ms",
                  "to_error_ms", "scale_error_pct", "waveform_rrmse_pct")
    out["by_venue"] = {}
    for label, mask in (("pooled", np.ones(len(rows), bool)), ("indoor", venue == "indoor"),
                        ("outdoor", venue == "outdoor")):
        if mask.any():
            out["by_venue"][label] = {k: float(np.nanmean(column(rows, k, mask))) for k in trial_keys}

    agree, calib, within, interval = {}, {}, {}, {}
    for name in DESCRIPTORS:
        e, r, h = column(rows, f"est:{name}"), column(rows, f"ref:{name}"), column(rows, f"half:{name}")
        m = np.isfinite(e) & np.isfinite(r)
        if m.sum() < 3:
            continue
        agree[name] = M.agreement(e[m], r[m])
        if np.isfinite(h[m]).any():
            calib[name] = {"coverage_pct": M.coverage(e[m], h[m], r[m]),
                           "ece": M.regression_ece(e[m], h[m] / 1.96, r[m]),
                           "mean_half_width": float(np.nanmean(h[m]))}
        if name in TABLE12:
            sem, mdc = M.sem_and_mdc(e[m], r[m])
            within[name] = {"pooled_r": agree[name]["r"],
                            "within_athlete_r": M.within_athlete_r(e[m], r[m], athlete[m]),
                            "per_athlete_mae": M.per_athlete_mae(e[m], r[m], athlete[m]),
                            "clustered_ci_mae": M.clustered_bootstrap_mae(e[m], r[m], athlete[m]),
                            "sem": sem, "mdc95": mdc,
                            "between_session_icc_est": M.between_session_icc(e[m], athlete[m], session[m]),
                            "between_session_icc_ref": M.between_session_icc(r[m], athlete[m], session[m])}
        if name in TABLE13 and np.isfinite(h[m]).any():
            sd_bio = M.between_trial_sd(r[m], athlete[m], session[m])
            rho, n_att = M.attempts_required(float(np.nanmean(h[m])), sd_bio)
            interval[name] = {"mean_half_width": float(np.nanmean(h[m])), "between_trial_sd": sd_bio,
                              "rho": rho, "attempts_required": n_att}
    out["descriptor_agreement"] = agree
    out["calibration"] = calib
    if calib:
        out["calibration_aggregate"] = {
            "coverage_pct": float(np.mean([c["coverage_pct"] for c in calib.values()])),
            "ece": float(np.mean([c["ece"] for c in calib.values()]))}
    out["within_athlete"] = within
    out["interval_vs_biological"] = interval
    return out


def print_tables(s):
    print("\nCoordinate, derivative and event accuracy (trial means)")
    for label, vals in s["by_venue"].items():
        print(f"  {label:8s} " + "  ".join(f"{k} {v:.2f}" for k, v in vals.items()))
    print("\nDescriptor agreement")
    print(f"  {'descriptor':32s} {'unit':6s} {'ref mean':>9s} {'est mean':>9s} {'MAE':>8s} {'RMSE':>8s} {'r':>6s} {'ICC':>6s}")
    for k, a in s["descriptor_agreement"].items():
        print(f"  {k:32s} {UNITS[k]:6s} {a['ref_mean']:9.3f} {a['est_mean']:9.3f} {a['mae']:8.3f} "
              f"{a['rmse']:8.3f} {a['r']:6.3f} {a['icc']:6.3f}")
    if s["calibration"]:
        print("\nCalibration of the 95 percent predictive interval")
        for k, c in s["calibration"].items():
            print(f"  {k:32s} coverage {c['coverage_pct']:5.1f}%  ECE {c['ece']:.3f}  half-width {c['mean_half_width']:.3f}")
    if s["interval_vs_biological"]:
        print("\nInterval half-width against between-trial variability")
        for k, c in s["interval_vs_biological"].items():
            print(f"  {k:32s} half {c['mean_half_width']:.3f}  SD {c['between_trial_sd']:.3f}  "
                  f"rho {c['rho']:.2f}  attempts {c['attempts_required']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--results", default="results/full")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="output directory (defaults to the results directory)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out or args.results)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in list_trials(args.results):
        cache_path = Path(args.cache) / path.name
        if not cache_path.exists():
            continue
        rows.append(trial_row(load_trial(path), load_trial(cache_path), cfg))
    if not rows:
        raise SystemExit("no matching results and caches")
    with open(out / "per_trial.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print_tables(summary)


if __name__ == "__main__":
    main()
