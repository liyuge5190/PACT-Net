# PACT-Net

Reference implementation of the method described in

> Q. Li, Y. Li. *Markerless quantification of takeoff mechanics and external limb loading in the long jump: kinematic validation on an outdoor runway and kinetic validation against a force platform.* Manuscript under review.

The pipeline recovers whole-body kinematics, centre-of-mass mechanics and ground reaction loading of the long jump takeoff from a synchronized multi-camera array, and reports a 95 percent predictive interval for every descriptor.

## Pipeline

| Stage | Manuscript | Module |
|---|---|---|
| Heteroscedastic 2D representation | Sec. 3.3.1, Eq. (3)–(7) | `pactnet/heatmap.py` |
| Robust information-form triangulation | Sec. 3.3.2, Eq. (8)–(12) | `pactnet/triangulation.py` |
| Camera calibration (intrinsics, keypoint bundle adjustment) | Appendix A | `pactnet/calibration.py` |
| Segmental centre of mass | Eq. (13) | `pactnet/skeleton.py` |
| Gravity-anchored metric self-calibration | Sec. 3.4, Eq. (14)–(16) | `pactnet/gravity.py` |
| Differentiable events, phase memberships | Sec. 3.5.1, Eq. (17)–(18) | `pactnet/model.py`, `pactnet/events.py` |
| Contact-gated attention, phase-normalized encoding, gated fusion | Sec. 3.5.2–3.5.3, Eq. (19)–(22) | `pactnet/model.py` |
| Phase-adaptive derivative estimation | Eq. (23) | `pactnet/derivatives.py` |
| Joint centre correction, bone-length projection | Sec. 3.6, Eq. (24)–(26) | `pactnet/anatomy.py` |
| Contact-phase physics refinement | Sec. 3.7, Eq. (27)–(30) | `pactnet/physics.py` |
| Joint angles, angular velocity | Eq. (31)–(32) | `pactnet/angles.py` |
| Takeoff descriptors and loading | Eq. (33) | `pactnet/descriptors.py` |
| Unscented uncertainty propagation | Eq. (34) | `pactnet/uncertainty.py` |
| Training objective | Sec. 3.9, Eq. (35) | `pactnet/losses.py` |
| Evaluation metrics, reliability | Table 2, Sec. 4.10–4.11 | `pactnet/metrics.py` |
| Force-platform reference | Sec. 4.1, Appendix C | `pactnet/forceplate.py` |

`configs/default.yaml` holds the settings of Table 3. `pactnet/config.py` defines the leave-one-out ablations of Figure 6.

## Installation

```bash
pip install -e .            # inference pipeline (NumPy, SciPy, PyYAML)
pip install -e ".[train]"   # adds PyTorch for the temporal network
pip install -e ".[calibration,test]"
```

Python 3.9 or later is required.

## Data

The long jump corpus is not included in this repository. `docs/data_format.md` specifies the per-trial layout expected by the loaders: per-view 2D keypoint moments, camera parameters, optoelectronic reference joint centres and the force-platform signal. The 2D detector (ViTPose-B in the manuscript) is run beforehand; `pactnet.heatmap.heatmap_moments` converts its heatmaps into the required moments.

`pactnet/synthetic.py` generates trials in the same layout. Synthetic trials have a centre of mass in exact free fall during every flight phase, ground reaction forces consistent with the centre-of-mass acceleration during each support, surface keypoints displaced from functional joint centres, anisotropic multi-view detection noise with occlusion and outliers, and extrinsics expressed in an arbitrary similarity frame. They serve to exercise the code end to end; they are not a substitute for the measured corpus.

## Usage

```bash
# 1. synthetic corpus (or place your own trials in data/raw)
python scripts/make_synthetic.py --out data/raw --athletes 6 --sessions 2 --trials 4

# 2. fusion, triangulation and gravity anchoring
python scripts/preprocess.py --raw data/raw --cache data/cache

# 3. temporal network, leave-one-subject-out
python scripts/train.py --cache data/cache --out runs/full

# 4. joint centre correction, physics refinement, descriptors and intervals
python scripts/infer.py --cache data/cache --checkpoints runs/full --out results/full

# 5. agreement, calibration and reliability statistics
python scripts/evaluate.py --cache data/cache --results results/full
```

Without `--checkpoints`, `infer.py` bypasses the temporal network: the triangulated trajectory and its posterior covariance are passed on directly and the contact events are estimated from the foot and centre-of-mass kinematics. This mode needs no PyTorch installation.

Ablations are selected with `--ablation`, for example

```bash
python scripts/preprocess.py --ablation no_gravity_anchoring --cache data/cache_no_gravity
python scripts/train.py --ablation no_discontinuity_attention --out runs/no_gating
python scripts/infer.py --ablation no_physics_refinement --checkpoints runs/full --out results/no_physics
```

Individual settings are overridden with `--set key.sub=value`, e.g. `--set derivatives.h_contact=3`.

`evaluate.py` writes `per_trial.csv` and `summary.json` with coordinate, derivative and event accuracy by venue, descriptor agreement (MAE, RMSE, Pearson r, ICC(2,1), Bland–Altman), interval calibration (coverage, ECE), within-athlete statistics (within-athlete r, clustered bootstrap intervals, SEM, MDC95) and the ratio of interval half-width to between-trial biological variability.

## Tests

```bash
pytest
```

## Implementation notes

- The segment model uses the 14 segments of de Leva (1996) with the hand placed on the forearm axis beyond the wrist, since no hand keypoint is detected.
- The flight windows are combined by a joint least-squares fit in which every window has its own position and velocity and all windows share one acceleration vector; this is the inverse-variance combination of the per-window estimates. Per-window scale estimates are retained for the internal consistency check.
- When reference kinematics are available, the metric reconstruction is registered to the runway frame by a rotation about the vertical and a horizontal translation estimated on the approach frames. Scale and vertical are never taken from the reference.
- In the physics refinement the centroidal angular momentum is computed from segment centres of mass, the sole is represented by the heel–toe segment with a fixed half-width, and the damping of Eq. (30) is scaled by the diagonal of the Gauss–Newton matrix.
- The propagated covariance is diagonal per coordinate and augmented with the scale and event uncertainties; the refined trajectory is propagated as the mean, without linearizing the refinement.
- The training loss uses the male de Leva parameters for the centre-of-mass term.

## License

MIT, see `LICENSE`.
