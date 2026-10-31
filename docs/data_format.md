# Data format

Each trial is stored as one compressed NumPy archive (`.npz`) in the raw
data directory. Times are expressed relative to the first markerless frame;
all devices are assumed to share one hardware clock (Appendix A).

## Required keys

| Key | Shape / type | Description |
|---|---|---|
| `kp2d_mu` | (V, T, J, 2) float | 2D keypoint locations in pixels (first moment of the heatmap, Eq. 4); NaN when not detected |
| `kp2d_cov` | (V, T, J, 2, 2) float | Anisotropic 2D covariance in px² (second central moment, Eq. 5) |
| `kp2d_vis` | (V, T, J) float | Visibility weight, one minus normalized heatmap entropy (Eq. 6) |
| `fps` | scalar | Markerless frame rate (200) |
| `cameras` | JSON string | List of V cameras: `K` (3×3), `R` (3×3), `t` (3), `dist` (k1, k2, p1, p2, k3), `name`. Extrinsics may be defined up to a similarity transform |
| `mass` | scalar | Body mass, kg |
| `sex` | string | `M` or `F` (selects the de Leva segment parameters) |
| `athlete` | string | Athlete identifier, e.g. `A01` |
| `session` | int | Session index |
| `trial` | int | Trial index within the session |

J = 20 joints in the order of `pactnet.skeleton.JOINTS`. When the 2D
detector outputs heatmaps, `pactnet.heatmap.heatmap_moments` converts them
to `kp2d_mu`, `kp2d_cov` and `kp2d_vis`.

## Reference keys (required for training and evaluation)

| Key | Shape / type | Description |
|---|---|---|
| `ref_joints` | (T_ref, J, 3) float | Functional joint centres from the optoelectronic system, metres, runway frame (x along the runway, z vertical) |
| `ref_fps` | scalar | Reference frame rate (400) |
| `grf` | (T_f, 3) float | Ground reaction force in newtons, runway axes; empty array when no platform is present |
| `force_fps` | scalar | Platform sampling rate (2000) |
| `has_force` | bool | Whether `grf` is available |
| `ref_td`, `ref_to` | scalar | Touchdown and toe-off in seconds; used when no force signal is present |

## Optional keys

`venue` (`indoor` or `outdoor`), `level`, `stature` (m; needed only for the
ablation without gravity anchoring).

## Preprocessed cache

`scripts/preprocess.py` writes one archive per trial to the cache directory
with the metric reconstruction (`joints`, `cov`), the estimated contact
(`td_init`, `to_init`, `stance_side`), the gravity-anchoring results
(`scale`, `scale_se`, `scale_windows`, `flight_windows`), the reference
resampled to the markerless frame rate and the metric cameras used for
augmentation during training.
