"""Training dataset for the temporal network (Appendix E).

Each item is a window of ``network_window`` frames centred on the estimated
touchdown of one cached trial. Augmentation comprises horizontal mirroring,
random dropout of one camera view, isotropic jitter of the two-dimensional
detections and temporal crop jitter; views and jitter are applied before
triangulation with the metric cameras stored in the cache.
"""

from __future__ import annotations

import numpy as np
import torch

from .camera import cameras_from_json
from .data import load_trial
from .heatmap import information_matrix, isotropic_information
from .pipeline import crop
from .skeleton import FLIP_PERM, mirror
from .triangulation import triangulate_sequence


class TakeoffDataset(torch.utils.data.Dataset):
    def __init__(self, cache_paths, cfg: dict, train: bool = True, seed: int = 0):
        self.items = [load_trial(p) for p in cache_paths]
        self.cfg = cfg
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.length = int(cfg["data"]["network_window"])
        aug = cfg["train"]["augment"]
        self.mirror_p = 0.5 if (train and aug["mirror"]) else 0.0
        self.view_dropout = aug["view_dropout"] if train else 0.0
        self.jitter = aug["keypoint_jitter_px"] if train else 0.0
        self.crop_jitter = int(aug["crop_jitter"]) if train else 0

    def __len__(self):
        return len(self.items)

    def _retriangulate(self, c, start, stop, drop_view):
        T = c["kp2d_mu"].shape[1]
        s, e = max(start, 0), min(stop, T)
        mu = np.asarray(c["kp2d_mu"][:, s:e], dtype=np.float64)
        if self.jitter > 0:
            mu = mu + self.rng.normal(0.0, self.jitter, mu.shape)
        cov = np.asarray(c["kp2d_cov"][:, s:e], dtype=np.float64)
        vis = np.asarray(c["kp2d_vis"][:, s:e], dtype=np.float64)
        f = self.cfg["fusion"]
        info = information_matrix(cov, vis, f["information_eps"]) if f["heteroscedastic"] else \
            isotropic_information(cov, vis, f["information_eps"])
        valid = np.isfinite(mu).all(-1)
        if drop_view is not None:
            valid[drop_view] = False
        Y, C = triangulate_sequence(cameras_from_json(c["cameras_metric"]), mu, info, valid,
                                    robust_scale=f["robust_scale"], iterations=f["iterations"])
        full_Y = np.full((T,) + Y.shape[1:], np.nan)
        full_C = np.full((T,) + C.shape[1:], np.nan)
        full_Y[s:e], full_C[s:e] = Y, C
        return crop(full_Y, start, stop), crop(full_C, start, stop)

    def __getitem__(self, i):
        c = self.items[i]
        shift = int(self.rng.integers(-self.crop_jitter, self.crop_jitter + 1)) if self.crop_jitter else 0
        start = int(round(float(c["td_init"]))) - self.length // 2 + shift
        stop = start + self.length
        drop = None
        if self.view_dropout > 0 and self.rng.random() < self.view_dropout:
            drop = int(self.rng.integers(0, c["kp2d_mu"].shape[0]))
        if drop is not None or self.jitter > 0:
            Y, C = self._retriangulate(c, start, stop, drop)
        else:
            Y = crop(np.asarray(c["joints"], dtype=np.float64), start, stop)
            C = crop(np.asarray(c["cov"], dtype=np.float64), start, stop)
        R = crop(np.asarray(c["ref_joints"], dtype=np.float64), start, stop)
        var = np.trace(np.nan_to_num(C, nan=1.0), axis1=-2, axis2=-1) / 3.0
        logvar = np.log(np.clip(var, 1e-10, 1.0))
        if self.rng.random() < self.mirror_p:
            Y, R = mirror(Y), mirror(R)
            logvar = logvar[:, FLIP_PERM]
        valid = np.isfinite(Y).all(axis=(1, 2))
        ref_valid = np.isfinite(R).all(axis=(1, 2))
        tau = np.array([float(c["ref_td"]) - start, float(c["ref_to"]) - start])
        return {
            "joints": torch.as_tensor(np.nan_to_num(Y), dtype=torch.float32),
            "logvar": torch.as_tensor(logvar, dtype=torch.float32),
            "valid": torch.as_tensor(valid),
            "ref": torch.as_tensor(np.nan_to_num(R), dtype=torch.float32),
            "ref_valid": torch.as_tensor(ref_valid),
            "tau_ref": torch.as_tensor(tau, dtype=torch.float32),
        }
