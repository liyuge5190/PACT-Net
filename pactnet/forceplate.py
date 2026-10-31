"""Reference kinetics from the force platform (Section 4.1, Appendix C).

Contact events come from a 20 N threshold on the unfiltered vertical force.
Impulses are integrated from the unfiltered signal; peaks and the loading
rate are taken after a zero-lag fourth-order Butterworth low-pass filter
(a second-order design applied forwards and backwards).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

G0 = 9.81
_trapz = getattr(np, "trapezoid", None) or np.trapz


def lowpass(x: np.ndarray, fs: float, cutoff: float = 100.0, order: int = 2) -> np.ndarray:
    b, a = butter(order, cutoff / (0.5 * fs))
    return filtfilt(b, a, x, axis=0)


def contact_events(fz: np.ndarray, fs: float, threshold: float = 20.0):
    """Touchdown and toe-off times (s) of the longest supra-threshold interval, linearly interpolated."""
    above = fz > threshold
    if not above.any():
        raise ValueError("no contact above threshold")
    m = np.concatenate([[False], above, [False]])
    d = np.diff(m.astype(np.int8))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    k = int(np.argmax(ends - starts))
    s, e = starts[k], ends[k] - 1

    def cross(i0, i1):
        f0, f1 = fz[i0], fz[i1]
        return i0 + (threshold - f0) / (f1 - f0) if f1 != f0 else float(i0)

    t_on = cross(s - 1, s) if s > 0 else float(s)
    t_off = cross(e, e + 1) if e + 1 < fz.size else float(e)
    return t_on / fs, t_off / fs


def kinetic_descriptors(grf: np.ndarray, fs: float, mass: float, cutoff: float | None = 100.0) -> dict:
    """Reference kinetic descriptors from a (N, 3) force signal in newtons (x forward, z up)."""
    bw = mass * G0
    td, to = contact_events(grf[:, 2], fs)
    i0, i1 = int(np.ceil(td * fs)), int(np.floor(to * fs))
    raw = grf[i0 : i1 + 1] / bw
    filt = (lowpass(grf, fs, cutoff) if cutoff else grf)[i0 : i1 + 1] / bw
    dt = 1.0 / fs
    kpk = int(np.argmax(filt[:, 2]))
    peak = float(filt[kpk, 2])
    rise = filt[: kpk + 1, 2]
    t20 = np.interp(0.2 * peak, rise, np.arange(rise.size)) if rise.size > 1 else 0.0
    t80 = np.interp(0.8 * peak, rise, np.arange(rise.size)) if rise.size > 1 else 0.0
    loading_rate = 0.6 * peak / max((t80 - t20) * dt, dt)
    return {
        "td": td,
        "to": to,
        "contact_time": (to - td) * 1000.0,
        "peak_vertical_force": peak,
        "time_to_peak_vertical_force": ((i0 + kpk) / fs - td) * 1000.0,
        "peak_braking_force": float(np.clip(-filt[:, 0], 0.0, None).max()),
        "vertical_impulse": float(_trapz(raw[:, 2], dx=dt)),
        "braking_impulse": float(_trapz(np.clip(-raw[:, 0], 0.0, None), dx=dt)),
        "propulsive_impulse": float(_trapz(np.clip(raw[:, 0], 0.0, None), dx=dt)),
        "loading_rate": float(loading_rate),
    }


def resample_waveform(grf: np.ndarray, fs: float, times: np.ndarray, mass: float, cutoff: float | None = None):
    """Force (in BW) sampled at ``times`` (s from the start of the signal)."""
    sig = lowpass(grf, fs, cutoff) if cutoff else grf
    t = np.arange(sig.shape[0]) / fs
    return np.stack([np.interp(times, t, sig[:, k]) for k in range(3)], -1) / (mass * G0)
