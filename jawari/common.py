"""Helpers shared by the 1-D scripts (validate1d.py, render1d.py, diag_bands.py) and the cross-checks.

Run every script from inside this directory, e.g.  cd jawari && python validate1d.py
Outputs go to jawari/out/1d (WAV, PNG, result text); override with TANPURA_OUT=/some/dir.
"""
import os
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))

from metrics import analyze, report, spectrogram_png  # noqa: E402,F401

OUT_ROOT = Path(os.environ.get("TANPURA_OUT", PKG / "out"))
HERE = str(OUT_ROOT / "1d")
os.makedirs(HERE, exist_ok=True)

# Mono mix weight of the horizontal bridge force (2-D model).  The bridge sits on a thin soundboard whose
# admittance normal to its plane is much larger than in-plane; -10 dB (0.3) is an assumed, not measured, ratio.
GZ_MONO = 0.3


def to_audio(y, fs, fs_out=44100):
    """Sub-audio DC block (2nd-order Butterworth high-pass at 20 Hz, causal) and, if needed,
    band-limited resampling to fs_out. Nothing else is done to the signal."""
    sos = ss.butter(2, 20.0, "highpass", fs=fs, output="sos")
    y = ss.sosfilt(sos, y)
    if int(round(fs)) != fs_out:
        g = np.gcd(int(round(fs)), fs_out)
        y = ss.resample_poly(y, fs_out // g, int(round(fs)) // g)
    return y


def metric_rows(m, times=(0.1, 0.25, 0.5, 1.0, 2.0)):
    out = {}
    for tt in times:
        i = int(np.argmin(np.abs(m["t"] - tt)))
        out[tt] = dict(level=m["frame_db"][i], centroid=m["centroid_hz"][i], hf=m["hf_frac"][i],
                       buzz=m["buzz_db"][i], peak=m["peak_hz"][i])
    return out


def mean_metric(m, key, t0, t1):
    sel = (m["t"] >= t0) & (m["t"] <= t1)
    return float(np.mean(m[key][sel]))


def smooth_track(m, key, win=0.25):
    """median-smoothed metric track (window win seconds) -> (t, values)."""
    t = m["t"]
    hop = t[1] - t[0]
    k = max(1, int(round(win / hop)))
    v = m[key]
    out = np.array([np.median(v[max(0, i - k // 2): i + k // 2 + 1]) for i in range(len(v))])
    return t, out
