"""Shared objective metrics for 'is this jawari?' — used by every prototype and judge.

Usage:
    python metrics.py file.wav [more.wav ...] [--png out_prefix]
    or: from metrics import analyze, report

Metrics per STFT frame (Hann, nperseg=4096, hop=1024):
  centroid_hz      spectral centroid
  hf_frac          fraction of energy above 1.5 kHz
  buzz_db          10*log10(E[2-8 kHz] / E[0-1 kHz])
  peak_hz          frequency of the maximum of the smoothed (≈160 Hz moving-average)
                   log spectrum between 300 Hz and 10 kHz -> tracks the jawari 'formant'
A plain linear plucked string has centroid that is constant or falls monotonically,
hf_frac ~ 0 and buzz_db << -30. A jawari tone shows hf_frac and buzz_db that stay
high for seconds, and a peak_hz that sweeps (typically downwards) as the tone decays.
"""
from __future__ import annotations
import sys
import numpy as np
import scipy.signal as ss

TIMES = (0.05, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def analyze(y: np.ndarray, sr: int) -> dict:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    f, t, Z = ss.stft(y, sr, window="hann", nperseg=4096, noverlap=4096 - 1024, boundary=None)
    S = np.abs(Z) ** 2 + 1e-30
    tot = S.sum(0)
    cen = (f[:, None] * S).sum(0) / tot
    hf = S[f > 1500].sum(0) / tot
    lo = S[f < 1000].sum(0)
    mid = S[(f >= 2000) & (f <= 8000)].sum(0)
    buzz = 10 * np.log10(mid / lo)
    logS = 10 * np.log10(S)
    k = 15
    sm = ss.convolve2d(logS, np.ones((k, 1)) / k, mode="same", boundary="symm")
    band = (f >= 300) & (f <= 10000)
    peak_hz = f[band][np.argmax(sm[band], axis=0)]
    frame_db = 10 * np.log10(tot / tot.max())
    return {"t": t, "f": f, "S": S, "centroid_hz": cen, "hf_frac": hf, "buzz_db": buzz,
            "peak_hz": peak_hz, "frame_db": frame_db, "sr": sr, "dur": len(y) / sr}


def report(m: dict, label: str = "") -> str:
    rows = []
    for tt in TIMES:
        if tt > m["dur"] - 0.05:
            continue
        i = int(np.argmin(np.abs(m["t"] - tt)))
        rows.append(f"  t={tt:5.2f}s  level={m['frame_db'][i]:6.1f}dB  centroid={m['centroid_hz'][i]:7.0f}Hz  "
                    f"hf_frac={m['hf_frac'][i]:.3f}  buzz={m['buzz_db'][i]:6.1f}dB  peak={m['peak_hz'][i]:6.0f}Hz")
    return f"[{label}] dur={m['dur']:.2f}s sr={m['sr']}\n" + "\n".join(rows)


def spectrogram_png(m: dict, path: str, title: str = "", fmax: float = 8000.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[3, 1], sharex=True)
    sel = m["f"] <= fmax
    db = 10 * np.log10(m["S"][sel] / m["S"].max())
    ax[0].pcolormesh(m["t"], m["f"][sel], db, vmin=-90, vmax=0, shading="auto", cmap="magma")
    ax[0].plot(m["t"], m["peak_hz"], "c.", ms=2, label="smoothed spectral peak")
    ax[0].set_ylabel("Hz"); ax[0].set_title(title); ax[0].legend(loc="upper right")
    ax[1].plot(m["t"], m["buzz_db"], label="buzz dB (2-8k vs <1k)")
    ax[1].plot(m["t"], m["frame_db"], label="frame level dB")
    ax[1].set_xlabel("s"); ax[1].legend(loc="upper right"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    import soundfile as sf
    args = sys.argv[1:]
    png = None
    if "--png" in args:
        i = args.index("--png"); png = args[i + 1]; del args[i:i + 2]
    for j, path in enumerate(args):
        y, sr = sf.read(path)
        m = analyze(y, sr)
        print(report(m, path))
        if png:
            spectrogram_png(m, f"{png}_{j}.png", title=path)
