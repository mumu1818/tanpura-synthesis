"""Time-resolved comparison of the model plucks with the real plucks, and which loss agrees with the ear.

    cd jawari
    python match_report.py                    # analysis only, no simulation (about 1-2 min on one core)
    -> out/report/match_report.html           # self-contained page (no network), open in any browser
    -> out/report/match_report.json           # every number the page shows

What it does (MATCH_REPORT.md explains the method and the results):
  1. Partial tracks.  For every signal, the level of each harmonic n*f0 over time (pitch-adaptive Hann window of
     6 periods, 10 ms hop, interpolated spectral peak within +-0.3 f0 of n*f0), up to 6 kHz.  This is an analysis of
     its own: it shares no feature code with fit2d.py / fit3.py, so it also checks them.
  2. Time evolution: level, band shares (partials 1-3 / 4-7 / 8-15 / 16-31 / 32-60), spectral centroid, jawari
     "formant" (centroid of partials >= 4), pitch, per-partial bloom time, peak and decay rate, and the attack
     (ERB bands, first 300 ms).
  3. Mismatch split, model minus recording, per partial and frame (dB):
        D_n(t) = c + s_n + e_n(t)
     c = overall offset, s_n = a fixed colour per partial (what an EQ, a microphone or a fixed body response could
     cause), e_n(t) = the part that changes with time (what only the string / jawari physics can cause).  s_n is
     split again into a smooth curve over frequency (EQ-like) and a rough part (single partials: pluck position,
     body modes).
  4. A simple auditory model (ERB bands, Schroeder spreading, Zwicker-type specific loudness) and the perceptual
     difference between model and recording over time and frequency, also for small-speaker playback (phone).
  5. Candidate losses, each for: model vs recording, and two real-vs-real yardsticks (the same pluck with other
     processing, and another real pluck), and which of them ranks the strings the way the user hears them.
  6. Recording forensics: which files are the same pluck (least-squares FIR that predicts one file from the other),
     and the EQ / level processing between them.

Inputs, read only: the recordings in $TANPURA_REFS (default /home/user/tanpura-app/assets, never copied into the
repository) and the model renders that fit3.py final wrote to out/fit3/wav/.  The report holds derived numbers only
(levels, curves), no audio.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal as ss

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
import fit2d as F2  # noqa: E402  (onset trim, f0 fit, REF_DIR, OUT_ROOT)
import fit3 as F3  # noqa: E402  (reference loader, D3)

SR = 44100
OUT = F2.OUT_ROOT / "report"
WAV3 = F2.OUT_ROOT / "fit3" / "wav"
TEMPLATE = PKG / "match_report_template.html"
STRINGS = ("kharaj", "pa", "sa")
NAMES = {"kharaj": "Kharaj", "pa": "Pa", "sa": "Sa"}
NOMINAL = {"kharaj": 69.33, "pa": 103.96, "sa": 138.54}
USER_VERDICT = {"sa": "close", "pa": "not close", "kharaj": "not close"}   # the user's listening, round-3 A/B files
FMAX_TRACK = 6000.0
HOP = 0.010
PERIODS = 6.0
T0 = 0.05                       # comparisons start 50 ms after the onset (the 6-period window smears the attack)
CLIP_DB = 30.0                  # per-cell level differences are clipped to +-30 dB (beyond that: absent vs present)
MASK_DB = -40.0                 # a cell counts if either signal's partial is within 40 dB of that frame's loudest
BANDS = ((1, 3), (4, 7), (8, 15), (16, 31), (32, 60))
DEC = 2                         # report data: every 2nd analysis frame (20 ms)

# the user's other files of the same strings (forensics: same pluck or not)
EXTRA = {
    "kharaj": ["Mumu_Tanpura/Archive/Kharaj_v2.mp3", "Mumu_Tanpura/Archive/kharaj_v3.mp3",
               "Mumu_Tanpura/Archive/mumu_kharaj_cs3_230426_20_27.mp3", "alt-jawari/mumu_EQ_Tanpura_CS3_Kharaj.mp3"],
    "pa": ["Mumu_Tanpura/Archive/Pa-mix.mp3", "Mumu_Tanpura/Archive/pa-mix2.mp3",
           "alt-jawari/mumu_Pa_Final230426_20_25pm.mp3"],
    "sa": ["Mumu_Tanpura/Archive/mumu_sa_v2.mp3", "Mumu_Tanpura/Archive/mumu_sa_v3.mp3",
           "Mumu_Tanpura/Archive/sa-mix.mp3", "alt-jawari/mumu_sa_Final_230426_20_26.mp3"],
}


def log(*a):
    print(*a, flush=True)


# ================================================================================================ loading
def load_file(rel):
    """Any recording under REF_DIR: mono, 44.1 kHz, trimmed 5 ms before its onset (fit2d.trim_onset).  Read only."""
    import soundfile as sf
    y, sr = sf.read(str(F2.REF_DIR / rel), always_2d=True)
    y = y.mean(1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        y = ss.resample_poly(y, SR // g, int(sr) // g)
    return F2.trim_onset(y), int(sr)


def load_wav(path):
    """A model render written by fit3.py final.  It already starts 5 ms before the finger's release (the model knows
    its onset), so it is NOT re-trimmed: a soft model attack (the Sa) only crosses fit2d.trim_onset's -30 dB threshold
    ~26 ms after the release, and re-trimming would shift its time axis."""
    import soundfile as sf
    y, sr = sf.read(str(path), always_2d=True)
    assert sr == SR, path
    return y.mean(1)


def signals(s):
    """The signals compared for string s.  Each starts 5 ms before its onset."""
    sig = {
        "rec_own": dict(y=F3.load_ref3(f"own_{s}"), label="Your recording", ref=f"own_{s}",
                        src=F3.REFS3[f"own_{s}"][0]),
        "rec_itan": dict(y=F3.load_ref3(f"itanpura_{s}"), label="iTanpura CS3", ref=f"itanpura_{s}",
                         src=F3.REFS3[f"itanpura_{s}"][0]),
        "rec_alt": dict(y=F3.load_ref3(f"alt_{s}"), label="iTanpura CS3Alt", ref=f"alt_{s}",
                        src=F3.REFS3[f"alt_{s}"][0]),
        "m3": dict(y=load_wav(WAV3 / f"{s}_model_chain_own.wav"), label="Model, round 3",
                   src=f"out/fit3/wav/{s}_model_chain_own.wav"),
        "m2": dict(y=load_wav(WAV3 / f"{s}_previous_fit2d_loudness_matched.wav"), label="Model, round 2",
                   src=f"out/fit3/wav/{s}_previous_fit2d_loudness_matched.wav"),
        "m3_dry": dict(y=load_wav(WAV3 / f"{s}_model_dry_bridge_force.wav"),
                       label="Model, round 3, string only (bridge force, no body)",
                       src=f"out/fit3/wav/{s}_model_dry_bridge_force.wav"),
    }
    for k, v in sig.items():
        v["dur"] = len(v["y"]) / SR
        v["t_end"] = F3.t_end_of(v["dur"]) if k.startswith("rec") else F3.T_LATE1
        v["f0"], v["B"] = F2.fit_f0_B(v["y"], NOMINAL[s])
    return sig


# ================================================================================================ partial tracks
def track_partials(y, f0, B, t_end, periods=PERIODS, hop=HOP, fmax=FMAX_TRACK):
    """Level of each harmonic over time.  Frame k is centred at k*hop after the file start; t is re the onset
    (5 ms into the file).  Returns t, A (amplitude, nmax x frames), F (peak frequency), c (nominal n f0 sqrt(1+Bn^2))."""
    nmax = int(fmax // f0)
    N = int(round(periods * SR / f0))
    N += N % 2
    nfft = 1 << int(np.ceil(np.log2(N * 8)))
    w = np.hanning(N)
    H = int(round(hop * SR))
    ypad = np.concatenate([np.zeros(N // 2), np.asarray(y, float), np.zeros(N)])
    nfr = min(int(np.floor((t_end + 0.06) * SR / H)) + 1, (len(ypad) - N) // H + 1)
    idx = np.arange(N)[None, :] + H * np.arange(nfr)[:, None]
    P = np.abs(np.fft.rfft(ypad[idx] * w[None, :], nfft, axis=1)) * (2.0 / w.sum())
    t = np.arange(nfr) * H / SR - 0.005
    df = SR / nfft
    n = np.arange(1, nmax + 1)
    c = n * f0 * np.sqrt(1 + B * n * n)
    A = np.full((nmax, nfr), 1e-12)
    F = np.full((nmax, nfr), np.nan)
    rows = np.arange(nfr)
    for i, cn in enumerate(c):
        lo = max(1, int(np.floor((cn - 0.3 * f0) / df)))
        hi = int(np.ceil((cn + 0.3 * f0) / df)) + 1
        if hi + 1 >= P.shape[1]:
            nmax = i
            break
        k = np.argmax(P[:, lo:hi], axis=1) + lo
        a = np.log(P[rows, k - 1] + 1e-300)
        b = np.log(P[rows, k] + 1e-300)
        cc = np.log(P[rows, k + 1] + 1e-300)
        den = a - 2 * b + cc
        d = np.where(den < 0, 0.5 * (a - cc) / np.where(den == 0, -1.0, den), 0.0)
        d = np.clip(d, -0.5, 0.5)
        A[i] = np.exp(b - 0.25 * (a - cc) * d)
        F[i] = (k + d) * df
    return dict(t=t, A=A[:nmax], F=F[:nmax], c=c[:nmax], f0=f0, B=B)


def smooth(v, k=5):
    v = np.asarray(v, float)
    if v.size < k:
        return v
    return np.convolve(np.pad(v, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def derive(tr, t_end):
    """Time series and per-partial descriptors of one signal's partial tracks."""
    t, A, F, c, f0, B = tr["t"], tr["A"], tr["F"], tr["c"], tr["f0"], tr["B"]
    E = A ** 2
    nmax = E.shape[0]
    n = np.arange(1, nmax + 1)
    tot = E.sum(0) + 1e-30
    keep = t <= t_end
    L = 10 * np.log10(tot)
    out = dict(t=t, level=L - L[keep].max())
    out["bands"] = {f"{a}-{b}": 10 * np.log10(E[a - 1:min(b, nmax)].sum(0) / tot + 1e-30) for a, b in BANDS
                    if a <= nmax}
    out["centroid_hz"] = (c[:, None] * E).sum(0) / tot
    e4 = E[3:].sum(0) + 1e-30
    out["formant_n"] = (n[3:, None] * E[3:]).sum(0) / e4
    k8 = min(8, nmax)
    ratio = F[:k8] / (n[:k8, None] * np.sqrt(1 + B * n[:k8, None] ** 2))
    wgt = E[:k8]
    fnow = np.nansum(ratio * wgt, 0) / (np.nansum(np.where(np.isnan(ratio), 0, wgt), 0) + 1e-30)
    ref = np.median(fnow[(t >= 0.5) & keep])
    out["pitch_cents"] = 1200 * np.log2(np.maximum(fnow, 1e-9) / ref)
    # per partial: bloom time, peak level, late decay rate
    gmax = 10 * np.log10(E[:, keep].max())
    tp, pk, dec = [], [], []
    for i in range(nmax):
        Ld = smooth(10 * np.log10(E[i] + 1e-30))
        Lk = np.where(keep, Ld, -np.inf)
        j = int(np.argmax(Lk))
        tp.append(float(t[j]))
        pk.append(float(Ld[j] - gmax))
        a0 = max(t[j] + 0.3, 1.0)
        sel = (t >= a0) & keep
        if sel.sum() * HOP >= 0.8 and Ld[j] - gmax > -50:
            dec.append(float(np.polyfit(t[sel], Ld[sel], 1)[0]))
        else:
            dec.append(None)
    out["t_peak"], out["peak_db"], out["decay"] = tp, pk, dec
    lv = out["level"]
    out["rise"] = float(lv[keep].max() - lv[np.argmin(np.abs(t - 0.05))])
    out["t_max"] = float(t[keep][np.argmax(lv[keep])])
    return out


def compare(trm, trr, t_end):
    """Model (or other) minus reference, per partial and frame, with the static / dynamic split."""
    nm = min(trm["A"].shape[0], trr["A"].shape[0])
    T = min(trm["A"].shape[1], trr["A"].shape[1])
    t = trr["t"][:T]
    Em, Er = trm["A"][:nm, :T] ** 2, trr["A"][:nm, :T] ** 2
    selt = (t >= T0) & (t <= t_end)
    Lm = 10 * np.log10(Em / Em[:, selt].sum() + 1e-30)
    Lr = 10 * np.log10(Er / Er[:, selt].sum() + 1e-30)
    rel_m = 10 * np.log10(Em / Em.max(0) + 1e-30)
    rel_r = 10 * np.log10(Er / Er.max(0) + 1e-30)
    valid = selt[None, :] & ((rel_r > MASK_DB) | (rel_m > MASK_DB))
    D = np.clip(Lm - Lr, -CLIP_DB, CLIP_DB)
    Dv = np.where(valid, D, np.nan)
    c0 = float(np.nanmean(Dv))
    cnt = valid.sum(1)
    s_n = np.where(cnt > 0, np.nansum(Dv - c0, 1) / np.maximum(cnt, 1), 0.0)
    e = Dv - c0 - s_n[:, None]
    lf = np.log2(trr["c"][:nm])
    use = cnt > 0
    deg = 3 if use.sum() > 8 else 1
    coef = np.polyfit(lf[use], s_n[use], deg, w=np.sqrt(cnt[use]))
    s_smooth = np.polyval(coef, lf)
    s_rough = s_n - s_smooth
    W = cnt / cnt.sum()
    ms_static_smooth = float((W * s_smooth ** 2).sum())
    ms_static_rough = float((W * s_rough ** 2).sum())
    ms_dyn = float(np.nanmean(e ** 2))
    ms_tot = float(np.nanmean((Dv - c0) ** 2))
    return dict(t=t, n=nm, D=Dv, offset=c0, s_n=s_n, s_smooth=s_smooth, s_rough=s_rough, freq=trr["c"][:nm],
                cells=int(valid.sum()),
                rms=dict(total=float(np.sqrt(ms_tot)), static=float(np.sqrt(ms_static_smooth + ms_static_rough)),
                         static_smooth=float(np.sqrt(ms_static_smooth)), static_rough=float(np.sqrt(ms_static_rough)),
                         dynamic=float(np.sqrt(ms_dyn)),
                         eq_invariant=float(np.sqrt(ms_static_rough + ms_dyn))),
                share=dict(static_smooth=ms_static_smooth / ms_tot, static_rough=ms_static_rough / ms_tot,
                           dynamic=ms_dyn / ms_tot))


def yin_pitch(y, f0, t1=0.8, hop=0.002, thresh=0.15):
    """Pitch over the first t1 s by YIN (de Cheveigne & Kawahara 2002): window of 3 periods (>= 25 ms), lag range
    f0/1.25 .. f0/0.8, cumulative-mean-normalised difference, first dip below `thresh` (else the minimum), parabolic
    interpolation.  t is re the onset (5 ms into the file); returns cents re f0 (the steady pitch) and the dip value."""
    W = int(round(max(3.0 / f0, 0.025) * SR))
    tmin, tmax = int(SR / (1.25 * f0)), int(SR / (0.8 * f0)) + 2
    H = int(round(hop * SR))
    y = np.asarray(y, float)
    starts = int(0.005 * SR) - W // 2 + H * np.arange(int(t1 / hop) + 1)
    nf = 1 << int(np.ceil(np.log2(W + tmax + 1)))
    t_out, c_out, q_out = [], [], []
    for s0 in starts:
        if s0 < 0 or s0 + W + tmax >= len(y):
            continue
        x = y[s0:s0 + W + tmax]
        a = x[:W]
        if np.sum(a * a) < 1e-12:
            continue
        r = np.fft.irfft(np.fft.rfft(x, nf) * np.conj(np.fft.rfft(a, nf)), nf)[:tmax + 1]
        cs = np.concatenate([[0.0], np.cumsum(x * x)])
        e1 = cs[W]
        tau = np.arange(tmax + 1)
        e2 = cs[tau + W] - cs[tau]
        d = e1 + e2 - 2 * r
        d[0] = 0.0
        cm = np.ones_like(d)
        cm[1:] = d[1:] * tau[1:] / np.maximum(np.cumsum(d[1:]), 1e-30)
        rng = np.arange(tmin, tmax)
        below = rng[cm[rng] < thresh]
        if below.size:
            k = below[0]
            while k + 1 < tmax and cm[k + 1] < cm[k]:
                k += 1
        else:
            k = rng[np.argmin(cm[rng])]
        a_, b_, c_ = cm[k - 1], cm[k], cm[k + 1]
        den = a_ - 2 * b_ + c_
        dk = 0.5 * (a_ - c_) / den if den > 0 else 0.0
        T = k + float(np.clip(dk, -1, 1))
        t_out.append((s0 + W / 2) / SR - 0.005)
        c_out.append(1200 * np.log2(SR / T / f0))
        q_out.append(float(b_))
    return np.array(t_out), np.array(c_out), np.array(q_out)


# ================================================================================================ ERB / auditory
def erb_number(f):
    return 21.4 * np.log10(1 + 0.00437 * np.asarray(f, float))


def erb_inv(e):
    return (10 ** (np.asarray(e, float) / 21.4) - 1) / 0.00437


def erb_filterbank(freqs, fmin, fmax, nb):
    """Triangular filters equally spaced on the ERB-number scale (half-overlapping).  Returns W (nb x bins), centres."""
    e = erb_number(freqs)
    ec = np.linspace(erb_number(fmin), erb_number(fmax), nb)
    de = ec[1] - ec[0]
    W = np.maximum(0.0, 1 - np.abs(e[None, :] - ec[:, None]) / de)
    return W, erb_inv(ec)


def band_spectrogram(y, nper, hop, fmin, fmax, nb, t1):
    seg = np.asarray(y[:int((t1 + 0.1) * SR) + nper], float)
    f, t, Z = ss.stft(seg, SR, window="hann", nperseg=nper, noverlap=nper - hop, boundary="zeros", padded=True)
    W, fc = erb_filterbank(f, fmin, fmax, nb)
    Eb = W @ (np.abs(Z) ** 2)
    return t - 0.005, Eb, fc


def bark(f):
    f = np.asarray(f, float)
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def threshold_quiet_db(f):
    k = np.asarray(f, float) / 1000.0
    return 3.64 * k ** -0.8 - 6.5 * np.exp(-0.6 * (k - 3.3) ** 2) + 1e-3 * k ** 4


class Ear:
    """Excitation (ERB bands + Schroeder spreading) and Zwicker-type specific loudness, 50 Hz - 15 kHz, 40 bands."""

    def __init__(self, nper=2048, hop=441, nb=40, fmin=50.0, fmax=15000.0):
        self.nper, self.hop, self.nb, self.fmin, self.fmax = nper, hop, nb, fmin, fmax
        f = np.fft.rfftfreq(nper, 1 / SR)
        self.W, self.fc = erb_filterbank(f, fmin, fmax, nb)
        z = bark(self.fc)
        dz = z[None, :] - z[:, None]                    # row: masker band i, column: masked band j
        x = dz + 0.474
        sf_db = 15.81 + 7.5 * x - 17.5 * np.sqrt(1 + x * x)
        self.S = 10 ** (sf_db / 10)
        self.Etq = 10 ** (threshold_quiet_db(self.fc) / 10)

    def power_bands(self, y, t1):
        seg = np.asarray(y[:int((t1 + 0.05) * SR)], float)
        f, t, Z = ss.stft(seg, SR, window="hann", nperseg=self.nper, noverlap=self.nper - self.hop,
                          boundary="zeros", padded=True)
        return t - 0.005, self.W @ (np.abs(Z) ** 2)

    def loudness(self, Pb, cal):
        """Pb: band powers (arbitrary units) x cal -> SPL power units.  Returns specific loudness N' (nb x frames)."""
        E = self.S.T @ (Pb * cal)
        Nsp = 0.08 * (self.Etq[:, None] ** 0.23) * ((0.5 + 0.5 * E / self.Etq[:, None]) ** 0.23 - 1.0)
        return np.maximum(Nsp, 0.0)


def perceptual(ear, ym, yr, t_end, hp_hz=None):
    """Specific loudness of model and recording, the model's gain matched for equal mean loudness over the window,
    the recording's loudest frame calibrated to 75 dB SPL.  hp_hz: listen through a small speaker (4th-order HP)."""
    if hp_hz:
        sos = ss.butter(4, hp_hz, "highpass", fs=SR, output="sos")
        ym, yr = ss.sosfiltfilt(sos, ym), ss.sosfiltfilt(sos, yr)
    t, Pr = ear.power_bands(yr, t_end)
    tm, Pm = ear.power_bands(ym, t_end)
    T = min(Pr.shape[1], Pm.shape[1])
    t, Pr, Pm = t[:T], Pr[:, :T], Pm[:, :T]
    sel = (t >= T0) & (t <= t_end)
    cal = 10 ** 7.5 / Pr.sum(0)[sel].max()
    Nr = ear.loudness(Pr, cal)
    target = Nr.sum(0)[sel].mean()
    lo, hi = -60.0, 60.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        Nm = ear.loudness(Pm, cal * 10 ** (mid / 10))
        if Nm.sum(0)[sel].mean() < target:
            lo = mid
        else:
            hi = mid
    Nm = ear.loudness(Pm, cal * 10 ** (0.5 * (lo + hi) / 10))
    Ntm, Ntr = Nm.sum(0), Nr.sum(0)
    diff = np.abs(Nm - Nr).sum(0)
    shape = np.abs(Nm / (Ntm + 1e-12) - Nr / (Ntr + 1e-12)).sum(0)
    wr = np.where(sel, Ntr, 0.0)
    out = dict(t=t, Nm=Nm, Nr=Nr, gain_db=0.5 * (lo + hi),
               d_t=np.where(sel, diff / (Ntr + 1e-12), np.nan),
               shape_t=np.where(sel, 50 * shape, np.nan),
               pld=float(100 * diff[sel].sum() / Ntr[sel].sum()),
               pld_shape=float(50 * (shape * wr).sum() / wr.sum()),
               loud_phon_rms=float(np.sqrt(np.mean((10 * np.log2((Ntm[sel] + 1e-9) / (Ntr[sel] + 1e-9))) ** 2))),
               phon_m=40 + 10 * np.log2(np.maximum(Ntm, 1e-6)), phon_r=40 + 10 * np.log2(np.maximum(Ntr, 1e-6)))
    return out


# ================================================================================================ other losses
def mss(ym, yr, t_end, sizes=(256, 512, 1024, 2048, 4096), fmax=8000.0, floor_db=80.0):
    """Multi-resolution log-magnitude STFT distance (DDSP-style), dB, both at equal energy over the window."""
    n = int(min(len(ym), len(yr), (t_end + 0.005) * SR))
    a, b = np.asarray(ym[:n], float), np.asarray(yr[:n], float)
    a = a * np.sqrt((b ** 2).sum() / ((a ** 2).sum() + 1e-30))
    vals = []
    for N in sizes:
        f, t, Za = ss.stft(a, SR, window="hann", nperseg=N, noverlap=N - N // 4)
        _, _, Zb = ss.stft(b, SR, window="hann", nperseg=N, noverlap=N - N // 4)
        k = f <= fmax
        La = 10 * np.log10(np.abs(Za[k]) ** 2 + 1e-30)
        Lb = 10 * np.log10(np.abs(Zb[k]) ** 2 + 1e-30)
        fl = Lb.max() - floor_db
        La, Lb = np.maximum(La, fl), np.maximum(Lb, fl)
        m = (La > fl) | (Lb > fl)
        vals.append(float(np.abs(La - Lb)[m].mean()))
    return float(np.mean(vals))


def mel_fb(nfft, nmel=40, fmin=60.0, fmax=8000.0):
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)
    f = np.fft.rfftfreq(nfft, 1 / SR)
    m = np.linspace(hz2mel(fmin), hz2mel(fmax), nmel + 2)
    fc = mel2hz(m)
    W = np.zeros((nmel, f.size))
    for i in range(nmel):
        lo, c, hi = fc[i], fc[i + 1], fc[i + 2]
        W[i] = np.maximum(0, np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)))
    return W


_MEL = mel_fb(2048)


def mcd(ym, yr, t_end, ncep=12):
    """Mel-cepstral distortion (dB), coefficients 1-12 (level-free), frames 50 ms - t_end within 40 dB of the
    recording's loudest frame."""
    from scipy.fft import dct
    n = int(min(len(ym), len(yr), (t_end + 0.005) * SR))
    out = []
    for y in (ym, yr):
        f, t, Z = ss.stft(np.asarray(y[:n], float), SR, window="hann", nperseg=2048, noverlap=2048 - 441)
        M = _MEL @ (np.abs(Z) ** 2)
        c = dct(0.5 * np.log(M + 1e-12), type=2, norm="ortho", axis=0)
        out.append((t - 0.005, c, M.sum(0)))
    (t, cm, _), (_, cr, er) = out
    T = min(cm.shape[1], cr.shape[1])
    sel = (t[:T] >= T0) & (t[:T] <= t_end) & (10 * np.log10(er[:T] / er[:T].max() + 1e-30) > -40)
    d = cm[1:ncep + 1, :T] - cr[1:ncep + 1, :T]
    return float(np.mean((10 / np.log(10)) * np.sqrt(2 * (d[:, sel] ** 2).sum(0))))


def d3(y, f0, ref_name):
    """fit3's distance D3 of a signal to one of the nine fit references (lower = closer)."""
    return float(F3.analyse_signal(y, f0, F3.ref3(ref_name))["D3"])


# ================================================================================================ forensics
def align(x, y, max_lag=0.25, n=2.0):
    """Lag (samples) that best aligns y with x (y[k + lag] ~ x[k])."""
    m = int(min(len(x), len(y), n * SR))
    xc = ss.correlate(y[:m], x[:m], mode="full", method="fft")
    lags = ss.correlation_lags(m, m, mode="full")
    sel = np.abs(lags) <= int(max_lag * SR)
    return int(lags[sel][np.argmax(np.abs(xc[sel]))])


def demodulate(y, f0, nmax, t1, hop=HOP, periods=PERIODS):
    """Complex amplitude of harmonics 1..nmax of y at f0 (Hann window of `periods` periods), phase re absolute time.
    Returns frame-centre times (file time) and Z (nmax x frames)."""
    N = int(round(periods * SR / f0))
    N += N % 2
    H = int(round(hop * SR))
    w = np.hanning(N)
    ypad = np.concatenate([np.zeros(N // 2), np.asarray(y, float), np.zeros(N)])
    nfr = min(int(t1 * SR / H) + 1, (len(ypad) - N) // H + 1)
    idx = np.arange(N)[None, :] + H * np.arange(nfr)[:, None]
    n = np.arange(1, nmax + 1)
    m = np.arange(N) - N // 2
    basis = w[:, None] * np.exp(-2j * np.pi * f0 * m[:, None] * n[None, :] / SR)
    tc = np.arange(nfr) * H / SR
    Z = (ypad[idx] @ basis) * np.exp(-2j * np.pi * f0 * n[None, :] * tc[:, None])
    return tc, Z.T


def same_vibration(x, y, f0x, f0y, t0=0.05, t1=2.5, nmin=4, nmax=30):
    """Is y the same string vibration as x, processed?  Two measures (calibrated in the report with controls):
    coherence: per harmonic n = 4..30, the phase of y re x over time, a straight line allowed (a pitch offset); the
    length of the mean unit phasor of what is left (1 = the same vibration, independent plucks ~0.4-0.5), weighted by
    energy.  resid: level of y re x per harmonic and frame, fitted as (fixed EQ per harmonic) + (common gain over
    time, i.e. compression or automation); the RMS of what is left, dB.  eq: the fixed EQ per harmonic (dB)."""
    lag = align(x, y)
    y2 = y[lag:] if lag >= 0 else np.concatenate([np.zeros(-lag), y])
    t, Zx = demodulate(x, f0x, nmax, t1 + 0.05)
    _, Zy = demodulate(y2, f0y, nmax, t1 + 0.05)
    T = min(Zx.shape[1], Zy.shape[1])
    Zx, Zy, t = Zx[:, :T], Zy[:, :T], t[:T]
    Ax, Ay = np.abs(Zx) + 1e-12, np.abs(Zy) + 1e-12
    vx = 20 * np.log10(Ax / Ax.max(0)) > -30
    vy = 20 * np.log10(Ay / Ay.max(0)) > -30
    sel = (t >= t0) & (t <= t1)
    Rn, wn = [], []
    for i in range(nmin - 1, nmax):
        v = vx[i] & vy[i] & sel
        if v.sum() < 20:
            continue
        d = np.unwrap(np.angle(Zy[i, v] * np.conj(Zx[i, v])))
        r = d - np.polyval(np.polyfit(t[v], d, 1), t[v])
        Rn.append(abs(np.mean(np.exp(1j * r))))
        wn.append(np.mean(Ax[i, v] ** 2))
    coh = float(np.dot(Rn, wn) / np.sum(wn)) if Rn else float("nan")
    # two-way level model on harmonics 1..nmax
    v = vx & vy & sel[None, :]
    D = np.where(v, 20 * np.log10(Ay / Ax), np.nan)
    a, b = np.zeros(nmax), np.zeros(T)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(60):
            a = np.nan_to_num(np.nanmean(D - b[None, :], 1))
            b = np.nan_to_num(np.nanmean(D - a[:, None], 0))
        resid = float(np.sqrt(np.nanmean((D - a[:, None] - b[None, :]) ** 2)))
        cols = np.isfinite(np.nanmean(D, 0)) & (t >= 0.2)
    span = float(np.ptp(b[cols])) if cols.any() else float("nan")
    ref = np.median(a[3:15])
    return dict(coh=coh, coh_med=float(np.median(Rn)) if Rn else float("nan"), resid=resid, gain_span=span,
                eq_f=(np.arange(1, nmax + 1) * f0x).tolist(), eq=(a - ref).tolist(), lag_ms=1000 * lag / SR,
                cents=float(1200 * np.log2(f0y / f0x)))


def classify(coh):
    if not np.isfinite(coh):
        return "unknown"
    if coh >= 0.9:
        return "same pluck, processed copy"
    if coh >= 0.65:
        return "largely the same pluck (heavier processing or mixed)"
    if coh >= 0.55:
        return "unclear"
    return "different pluck"


def compressor(y, thr_db=-20.0, ratio=4.0, att=0.005, rel=0.3):
    """Simple feed-forward compressor (control signal for the same-pluck test)."""
    env = np.abs(ss.hilbert(y))
    peak = env.max()
    aa, rr = np.exp(-1 / (att * SR)), np.exp(-1 / (rel * SR))
    out, lvl = np.empty_like(y), 0.0
    for i in range(len(y)):
        c = aa if env[i] > lvl else rr
        lvl = c * lvl + (1 - c) * env[i]
        ldb = 20 * np.log10(lvl / peak + 1e-12)
        out[i] = y[i] * 10 ** ((min(0.0, (thr_db - ldb) * (1 - 1 / ratio)) if ldb > thr_db else 0.0) / 20)
    return out


def controls(own):
    """Known same-pluck controls: the own file through a 120 Hz high-pass + 3 kHz low-pass, a 4:1 compressor and a
    +1.5 cent pitch shift; and the own file plus a 0.8 s noise reverb at -6 dB."""
    seg = np.asarray(own[:int(3.4 * SR)], float)
    sos = np.vstack([ss.butter(2, 120, "highpass", fs=SR, output="sos"), ss.butter(1, 3000, "lowpass", fs=SR,
                                                                                     output="sos")])
    a = ss.resample_poly(compressor(ss.sosfilt(sos, seg)), 1153, 1154)      # played at SR: +1.50 cents
    rng = np.random.default_rng(1)
    L = int(0.8 * SR)
    ir = rng.standard_normal(L) * np.exp(-6.9 * np.arange(L) / L)
    ir /= np.sqrt((ir ** 2).sum())
    b = seg + 0.5 * ss.fftconvolve(seg, ir)[:seg.size]
    return [("control: your file, EQ + 4:1 compressor + 1.5 cent pitch shift", a),
            ("control: your file + 0.8 s reverb at -6 dB", b)]


def forensics(s, sig):
    """Which of the user's files of string s are the same pluck as the iTanpura CS3 file and as the own file."""
    rows, eqs = [], {}
    x_own, x_it = sig["rec_own"]["y"], sig["rec_itan"]["y"]
    f_own, f_it = sig["rec_own"]["f0"], sig["rec_itan"]["f0"]
    files = [(sig["rec_own"]["src"], x_own, 48000, "own"), (sig["rec_alt"]["src"], sig["rec_alt"]["y"], 44100, "alt")]
    for rel in EXTRA[s]:
        try:
            y, sr = load_file(rel)
        except Exception as exc:  # a missing file is reported, not fatal
            rows.append(dict(file=rel, error=str(exc)))
            continue
        files.append((rel, y, sr, "extra"))
    for lab, y in controls(x_own):
        files.append((lab, y, SR, "control"))
    for rel, y, sr, kind in files:
        f0, _ = F2.fit_f0_B(y, NOMINAL[s])
        vi = same_vibration(x_it, y, f_it, f0)
        row = dict(file=rel, kind=kind, sr=sr, dur=round(len(y) / SR, 2), f0=round(f0, 3),
                   cents_vs_itan=round(vi["cents"], 2), coh_itan=round(vi["coh"], 2), resid_itan=round(vi["resid"], 1),
                   gain_span_itan=round(vi["gain_span"], 1), verdict_itan=classify(vi["coh"]))
        if kind == "control":     # controls are compared with the file they were made from
            vo = same_vibration(x_own, y, f_own, f0)
            row.update(coh_own=round(vo["coh"], 2), resid_own=round(vo["resid"], 1), verdict_own=classify(vo["coh"]))
        elif kind != "own":
            vo = same_vibration(x_own, y, f_own, f0)
            row.update(coh_own=round(vo["coh"], 2), resid_own=round(vo["resid"], 1), verdict_own=classify(vo["coh"]))
        rows.append(row)
        if kind in ("own", "extra") and vi["coh"] >= 0.65:
            eqs[rel] = dict(f=q(vi["eq_f"], nd=0), eq=q(vi["eq"]), coh=vi["coh"], resid=vi["resid"])
        log(f"  forensics {s}: {rel}: f0 {f0:.3f} ({vi['cents']:+.2f} c re iTanpura), coherence vs iTanpura "
            f"{vi['coh']:.2f}, level residual {vi['resid']:.1f} dB -> {row['verdict_itan']}"
            + (f" | vs own {row['coh_own']:.2f}" if "coh_own" in row else ""))
    return dict(rows=rows, eq=eqs)


# ================================================================================================ helpers for JSON
def _r(v, nd):
    if not np.isfinite(v):
        return None
    return int(round(float(v))) if nd == 0 else round(float(v), nd)


def q(a, scale=1.0, nd=1):
    """list of rounded numbers (None for nan / inf)"""
    a = np.asarray(a, float) * scale
    return [_r(v, nd) for v in a.ravel()]


def qmat(M, floor=None, nd=1, dec=DEC):
    M = np.asarray(M, float)[:, ::dec]
    if floor is not None:
        M = np.maximum(M, floor)
    return [[_r(v, nd) for v in row] for row in M]


# ================================================================================================ main analysis
PAIRS = (   # (key, test signal, reference signal, label, kind)
    ("m3_own", "m3", "rec_own", "Model round 3 vs your recording", "model"),
    ("m2_own", "m2", "rec_own", "Model round 2 vs your recording", "model"),
    ("itan_own", "rec_itan", "rec_own", "iTanpura CS3 vs your recording", "real"),
    ("alt_itan", "rec_alt", "rec_itan", "CS3Alt vs iTanpura CS3 (another real pluck)", "real"),
    ("alt_own", "rec_alt", "rec_own", "CS3Alt vs your recording", "real"),
)
LOSSES = (  # key, label, unit, what it measures
    ("d3", "D3 (the fit's current loss)", "", "hand-picked features: band shares at 5 times, attack, level curve, decay, bloom"),
    ("hls", "Harmonic level difference", "dB RMS", "every partial, every 10 ms, levels in dB (clipped at 30 dB)"),
    ("hls_eqinv", "Same, EQ-invariant", "dB RMS", "a smooth fixed EQ curve removed first (recording chain / mic)"),
    ("hls_dyn", "Time-evolution only", "dB RMS", "every fixed per-partial colour removed: only how the sound changes"),
    ("pld", "Perceptual loudness difference", "%", "auditory model: ERB bands, masking, loudness compression"),
    ("pld_shape", "Perceptual timbre difference", "%", "the same with each frame's total loudness divided out"),
    ("pld_phone", "Perceptual, small speaker", "%", "the same through a phone-like speaker (no sound below 250 Hz)"),
    ("loud_curve", "Loudness-curve difference", "phon RMS", "how the overall loudness swells and fades"),
    ("mss", "Multi-resolution spectrogram", "dB", "standard neural-synthesis loss (5 FFT sizes, fine detail)"),
    ("mcd", "Mel-cepstral distortion", "dB", "spectral envelope only, per frame"),
)


def analyse_string(s, ear):
    t_start = time.time()
    sig = signals(s)
    t_end = sig["rec_own"]["t_end"]
    log(f"[{s}] f0: " + ", ".join(f"{k} {v['f0']:.3f} Hz (B {v['B']:.1e})" for k, v in sig.items()) +
        f" | t_end {t_end}")
    tr = {k: track_partials(v["y"], v["f0"], v["B"], max(t_end, v["t_end"])) for k, v in sig.items()}
    der = {k: derive(tr[k], v["t_end"] if k.startswith("rec") else t_end) for k, v in sig.items()}
    res = dict(string=s, name=NAMES[s], verdict=USER_VERDICT[s], t_end=t_end,
               signals={k: dict(label=v["label"], src=v["src"], f0=v["f0"], B=v["B"], dur=v["dur"],
                                t_end=v["t_end"]) for k, v in sig.items()})
    # ---- comparisons and losses
    comps, losses, percs = {}, {}, {}
    for key, a, b, label, kind in PAIRS:
        te = min(sig[a]["t_end"], sig[b]["t_end"], t_end) if not a.startswith("m") else sig[b]["t_end"]
        cp = compare(tr[a], tr[b], te)
        pc = perceptual(ear, sig[a]["y"], sig[b]["y"], te)
        pp = perceptual(ear, sig[a]["y"], sig[b]["y"], te, hp_hz=250.0)
        L = dict(d3=d3(sig[a]["y"], sig[a]["f0"], sig[b]["ref"]), hls=cp["rms"]["total"],
                 hls_eqinv=cp["rms"]["eq_invariant"], hls_dyn=cp["rms"]["dynamic"], pld=pc["pld"],
                 pld_shape=pc["pld_shape"], pld_phone=pp["pld"], loud_curve=pc["loud_phon_rms"],
                 mss=mss(sig[a]["y"], sig[b]["y"], te), mcd=mcd(sig[a]["y"], sig[b]["y"], te))
        comps[key], losses[key], percs[key] = cp, L, pc
        log(f"  {key:9s} t_end {te}: " + "  ".join(f"{k} {v:.2f}" for k, v in L.items()) +
            f" | split smooth/rough/dyn {cp['share']['static_smooth']:.2f}/{cp['share']['static_rough']:.2f}/"
            f"{cp['share']['dynamic']:.2f}")
    res["losses"] = losses
    res["split"] = {k: dict(rms=v["rms"], share=v["share"], offset=v["offset"], cells=v["cells"]) for k, v in comps.items()}
    res["static"] = {k: dict(freq=q(v["freq"], nd=1), s=q(v["s_n"]), smooth=q(v["s_smooth"])) for k, v in comps.items()}
    # ---- heatmaps (dB re the signal's own loudest partial), difference maps
    heat = {}
    for k in ("rec_own", "m3", "m2", "rec_itan", "rec_alt"):
        E = tr[k]["A"] ** 2
        T = int(np.searchsorted(tr[k]["t"], t_end + 1e-9))
        Ld = 10 * np.log10(E[:, :T] / E[:, :T].max() + 1e-30)
        heat[k] = dict(L=qmat(Ld, floor=-70.0, nd=0))
    tt = tr["rec_own"]["t"]
    T = int(np.searchsorted(tt, t_end + 1e-9))
    res["heat"] = dict(t=q(tt[:T:DEC], nd=3), f0=sig["rec_own"]["f0"], maps=heat,
                       diff={k: qmat(comps[k]["D"][:, :T], nd=0) for k in ("m3_own", "m2_own", "itan_own", "alt_own")})
    # ---- time series (every 20 ms)
    ser = {}
    for k in ("rec_own", "m3", "m2", "rec_itan", "rec_alt", "m3_dry"):
        d = der[k]
        T = int(np.searchsorted(d["t"], t_end + 1e-9))
        ser[k] = dict(level=q(d["level"][:T:DEC]), bands={b: q(v[:T:DEC]) for b, v in d["bands"].items()},
                      centroid=q(d["centroid_hz"][:T:DEC], nd=0), formant=q(d["formant_n"][:T:DEC], nd=2),
                      pitch=q(d["pitch_cents"][:T:DEC], nd=2), rise=d["rise"], t_max=d["t_max"])
    res["series"] = dict(t=q(tt[:int(np.searchsorted(tt, t_end + 1e-9)):DEC], nd=3), sig=ser)
    yin = {}
    for k in ("rec_own", "m3", "m2", "rec_itan", "rec_alt"):
        ty, cy, qy = yin_pitch(sig[k]["y"], sig[k]["f0"])
        cy = np.where((qy < 0.3) & (np.abs(cy) < 60), cy, np.nan)   # no clear period, or an octave / period slip
        med = ss.medfilt(np.nan_to_num(cy, nan=1e3), 5)            # 5-frame (10 ms) median against single-frame slips
        cy = np.where(np.isfinite(cy) & (np.abs(med) < 60), med, np.nan)
        yin[k] = dict(cents=q(cy, nd=1))
        yin["t"] = q(ty, nd=3)
        early = cy[(ty >= 0.0) & (ty <= 0.05)]
        yin[k]["start_cents"] = float(np.nanmedian(early)) if np.isfinite(early).any() else None
        mid = cy[(ty >= 0.2) & (ty <= 0.6)]
        yin[k]["settled_cents"] = float(np.nanmedian(mid)) if np.isfinite(mid).any() else None
    res["yin"] = yin
    log(f"  YIN start (0-50 ms) / settled (0.2-0.6 s) cents: " + ", ".join(
        f"{k} {yin[k]['start_cents']:+.1f}/{yin[k]['settled_cents']:+.1f}" for k in ("rec_own", "m3", "m2", "rec_itan", "rec_alt")
        if yin[k]['start_cents'] is not None and yin[k]['settled_cents'] is not None))
    # ---- per partial
    res["partials"] = {k: dict(t_peak=q(der[k]["t_peak"], nd=3), peak=q(der[k]["peak_db"]),
                               decay=[None if v is None else round(v, 2) for v in der[k]["decay"]],
                               freq=q(tr[k]["c"], nd=1)) for k in ("rec_own", "m3", "m2", "rec_itan", "rec_alt")}
    # ---- attack (ERB bands, first 300 ms; 11.6 ms window, 2 ms hop)
    att = {}
    for k in ("rec_own", "m3", "m2"):
        ta, Eb, fc = band_spectrogram(sig[k]["y"], 512, 88, 80.0, 12000.0, 32, 0.30)
        keep = (ta >= -0.004) & (ta <= 0.30)
        Lb = 10 * np.log10(Eb[:, keep] / Eb[:, keep].max() + 1e-30)
        _, Eb2, _ = band_spectrogram(sig[k]["y"], 512, 441, 80.0, 12000.0, 32, 1.5)
        t2 = np.arange(Eb2.shape[1]) * 441 / SR - 0.005
        early = Eb[:, (ta >= 0.0) & (ta <= 0.04)].mean(1)
        late = Eb2[:, (t2 >= 0.5) & (t2 <= 1.5)].mean(1)
        contrast = 10 * np.log10(early / early.sum() + 1e-30) - 10 * np.log10(late / late.sum() + 1e-30)
        att[k] = dict(L=qmat(Lb, floor=-60.0, nd=0, dec=1), contrast=q(contrast))
        att["t"], att["fc"] = q(ta[keep], nd=3), q(fc, nd=0)
    res["attack"] = att
    # ---- perceptual maps and curves (model round 3 and round 2 vs your recording)
    pm = percs["m3_own"]
    T = int(np.searchsorted(pm["t"], t_end + 1e-9))
    res["perc"] = dict(t=q(pm["t"][:T:DEC], nd=3), fc=q(ear.fc, nd=0),
                       Nr=qmat(pm["Nr"][:, :T], nd=3), Nm=qmat(pm["Nm"][:, :T], nd=3),
                       Nm2=qmat(percs["m2_own"]["Nm"][:, :T], nd=3),
                       d={k: q(100 * percs[k]["d_t"][:T:DEC], nd=1) for k in percs},
                       phon={"rec_own": q(pm["phon_r"][:T:DEC]), "m3": q(pm["phon_m"][:T:DEC]),
                             "m2": q(percs["m2_own"]["phon_m"][:T:DEC])})
    log(f"[{s}] done in {time.time() - t_start:.0f} s")
    return res, sig


def agreement(results):
    """For each loss: the strings ranked by model-round-3 distance (raw and divided by the real-vs-real yardstick
    'another real pluck'), and whether Sa (the one the user hears as close) comes out closest."""
    out = {}
    for key, label, unit, what in LOSSES:
        raw = {s: results[s]["losses"]["m3_own"][key] for s in STRINGS}
        yard = {s: results[s]["losses"]["alt_itan"][key] for s in STRINGS}
        same = {s: results[s]["losses"]["itan_own"][key] for s in STRINGS}
        norm = {s: raw[s] / yard[s] if yard[s] else None for s in STRINGS}
        rank_raw = sorted(STRINGS, key=lambda s: raw[s])
        rank_norm = sorted(STRINGS, key=lambda s: norm[s])
        out[key] = dict(label=label, unit=unit, what=what, raw=raw, yard=yard, same=same, norm=norm,
                        rank_raw=rank_raw, rank_norm=rank_norm,
                        agrees_raw=rank_raw[0] == "sa", agrees_norm=rank_norm[0] == "sa",
                        m2={s: results[s]["losses"]["m2_own"][key] for s in STRINGS})
    return out


def _at(series_t, values, x):
    t = np.asarray(series_t, float)
    v = np.asarray([np.nan if a is None else a for a in values], float)
    return float(v[int(np.argmin(np.abs(t - x)))])


def _glide(res, k):
    """pitch change from the start (0.05-0.15 s) to 0.5-0.75 s, cents (YIN); negative = falls like a plucked string"""
    Y = res["yin"]
    t = np.asarray(Y["t"], float)
    c = np.asarray([np.nan if a is None else a for a in Y[k]["cents"]], float)
    a = np.nanmedian(c[(t >= 0.05) & (t <= 0.15)])
    b = np.nanmedian(c[(t >= 0.5) & (t <= 0.75)])
    return float(a), float(b), float(b - a)


def difference_table(results):
    """Model (round 3) minus your recording for the features a listener might notice, per string, and whether the
    size of the difference follows the user's verdict (smallest on Sa, the string heard as close)."""
    rows = []

    def add(key, label, unit, fn, what, nd=1):
        vals, real, model = {}, {}, {}
        for s in STRINGS:
            r = results[s]
            a, b = fn(r, "rec_own"), fn(r, "m3")
            real[s], model[s], vals[s] = round(a, nd), round(b, nd), round(b - a, nd)
        mag = {s: abs(vals[s]) for s in STRINGS}
        tracks = mag["sa"] < min(mag["kharaj"], mag["pa"])
        rows.append(dict(key=key, label=label, unit=unit, what=what, diff=vals, real=real, model=model, tracks_ear=tracks))

    add("t_max", "Swell peak time", "s", lambda r, k: r["series"]["sig"][k]["t_max"],
        "when the overall level is highest", nd=2)
    add("rise", "Swell size", "dB", lambda r, k: r["series"]["sig"][k]["rise"], "peak level minus the level at 50 ms")
    add("glide", "Pitch slide, first 0.75 s", "cents", lambda r, k: _glide(r, k)[2],
        "negative: the pitch falls after the pluck, as on a real string")
    add("end", "Level at the end", "dB", lambda r, k: _at(r["series"]["t"], r["series"]["sig"][k]["level"], r["t_end"]),
        "re the peak, at 3.8 s (Sa 3.0 s)")
    add("b13", "Partials 1-3 at 1 s", "dB", lambda r, k: _at(r["series"]["t"], r["series"]["sig"][k]["bands"]["1-3"], 1.0),
        "share of the sound in the lowest three harmonics")
    add("b1631", "Partials 16-31 at 1 s", "dB",
        lambda r, k: _at(r["series"]["t"], r["series"]["sig"][k]["bands"]["16-31"], 1.0), "share in the upper harmonics")
    add("formant", "Jawari formant at 1 s", "partial", lambda r, k: _at(r["series"]["t"], r["series"]["sig"][k]["formant"], 1.0),
        "centre of the energy above partial 3, in partial numbers")

    def click(r, k):
        fc = np.asarray(r["attack"]["fc"], float)
        c = np.asarray(r["attack"][k]["contrast"], float)
        return float(np.nanmean(c[(fc >= 2000) & (fc < 5000)]))
    add("click", "Attack brightness 2-5 kHz", "dB", click, "first 40 ms re the sustained sound (0.5-1.5 s)")
    return rows


def findings(results, agree, table, foren):
    """Plain-English findings with the numbers, for the top of the page (all from the data above)."""
    R = results
    f = []
    tm = {s: (R[s]["series"]["sig"]["rec_own"]["t_max"], R[s]["series"]["sig"]["m3"]["t_max"]) for s in STRINGS}
    f.append(dict(title="The swell's timing follows what you hear",
                  text=f"The level peaks at {tm['kharaj'][1]:.2f} s in the kharaj model against {tm['kharaj'][0]:.2f} s in "
                       f"your recording, and at {tm['pa'][1]:.2f} s against {tm['pa'][0]:.2f} s on Pa. Sa, the string "
                       f"you hear as close, is the nearest: {tm['sa'][1]:.2f} s against {tm['sa'][0]:.2f} s."))
    g = {s: (_glide(R[s], "rec_own")[2], _glide(R[s], "m3")[2]) for s in STRINGS}
    f.append(dict(title="So does the pitch slide after the pluck",
                  text=f"Every real pluck starts a few cents sharp and slides down (kharaj {g['kharaj'][0]:+.1f}, Pa "
                       f"{g['pa'][0]:+.1f}, Sa {g['sa'][0]:+.1f} cents over the first 0.75 s). The kharaj and Pa models slide "
                       f"up instead ({g['kharaj'][1]:+.1f} and {g['pa'][1]:+.1f} cents): the jawari contact grows during the "
                       f"swell and raises the pitch. The Sa model slides down ({g['sa'][1]:+.1f} cents), like the real string. "
                       f"The fit's loss has no pitch term at all."))
    b = {s: (_at(R[s]["series"]["t"], R[s]["series"]["sig"]["rec_own"]["bands"]["1-3"], 1.0),
             _at(R[s]["series"]["t"], R[s]["series"]["sig"]["m3"]["bands"]["1-3"], 1.0)) for s in STRINGS}
    u = {s: (_at(R[s]["series"]["t"], R[s]["series"]["sig"]["rec_own"]["bands"]["16-31"], 1.0),
             _at(R[s]["series"]["t"], R[s]["series"]["sig"]["m3"]["bands"]["16-31"], 1.0)) for s in STRINGS}
    f.append(dict(title="The tone balance does not follow what you hear",
                  text=f"At 1 s the model has too much in partials 1-3 (Sa {b['sa'][1] - b['sa'][0]:+.0f} dB, kharaj "
                       f"{b['kharaj'][1] - b['kharaj'][0]:+.0f}, Pa {b['pa'][1] - b['pa'][0]:+.0f}) and too little in partials "
                       f"16-31 (Sa {u['sa'][1] - u['sa'][0]:+.0f} dB, kharaj {u['kharaj'][1] - u['kharaj'][0]:+.0f}, Pa "
                       f"{u['pa'][1] - u['pa'][0]:+.0f}). Sa is the worst on both, yet sounds closest. These differences are "
                       f"real, but they are not what makes the kharaj and Pa sound wrong to you."))
    sp = R["kharaj"]["split"]["m3_own"]["share"]
    sp_it = R["kharaj"]["split"]["itan_own"]["rms"]["total"]
    f.append(dict(title="About a quarter of the kharaj mismatch is a fixed tone colour",
                  text=f"Splitting the model-minus-recording difference: {100 * sp['static_smooth']:.0f}% is a smooth fixed "
                       f"tilt (too much bass, too little treble: what an EQ or a microphone could also cause), "
                       f"{100 * sp['static_rough']:.0f}% is fixed per partial, and {100 * sp['dynamic']:.0f}% is how the "
                       f"sound changes over time. The same pluck processed two ways (your file and the iTanpura file) already "
                       f"differs by {sp_it:.1f} dB RMS."))
    d3k = agree["d3"]
    f.append(dict(title="The current loss measures processing as much as physics",
                  text=f"By the fit's distance D3, the kharaj model ({d3k['raw']['kharaj']:.0f}) is closer to your file than "
                       f"the iTanpura file is ({d3k['same']['kharaj']:.0f}), although that is largely the same pluck. D3 ranks "
                       f"Sa the worst string ({d3k['raw']['sa']:.0f}). A perceptual distance ranks the same pluck processed "
                       f"twice as clearly closest and the model about as far away as another real pluck."))
    own_rows = [r for s in STRINGS for r in foren[s]["rows"] if r.get("kind") == "own"]
    coh = {r["file"].split("_")[-1].split(".")[0]: r["coh_itan"] for r in own_rows}
    f.append(dict(title="Your reference files are mostly the iTanpura plucks, processed",
                  text=f"Your Male/tt_CS3 kharaj and Sa share most of their vibration with the iTanpura CS3 files (phase "
                       f"coherence {coh.get('kharaj', float('nan')):.2f} and {coh.get('sa', float('nan')):.2f}; independent "
                       f"plucks give 0.34-0.53, processed copies 0.94-1.00). Pa is unclear "
                       f"({coh.get('pa', float('nan')):.2f}, re-tuned by about +1.5 cents). So the fit has in effect one real "
                       f"pluck per string, plus CS3Alt."))
    return f


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ear = Ear()
    results, foren = {}, {}
    for s in STRINGS:
        results[s], sig = analyse_string(s, ear)
        foren[s] = forensics(s, sig)
    agree = agreement(results)
    for k, a in agree.items():
        log(f"{k:10s} raw " + " ".join(f"{s} {a['raw'][s]:.2f}" for s in STRINGS) + " | norm " +
            " ".join(f"{s} {a['norm'][s]:.2f}" for s in STRINGS) + f" | Sa closest: raw {a['agrees_raw']} "
            f"norm {a['agrees_norm']}")
    table = difference_table(results)
    found = findings(results, agree, table, foren)
    for r in table:
        log(f"{r['label']:30s} model-real " + " ".join(f"{s} {r['diff'][s]:+.2f}" for s in STRINGS) +
            f" | tracks the ear: {r['tracks_ear']}")
    data = dict(generated=time.strftime("%Y-%m-%d %H:%M"), strings=list(STRINGS), table=table, findings=found,
                losses_def=[
        dict(key=k, label=l, unit=u, what=w) for k, l, u, w in LOSSES], pairs=[
        dict(key=k, label=l, kind=kd) for k, a, b, l, kd in PAIRS], results=results, forensics=foren,
        agreement=agree, bands=[f"{a}-{b}" for a, b in BANDS], verdict=USER_VERDICT,
        method=dict(periods=PERIODS, hop_ms=HOP * 1000, report_hop_ms=HOP * DEC * 1000, fmax=FMAX_TRACK,
                    clip_db=CLIP_DB, mask_db=MASK_DB, t0=T0))
    js = json.dumps(data, separators=(",", ":"), allow_nan=False)
    (OUT / "match_report.json").write_text(js)
    if TEMPLATE.exists():
        html = TEMPLATE.read_text().replace("/*__DATA__*/null", js.replace("</", "<\\/"))
        (OUT / "match_report.html").write_text(html)
        log(f"wrote {OUT / 'match_report.html'} ({len(html) / 1e6:.2f} MB)")
    log(f"wrote {OUT / 'match_report.json'} ({len(js) / 1e6:.2f} MB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main(sys.argv[1:])
