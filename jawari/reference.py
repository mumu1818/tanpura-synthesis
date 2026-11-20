"""Reference analyser: one recorded tanpura pluck (real instrument or a sample-app recording) vs a model render.

    cd jawari
    python reference.py analyse REF.wav --string kharaj            # features + figures -> out/reference/<name>/
    python reference.py compare REF.wav MODEL.wav --string kharaj  # side-by-side report + ranked mismatches
    python reference.py compare REF.wav --render kharaj            # render the kharaj preset now and compare
    python reference.py fit REF.wav --string kharaj --params target_gap,delta_ve,Yb,xp --budget 48 --jobs 3

    from reference import analyse, compare, fit
    A = analyse("references/kharaj.wav", string="kharaj")   # dict of features (see analyse())
    rep = compare("references/kharaj.wav", y_model, sr=44100, string="kharaj")

See REFERENCE.md for how to record references, what every output means and how the fit works.  The validation
(synthetic signals, model round trips, fit recovery) is validate_reference.py.

Analysis pipeline (analyse):
  1. load (mono or stereo), detect every pluck onset (log-band spectral flux), pick one, refine its onset to the
     sample (-15 dB crossing of the 1 ms high-passed envelope, linear interpolation), truncate the analysis at the
     next pluck (iTanpura-style overlapping strings) and at max_dur.
  2. f0 and inharmonicity: long-window (Blackman-Harris, zero-padded) peak picking of partials 1..N, robust fit of
     f_n = n f0 sqrt(1 + B n^2) (soft-L1 least squares in cents).
  3. partial envelopes: a frequency-domain heterodyne filter bank -- one power-complementary band per partial
     (edges midway between neighbouring partials), shifted to baseband and decimated: |z_n(t)| is the amplitude of
     everything in partial n's band (robust to the jawari's spectral spreading; bands tile the spectrum).
     Instantaneous frequency per band from the phase -> f0(t) drift/glide in cents.
  4. per-partial decay: robust (Huber) line fits of the dB envelope: early (first 10-15 dB after the partial's
     peak) and late (from 10 dB below the peak down to 10 dB above the noise floor) -> T60.
  5. pluck signature: early (10-90 ms) partial levels, comb fit |sin(n pi beta)| x smooth trend -> beta (and
     1-beta: indistinguishable from the spectrum), notch list, excitation roll-off.
  6. jawari descriptors (pitch-relative, work for a 69 Hz kharaj): buzz index BI_k = energy of partials >= k over
     partials < k; formant centre / peak / upper edge of the partial band (n >= n_lo) vs time; bloom onset, time
     to peak, plateau, persistence.
  7. attack: rise time, transient HF burst (click/finger noise), aperiodicity of the first periods.
  8. long-term average spectrum (Welch) and a smooth partial-peak envelope (1/3-octave) -> body EQ ratio.
  9. stereo: inter-channel coherence vs frequency, L/R correlation per octave, side/mid ratio, room tail if the
     note is damped before the end of the file.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal as ss
from scipy.optimize import least_squares

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
OUT_ROOT = Path(os.environ.get("TANPURA_OUT", PKG / "out"))
OUT = OUT_ROOT / "reference"

# nominal open-string pitches of the user's male tanpura in C#: kharaj C#2 69.4 Hz (shared/params.py), Pa a just
# fifth above it (tanpuras are tuned by ear to beatless intervals), Sa one octave above the kharaj.
STRINGS = {"kharaj": 69.4, "pa": 69.4 * 1.5, "sa": 69.4 * 2.0}

DEFAULTS = dict(
    max_dur=12.0,        # s analysed after the onset (also cut at the next pluck)
    pre=0.05,            # s kept before the onset anchor
    f_max=10000.0,       # highest partial frequency tracked
    n_max=150,           # most partials tracked
    hop=0.005,           # s, envelope time grid
    smooth=0.020,        # s, moving-average smoothing of the partial power envelopes
    spec_win=(0.10, 2.5),  # s after the onset: window of the long-window spectrum for partial frequencies
    early=(0.0, 0.040),  # s after the onset: window of the early (pluck) partial levels (the jawari fills
    #                       the pluck-comb notches within a few periods, so this is kept short)
    buzz_k=(5, 10, 20),  # buzz index split partials
    buzz_main=10,        # the headline buzz index
    floor_margin=10.0,   # dB above the per-partial noise floor that still counts as signal
    dyn_range=100.0,     # dB below the envelope maximum treated as silence (model renders have no noise)
    guard=0.010,         # s cut before the next detected pluck
    onset_db=-6.0,       # dB re the attack peak (first 40 ms): onset anchor threshold
    min_ioi=0.15,        # s, minimum time between two detected plucks
    truncate=True,       # stop the analysis at the next detected pluck (False: a knock / cough was detected)
    onset_abs_db=0.7,    # absolute onset-flux threshold (dB, band-averaged rise); see detect_onsets
)

def _settings(over=None):
    s = dict(DEFAULTS)
    if over:
        unknown = set(over) - set(s)
        if unknown:
            raise TypeError(f"unknown analysis setting(s) {sorted(unknown)}")
        s.update(over)
    return s


# ======================================================================================== small numerical helpers
def _db(p, floor=1e-30):
    return 10.0 * np.log10(np.maximum(p, floor))


def _next_pow2(n):
    return 1 << int(math.ceil(math.log2(max(2, n))))


def _movavg(x, k, axis=-1):
    """centred moving average of length k (k >= 1) along axis, edges by shrinking the window."""
    k = int(max(1, k))
    if k == 1:
        return np.asarray(x, float)
    x = np.asarray(x, float)
    c = np.cumsum(np.concatenate([np.zeros_like(np.take(x, [0], axis=axis)), x], axis=axis), axis=axis)
    n = x.shape[axis]
    i = np.arange(n)
    lo = np.clip(i - k // 2, 0, n)
    hi = np.clip(i - k // 2 + k, 0, n)
    num = np.take(c, hi, axis=axis) - np.take(c, lo, axis=axis)
    shape = [1] * x.ndim
    shape[axis] = n
    return num / (hi - lo).reshape(shape)


def robust_line(t, y, w=None, c=1.345, iters=30):
    """Huber IRLS straight line y = a + b t.  Returns (a, b, resid_scale)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    w0 = np.ones_like(t) if w is None else np.asarray(w, float)
    ww = w0.copy()
    a = b = 0.0
    s = 0.0
    for _ in range(iters):
        W = ww.sum()
        if W <= 0 or t.size < 2:
            return np.nan, np.nan, np.nan
        tm = (ww * t).sum() / W
        ym = (ww * y).sum() / W
        stt = (ww * (t - tm) ** 2).sum()
        if stt <= 0:
            return np.nan, np.nan, np.nan
        b_new = (ww * (t - tm) * (y - ym)).sum() / stt
        a_new = ym - b_new * tm
        r = y - a_new - b_new * t
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = np.abs(r) / (c * s)
        ww = w0 * np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))
        if abs(b_new - b) < 1e-9 * (abs(b) + 1e-9) and abs(a_new - a) < 1e-9 * (abs(a) + 1e-9):
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    return float(a), float(b), float(s)


def cents(f, fref):
    return 1200.0 * np.log2(np.asarray(f, float) / fref)


# ======================================================================================== loading
def load(src, sr=None):
    """(x (N, C) float64, sr, name).  src: path or array ((N,) or (N, C); then sr is required)."""
    if isinstance(src, (str, Path)):
        import soundfile as sf
        try:
            x, sr = sf.read(str(src), always_2d=True, dtype="float64")
        except Exception as e:
            # (verify3) phone voice memos / screen recordings (.m4a, .mp4, .mov, .aac, ...): decode with ffmpeg,
            # keeping the file's own sample rate and channels (lossless 32-bit float WAV in a temp dir)
            import shutil
            import subprocess
            import tempfile
            if shutil.which("ffmpeg") is None:
                raise RuntimeError(f"cannot read {src} ({e}); install ffmpeg or convert it: "
                                   f"ffmpeg -i '{src}' out.wav") from e
            with tempfile.TemporaryDirectory() as td:
                tmp = os.path.join(td, "decoded.wav")
                subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-c:a",
                                "pcm_f32le", tmp], check=True)
                x, sr = sf.read(tmp, always_2d=True, dtype="float64")
        name = Path(src).stem
    else:
        if sr is None:
            raise ValueError("sr is required for an array input")
        x = np.asarray(src, float)
        if x.ndim == 1:
            x = x[:, None]
        name = "array"
    if x.shape[1] > 2:
        x = x[:, :2]
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite samples")
    return x, int(sr), name


# ======================================================================================== onsets
def onset_function(y, sr, hop=256, nfft=2048, n_bands=24, fmin=60.0, fmax=10000.0):
    """Log-band spectral flux (positive part of the rise of each band's log energy over the max of the previous
    3 frames; SuperFlux-like, robust to slow jawari blooms and beating).  Returns (t, flux_db, band_db)."""
    f, t, Z = ss.stft(y, sr, window="hann", nperseg=nfft, noverlap=nfft - hop, boundary="zeros", padded=False)
    P = np.abs(Z) ** 2
    edges = np.geomspace(fmin, min(fmax, 0.45 * sr), n_bands + 1)
    B = np.stack([P[(f >= lo) & (f < hi)].sum(0) for lo, hi in zip(edges[:-1], edges[1:])])
    Bd = _db(B + 1e-30)
    top = Bd.max()
    Bd = np.maximum(Bd, top - 80.0)                      # 80 dB below the loudest band-frame = silence
    prev = np.full_like(Bd, -np.inf)
    for lag in (1, 2, 3):
        sh = np.concatenate([np.full((Bd.shape[0], lag), Bd[:, :1].min()), Bd[:, :-lag]], axis=1)
        prev = np.maximum(prev, sh)
    rise = np.maximum(Bd - prev, 0.0)
    flux = rise.mean(0)
    return t, flux, Bd, rise, edges


def detect_onsets(y, sr, min_ioi=0.15, rel=0.04, abs_db=0.7):
    """Times (s) of pluck onsets: peaks of the band-averaged dB flux above max(abs_db, rel * strongest peak,
    median + 8 MAD), at least min_ioi apart, not in the last 0.1 s.  Calibration: a pluck from silence gives
    ~35 dB; a pluck over one still-ringing string 2.9-7 dB (validate_reference.py); a pluck inside the full
    4-string model drone (render_strings.py) 0.76-2.1 dB; the chaotic jawari fluctuations of the model
    <= 0.59 dB.  The margin inside a dense drone is thin: check the detected onsets in the report and pass
    onsets=[...] to analyse() (CLI --onsets) if needed."""
    t, flux, Bd, rise, edges = onset_function(y, sr)
    if flux.size < 3:
        return np.array([0.0]), dict(t=t, flux=flux, thr=0.0)
    # (verify3 fix) broadband frame level, and two guards for files that start with room noise (a 1 s pre-roll,
    # as REFERENCE.md asks for): (1) the first frames are compared with the STFT's zero padding, so stationary
    # noise at the file start looked like a pluck at t = 0 (any SNR <= ~60 dB); keep a start candidate only if the
    # file starts loud (within 15 dB of its level peak over the first 3 s: a pluck at the start, or a recording
    # that starts mid-drone).  (2) noise flux in a quiet pre-roll (40-60 dB SNR) passed the 0.7 dB threshold;
    # a candidate that rises out of near-silence must raise the level to >= 15 dB above the quiet floor.
    hop_s = float(t[1] - t[0]) if t.size > 1 else 256.0 / sr
    lev = _db(np.sum(10 ** (Bd / 10), axis=0))
    start = t < 2048.0 / sr                                  # frames whose window reaches into the padding
    if start.any() and lev[start].max() < lev[t < 3.0].max() - 15.0:
        flux = flux.copy()
        flux[start] = 0.0
    floor_lev = float(np.percentile(lev, 2))
    quiet_file = floor_lev < lev.max() - 30.0
    med = np.median(flux)
    mad = 1.4826 * np.median(np.abs(flux - med)) + 1e-9
    pk, _ = ss.find_peaks(np.concatenate([[0.0], flux, [0.0]]))
    pk = pk - 1
    if pk.size == 0:
        return np.array([0.0]), dict(t=t, flux=flux, thr=0.0)
    thr = max(abs_db, rel * flux[pk].max(), med + 8 * mad)
    pk = pk[(flux[pk] >= thr) & (t[pk] < y.size / sr - 0.1)]
    if quiet_file and pk.size:
        nb, na = max(1, int(round(0.012 / hop_s))), max(2, int(round(0.05 / hop_s)))
        keep_q = []
        for k in pk:
            before = lev[max(0, k - 4 * nb):max(1, k - nb)].max() if k >= nb else floor_lev
            after = float(np.mean(lev[k:k + na]))
            keep_q.append(not (before < floor_lev + 10.0 and after < floor_lev + 15.0))
        pk = pk[np.array(keep_q, bool)]
    if pk.size == 0:
        return np.array([0.0]), dict(t=t, flux=flux, thr=thr)
    keep = []
    for k in pk:                                     # greedy: strongest within min_ioi wins
        if keep and t[k] - t[keep[-1]] < min_ioi:
            if flux[k] > flux[keep[-1]]:
                keep[-1] = k
            continue
        keep.append(k)
    # the STFT frame time is the window centre; the flux peaks when the onset is in the leading half of the
    # window -> report a quarter window earlier (refine_onset searches -50 ms .. +150 ms around it).
    # bands: per onset, the 3 adjacent bands with the largest summed rise (where the NEW string's energy is)
    keep = np.array(keep, int)
    bands = []
    for k in keep:
        r3 = np.convolve(rise[:, k], np.ones(3), mode="valid")
        b = int(np.argmax(r3))
        bands.append((float(edges[b]), float(edges[b + 3])))
    return np.maximum(t[keep] - 0.25 * (2048 / sr), 0.0), dict(t=t, flux=flux, thr=thr, bands=bands, rise=rise,
                                                             edges=edges)


def _onset_band(odet, t0):
    """Band (3 adjacent onset bands) with the largest rise in the flux frame of largest flux within
    [t0 - 30 ms, t0 + 90 ms] (for a manually given onset)."""
    t, fl = odet["t"], odet["flux"]
    m = np.where((t >= t0 - 0.03) & (t <= t0 + 0.09))[0]
    if m.size == 0 or "rise" not in odet:
        return None
    k = m[int(np.argmax(fl[m]))]
    r3 = np.convolve(odet["rise"][:, k], np.ones(3), mode="valid")
    b = int(np.argmax(r3))
    return float(odet["edges"][b]), float(odet["edges"][b + 3])


def refine_onset(y, sr, t_coarse, f0, onset_db=-6.0, start_db=-40.0, attack_win=0.04, band=None):
    """Sample-accurate onset of the pluck near t_coarse.

    Envelope: zero-phase high-pass at max(100 Hz, 1.5 f0) (removes the model's slow finger pull and room
    rumble), squared, 1 ms moving average.
      START  first sample in [t_coarse - 50 ms, t_coarse + 150 ms] whose CAUSAL envelope (same high-pass,
             0.25 ms average) exceeds max(window max + start_db, pre-onset floor + 10 dB);
      ATTACK PEAK  envelope maximum within attack_win (40 ms) after START (the initial transient, not the
             later jawari bloom, which can be louder in the high-passed band);
      ANCHOR first crossing of (attack peak + onset_db) between START and the attack peak, linearly interpolated
             (fractional sample).  t = 0 of every feature is the ANCHOR.
    Rise time: 10 % -> 90 % of the attack-peak amplitude of the broadband (> 30 Hz) envelope, smoothed over a
    quarter period, peak within max(40 ms, 3 periods) of START.
    band=(lo, hi): use a band-pass instead of the high-pass (later plucks over strongly ringing strings: the band
    where the onset flux rose most, i.e. the new string's own partials)."""
    fc = max(100.0, 1.5 * f0)
    if band is not None:
        sos = ss.butter(3, [band[0], min(band[1], 0.45 * sr)], "bandpass", fs=sr, output="sos")
        fc = band
    else:
        sos = ss.butter(4, fc, "highpass", fs=sr, output="sos")
    e = ss.sosfiltfilt(sos, y)
    env = _movavg(e * e, max(1, int(round(1e-3 * sr))))
    n = y.size
    i_lo = max(0, int((t_coarse - 0.05) * sr))
    i_hi = min(n, int((t_coarse + 0.15) * sr))
    if i_hi <= i_lo + 2:
        i_lo, i_hi = 0, n
    # background (strings still ringing from earlier plucks, or noise): median power just before the window,
    # subtracted so that only the NEW pluck's power is thresholded (no-op for a pluck from silence)
    b0, b1 = max(0, i_lo - int(0.06 * sr)), max(0, i_lo - int(0.01 * sr))
    bg = float(np.median(env[b0:b1])) if b1 - b0 > int(0.01 * sr) else 0.0
    # residual fluctuation of the background (beats of the ringing strings) after the subtraction
    bg_res = float(np.percentile(np.abs(env[b0:b1] - bg), 95)) if bg > 0 else 0.0
    env = np.maximum(env - bg, 1e-3 * bg + 1e-300)
    edb = _db(env)
    wmax = float(edb[i_lo:i_hi].max())
    pre = edb[max(0, i_lo - int(0.5 * sr)):max(0, i_lo - int(0.005 * sr))]
    floor_db = float(np.percentile(pre, 20)) if pre.size > int(0.02 * sr) else -np.inf
    if bg > 0:
        floor_db = float(_db(bg_res))   # the background is subtracted: its residual fluctuation is the floor
    lim = max(wmax + start_db, floor_db + 10.0)
    above = np.where(edb[i_lo:i_hi] > lim)[0]
    j = i_lo + int(above[0]) if above.size else i_lo
    # START from a CAUSAL envelope (the zero-phase one pre-rings ~10 ms before a sharp onset)
    ec = ss.sosfilt(sos, y[max(0, i_lo - int(0.5 * sr)):i_hi])
    off = max(0, i_lo - int(0.5 * sr))
    envc = _movavg(ec * ec, max(1, int(round(0.25e-3 * sr))))
    bgc = float(np.median(envc[b0 - off:b1 - off])) if b1 - b0 > int(0.01 * sr) else 0.0
    bgc_res = float(np.percentile(np.abs(envc[b0 - off:b1 - off] - bgc), 95)) if bgc > 0 else 0.0
    envc = _db(np.maximum(envc - bgc, 1e-3 * bgc + 1e-300))
    pre_c = envc[:max(1, i_lo - off - int(0.005 * sr))]
    floor_c = float(np.percentile(pre_c, 20)) if pre_c.size > int(0.02 * sr) else -np.inf
    if bgc > 0:
        floor_c = float(_db(bgc_res))
    lim_c = max(float(envc[i_lo - off:].max()) + start_db, floor_c + 10.0)
    above_c = np.where(envc[i_lo - off:] > lim_c)[0]
    j_c = i_lo + int(above_c[0]) if above_c.size else j
    k1 = min(n, j + int(attack_win * sr))
    ipk = j + int(np.argmax(env[j:k1]))
    peak_db = float(edb[ipk])
    thr = peak_db + onset_db
    k = j + int(np.argmax(edb[j:ipk + 1] >= thr))
    if k > 0 and edb[k - 1] < thr <= edb[k]:
        anchor = k - 1 + (thr - edb[k - 1]) / max(edb[k] - edb[k - 1], 1e-12)
    else:
        anchor = float(k)
    # broadband rise time on an envelope smoothed over a quarter period (a 1 ms envelope of a 69 Hz string is
    # dominated by the waveform's peaks within each period)
    sos2 = ss.butter(2, 30.0, "highpass", fs=sr, output="sos")
    e2 = ss.sosfiltfilt(sos2, y)
    a2 = np.sqrt(_movavg(e2 * e2, max(1, int(round(0.25 * sr / f0)))))
    w0 = max(0, j - int(0.005 * sr))
    w1 = min(n, j + int(max(attack_win, 3.0 / f0) * sr))
    seg = a2[w0:w1]
    if seg.size > 3:
        ip = int(np.argmax(seg))
        pk = seg[ip]
        i10 = int(np.argmax(seg[:ip + 1] >= 0.1 * pk))
        i90 = int(np.argmax(seg[:ip + 1] >= 0.9 * pk))
        rise = (i90 - i10) / sr
        t_peak = (w0 + ip - anchor) / sr
    else:
        rise, t_peak = np.nan, np.nan
    return dict(anchor=float(anchor), t_anchor=float(anchor / sr), start=int(j_c), t_start=float(j_c / sr),
                hp_hz=fc, background_db=float(_db(bg)) if bg > 0 else -np.inf, attack_peak_db=peak_db, window_max_db=wmax, floor_db=float(floor_db),
                rise_10_90=float(rise), t_amp_peak=float(t_peak))


# ======================================================================================== spectrum / partials
def long_spectrum(y, sr, t0, t1, pad=8):
    """Blackman-Harris zero-padded amplitude spectrum (dB re a unit-amplitude sinusoid)."""
    i0, i1 = max(0, int(t0 * sr)), min(y.size, int(t1 * sr))
    seg = y[i0:i1]
    if seg.size < 16:
        raise ValueError("spectrum window too short")
    w = ss.windows.blackmanharris(seg.size)
    nfft = _next_pow2(seg.size * pad)
    X = np.fft.rfft((seg - seg.mean()) * w, nfft)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    return f, 20.0 * np.log10(np.abs(X) / (0.5 * w.sum()) + 1e-300)


def _pick(f, Sdb, fp, half):
    """Largest peak of Sdb within fp +- half; parabolic interpolation.  Returns (freq, level_dB, local_floor_dB)."""
    df = f[1] - f[0]
    lo = max(1, int((fp - half) / df))
    hi = min(f.size - 2, int((fp + half) / df) + 1)
    if hi <= lo + 2:
        return np.nan, np.nan, np.nan
    k = lo + int(np.argmax(Sdb[lo:hi]))
    a, b, c = Sdb[k - 1], Sdb[k], Sdb[k + 1]
    den = a - 2 * b + c
    d = 0.5 * (a - c) / den if den < 0 else 0.0
    d = float(np.clip(d, -0.5, 0.5))
    lo2 = max(0, int((fp - 2 * half) / df))
    hi2 = min(f.size, int((fp + 2 * half) / df) + 1)
    floor = float(np.percentile(Sdb[lo2:hi2], 25))
    return float(f[k] + d * df), float(b - 0.25 * (a - c) * d), floor


def stretched(n, f0, B):
    n = np.asarray(n, float)
    return n * f0 * np.sqrt(1.0 + B * n * n)


def fit_inharmonicity(n, fn, w=None, f0_init=None, B_init=1e-4):
    """Robust fit of f_n = n f0 sqrt(1 + B n^2) in cents (soft-L1).  Returns (f0, B, residual cents)."""
    n = np.asarray(n, float)
    fn = np.asarray(fn, float)
    w = np.ones_like(n) if w is None else np.asarray(w, float)
    ok = np.isfinite(fn) & (w > 0)
    n, fn, w = n[ok], fn[ok], w[ok]
    if n.size < 3:
        f0 = float(np.median(fn / n)) if n.size else np.nan
        return f0, 0.0, np.full(n.size, np.nan)
    f0_init = float(np.median(fn[:3] / n[:3])) if f0_init is None else f0_init
    sw = np.sqrt(w / w.max())

    def res(p):
        return sw * 1200.0 * np.log2(fn / stretched(n, p[0], max(p[1], 0.0) * 1e-4))

    r = least_squares(res, [f0_init, float(np.clip(B_init * 1e4, 0.0, 99.0))], loss="soft_l1", f_scale=3.0,
                      bounds=([0.5 * f0_init, 0.0], [2.0 * f0_init, 100.0]))
    f0, B = float(r.x[0]), float(r.x[1] * 1e-4)
    return f0, B, 1200.0 * np.log2(fn / stretched(n, f0, B))


def estimate_f0(y, sr, t0, t1, fmin=40.0, fmax=600.0):
    """Hint-free f0: harmonic sum of the long-window log spectrum over 8 partials, sub-harmonic-safe."""
    f, S = long_spectrum(y, sr, t0, t1, pad=4)
    S = np.maximum(S, S.max() - 80.0)
    cand = np.geomspace(fmin, fmax, 2000)
    score = np.zeros_like(cand)
    for k in range(1, 9):
        score += np.interp(k * cand, f, S, right=S.min()) / k ** 0.5
    best = cand[int(np.argmax(score))]
    # prefer the octave above when it scores nearly as well (sub-harmonic errors)
    for _ in range(2):
        up = 2 * best
        if up <= fmax:
            s_up = sum(np.interp(k * up, f, S) / k ** 0.5 for k in range(1, 9))
            s_b = sum(np.interp(k * best, f, S) / k ** 0.5 for k in range(1, 9))
            if s_up > s_b - 3.0:
                best = up
    return float(best)


def track_partials(y, sr, f0_hint, t0, t1, f_max=10000.0, n_max=100, snr_min=10.0):
    """Partial frequencies from the long-window spectrum, picked iteratively along the fitted stretch curve.
    Returns dict(n, f (nan = not found), level_db, snr_db, f0, B, resid_cents, f1 (partial 1 itself))."""
    f, S = long_spectrum(y, sr, t0, t1, pad=8)
    # 1. refine f0 with B = 0 on partials 1..6 (grid search +- 60 cents)
    grid = f0_hint * 2.0 ** (np.linspace(-60, 60, 1201) / 1200.0)
    sc = np.zeros_like(grid)
    for k in range(1, 7):
        sc += np.interp(k * grid, f, S)
    f0 = float(grid[int(np.argmax(sc))])
    B = 0.0
    nmax = int(min(n_max, np.floor(min(f_max, 0.45 * sr) / f0)))
    fr = np.full(nmax, np.nan)
    lv = np.full(nmax, np.nan)
    snr = np.full(nmax, np.nan)
    last_fit = 0
    for n in range(1, nmax + 1):
        fp = stretched(n, f0, B)
        if fp > min(f_max, 0.45 * sr):
            break
        half = 0.3 * f0 * np.sqrt(1 + 3 * B * n * n)
        fk, lk, fl = _pick(f, S, fp, half)
        fr[n - 1], lv[n - 1], snr[n - 1] = fk, lk, lk - fl
        if n - last_fit >= 4 and n >= 6:
            ok = np.isfinite(fr[:n]) & (snr[:n] >= snr_min)
            if ok.sum() >= 4:
                nn = np.arange(1, n + 1)[ok]
                f0, B, _ = fit_inharmonicity(nn, fr[:n][ok], w=np.minimum(snr[:n][ok], 40.0), f0_init=f0,
                                             B_init=B if B else 1e-4)
                last_fit = n
    nn = np.arange(1, nmax + 1)
    good = np.isfinite(fr) & (snr >= snr_min)
    f0, B, res_c = fit_inharmonicity(nn[good], fr[good], w=np.minimum(snr[good], 40.0), f0_init=f0,
                                     B_init=max(B, 1e-6))
    resid = np.full(nmax, np.nan)
    resid[good] = res_c
    # B from low and high partials separately (jawari contact can pull high partials towards harmonic)
    sub = {}
    for lab, sel in (("lo", good & (nn <= 12)), ("hi", good & (nn > 12))):
        if sel.sum() >= 4:
            sub[lab] = fit_inharmonicity(nn[sel], fr[sel], w=np.minimum(snr[sel], 40.0), f0_init=f0,
                                         B_init=max(B, 1e-6))[:2]
    f0_lo, B_lo = sub.get("lo", (f0, B))
    # keep only partials whose (measured or predicted) frequency is inside the analysis band
    lim = min(f_max, 0.45 * sr)
    nv = int(np.sum(np.where(np.isfinite(fr), fr, stretched(nn, f0, B)) < lim))
    nv = max(nv, 3)
    nn, fr, lv, snr, good, resid = nn[:nv], fr[:nv], lv[:nv], snr[:nv], good[:nv], resid[:nv]
    return dict(n=nn, f=fr, level_db=lv, snr_db=snr, good=good, f0=f0_lo, B=B_lo, f0_all=f0, B_all=B,
                resid_cents=resid, f1=float(fr[0]) if np.isfinite(fr[0]) and snr[0] >= snr_min else np.nan,
                B_lo=B_lo, B_hi=sub.get("hi", (np.nan, np.nan))[1], window=(t0, t1))


def band_edges(fc, f0):
    """Lower/upper edges of the per-partial bands (midpoints between neighbouring centres)."""
    fc = np.asarray(fc, float)
    mid = 0.5 * (fc[1:] + fc[:-1])
    lo = np.concatenate([[max(0.5 * fc[0], fc[0] - 0.5 * f0)], mid])
    hi = np.concatenate([mid, [fc[-1] + 0.5 * (fc[-1] - fc[-2]) if fc.size > 1 else fc[0] + 0.5 * f0]])
    return lo, hi


def filterbank(y, sr, fc, f0, tw_frac=0.15, guard_s=0.5):
    """Frequency-domain heterodyne filter bank.  One band per centre fc[n]; band edges midway between neighbours;
    power-complementary sin/cos cross-fades of width tw_frac*f0 at each edge (the bands tile the spectrum, so the
    band powers add up to the signal power).  Each band is shifted to baseband and decimated.
    Returns (t (s, from the first sample of y), z (n_bands, n_t) complex amplitudes: |z| = A for a sinusoid
    A cos(2 pi f t) inside the band), fs_d."""
    fc = np.asarray(fc, float)
    n = y.size
    nfft = _next_pow2(n + int(guard_s * sr))
    X = np.fft.rfft(y, nfft)
    df = sr / nfft
    lo, hi = band_edges(fc, f0)
    tw = tw_frac * f0
    width = (hi - lo).max() + tw
    Nd = _next_pow2(int(np.ceil(4.0 * width / df)))
    fs_d = Nd * df
    m_valid = int(np.ceil(n * fs_d / sr))
    z = np.zeros((fc.size, m_valid), complex)
    for b in range(fc.size):
        k0 = max(1, int(np.floor((lo[b] - tw / 2) / df)))
        k1 = min(X.size - 1, int(np.ceil((hi[b] + tw / 2) / df)))
        k = np.arange(k0, k1 + 1)
        fk = k * df
        g = np.ones(k.size)
        # lower edge: rises from 0 to 1 over [lo - tw/2, lo + tw/2] (sin), upper edge: cos
        u = np.clip((fk - (lo[b] - tw / 2)) / tw, 0, 1)
        g *= np.sin(0.5 * np.pi * u)
        if b == 0:
            g[:] = np.where(fk < lo[b] - tw / 2, 0.0, np.where(fk < lo[b] + tw / 2, np.sin(0.5 * np.pi * u), 1.0))
        u2 = np.clip((fk - (hi[b] - tw / 2)) / tw, 0, 1)
        g *= np.cos(0.5 * np.pi * u2)
        Zb = np.zeros(Nd, complex)
        kc = int(round(fc[b] / df))
        idx = (k - kc) % Nd
        Zb[idx] = X[k] * g
        zb = np.fft.ifft(Zb) * (2.0 * Nd / nfft)
        # the band was shifted by kc*df: multiply back the residual of the true centre so the phase refers to fc[b]
        z[b] = zb[:m_valid] * np.exp(-2j * np.pi * (fc[b] - kc * df) * np.arange(m_valid) / fs_d)
    t = np.arange(m_valid) / fs_d
    return t, z, fs_d


def interharmonic_floor(y, sr, fc, f0, t0, t1, nper=8192):
    """Per-band noise power in |z|^2 units (as the filter bank: |z|^2 = A^2 for A cos) from the median Welch PSD
    of the bins of each band that are > 0.3 f0 away from its partial, over [t0, t1] s of y.  (verify3)"""
    i0, i1 = max(0, int(t0 * sr)), min(y.size, int(t1 * sr))
    if i1 - i0 < 2 * nper:
        return None
    f, Pw = ss.welch(y[i0:i1], sr, nperseg=nper)
    lo, hi = band_edges(fc, f0)
    out = np.full(np.asarray(fc).size, np.nan)
    for b in range(out.size):
        sel = (f >= lo[b]) & (f < hi[b]) & (np.abs(f - fc[b]) > 0.3 * f0)
        if sel.sum() >= 3:
            out[b] = 2.0 * np.median(Pw[sel]) * (hi[b] - lo[b])
    return out


# ======================================================================================== the analysis
def analyse(src, sr=None, f0=None, string=None, name=None, pluck=0, settings=None, verbose=False, onsets=None):
    """Analyse ONE pluck of a recording or render.

    src     path to a WAV/FLAC (mono or stereo) or an array ((N,) or (N, 2); then give sr).
    f0      nominal pitch hint (Hz).  Default: STRINGS[string] if string is given, else estimated.
    string  'kharaj' | 'pa' | 'sa' (sets the f0 hint and the labels).
    pluck   index of the pluck to analyse if the file contains several (iTanpura drone): the analysis stops
            at the next detected pluck.
    onsets  optional list of pluck times (s) replacing the automatic detection (each is still refined).
    Returns a dict of features (arrays as numpy; summary(A) gives the JSON-able scalars)."""
    st = _settings(settings)
    x, sr, nm = load(src, sr)
    name = name or nm
    y_all = x.mean(axis=1)
    if not np.any(y_all):
        raise ValueError("silent input")
    # ---- onsets
    onsets_override = onsets
    onsets, odet = detect_onsets(y_all, sr, min_ioi=st["min_ioi"], abs_db=st["onset_abs_db"])
    if onsets_override is not None and len(onsets_override):
        onsets = np.sort(np.asarray(onsets_override, float))
        odet = dict(odet, manual=True, bands=[_onset_band(odet, float(x)) for x in onsets])
    if pluck >= len(onsets):
        raise ValueError(f"pluck index {pluck} but only {len(onsets)} onset(s) detected")
    t_coarse = float(onsets[pluck])
    nominal = f0 if f0 is not None else (STRINGS.get(string) if string else None)
    f0_hint = nominal
    if f0_hint is None:
        f0_hint = estimate_f0(y_all, sr, t_coarse + 0.1, min(y_all.size / sr, t_coarse + 2.0))
        nominal = f0_hint
    on = refine_onset(y_all, sr, t_coarse, f0_hint, onset_db=st["onset_db"])
    # (verify3 fix) quality of a refinement = how far the attack rises above the background's residual
    # fluctuation (window max - floor).  Over ringing strings the broadband refinement can be off by -66..+23 ms
    # with a margin < 10 dB while the old switch (background > window max - 15 dB) kept it; the band refinement
    # is within +-2 ms at margins >= 19 dB, +12 ms at 15 dB and fails (-61 ms) at 8 dB (verify3 tests).
    qual = lambda o: o["window_max_db"] - o["floor_db"]               # noqa: E731
    if (on["background_db"] > on["window_max_db"] - 15.0 or qual(on) < 15.0) and odet.get("bands") \
            and odet["bands"][pluck]:
        # the new pluck hardly rises above the strings still ringing in the broad band: refine in its own band
        onb = refine_onset(y_all, sr, t_coarse, f0_hint, onset_db=st["onset_db"], band=odet["bands"][pluck])
        onb["refined_in_band"] = True
        if qual(onb) >= qual(on) or on["background_db"] > on["window_max_db"] - 15.0:
            on = onb
    on["margin_db"] = float(qual(on))
    if on["margin_db"] < 12.0:
        # neither refinement resolves the attack: the onset-flux time is 11-24 ms early in these cases (mean 17)
        a_ = (t_coarse + (0.0 if onsets_override is not None and len(onsets_override) else 0.017)) * sr
        on.update(anchor=float(a_), t_anchor=float(a_ / sr), start=int(a_), t_start=float(int(a_) / sr),
                  rise_10_90=np.nan, t_amp_peak=np.nan,
                  quality=f"coarse (+-10 ms): the attack rises only {on['margin_db']:.0f} dB above the ringing "
                          "strings' fluctuation; check it or pass --onsets")
    elif on["margin_db"] < 18.0:
        on["quality"] = (f"approximate (+-12 ms): the attack rises {on['margin_db']:.0f} dB above the ringing "
                         "strings' fluctuation")
    t_on = on["t_anchor"]
    next_on = [float(t) for t in onsets if t_on + st["min_ioi"] < t < y_all.size / sr - 0.1] if st["truncate"] else []
    t_next = next_on[0] if next_on else None
    t_end = y_all.size / sr
    if t_next is not None:
        t_end = min(t_end, t_next - st["guard"])
    t_end = min(t_end, t_on + st["max_dur"])
    if t_end - t_on < 0.5:
        raise ValueError(f"only {t_end - t_on:.2f} s until the next pluck / end of file: too short to analyse")
    i0 = max(0, int(round((t_on - st["pre"]) * sr)))
    i1 = int(round(t_end * sr))
    seg = x[i0:i1].copy()
    # half-Hann fades: 20 ms at the start of the pre-roll, 50 ms at the end (a cut at the next pluck or the end
    # of the file would otherwise leak into every band); the last 0.1 s is dropped from the time grid below
    nf0 = min(int(0.02 * sr), seg.shape[0] // 4)
    nf1 = min(int(0.05 * sr), seg.shape[0] // 4)
    seg[:nf0] *= np.sin(0.5 * np.pi * np.arange(nf0) / nf0)[:, None] ** 2
    seg[seg.shape[0] - nf1:] *= np.cos(0.5 * np.pi * np.arange(nf1) / nf1)[:, None] ** 2
    y = seg.mean(axis=1)
    t_off = i0 / sr - t_on                       # time of seg[0] relative to the anchor (<= 0)
    # pre-onset noise (for floors): up to 0.5 s ending 5 ms before the onset start
    j1 = max(0, on["start"] - int(0.005 * sr))
    noise = y_all[max(0, j1 - int(0.5 * sr)):j1]
    # ---- level normalisation: power of the first second after the onset = 0 dB
    g1 = y[int((-t_off) * sr):int((-t_off + 1.0) * sr)]
    norm = 1.0 / max(np.sqrt(np.mean(g1 ** 2)), 1e-300)
    y = y * norm
    seg = seg * norm
    noise = noise * norm
    # ---- f0, partials, B
    w0, w1 = st["spec_win"]
    w1 = min(w1, t_end - t_on)
    w0 = min(w0, 0.5 * w1)
    tp = track_partials(y, sr, f0_hint, w0 - t_off, w1 - t_off, f_max=st["f_max"], n_max=st["n_max"])
    f0m = tp["f0"]
    N = tp["n"].size
    fc = np.where(tp["good"], tp["f"], stretched(tp["n"], tp["f0_all"], tp["B_all"]))
    # keep the centres monotone and inside (0, nyquist)
    for k in range(1, N):
        if fc[k] <= fc[k - 1] + 0.5 * f0m:
            fc[k] = max(stretched(k + 1, tp["f0_all"], tp["B_all"]), fc[k - 1] + 0.9 * f0m)
    B = tp["B"]
    # ---- filter bank -> partial envelopes on the common time grid
    tz, z, fs_d = filterbank(y, sr, fc, f0m)
    P_raw = np.abs(z) ** 2
    ks = max(1, int(round(st["smooth"] * fs_d)))
    Ps = _movavg(P_raw, ks, axis=1)
    tg = np.arange(0.0, (t_end - t_on) - 0.1, st["hop"])
    tz_rel = tz + t_off
    P = np.stack([np.interp(tg, tz_rel, Ps[b]) for b in range(N)])
    # instantaneous frequency per band (power-weighted over 50 ms), on the grid
    dph = np.angle(z[:, 1:] * np.conj(z[:, :-1])) * fs_d / (2 * np.pi)
    wgt = P_raw[:, 1:]
    kw = max(1, int(round(0.05 * fs_d)))
    IF = fc[:, None] + _movavg(dph * wgt, kw, axis=1) / np.maximum(_movavg(wgt, kw, axis=1), 1e-300)
    IFg = np.stack([np.interp(tg, tz_rel[1:], IF[b]) for b in range(N)])
    # noise floor per band: from the pre-onset noise if there is enough of it, else from the envelope minimum
    Pmax = P.max()
    floor = np.full(N, Pmax * 10 ** (-st["dyn_range"] / 10))
    if noise.size > int(0.1 * sr):
        tzn, zn, _ = filterbank(noise, sr, fc, f0m)
        pn = np.median(np.abs(zn[:, int(0.05 * zn.shape[1]):]) ** 2, axis=1)
        floor = np.maximum(floor, pn)
        floor_src = "pre-onset noise"
    else:
        # without pre-roll noise the floor is unknown.  (verify3 fix: before, only the dynamic-range limit was
        # applied, so a recording trimmed to start at the pluck let the late decay fits run into the noise:
        # median T60 error +1475 % at 60 dB SNR.)  Noise fills the spectrum BETWEEN the partials and is stationary;
        # the leakage of a decaying partial is neither.  Per band: the median Welch PSD between the partials over
        # the last 1 s and the 1 s before it; where the two agree within 2 dB, it is taken as the band's floor.
        # On clean renders this floor stays >= ~40 dB below the partial (window leakage), so the fits are unchanged.
        floor_src = "dynamic range only (no pre-onset noise)"
        tl = (y.size / sr) - 0.1
        if tl - 2.0 >= 1.5:
            f_late = interharmonic_floor(y, sr, fc, f0m, tl - 1.0, tl)
            f_prev = interharmonic_floor(y, sr, fc, f0m, tl - 2.0, tl - 1.0)
            if f_late is not None and f_prev is not None:
                stat = np.isfinite(f_late) & np.isfinite(f_prev) & (np.abs(_db(f_late) - _db(f_prev)) < 2.0)
                raised = stat & (f_late > floor)
                floor = np.where(raised, np.maximum(floor, f_late), floor)
                if raised.sum() >= max(4, 0.5 * N):
                    # most bands show a stationary floor: broadband noise.  Bands whose partial was still fading
                    # into it fail the stationarity test; give them the floor interpolated over log f (dB / Hz)
                    dens = _db(f_late[raised] / np.diff(np.stack(band_edges(fc, f0m)), axis=0)[0][raised])
                    lf = np.log(fc)
                    di = np.interp(lf, lf[raised], ss.medfilt(dens, 5) if dens.size >= 5 else dens)
                    fi = 10 ** (di / 10) * np.diff(np.stack(band_edges(fc, f0m)), axis=0)[0]
                    floor = np.maximum(floor, np.where(raised, floor, fi))
                if raised.any():
                    floor_src = (f"no pre-onset noise: stationary floor between the partials in {int(raised.sum())} "
                                 f"of {N} bands (last 2 s), else dynamic range")
    Pdb = _db(P)
    floor_db = _db(floor)
    # partials whose pre-onset level (previous plucks still ringing, noise) is within 20 dB of their own peak:
    # their decay and comb values are unreliable (the floor limits every fit to floor + margin anyway)
    contaminated = np.where(Pdb.max(1) - floor_db < 20.0)[0] + 1
    if floor_src == "pre-onset noise":
        # refit f0 / B without partials whose pre-onset level is within 15 dB of their own peak (another string
        # ringing at the same frequency -- Sa = 2x, Pa = 1.5x kharaj -- biases their frequencies)
        clean = (Pdb.max(1) - floor_db) >= 15.0
        g2 = tp["good"] & clean
        nn = tp["n"]
        if (g2 & (nn <= 12)).sum() >= 4 and (~clean[:12]).any():
            w2 = np.minimum(tp["snr_db"][g2], 40.0)
            f0a, Ba, _ = fit_inharmonicity(nn[g2], tp["f"][g2], w=w2, f0_init=tp["f0"], B_init=max(tp["B"], 1e-6))
            sel = g2 & (nn <= 12)
            f0l, Bl, _ = fit_inharmonicity(nn[sel], tp["f"][sel], w=np.minimum(tp["snr_db"][sel], 40.0),
                                           f0_init=tp["f0"], B_init=max(tp["B"], 1e-6))
            tp.update(f0=f0l, B=Bl, B_lo=Bl, f0_all=f0a, B_all=Ba, refit_excluded=[int(k) for k in nn[~clean]])
            f0m, B = f0l, Bl
    # ---- the pieces
    f0tr = f0_track(tg, IFg, Pdb, floor_db, tp, nominal)
    dec = decays(tg, Pdb, floor_db, margin=st["floor_margin"])
    early = early_levels(tg, Pdb, st["early"])
    comb = pluck_comb(early, floor_db, f0m)
    jaw = jawari(tg, P, floor, f0m, fc, st)
    att = attack(y, sr, -t_off, f0m, fc, B, on)
    lt = ltas(seg, sr, -t_off, min(t_end - t_on, 8.0), fc, P, tg, f0m)
    stereo = stereo_features(seg, sr, -t_off, min(t_end - t_on, 4.0)) if seg.shape[1] == 2 else None
    room = room_tail(y_all * norm, sr, t_on, (y_all.size / sr) if t_next is None else t_next)
    A = dict(name=name, sr=sr, string=string, nominal_f0=nominal, f0_hint=f0_hint, settings=st,
             n_channels=x.shape[1], file_dur=x.shape[0] / sr, onsets=onsets, onset=on, t_onset=t_on,
             t_next=t_next, t_end=t_end, dur=t_end - t_on, truncated_by_next=t_next is not None and
             t_next - st["guard"] <= t_on + st["max_dur"], norm_gain_db=20 * np.log10(norm),
             partials=tp, n=tp["n"], fc=fc, t=tg, P=P, Pdb=Pdb, IF=IFg, floor_db=floor_db, floor_src=floor_src,
             contaminated=contaminated,
             f0=f0tr, decay=dec, early=early, comb=comb, jawari=jaw, attack=att, ltas=lt, stereo=stereo, room=room,
             y=y, seg_t0=t_off, onset_det=odet)
    if verbose:
        print(text_summary(A))
    return A


# ======================================================================================== feature blocks
def f0_track(tg, IFg, Pdb, floor_db, tp, nominal, n_use=(4, 12)):
    """f0(t): power-weighted MEDIAN over partials n_use[0]..n_use[1] of IF_n(t) / (n sqrt(1 + B n^2)) (B from the
    low-partial fit).  Partials 1-3 are left out: phone microphones roll them off and body / room modes near them
    pull their instantaneous frequency during the attack.  (verify3 fix: with partials 2-8 the glide of ONE kharaj
    render read -3.3 / +6.3 / +0.2 c through the three body presets and -0.9 c dry, a 9.6 c spread from the body
    alone, and +2.1 c for a synthetic +4.9 c glide; partials 4-12 give a 1.6 c spread and +4.8 c.)  Partials > 40 c off the fitted f0, or < 20 dB above their
    floor, are ignored at that time.  glide = mean(30-150 ms) - mean(0.5-1.5 s); drift = robust slope after 1 s."""
    f0m, B = tp["f0"], tp["B"]
    a, b = n_use
    b = min(b, IFg.shape[0])
    nn = np.arange(a, b + 1)
    est = IFg[a - 1:b] / stretched(nn, 1.0, B)[:, None]
    snr = Pdb[a - 1:b] - floor_db[a - 1:b, None]
    dev = np.abs(1200 * np.log2(np.maximum(est, 1e-9) / f0m))
    w = np.where((snr > 20.0) & (dev < 40.0), 10 ** (Pdb[a - 1:b] / 20), 0.0)
    f0t = np.full(est.shape[1], np.nan)
    order = np.argsort(est, axis=0)
    es = np.take_along_axis(est, order, 0)
    ws = np.take_along_axis(w, order, 0)
    cw = np.cumsum(ws, 0)
    tot = cw[-1]
    ok = tot > 0
    idx = np.argmax(cw >= 0.5 * tot[None, :], axis=0)
    f0t[ok] = es[idx[ok], np.where(ok)[0]]
    c = cents(f0t, nominal)
    sm = _nan_smooth(c, 10)

    def mean_c(t0, t1):
        sel = (tg >= t0) & (tg < t1) & np.isfinite(sm)
        return float(np.mean(sm[sel])) if sel.any() else np.nan
    early_c = mean_c(0.03, 0.15)
    mid_c = mean_c(0.5, 1.5)
    late_sel = (tg >= 1.0) & np.isfinite(sm)
    drift = np.nan
    if late_sel.sum() > 20 and tg[late_sel][-1] - tg[late_sel][0] > 1.0:
        drift = robust_line(tg[late_sel], sm[late_sel])[1]
    return dict(t=tg, f0=f0t, cents=sm, f0_fit=f0m, f0_all=tp["f0_all"], f1=tp["f1"],
                cents_fit=float(cents(f0m, nominal)), cents_all=float(cents(tp["f0_all"], nominal)),
                cents_f1=float(cents(tp["f1"], nominal)) if np.isfinite(tp["f1"]) else np.nan,
                B=B, glide_cents=early_c - mid_c if np.isfinite(early_c) and np.isfinite(mid_c) else np.nan,
                early_cents=early_c, mid_cents=mid_c, drift_cents_per_s=float(drift))


def _nan_smooth(v, k):
    ok = np.isfinite(v)
    num = _movavg(np.where(ok, v, 0.0), k)
    den = _movavg(ok.astype(float), k)
    return np.where(den > 0.3, num / np.maximum(den, 1e-12), np.nan)


def decays(tg, Pdb, floor_db, margin=10.0, step=0.02):
    """Per-partial decay from the dB envelopes (Huber line fits on a `step` grid).
      early: [t_peak, t_peak + 0.5 s], stopping where the level is 15 dB below the peak   -> T60_early (EDT-like)
      (t_peak = time of the envelope maximum within the first 4 s)
      late : from t_a = min(t_peak + 1 s, the time the level is 15 dB below the peak) (>= t_peak + 0.2 s) to the
             last time the level is `margin` dB above the floor; needs >= 0.4 s -> T60_late.
    se_* = standard error of the T60 from the robust residual scale (beats and jawari fluctuations make it large);
    curv = slope of the first half of the late window / slope of the second half (> 1: the decay slows = two-stage).
    """
    N, nt = Pdb.shape
    hop = tg[1] - tg[0]
    k = max(1, int(round(step / hop)))
    t = tg[::k]
    E = Pdb[:, ::k]
    keys = ("T60_early", "T60_late", "se_early", "se_late", "t_peak", "L_peak", "t_valid", "fall_db", "curv",
            "late_t0", "late_t1", "T60_late_raw")
    out = {key: np.full(N, np.nan) for key in keys}

    def fit(tt, ee):
        a_, s_, r_ = robust_line(tt, ee)
        if not np.isfinite(s_) or s_ >= 0:
            return np.nan, np.nan
        se = r_ / np.sqrt(max(np.sum((tt - tt.mean()) ** 2), 1e-12))
        T = -60.0 / s_
        return T, T * se / abs(s_)

    for b in range(N):
        e = E[b]
        above = e > floor_db[b] + margin
        if not above.any():
            continue
        j_last = int(np.where(above)[0][-1])
        jp = int(np.argmax(e[:min(j_last, int(np.searchsorted(t, 4.0))) + 1]))
        # (verify3 fix) end of the decay = where the level drops below floor + margin and STAYS below for 0.5 s, not
        # the last sample above it: near the floor (30-40 dB SNR) one late noise excursion otherwise stretched the
        # late window over seconds of noise plateau (T60 of 60-440 s for partials whose true T60 is ~5 s)
        kq = max(2, int(round(0.5 / (t[1] - t[0])))) if t.size > 1 else 2
        bl = np.flatnonzero(~above[jp:j_last + 1])
        if bl.size:
            runs = np.split(bl, np.flatnonzero(np.diff(bl) > 1) + 1)
            for rr in runs:
                if rr.size >= kq:
                    j_last = jp + int(rr[0]) - 1
                    break
        pk = e[jp]
        out["t_peak"][b], out["L_peak"][b], out["t_valid"][b] = t[jp], pk, t[j_last]
        out["fall_db"][b] = pk - e[j_last]
        j = jp
        while j < j_last and e[j] > pk - 15.0 and t[j] - t[jp] < 0.5:
            j += 1
        if j - jp >= 4:
            out["T60_early"][b], out["se_early"][b] = fit(t[jp:j + 1], e[jp:j + 1])
        below = np.where(e[jp:j_last + 1] <= pk - 15.0)[0]
        ta = t[jp] + 1.0
        if below.size:
            ta = min(ta, t[jp + int(below[0])])
        ta = max(ta, t[jp] + 0.2)
        ja = int(np.searchsorted(t, ta))
        if ja < j_last and t[j_last] - t[ja] >= 0.4:
            tt, ee = t[ja:j_last + 1], e[ja:j_last + 1]
            out["T60_late"][b], out["se_late"][b] = fit(tt, ee)
            out["T60_late_raw"][b] = out["T60_late"][b]
            # (verify3 fix) a late fit with a standard error > 30 % of T60 (short window just above the noise
            # floor: 30-40 dB SNR recordings gave 1.5-4x too long T60s there) is not used; none of the model
            # renders' late fits (12 s, all strings) exceeds 30 %.  The raw value is kept in T60_late_raw.
            if np.isfinite(out["se_late"][b]) and out["se_late"][b] > 0.3 * out["T60_late"][b]:
                out["T60_late"][b] = np.nan
            out["late_t0"][b], out["late_t1"][b] = tt[0], tt[-1]
            h = tt.size // 2
            if h >= 5:
                s1 = robust_line(tt[:h], ee[:h])[1]
                s2 = robust_line(tt[h:], ee[h:])[1]
                if s1 < 0 and s2 < 0:
                    out["curv"][b] = s1 / s2
    return out


def early_levels(tg, Pdb, win):
    sel = (tg >= win[0]) & (tg <= win[1])
    if not sel.any():
        sel = tg <= win[1] + 0.05
    L = _db(np.mean(10 ** (Pdb[:, sel] / 10), axis=1))
    return dict(L=L, win=win)


def comb_shape(n, beta, depth_db=30.0):
    return 20 * np.log10(np.maximum(np.abs(np.sin(np.pi * np.asarray(n) * beta)), 10 ** (-depth_db / 20)))


def pluck_comb(early, floor_db, f0, n_fit=30, depths=(8.0, 12.0, 18.0, 30.0)):
    """Fit the early partial levels L_n = comb(beta, depth) + smooth trend (quadratic in log n), n = 1..n_fit.
    comb = 20 log10 max(|sin(n pi beta)|, 10^(-depth/20)); the notch depth is free because the jawari (and the
    body / room) partly fill the notches within the first periods.
    Returns beta in (0, 0.5] -- the pluck is at beta OR 1 - beta of the speaking length from the BRIDGE: the
    magnitude spectrum cannot tell which --, the depth, the rms residual with and without the comb (evidence_db;
    < ~1.5 dB = no clear comb), the notch list (>= 6 dB below both neighbours and the trend) and the excitation
    roll-off (trend slope, dB/oct, partials 4-16)."""
    L = early["L"]
    N = min(n_fit, L.size)
    n = np.arange(1, N + 1)
    ok = L[:N] > floor_db[:N] + 15.0
    base = dict(beta=np.nan, beta_alt=np.nan, depth_db=np.nan, evidence_db=np.nan, rms_comb=np.nan,
                rms_nocomb=np.nan, notches=[], n=n, L=L[:N], ok=ok, tilt_db_oct=np.nan, trend=np.full(N, np.nan),
                comb=np.full(N, np.nan), betas=np.array([]), rms_beta=np.array([]))
    if ok.sum() < 6:
        return base
    ln = np.log(n)
    Xb = np.stack([np.ones(N), ln, ln ** 2], 1)

    def fit(target):
        w = ok.astype(float)
        coef = np.linalg.lstsq(Xb[ok], target[ok], rcond=None)[0]
        for _ in range(8):                          # Huber reweighting
            r = target - Xb @ coef
            sc = 1.4826 * np.median(np.abs(r[ok])) + 0.5
            w = np.where(ok, np.minimum(1.0, 1.345 * sc / np.maximum(np.abs(r), 1e-12)), 0.0)
            W = np.sqrt(w)
            coef = np.linalg.lstsq(Xb * W[:, None], target * W, rcond=None)[0]
        r = target - Xb @ coef
        return coef, float(np.sqrt(np.sum(ok * r ** 2) / max(ok.sum(), 1))), r

    Ln = L[:N]
    c0, rms0, _ = fit(Ln)
    betas = np.arange(0.08, 0.5001, 0.0025)
    best = (np.inf, None, None)
    curves = {}
    for dep in depths:
        rms = np.array([fit(Ln - comb_shape(n, bb, dep))[1] for bb in betas])
        curves[dep] = rms
        jb = int(np.argmin(rms))
        if rms[jb] < best[0]:
            best = (rms[jb], betas[jb], dep)
    _, b0, dep = best
    fine = np.linspace(max(0.08, b0 - 0.0025), min(0.5, b0 + 0.0025), 21)
    rf = np.array([fit(Ln - comb_shape(n, bb, dep))[1] for bb in fine])
    beta = float(fine[int(np.argmin(rf))])
    coef, rms_c, r = fit(Ln - comb_shape(n, beta, dep))
    trend = Xb @ coef
    notches = [int(k) for k in range(2, N) if ok[k - 1] and Ln[k - 1] < min(Ln[k - 2], Ln[k]) - 6.0
               and Ln[k - 1] < trend[k - 1] - 6.0]
    na, nb = 4, min(16, N)
    poly = [coef[2], coef[1], coef[0]]
    tilt = (np.polyval(poly, np.log(nb)) - np.polyval(poly, np.log(na))) / np.log2(nb / na)
    base.update(beta=beta, beta_alt=1 - beta, depth_db=dep, rms_comb=rms_c, rms_nocomb=rms0,
                evidence_db=rms0 - rms_c, notches=notches, trend=trend, comb=comb_shape(n, beta, dep) + trend,
                tilt_db_oct=float(tilt), betas=betas, rms_beta=curves[dep])
    return base


def _smooth_partials(P, n, n_lo, oct_sigma=1.0 / 6.0):
    """Gaussian smoothing of partial powers along log2(n) (sigma = 1/6 octave ~ 1/3-octave band), n >= n_lo."""
    nb = n[n_lo - 1:]
    l2 = np.log2(nb)
    K = np.exp(-0.5 * ((l2[:, None] - l2[None, :]) / oct_sigma) ** 2)
    K /= K.sum(1, keepdims=True)
    return nb, K @ P[n_lo - 1:]


OCT_BANDS = ((1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127))


def jawari(tg, P, floor, f0, fc, st):
    """Pitch-relative jawari descriptors from the partial powers P[n, t] (linear, floor-subtracted).

    BI_k(t)    buzz index: 10 log10( sum_{n >= k} P_n / sum_{n < k} P_n ), k in st['buzz_k'] (50 ms smoothing)
    octaves    level of partial-octave bands (1, 2-3, 4-7, 8-15, 16-31, 32-63, 64-127) re the total, dB
    For the partial band n >= n_lo (n_lo = max(3, round(300 Hz / f0))), 100 ms smoothing:
      centre   loudness-weighted mean partial number (weights P_n^0.3, ~ specific loudness)
      peak     the most prominent local maximum (>= 3 dB prominence) of the level after 1/3-octave smoothing
               along n: the 'formant' hump (nan when the profile has no hump, e.g. a flat buzz plateau)
      upper    highest partial whose 1/3-octave-smoothed level is within 30 dB of that maximum (upper edge)
      e95      partial below which 95 % of the loudness weights lie
    Bloom (from the headline BI, 100 ms smoothing): start = mean over 20-60 ms; onset = first time it has risen by
    10 % of the bloom gain; peak time; plateau = total time within 3 dB of the max; persistence = last time within
    10 dB of the max."""
    N = P.shape[0]
    hop = tg[1] - tg[0]
    Pm = np.maximum(P - floor[:, None], 0.0)
    Ps = _movavg(Pm, max(1, int(round(0.05 / hop))), axis=1)
    P1 = _movavg(Pm, max(1, int(round(0.1 / hop))), axis=1)
    valid_t = Ps.sum(0) > 10 * floor.sum()
    out = dict(t=tg, k=list(st["buzz_k"]))
    for k in st["buzz_k"]:
        if k >= N:
            continue
        hi = Ps[k - 1:].sum(0)
        lo = Ps[:k - 1].sum(0)
        out[f"BI{k}"] = np.where(valid_t & (hi > 0), _db(hi) - _db(lo), np.nan)
    n = np.arange(1, N + 1)
    tot = P1.sum(0)
    octs = []
    for a, b in OCT_BANDS:
        if a > N:
            break
        octs.append(np.where(valid_t, _db(P1[a - 1:min(b, N)].sum(0)) - _db(tot), np.nan))
    out["octaves"] = np.array(octs)
    out["oct_bands"] = OCT_BANDS[:len(octs)]
    n_lo = int(max(3, round(300.0 / f0)))
    out["n_lo"] = n_lo
    band = P1[n_lo - 1:]
    nb = n[n_lo - 1:]
    wl = band ** 0.3
    W = wl.sum(0)
    cen = np.where(W > 0, (nb[:, None] * wl).sum(0) / np.maximum(W, 1e-300), np.nan)
    cum = np.cumsum(wl, axis=0)
    e95 = nb[np.argmax(cum >= 0.95 * cum[-1:], axis=0)].astype(float)
    nb2, Sm = _smooth_partials(P1, n, n_lo)
    Sdb = _db(Sm)
    pk = np.full(Sdb.shape[1], np.nan)
    for jj in range(Sdb.shape[1]):
        col = Sdb[:, jj]
        if not np.isfinite(col).all() or col.max() <= -290:
            continue
        ii, pr = ss.find_peaks(col, prominence=3.0)
        if ii.size:
            pk[jj] = nb2[ii[int(np.argmax(pr["prominences"]))]]
    top = Sdb.max(0)
    upper = np.array([nb2[np.where(Sdb[:, j] >= top[j] - 30.0)[0][-1]] for j in range(Sdb.shape[1])], float)
    for key, v in (("centre", cen), ("peak", pk), ("upper", upper), ("e95", e95)):
        v = np.where(valid_t & (W > 0), v, np.nan)
        out[key] = v
        out[key + "_hz"] = np.interp(v, n, fc, left=np.nan, right=np.nan)
    km = st["buzz_main"] if f"BI{st['buzz_main']}" in out else [k for k in st["buzz_k"] if f"BI{k}" in out][0]
    bi = _nan_smooth(out[f"BI{km}"], max(1, int(round(0.1 / hop))))
    out["BI_main"] = km
    out["BI_smooth"] = bi
    sel0 = (tg >= 0.02) & (tg <= 0.06) & np.isfinite(bi)
    b0 = float(np.nanmean(bi[sel0])) if sel0.any() else np.nan
    win = (tg <= 6.0) & np.isfinite(bi)
    bmax = gain = t_peak = t_on = plateau = persist = last_valid = np.nan
    if win.any() and np.isfinite(b0):
        jmax = int(np.argmax(np.where(win, bi, -np.inf)))
        bmax = float(bi[jmax])
        gain = bmax - b0
        t_peak = float(tg[jmax])
        if gain >= 3.0:
            up = np.where((tg <= t_peak) & (bi >= b0 + 0.1 * gain) & np.isfinite(bi))[0]
            t_on = float(tg[up[0]]) if up.size else np.nan
        plateau = float(np.sum(np.isfinite(bi) & (bi >= bmax - 3.0)) * hop)
        after = np.where(np.isfinite(bi) & (bi >= bmax - 10.0))[0]
        persist = float(tg[after[-1]]) if after.size else np.nan
        last_valid = float(tg[np.where(np.isfinite(bi))[0][-1]])
    out.update(BI_start=b0, BI_max=bmax, bloom_gain_db=gain, t_bloom_onset=t_on, t_bloom_peak=t_peak,
               plateau_s=plateau, persistence_s=persist, BI_valid_until=last_valid)
    return out


def attack(y, sr, i_on_s, f0, fc, B, on):
    """Attack descriptors.  y starts at seg_t0 (the anchor is at time i_on_s from y[0])."""
    a = int(round(i_on_s * sr))
    seg = lambda t0, t1: y[max(0, a + int(t0 * sr)):max(0, a + int(t1 * sr))]   # noqa: E731
    out = dict(rise_10_90_ms=1e3 * on["rise_10_90"], t_amp_peak_ms=1e3 * on["t_amp_peak"])
    # transient HF burst: energy > max(4 kHz, 40 f0) in the first 20 ms vs 20-200 ms (per unit time)
    fh = min(max(4000.0, 40 * f0), 0.45 * sr)
    sos = ss.butter(4, fh, "highpass", fs=sr, output="sos")
    h = ss.sosfiltfilt(sos, y)
    e_a = np.mean(h[max(0, a - int(0.002 * sr)):a + int(0.020 * sr)] ** 2)
    e_b = np.mean(h[a + int(0.020 * sr):a + int(0.200 * sr)] ** 2)
    out["hf_burst_db"] = float(_db(e_a) - _db(e_b))
    out["hf_burst_hz"] = fh
    # aperiodicity of the first periods below 12 f0: residual of a one-period-delay prediction (gain-optimal)
    sos2 = ss.butter(4, min(12 * f0, 0.45 * sr), "lowpass", fs=sr, output="sos")
    yl = ss.sosfiltfilt(sos2, y)
    T0 = sr / f0

    def aper(t0, t1):
        i0, i1 = a + int(t0 * sr), a + int(t1 * sr)
        if i0 - int(T0) - 2 < 0 or i1 >= yl.size:
            return np.nan
        idx = np.arange(i0, i1)
        d = np.interp(idx - T0, np.arange(yl.size), yl)
        cur = yl[idx]
        g = np.dot(cur, d) / max(np.dot(d, d), 1e-300)
        r = cur - g * d
        return float(_db(np.mean(r ** 2)) - _db(np.mean(cur ** 2)))
    out["aperiodicity_early_db"] = aper(0.0, 0.03)          # first ~2 periods
    out["aperiodicity_late_db"] = aper(0.3, 0.6)
    # attack spectrum: centroid (in partials) of the first 50 ms vs 50-300 ms
    def centroid(t0, t1):
        s_ = seg(t0, t1)
        if s_.size < 64:
            return np.nan
        f, Pw = ss.periodogram(s_ * np.hanning(s_.size), sr)
        return float((f * Pw).sum() / max(Pw.sum(), 1e-300) / f0)
    out["centroid_attack_n"] = centroid(0.0, 0.05)
    out["centroid_after_n"] = centroid(0.05, 0.3)
    # broadband envelope (dB, 1 ms) from -20 ms to +300 ms for plots
    sos3 = ss.butter(2, 30.0, "highpass", fs=sr, output="sos")
    e3 = ss.sosfilt(sos3, y)
    env = _db(_movavg(e3 * e3, max(1, int(1e-3 * sr))))
    i0 = max(0, a - int(0.02 * sr))
    i1 = min(y.size, a + int(0.3 * sr))
    out["env_t"] = (np.arange(i0, i1) - a) / sr
    out["env_db"] = env[i0:i1] - env[i0:i1].max()
    return out


def third_octaves(fmin=40.0, fmax=16000.0):
    k = np.arange(np.floor(3 * np.log2(fmin / 1000.0)), np.ceil(3 * np.log2(fmax / 1000.0)) + 1)
    return 1000.0 * 2.0 ** (k / 3.0)


def ltas(seg, sr, t_on_s, dur, fc, P, tg, f0):
    """Welch LTAS (channels averaged) over [onset, onset + dur] and the smooth partial-peak envelope: time-mean
    partial power at fc, interpolated on 1/3-octave centres (log f) with 1/3-octave smoothing."""
    a = int(round(t_on_s * sr))
    b = min(seg.shape[0], a + int(dur * sr))
    s_ = seg[a:b]
    nper = min(1 << 15, _next_pow2(s_.shape[0]) // 2)
    f, Pw = ss.welch(s_, sr, nperseg=nper, axis=0)
    Pw = Pw.mean(axis=1)
    sel = tg <= dur
    Pm = P[:, sel].mean(1)
    Ldb = _db(Pm)
    fcen = third_octaves(max(30.0, 0.7 * fc[0]), min(16000.0, 1.2 * fc[-1]))
    env = np.full(fcen.size, np.nan)
    lf = np.log2(fc)
    for i, c in enumerate(fcen):
        w = np.exp(-0.5 * ((lf - np.log2(c)) / (1.0 / 6.0)) ** 2)       # ~1/3-octave Gaussian in log f
        if w.sum() > 1e-3:
            env[i] = _db(np.sum(w * Pm) / np.sum(w))
    # 1/3-octave band levels of the LTAS (energy sums) as the robust alternative
    band = np.array([_db(Pw[(f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))].sum() + 1e-300) for c in fcen])
    return dict(f=f, psd_db=_db(Pw), f_c=fcen, env_db=env, band_db=band, partial_mean_db=Ldb, dur=dur)


def stereo_features(seg, sr, t_on_s, dur):
    a = int(round(t_on_s * sr))
    b = min(seg.shape[0], a + int(dur * sr))
    L, R = seg[a:b, 0], seg[a:b, 1]
    f, C = ss.coherence(L, R, sr, nperseg=4096)
    octs = 125.0 * 2.0 ** np.arange(0, 7)
    corr = {}
    coh = {}
    for c in octs:
        lo, hi = c / np.sqrt(2), c * np.sqrt(2)
        if hi >= 0.45 * sr:
            continue
        sos = ss.butter(4, [lo, hi], "bandpass", fs=sr, output="sos")
        l_, r_ = ss.sosfilt(sos, L), ss.sosfilt(sos, R)
        corr[float(c)] = float(np.dot(l_, r_) / np.sqrt(np.dot(l_, l_) * np.dot(r_, r_) + 1e-300))
        sel = (f >= lo) & (f < hi)
        coh[float(c)] = float(np.mean(C[sel]))
    M, S = 0.5 * (L + R), 0.5 * (L - R)
    return dict(f=f, coherence=C, corr_oct=corr, coh_oct=coh,
                corr_broad=float(np.dot(L, R) / np.sqrt(np.dot(L, L) * np.dot(R, R) + 1e-300)),
                side_mid_db=float(_db(np.mean(S ** 2)) - _db(np.mean(M ** 2))),
                lr_level_db=float(_db(np.mean(L ** 2)) - _db(np.mean(R ** 2))))


def room_tail(y, sr, t_on, t_stop):
    """Room decay after the note is DAMPED inside the file: look for a drop of the 10 ms broadband envelope of
    > 20 dB within 150 ms after t_on + 1 s and before t_stop; fit the decay after the drop (T60 from -5 to -25 dB
    of the Schroeder curve).  Returns None when the file contains no damping (the usual case)."""
    i0 = int((t_on + 1.0) * sr)
    i1 = min(y.size, int(t_stop * sr))
    if i1 - i0 < int(0.5 * sr):
        return None
    env = _db(_movavg(y[i0:i1] ** 2, int(0.01 * sr)))[::int(0.005 * sr)]
    d = int(0.15 / 0.005)
    if env.size <= d + 10:
        return None
    drop = env[:-d] - env[d:]
    j = int(np.argmax(drop))
    if drop[j] < 20.0:
        return None
    k0 = i0 + (j + d) * int(0.005 * sr)
    tail = y[k0:i1]
    if tail.size < int(0.2 * sr):
        return None
    edc = _db(np.cumsum(tail[::-1] ** 2)[::-1])
    edc -= edc[0]
    tt = np.arange(edc.size) / sr
    sel = (edc <= -5) & (edc >= -25)
    if sel.sum() < 10:
        return dict(t_damp=k0 / sr - t_on, T60=np.nan)
    _, s, _ = robust_line(tt[sel], edc[sel])
    return dict(t_damp=k0 / sr - t_on, T60=float(-60 / s) if s < 0 else np.nan)


# ======================================================================================== summaries
def _group_median(v, n, groups):
    out = []
    for a, b in groups:
        sel = (n >= a) & (n <= b) & np.isfinite(v)
        out.append(float(np.median(v[sel])) if sel.sum() >= min(2, b - a + 1) else np.nan)
    return np.array(out)


def partial_groups(N):
    g = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 16), (17, 25), (26, 40), (41, 60), (61, 100)]
    return [(a, min(b, N)) for a, b in g if a <= N]


def summary(A):
    """JSON-able scalar summary of an analysis."""
    tp, f0, d, c, j, at = A["partials"], A["f0"], A["decay"], A["comb"], A["jawari"], A["attack"]
    N = A["n"].size
    G = partial_groups(N)
    ts = [t for t in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0) if t < A["dur"] - 0.05]

    def at_t(v, t):
        return float(v[int(np.argmin(np.abs(A["t"] - t)))])
    s = dict(name=A["name"], string=A["string"], sr=A["sr"], channels=A["n_channels"],
             onsets_s=[round(float(t), 4) for t in A["onsets"]], t_onset_s=round(A["t_onset"], 6),
             t_start_s=round(A["onset"]["t_start"], 6), analysed_s=round(A["dur"], 3),
             onset_margin_db=A["onset"].get("margin_db"), onset_quality=A["onset"].get("quality", "ok"),
             truncated_by_next_pluck=bool(A["truncated_by_next"]), n_partials=int(N),
             n_partials_found=int(tp["good"].sum()), floor=A["floor_src"],
             n_contaminated=int(len(A["contaminated"])), contaminated=[int(k) for k in A["contaminated"][:40]],
             nominal_f0=A["nominal_f0"], f0_fit=f0["f0_fit"], f1=f0["f1"], cents_fit=f0["cents_fit"],
             cents_f1=f0["cents_f1"], f0_all=f0["f0_all"], cents_all=f0["cents_all"], B=tp["B"], B_lo=tp["B_lo"],
             B_hi=tp["B_hi"], B_all=tp["B_all"],
             glide_cents=f0["glide_cents"], drift_cents_per_s=f0["drift_cents_per_s"],
             T60_late_groups={f"{a}-{b}": v for (a, b), v in zip(G, _group_median(d["T60_late"], A["n"], G))},
             T60_early_groups={f"{a}-{b}": v for (a, b), v in zip(G, _group_median(d["T60_early"], A["n"], G))},
             T60_late_se_groups={f"{a}-{b}": v for (a, b), v in zip(G, _group_median(d["se_late"], A["n"], G))},
             pluck_beta=c["beta"], pluck_beta_alt=c["beta_alt"], comb_evidence_db=c["evidence_db"],
             comb_depth_db=c["depth_db"],
             notches=c["notches"], excitation_tilt_db_oct=c.get("tilt_db_oct", np.nan),
             BI_main=j["BI_main"], BI_start=j["BI_start"], BI_max=j["BI_max"], bloom_gain_db=j["bloom_gain_db"],
             t_bloom_onset=j["t_bloom_onset"], t_bloom_peak=j["t_bloom_peak"], plateau_s=j["plateau_s"],
             persistence_s=j["persistence_s"],
             formant_centre={t: at_t(j["centre"], t) for t in ts}, formant_peak={t: at_t(j["peak"], t) for t in ts},
             formant_upper={t: at_t(j["upper"], t) for t in ts}, formant_e95={t: at_t(j["e95"], t) for t in ts},
             octave_levels={f"{a}-{b}": {t: at_t(j["octaves"][i], t) for t in ts}
                            for i, (a, b) in enumerate(j["oct_bands"])},
             BI_track={t: at_t(j["BI_smooth"], t) for t in ts},
             rise_10_90_ms=at["rise_10_90_ms"], hf_burst_db=at["hf_burst_db"],
             aperiodicity_early_db=at["aperiodicity_early_db"], aperiodicity_late_db=at["aperiodicity_late_db"],
             centroid_attack_n=at["centroid_attack_n"], centroid_after_n=at["centroid_after_n"])
    if A["stereo"] is not None:
        st = A["stereo"]
        s.update(stereo_corr=st["corr_broad"], stereo_side_mid_db=st["side_mid_db"], stereo_coh_oct=st["coh_oct"])
    if A["room"] is not None:
        s.update(room=A["room"])
    return _jsonable(s)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def _f(v, fmt="{:.2f}", na="n/a"):
    return na if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


def text_summary(A):
    s = summary(A)
    L = [f"== {s['name']} ({s['string'] or 'string ?'}; {s['channels']} ch, {s['sr']} Hz) ==",
         f"onset {s['t_onset_s']:.4f} s (start {s['t_start_s']:.4f} s); plucks detected at {s['onsets_s']}; "
         f"analysed {s['analysed_s']:.2f} s" + (" (cut at the next pluck)" if s["truncated_by_next_pluck"] else "")
         + ("" if s["onset_quality"] == "ok" else f"; ONSET {s['onset_quality']}"),
         f"noise floor: {s['floor']}; {s['n_contaminated']} partials within 20 dB of the floor"
         + (f" (first: {s['contaminated'][:12]})" if s["n_contaminated"] else ""),
         f"f0 {_f(s['f0_fit'], '{:.3f}')} Hz ({_f(s['cents_fit'], '{:+.1f}')} c re {s['nominal_f0']:.2f}; partials <= 12)"
         f", partial 1 {_f(s['f1'], '{:.3f}')} Hz, all partials {_f(s['f0_all'], '{:.3f}')} Hz; B {_f(s['B'], '{:.2e}')} "
         f"(partials <= 12; > 12: {_f(s['B_hi'], '{:.2e}')}; all: {_f(s['B_all'], '{:.2e}')}); glide {_f(s['glide_cents'], '{:+.1f}')} c (30-150 ms vs 0.5-1.5 s), drift "
         f"{_f(s['drift_cents_per_s'], '{:+.2f}')} c/s; {s['n_partials_found']}/{s['n_partials']} partials found",
         "T60 late  by partial group: " + ", ".join(f"{k}: {_f(v, '{:.2f}')}" for k, v in s["T60_late_groups"].items()),
         "T60 early by partial group: " + ", ".join(f"{k}: {_f(v, '{:.2f}')}" for k, v in s["T60_early_groups"].items()),
         f"pluck comb: beta {_f(s['pluck_beta'], '{:.3f}')} (or {_f(s['pluck_beta_alt'], '{:.3f}')}) of the length "
         f"from the bridge; notch depth {_f(s['comb_depth_db'], '{:.0f}')} dB; comb evidence "
         f"{_f(s['comb_evidence_db'], '{:.1f}')} dB; notches at {s['notches']}; "
         f"excitation tilt {_f(s['excitation_tilt_db_oct'], '{:+.1f}')} dB/oct (partials 4-16)",
         f"jawari: BI{s['BI_main']} start {_f(s['BI_start'], '{:.1f}')} dB, max {_f(s['BI_max'], '{:.1f}')} dB "
         f"(bloom {_f(s['bloom_gain_db'], '{:+.1f}')} dB), onset {_f(s['t_bloom_onset'], '{:.2f}')} s, peak "
         f"{_f(s['t_bloom_peak'], '{:.2f}')} s, plateau {_f(s['plateau_s'], '{:.2f}')} s, persists to "
         f"{_f(s['persistence_s'], '{:.2f}')} s (within 10 dB of max)",
         "  formant centre / peak / upper edge / e95 (partial no.) at " + "; ".join(
             f"{t}s: {_f(s['formant_centre'][t], '{:.1f}')}/{_f(s['formant_peak'][t], '{:.0f}')}/"
             f"{_f(s['formant_upper'][t], '{:.0f}')}/{_f(s['formant_e95'][t], '{:.0f}')}" for t in s["formant_centre"]),
         "  partial-octave levels re total (dB) at " + "; ".join(
             f"{t}s: " + "/".join(_f(v[t], '{:.0f}') for v in s["octave_levels"].values())
             for t in s["formant_centre"]) + f"  (bands {', '.join(s['octave_levels'])})",
         "  BI track: " + ", ".join(f"{t}s {_f(v, '{:.1f}')}" for t, v in s["BI_track"].items()),
         f"attack: rise {_f(s['rise_10_90_ms'], '{:.1f}')} ms; HF burst {_f(s['hf_burst_db'], '{:+.1f}')} dB; "
         f"aperiodicity first 30 ms {_f(s['aperiodicity_early_db'], '{:.1f}')} dB vs 0.3-0.6 s "
         f"{_f(s['aperiodicity_late_db'], '{:.1f}')} dB; centroid {_f(s['centroid_attack_n'], '{:.1f}')} f0 "
         f"(0-50 ms) -> {_f(s['centroid_after_n'], '{:.1f}')} f0 (50-300 ms)"]
    if "stereo_corr" in s:
        L.append(f"stereo: L/R corr {s['stereo_corr']:.2f}, side/mid {s['stereo_side_mid_db']:.1f} dB, coherence per "
                 "octave " + ", ".join(f"{float(k):.0f}: {v:.2f}" for k, v in s["stereo_coh_oct"].items()))
    if "room" in s:
        L.append(f"abrupt end (hand damping or a fade) at {_f(s['room'].get('t_damp'))} s: decay after it (room + "
                 f"body ringing) T60 {_f(s['room'].get('T60'))} s")
    return "\n".join(L)


# ======================================================================================== EQ (body colouring)
def eq_curve(A, M, t_max=4.0, oct_sigma=0.25):
    """Smooth EQ (dB per model partial) that maps the model's long-term partial levels onto the reference's:
    time-mean partial power over [0, t_max], difference ref - model, Gaussian-smoothed along log2(f)
    (sigma = oct_sigma octave), evaluated at the model's partials and at 1/3-octave centres.  Partials below
    their floor + margin in either signal are left out of the smoothing."""
    N = min(A["n"].size, M["n"].size)
    tm = min(t_max, A["t"][-1], M["t"][-1])
    sa, sm = A["t"] <= tm, M["t"] <= tm
    La = _db(A["P"][:N, sa].mean(1))
    Lm = _db(M["P"][:N, sm].mean(1))
    ok = (La > A["floor_db"][:N] + 10) & (Lm > M["floor_db"][:N] + 10)
    f = M["fc"][:N]
    d = La - Lm
    l2 = np.log2(f)

    def smooth_at(fq):
        w = np.exp(-0.5 * ((np.log2(fq)[:, None] - l2[None, ok]) / oct_sigma) ** 2)
        ws = w.sum(1)
        return np.where(ws > 0.05, (w * d[None, ok]).sum(1) / np.maximum(ws, 1e-12), np.nan)
    eq_n = smooth_at(f) if ok.sum() >= 3 else np.zeros(N)
    eq_n = np.where(np.isfinite(eq_n), eq_n, 0.0)
    f_c = third_octaves(max(30.0, 0.7 * f[0]), min(16000.0, 1.2 * f[-1]))
    return dict(n=np.arange(1, N + 1), f=f, eq_n=eq_n, raw=np.where(ok, d, np.nan), f_c=f_c, eq_c=smooth_at(f_c),
                t_max=tm, sigma_oct=oct_sigma)


def with_eq(M, eq_n, settings=None):
    """Copy of analysis M with the per-partial EQ (dB) applied to its partial powers; the level-dependent blocks
    (early levels, comb, jawari descriptors) are recomputed."""
    st = M["settings"] if settings is None else settings
    N = eq_n.size
    g = 10 ** (np.asarray(eq_n) / 10)[:, None]
    M2 = dict(M)
    M2["P"] = M["P"][:N] * g
    M2["Pdb"] = _db(M2["P"])
    M2["floor_db"] = M["floor_db"][:N] + eq_n
    M2["n"] = M["n"][:N]
    M2["fc"] = M["fc"][:N]
    fl = 10 ** (M2["floor_db"] / 10)
    M2["early"] = early_levels(M2["t"], M2["Pdb"], st["early"])
    M2["comb"] = pluck_comb(M2["early"], M2["floor_db"], M["f0"]["f0_fit"])
    M2["jawari"] = jawari(M2["t"], M2["P"], fl, M["f0"]["f0_fit"], M2["fc"], st)
    M2["eq_applied"] = np.asarray(eq_n)
    return M2


# ======================================================================================== comparison
TOL = dict(f0_cents=5.0, glide_cents=3.0, stretch_c20=10.0, t60_log2=np.log2(1.3), oct_db=3.0, bi_db=3.0,
           bloom_t_rel=0.3, bloom_t_min=0.1, bloom_gain_db=3.0, persist_rel=0.25, persist_min=0.5,
           formant_log2=0.25, beta=0.03, rise_log2=1.0, burst_db=6.0, aper_db=6.0, eq_db=6.0, stereo_corr=0.15,
           image_db=6.0)
EVAL_T = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0])


def _at(t, v, ts):
    """Mean of v over log-time bins around each evaluation time ts[i] (edges at the geometric means of
    neighbouring ts; the first bin starts at 0.8 ts[0]): the jawari tracks fluctuate chaotically, so point
    samples are not comparable, bin means are."""
    ts = np.asarray(ts, float)
    edges = np.concatenate([[0.8 * ts[0]], np.sqrt(ts[1:] * ts[:-1]), [ts[-1] * 1.2]])
    out = np.full(ts.size, np.nan)
    for i in range(ts.size):
        sel = (t >= edges[i]) & (t < edges[i + 1]) & np.isfinite(v)
        if sel.any():
            out[i] = float(np.mean(v[sel]))
    return out


def _stretch_c20(B):
    return 1200 * np.log2(np.sqrt(1 + B * 400.0)) if np.isfinite(B) else np.nan


def _hz(n, fc):
    return float(np.interp(n, np.arange(1, fc.size + 1), fc)) if np.isfinite(n) else np.nan


def compare_features(A, M, eq=True):
    """Feature-by-feature differences, model M vs reference A.  Returns (items, info): items = list of dicts
    (key, group, severity, text, ref, model); severity = |difference| / tolerance (TOL; ~1 = just noticeable,
    the tolerances are engineering choices, not measured JNDs).  Level-shape items (jawari, comb) are computed
    after matching the long-term spectral balance with a smooth EQ (eq=True); the EQ itself is one item."""
    items = []

    def add(key, group, sev, text, ref=None, model=None, fit=True):
        if sev is None or not np.isfinite(sev):
            return
        items.append(dict(key=key, group=group, severity=float(sev), text=text, ref=ref, model=model, fit=fit))

    E = eq_curve(A, M)
    N = E["n"].size
    Me = with_eq(M, E["eq_n"]) if eq else (with_eq(M, np.zeros(N)) if M["n"].size != N else M)
    if A["n"].size != N:               # descriptors over the same partials in both
        A = with_eq(A, np.zeros(N))
    dur = min(A["dur"], M["dur"]) - 0.15
    ts = EVAL_T[EVAL_T <= dur]
    fa, fm = A["f0"], M["f0"]
    # ---- pitch
    d = fm["cents_fit"] - fa["cents_fit"]
    add("f0", "pitch", abs(d) / TOL["f0_cents"],
        f"model is {abs(d):.1f} cents {'sharp' if d > 0 else 'flat'} (f0 {fm['f0_fit']:.3f} vs {fa['f0_fit']:.3f} Hz, "
        "partials <= 12)", fa["f0_fit"], fm["f0_fit"], fit=False)
    if np.isfinite(fa["glide_cents"]) and np.isfinite(fm["glide_cents"]):
        d = fm["glide_cents"] - fa["glide_cents"]
        add("glide", "pitch", abs(d) / TOL["glide_cents"],
            f"attack pitch glide {fa['glide_cents']:+.1f} c in the reference vs {fm['glide_cents']:+.1f} c in the model "
            "(30-150 ms re 0.5-1.5 s; + = starts sharp, as with tension modulation)", fa["glide_cents"],
            fm["glide_cents"])
    if np.isfinite(fa["drift_cents_per_s"]) and np.isfinite(fm["drift_cents_per_s"]):
        d = fm["drift_cents_per_s"] - fa["drift_cents_per_s"]
        add("drift", "pitch", abs(d) / 1.0,
            f"slow pitch drift {fa['drift_cents_per_s']:+.2f} c/s in the reference vs {fm['drift_cents_per_s']:+.2f} c/s",
            fa["drift_cents_per_s"], fm["drift_cents_per_s"])
    for lab, key in (("partials <= 12", "B_lo"), ("partials > 12", "B_hi")):
        ca, cm = _stretch_c20(A["partials"][key]), _stretch_c20(M["partials"][key])
        if np.isfinite(ca) and np.isfinite(cm):
            add(f"stretch_{key}", "pitch", abs(cm - ca) / TOL["stretch_c20"],
                f"inharmonicity of {lab}: B = {A['partials'][key]:.2e} (ref) vs {M['partials'][key]:.2e} (model), i.e. "
                f"partial 20 stretched {ca:+.0f} c vs {cm:+.0f} c" + (" (the jawari locks partials to harmonic)"
                                                                     if min(ca, cm) < 3 else ""), ca, cm)
    # ---- decay
    G = partial_groups(N)
    for key, lab in (("T60_late", "late"), ("T60_early", "early")):
        ga = _group_median(A["decay"][key][:N], A["n"][:N], G)
        gm = _group_median(M["decay"][key][:N], M["n"][:N], G)
        for (a, b), ra, rm in zip(G, ga, gm):
            if np.isfinite(ra) and np.isfinite(rm) and ra > 0 and rm > 0:
                r = rm / ra
                grp = f"{a}" if a == b else f"{a}-{b}"
                add(f"{key}_{grp}", "decay" if lab == "late" else "decay_early", abs(np.log2(r)) / TOL["t60_log2"],
                    f"model partial{'s' if a != b else ''} {grp} decay{'s' if a == b else ''} {max(r, 1 / r):.1f}x "
                    f"{'faster' if r < 1 else 'slower'} ({lab} T60 {rm:.2f} s vs {ra:.2f} s in the reference)", ra, rm)
    # ---- jawari (after EQ)
    ja, jm = A["jawari"], Me["jawari"]
    k = ja["BI_main"]
    if f"BI{k}" in jm:
        va = _at(A["t"], ja["BI_smooth"], ts)
        vm = _at(Me["t"], jm["BI_smooth"], ts)
        ok = np.isfinite(va) & np.isfinite(vm)
        if ok.sum() >= 3:
            dd = vm[ok] - va[ok]
            add("BI", "jawari", np.sqrt(np.mean(dd ** 2)) / TOL["bi_db"],
                f"jawari buzz index BI{k} (partials >= {k} vs below) is {abs(dd.mean()):.1f} dB "
                f"{'stronger' if dd.mean() > 0 else 'weaker'} in the model on average ({ts[ok][0]:g}-{ts[ok][-1]:g} s, "
                f"rms difference {np.sqrt(np.mean(dd ** 2)):.1f} dB)", va.tolist(), vm.tolist())
    for i, (a, b) in enumerate(ja["oct_bands"]):
        if i >= len(jm["oct_bands"]) or b > N + 30:
            continue
        va = _at(A["t"], ja["octaves"][i], ts)
        vm = _at(Me["t"], jm["octaves"][i], ts)
        ok = np.isfinite(va) & np.isfinite(vm) & (va > -60) & (vm > -60)
        if ok.sum() >= 3:
            dd = vm[ok] - va[ok]
            rms = float(np.sqrt(np.mean(dd ** 2)))
            # where is the difference largest?
            j = int(np.argmax(np.abs(dd)))
            grp = f"{a}" if a == b else f"{a}-{min(b, N)}"
            add(f"oct_{grp}", "jawari", rms / TOL["oct_db"],
                f"partial{'s' if a != b else ''} {grp} (share of the whole tone): model {dd.mean():+.1f} dB on "
                f"average, largest {dd[j]:+.1f} dB at {ts[ok][j]:g} s", va.tolist(), vm.tolist())
    for key, lab in (("t_bloom_peak", "peaks"), ("t_bloom_onset", "starts to bloom")):
        a_, m_ = ja[key], jm[key]
        if np.isfinite(a_) and np.isfinite(m_):
            tol = max(TOL["bloom_t_min"], TOL["bloom_t_rel"] * a_)
            add(key, "jawari", abs(m_ - a_) / tol,
                f"jawari {lab} {abs(m_ - a_):.2f} s {'later' if m_ > a_ else 'earlier'} in the model "
                f"({m_:.2f} s vs {a_:.2f} s; BI{k})", a_, m_)
        elif np.isfinite(a_) != np.isfinite(m_) and key == "t_bloom_onset":
            who = "reference" if np.isfinite(a_) else "model"
            add(key, "jawari", 2.0, f"only the {who} shows a jawari bloom (BI{k} rising >= 3 dB after the attack)",
                a_, m_)
    if np.isfinite(ja["bloom_gain_db"]) and np.isfinite(jm["bloom_gain_db"]):
        d = jm["bloom_gain_db"] - ja["bloom_gain_db"]
        add("bloom_gain", "jawari", abs(d) / TOL["bloom_gain_db"],
            f"jawari bloom (BI{k} rise after the attack) {jm['bloom_gain_db']:+.1f} dB in the model vs "
            f"{ja['bloom_gain_db']:+.1f} dB", ja["bloom_gain_db"], jm["bloom_gain_db"])
    if np.isfinite(ja["persistence_s"]) and np.isfinite(jm["persistence_s"]):
        lim = min(A["dur"], M["dur"]) - 0.2
        a_, m_ = min(ja["persistence_s"], lim), min(jm["persistence_s"], lim)
        tol = max(TOL["persist_min"], TOL["persist_rel"] * a_)
        add("persistence", "jawari", abs(m_ - a_) / tol,
            f"the jawari lasts {abs(m_ - a_):.1f} s {'longer' if m_ > a_ else 'shorter'} in the model (BI{k} within "
            f"10 dB of its maximum until {m_:.1f} s vs {a_:.1f} s{' = end of analysis' if max(a_, m_) >= lim else ''})",
            a_, m_)
    for key, lab in (("centre", "centred"), ("e95", "extends (95 % of loudness)"), ("upper", "upper edge")):
        va = _at(A["t"], ja[key], ts)
        vm = _at(Me["t"], jm[key], ts)
        ok = np.isfinite(va) & np.isfinite(vm) & (va > 0) & (vm > 0)
        if ok.sum() >= 3:
            dd = np.log2(vm[ok] / va[ok])
            na_, nm_ = float(np.median(va[ok])), float(np.median(vm[ok]))
            # 'upper' jumps when the profile is a flat plateau near the -30 dB threshold: report only
            add(f"formant_{key}", "jawari", np.sqrt(np.mean(dd ** 2)) / TOL["formant_log2"],
                f"buzz band {lab}: partial {nm_:.0f} in the model vs {na_:.0f} in the reference (median "
                f"{_hz(nm_, M['fc']):.0f} vs {_hz(na_, A['fc']):.0f} Hz; rms log2 ratio over time "
                f"{np.sqrt(np.mean(dd ** 2)):.2f})", va.tolist(), vm.tolist(), fit=(key != "upper"))
    # ---- pluck comb
    ca, cm = A["comb"], Me["comb"]
    ev_a = ca["evidence_db"] if np.isfinite(ca["evidence_db"]) else 0.0
    ev_m = cm["evidence_db"] if np.isfinite(cm["evidence_db"]) else 0.0
    if ev_a >= 1.5 and ev_m >= 1.5:
        d = cm["beta"] - ca["beta"]
        add("pluck_beta", "pluck", abs(d) / TOL["beta"],
            f"pluck comb: reference notches at {ca['notches']} -> pluck at {ca['beta']:.3f} (or {1 - ca['beta']:.3f}) "
            f"of the length from the bridge; model {cm['beta']:.3f} (notches {cm['notches']})", ca["beta"], cm["beta"])
    elif ev_a >= 1.5 > ev_m:
        add("pluck_beta", "pluck", 2.0,
            f"reference shows a pluck comb (notches at {ca['notches']} -> pluck at {ca['beta']:.3f} or "
            f"{1 - ca['beta']:.3f} of the length from the bridge) that the model's jawari fills in", ca["beta"], np.nan)
    elif ev_m >= 1.5 > ev_a:
        add("pluck_beta", "pluck", 2.0,
            f"model shows a pluck comb (notches at {cm['notches']}, beta {cm['beta']:.3f}) that the reference lacks "
            "(filled by the jawari, body or room in the reference)", np.nan, cm["beta"])
    # ---- attack
    aa, am = A["attack"], M["attack"]
    if np.isfinite(aa["rise_10_90_ms"]) and np.isfinite(am["rise_10_90_ms"]) and min(aa["rise_10_90_ms"],
                                                                                         am["rise_10_90_ms"]) > 0:
        r = am["rise_10_90_ms"] / aa["rise_10_90_ms"]
        add("rise", "attack", abs(np.log2(r)) / TOL["rise_log2"],
            f"attack rise time (10-90 %) {am['rise_10_90_ms']:.1f} ms in the model vs {aa['rise_10_90_ms']:.1f} ms",
            aa["rise_10_90_ms"], am["rise_10_90_ms"])
    d = am["hf_burst_db"] - aa["hf_burst_db"]
    add("hf_burst", "attack", abs(d) / TOL["burst_db"],
        f"attack HF burst (> {aa['hf_burst_hz']:.0f} Hz, first 20 ms re 20-200 ms): {aa['hf_burst_db']:+.1f} dB in "
        f"the reference vs {am['hf_burst_db']:+.1f} dB in the model" +
        (" -> the reference has a click / finger-noise transient" if d < -TOL["burst_db"] else ""),
        aa["hf_burst_db"], am["hf_burst_db"])
    if np.isfinite(aa["aperiodicity_early_db"]) and np.isfinite(am["aperiodicity_early_db"]):
        d = am["aperiodicity_early_db"] - aa["aperiodicity_early_db"]
        add("aperiodicity", "attack", abs(d) / TOL["aper_db"],
            f"aperiodicity of the first 30 ms (< 12 f0): {aa['aperiodicity_early_db']:.1f} dB (ref) vs "
            f"{am['aperiodicity_early_db']:.1f} dB (model)", aa["aperiodicity_early_db"], am["aperiodicity_early_db"])
    # ---- colour (EQ)
    ec = E["eq_c"]
    okc = np.isfinite(ec)
    if okc.sum() >= 3:
        e0 = ec - np.nanmedian(ec[okc])
        j1 = int(np.nanargmax(np.where(okc, e0, -np.inf)))
        j2 = int(np.nanargmin(np.where(okc, e0, np.inf)))
        add("eq", "colour", max(abs(e0[j1]), abs(e0[j2])) / TOL["eq_db"],
            f"long-term colour (smooth EQ ref/model, re its median): {e0[j1]:+.0f} dB around {E['f_c'][j1]:.0f} Hz, "
            f"{e0[j2]:+.0f} dB around {E['f_c'][j2]:.0f} Hz (body / microphone, or overall jawari strength)",
            None, None, fit=False)
    # ---- stereo
    if A["stereo"] is not None and M["stereo"] is not None:
        d = M["stereo"]["corr_broad"] - A["stereo"]["corr_broad"]
        add("stereo", "stereo", abs(d) / TOL["stereo_corr"],
            f"L/R correlation {M['stereo']['corr_broad']:.2f} in the model vs {A['stereo']['corr_broad']:.2f}",
            A["stereo"]["corr_broad"], M["stereo"]["corr_broad"], fit=False)
    elif A["stereo"] is not None:
        add("stereo", "stereo", 0.5, f"reference is stereo (L/R correlation {A['stereo']['corr_broad']:.2f}); "
            "the model is mono (render with body.apply_body(..., stereo=True))", fit=False)
    # ---- whole partial-envelope image (after EQ), dB clipped 70 dB below each max
    img = image_distance(A, Me, ts)
    add("image", "overall", img["rms_db"] / TOL["image_db"],
        f"partial-envelope image rms difference {img['rms_db']:.1f} dB (after EQ; partials 1-{img['N']}, "
        f"{ts[0]:g}-{ts[-1]:g} s, 70 dB range)", fit=False)
    items.sort(key=lambda d_: -d_["severity"])
    return items, dict(eq=E, model_eq=Me, eval_t=ts, image=img)


def image_distance(A, M, ts, rng=70.0):
    N = min(A["n"].size, M["n"].size)
    Ia = _db(np.stack([_at(A["t"], A["P"][b], ts) for b in range(N)]))
    Im = _db(np.stack([_at(M["t"], M["P"][b], ts) for b in range(N)]))
    ra = Ia.max() - rng
    rm = Im.max() - rng
    Ia = np.maximum(Ia - Ia.max(), -rng)
    Im = np.maximum(Im - Im.max(), -rng)
    D = Im - Ia
    w = ((Ia > -rng) | (Im > -rng)).astype(float)
    _ = ra, rm
    return dict(N=N, D=D, rms_db=float(np.sqrt((w * D ** 2).sum() / max(w.sum(), 1))))


def feature_distance(items, weights=None, cap=5.0):
    """Scalar fit loss: sum of capped squared severities of the fit-relevant items, by group weight."""
    # early decays of high partials fluctuate between 'twin' renders (1e-12 parameter changes; chaotic contact):
    # severities up to ~1.7, so they get half weight; every other item stayed < 0.1 between twins
    W = dict(pitch=0.5, decay=1.0, decay_early=0.5, jawari=1.0, pluck=0.5, attack=0.5, overall=0.0, colour=0.0,
             stereo=0.0)
    if weights:
        W.update(weights)
    tot = 0.0
    per = {}
    for it in items:
        if not it["fit"]:
            continue
        v = W.get(it["group"], 1.0) * min(it["severity"], cap) ** 2
        per[it["key"]] = v
        tot += v
    return tot, per


def compare(ref, model, sr=None, model_sr=None, string=None, f0=None, name=None, model_label="model", out=True,
            plots=True, settings=None, pluck=0, eq=True, onsets=None):
    """Analyse a reference and a model render the same way and report the differences.

    ref, model  paths or arrays (then sr / model_sr), or analysis dicts from analyse().
    Writes report.txt, features.json and figures to out/reference/<name>/ (out=True) or to the directory `out`.
    Returns dict(ref=A, model=M, items=ranked mismatches, text=report, dir=output dir)."""
    A = ref if isinstance(ref, dict) else analyse(ref, sr, f0=f0, string=string, settings=settings, pluck=pluck,
                                                  onsets=onsets)
    st_m = dict(settings or {})
    st_m["max_dur"] = min(st_m.get("max_dur", DEFAULTS["max_dur"]), A["dur"] + 0.1)
    M = model if isinstance(model, dict) else analyse(model, model_sr or sr, f0=A["nominal_f0"],
                                                      string=string or A["string"], settings=st_m,
                                                      name=None if isinstance(model, (str, Path)) else model_label)
    items, info = compare_features(A, M, eq=eq)
    loss, per = feature_distance(items)
    name = name or f"{A['name']}__vs__{M['name']}"
    lines = [f"==== reference {A['name']}  vs  {model_label} {M['name']}  ({time.strftime('%Y-%m-%d %H:%M')}) ====",
             "", "Biggest mismatches (severity = difference / tolerance; ~1 just noticeable, > 3 large):"]
    for i, it in enumerate(items[:12]):
        lines.append(f"  {i + 1:2d}. [{it['group']:7s} {it['severity']:5.1f}] {it['text']}")
    lines += ["", f"feature distance (fit loss, capped): {loss:.1f}", "", "-- reference --", text_summary(A), "",
              f"-- {model_label} --", text_summary(M), "", "-- all items --"]
    lines += [f"  [{it['group']:7s} {it['severity']:5.1f}] {it['text']}" for it in items]
    text = "\n".join(lines)
    d = None
    if out:
        d = Path(out) if isinstance(out, (str, Path)) else OUT / _safe(name)
        d.mkdir(parents=True, exist_ok=True)
        (d / "report.txt").write_text(text + "\n")
        (d / "features.json").write_text(json.dumps(dict(reference=summary(A), model=summary(M), loss=loss,
                                                         items=_jsonable(items),
                                                         eq_third_octave=_jsonable(dict(f=info["eq"]["f_c"],
                                                                                        db=info["eq"]["eq_c"]))),
                                                    indent=1))
        if plots:
            plot_all(A, M, info, d, model_label=model_label)
    return dict(ref=A, model=M, items=items, info=info, loss=loss, loss_items=per, text=text, dir=d)


def _safe(s):
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s))[:120]


def analyse_report(src, sr=None, string=None, f0=None, name=None, pluck=0, settings=None, out=True, plots=True,
                   onsets=None):
    """analyse() + report.txt / features.json / figures for a single file (no model)."""
    A = analyse(src, sr, f0=f0, string=string, name=name, pluck=pluck, settings=settings, onsets=onsets)
    d = None
    if out:
        d = Path(out) if isinstance(out, (str, Path)) else OUT / _safe(A["name"])
        d.mkdir(parents=True, exist_ok=True)
        (d / "report.txt").write_text(text_summary(A) + "\n")
        (d / "features.json").write_text(json.dumps(summary(A), indent=1))
        if plots:
            plot_all(A, None, None, d)
    return A, d


# ======================================================================================== figures
C_REF, C_MOD = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
                         "legend.frameon": False})
    seq = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#b9d3f2", "#2a78d6", "#0d2f5c"])
    div = LinearSegmentedColormap.from_list("div", [C_MOD, "#f3c3ad", "#e8e8e6", "#b9d3f2", C_REF])
    return plt, seq, div


def plot_all(A, M, info, d, model_label="model"):
    plt, seq, div = _mpl()
    both = [(A, "reference", C_REF)] + ([(M, model_label, C_MOD)] if M is not None else [])
    d = Path(d)
    # 1. spectrograms
    fig, axs = plt.subplots(1, len(both), figsize=(6.5 * len(both), 4.2), squeeze=False, sharey=True)
    for ax, (X, lab, _) in zip(axs[0], both):
        sr = X["sr"]
        f, t, Z = ss.stft(X["y"], sr, nperseg=4096, noverlap=4096 - 512)
        fmax = min(0.45 * sr, 8000.0)
        sel = f <= fmax
        S = _db(np.abs(Z[sel]) ** 2)
        ax.pcolormesh(t + X["seg_t0"], f[sel], S - S.max(), vmin=-80, vmax=0, cmap=seq, shading="auto",
                      rasterized=True)
        ax.set_title(f"{lab}: {X['name']}", color=INK)
        ax.set_xlabel("time after onset (s)")
        ax.grid(False)
    axs[0, 0].set_ylabel("Hz")
    fig.tight_layout()
    fig.savefig(d / "spectrograms.png", dpi=110)
    plt.close(fig)
    # 2. partial-envelope images (+ difference)
    ncol = len(both) + (1 if M is not None else 0)
    fig, axs = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.4), squeeze=False, sharey=True)
    for ax, (X, lab, _) in zip(axs[0], both):
        if M is not None and X is M:
            X = info["model_eq"]
            lab += " (+ EQ to the reference's colour)"
        Pd = X["Pdb"] - X["Pdb"].max()
        N = Pd.shape[0]
        ax.pcolormesh(X["t"], np.arange(1, N + 1), Pd, vmin=-70, vmax=0, cmap=seq, shading="auto", rasterized=True)
        j = X["jawari"]
        ax.plot(X["t"], j["centre"], color=INK, lw=1.0, label="centre (loudness-weighted)")
        ax.plot(X["t"], j["e95"], color=INK2, lw=0.8, ls="--", label="95 % of loudness")
        ax.set_title(lab, color=INK)
        ax.set_xlabel("time after onset (s)")
        ax.grid(False)
        ax.legend(loc="upper right", fontsize=7)
    axs[0, 0].set_ylabel("partial number")
    if M is not None:
        img = info["image"]
        ts = info["eval_t"]
        ax = axs[0, -1]
        lim = 20
        ax.pcolormesh(np.arange(ts.size + 1), np.arange(1, img["N"] + 2) - 0.5, img["D"], vmin=-lim, vmax=lim,
                      cmap=div.reversed(), shading="flat", rasterized=True)
        ax.set_xticks(np.arange(ts.size) + 0.5)
        ax.set_xticklabels([f"{x:g}" for x in ts], rotation=60, fontsize=7)
        ax.set_title(f"model - reference (dB; orange = model louder)\nrms {img['rms_db']:.1f} dB", color=INK)
        ax.set_xlabel("time after onset (s, sampled)")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(d / "partial_envelopes.png", dpi=110)
    plt.close(fig)
    # 3. decay
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, key, title in ((axs[0], "T60_late", "late decay (T60)"), (axs[1], "T60_early", "early decay (EDT-like T60)")):
        for X, lab, c in both:
            v = X["decay"][key]
            ax.semilogy(X["n"], v, "o", ms=3, color=c, alpha=0.5)
            G = partial_groups(X["n"].size)
            gm = _group_median(v, X["n"], G)
            ax.semilogy([0.5 * (a + b) for a, b in G], gm, "-", color=c, label=f"{lab} (group median)")
        ax.set_xscale("log")
        ax.set_xlabel("partial number")
        ax.set_title(title, color=INK)
        ax.legend()
    axs[0].set_ylabel("s")
    fig.tight_layout()
    fig.savefig(d / "decay.png", dpi=110)
    plt.close(fig)
    # 4. pitch
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for X, lab, c in both:
        f0 = X["f0"]
        axs[0].plot(f0["t"], f0["cents"], color=c, label=f"{lab}: f0 {f0['f0_fit']:.3f} Hz")
        tp = X["partials"]
        dev = 1200 * np.log2(tp["f"] / (tp["n"] * tp["f0"]))
        axs[1].plot(tp["n"][tp["good"]], dev[tp["good"]], ".", color=c, ms=4,
                    label=f"{lab}: B {tp['B']:.1e} (n<=12), {tp['B_hi']:.1e} (n>12)")
        nn = np.arange(1, tp["n"].size + 1)
        axs[1].plot(nn, 1200 * np.log2(np.sqrt(1 + tp["B"] * nn ** 2)), "-", color=c, lw=0.8)
    devs = np.concatenate([1200 * np.log2(X["partials"]["f"][X["partials"]["good"]] /
                                          (X["partials"]["n"][X["partials"]["good"]] * X["partials"]["f0"]))
                           for X, _, _ in both])
    if devs.size:
        lo_, hi_ = np.percentile(devs, [1, 99])
        axs[1].set_ylim(min(lo_, -5) - 5, max(hi_, 5) + 5)
    axs[0].set_xscale("symlog", linthresh=0.5)
    axs[0].set_xlabel("time after onset (s)")
    axs[0].set_ylabel(f"cents re {A['nominal_f0']:.2f} Hz")
    axs[0].set_title("f0 track (partials 4-12, weighted median)", color=INK)
    axs[0].legend()
    axs[1].set_xlabel("partial number")
    axs[1].set_ylabel("cents re n f0")
    axs[1].set_title("partial frequencies (line: stretch from the n <= 12 fit)", color=INK)
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "pitch.png", dpi=110)
    plt.close(fig)
    # 5. pluck comb
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for X, lab, c in both:
        cb = X["comb"]
        L0 = np.nanmax(cb["L"])
        ax.plot(cb["n"], cb["L"] - L0, "o-", color=c, ms=3, lw=1,
                label=f"{lab}: beta {cb['beta']:.3f} / {cb['beta_alt']:.3f}, evidence {cb['evidence_db']:.1f} dB")
        if np.isfinite(cb["beta"]):
            ax.plot(cb["n"], cb["comb"] - L0, ":", color=c, lw=1)
    ax.set_xlabel("partial number")
    ax.set_ylabel("dB (first 40 ms)")
    ax.set_title("early partial levels and the fitted pluck comb (dotted)", color=INK)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "pluck.png", dpi=110)
    plt.close(fig)
    # 6. jawari
    Xs = [(A, "reference", C_REF)] + ([(info["model_eq"], model_label + " + EQ", C_MOD)] if M is not None else [])
    nb = len(A["jawari"]["oct_bands"])
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.4))
    ax = axs[0, 0]
    for X, lab, c in Xs:
        j = X["jawari"]
        ax.plot(X["t"], j["BI_smooth"], color=c, label=lab)
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_title(f"buzz index BI{A['jawari']['BI_main']} (dB)", color=INK)
    ax.legend(fontsize=7)
    ax = axs[0, 1]
    for X, lab, c in Xs:
        j = X["jawari"]
        ax.plot(X["t"], j["centre"], color=c, label=f"{lab} centre")
        ax.plot(X["t"], j["e95"], color=c, ls="--", lw=1, label=f"{lab} 95 %")
    ax.set_yscale("log")
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_title("buzz band: centre / 95 % (partial no.)", color=INK)
    ax.legend(fontsize=7)
    for i in range(min(nb, 6)):
        ax = axs.flat[2 + i]
        a, b = A["jawari"]["oct_bands"][i]
        for X, lab, c in Xs:
            j = X["jawari"]
            if i < len(j["oct_bands"]):
                ax.plot(X["t"], j["octaves"][i], color=c, label=lab)
        ax.set_xscale("symlog", linthresh=0.5)
        ax.set_ylim(-60, 2)
        ax.set_title(f"partials {a}-{b}: dB re total", color=INK)
    for ax in axs[1]:
        ax.set_xlabel("time after onset (s)")
    fig.tight_layout()
    fig.savefig(d / "jawari.png", dpi=110)
    plt.close(fig)
    # 7. attack + colour
    fig, axs = plt.subplots(1, 3 if M is not None else 2, figsize=(15 if M is not None else 10, 3.8))
    for X, lab, c in both:
        at = X["attack"]
        axs[0].plot(1e3 * at["env_t"], at["env_db"], color=c, lw=1,
                    label=f"{lab}: rise {at['rise_10_90_ms']:.1f} ms, HF burst {at['hf_burst_db']:+.1f} dB")
        lt = X["ltas"]
        axs[1].semilogx(lt["f_c"], lt["env_db"] - np.nanmax(lt["env_db"]), "-o", ms=3, color=c, label=lab)
    axs[0].set_xlabel("ms after onset")
    axs[0].set_ylabel("dB")
    axs[0].set_ylim(-60, 3)
    axs[0].set_title("attack envelope (> 30 Hz, 1 ms)", color=INK)
    axs[0].legend(fontsize=7)
    axs[1].set_xlabel("Hz")
    axs[1].set_ylabel("dB")
    axs[1].set_title("long-term partial-peak envelope (1/3 octave)", color=INK)
    axs[1].legend()
    if M is not None:
        E = info["eq"]
        axs[2].semilogx(E["f"], E["raw"], ".", color=INK2, ms=3, label="per partial")
        axs[2].semilogx(E["f_c"], E["eq_c"], "-", color=INK, label="smooth EQ (1/4-oct)")
        axs[2].axhline(0, color=INK2, lw=0.8)
        axs[2].set_xlabel("Hz")
        axs[2].set_ylabel("dB")
        axs[2].set_title("colour: reference / model (time-mean partial levels)", color=INK)
        axs[2].legend()
    fig.tight_layout()
    fig.savefig(d / "attack_colour.png", dpi=110)
    plt.close(fig)


# ======================================================================================== model renders
GEOM_DYN_KEYS = ("fs", "fmax", "M", "kc", "wc", "thread", "tune")     # dynamics keys that the statics depend on


def _code_version():
    import hashlib
    h = hashlib.sha1()
    for f in ("presets.py", "statics.py", "tanpura1d.py"):
        h.update((PKG / f).read_bytes())
    return h.hexdigest()[:12]


def model_params(string="kharaj", dur=6.0, **over):
    """presets.preset(string, dur=dur, **over) with a disk cache of the statics (out/reference/cache/): the
    geometry is solved once per (preset, geometry overrides, code version) and the other overrides are applied on
    top, exactly as preset() does.  xp_frac (pluck position as a fraction of the speaking length) is accepted."""
    import hashlib
    import pickle
    import presets as P
    over = dict(over)
    if "xp_frac" in over:
        xf = over.pop("xp_frac")
    else:
        xf = None
    geom = {k: over.pop(k) for k in list(over) if k in P.GEOM_KEYS or k in GEOM_DYN_KEYS}
    key = json.dumps(dict(name=string, geom={k: repr(v) for k, v in sorted(geom.items())}, v=_code_version()),
                     sort_keys=True)
    path = OUT / "cache" / (hashlib.sha1(key.encode()).hexdigest()[:20] + ".pkl")
    p = None
    if path.exists():
        try:
            with open(path, "rb") as fh:
                p = pickle.load(fh)
        except Exception:
            p = None
    if p is None:
        p = P.preset(string, **geom)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(p, fh)
        os.replace(tmp, path)
    p = dict(p)
    p.update(over)
    p["dur"] = float(dur)
    if xf is not None:
        p["xp"] = float(xf) * p["L"]
    if p.get("pluck") == "finger" and ("xp" in over or xf is not None):
        p["x_out"] = p["xp"]
    return p


def render_model(string="kharaj", dur=6.0, body="tanpura_male", stereo=False, room=0.0, return_params=False,
                 **over):
    """Render one pluck: preset -> tanpura1d -> common.to_audio (44.1 kHz) -> body.apply_body (unless body=None).
    Returns (y, 44100) or (y, 44100, params, stats)."""
    import tanpura1d as T1
    from common import to_audio
    p = model_params(string, dur, **over)
    meta = p.pop("meta", None)
    p, s, r = T1.simulate(p)
    y = to_audio(r["Fbridge"], p["fs"])
    if body:
        import body as BD
        y = BD.apply_body(y, 44100, body, stereo=stereo, room=room)
    if return_params:
        st = dict(r["stats"])
        st.update(cpu=r["cpu"], meta=meta)
        return y, 44100, p, st
    return y, 44100


# ======================================================================================== fitting
FIT_SPACE = {
    # name: (lo, hi, scale, unit factor for printing, unit)
    "target_gap": (-2.0e-6, 3.0e-6, "lin", 1e6, "um"),        # crest gap of the thread-only rest shape (+ = open)
    "thread_shift": (-0.15e-3, 0.15e-3, "lin", 1e3, "mm"),   # thread moved from grazing (+ = towards the nut)
    "delta_ve": (1.0e-3, 2.0e-2, "log", 1.0, ""),           # visco-elastic loss factor
    "Yb": (1.0e-3, 2.5e-2, "log", 1.0, "s/kg"),             # body-drain mobility
    "Qte_inv": (3.0e-5, 2.0e-3, "log", 1.0, ""),            # frequency-independent internal loss
    "xp_frac": (0.10, 0.90, "lin", 1.0, "L"),               # pluck position / speaking length, from the bridge
    "wp": (0.004, 0.030, "lin", 1e3, "mm"),                 # finger-pad width
    "pluck_release": (0.3e-3, 8.0e-3, "log", 1e3, "ms"),    # finger release time
    "pluck_disp": (0.5e-3, 5.0e-3, "log", 1e3, "mm"),       # pluck strength (static pad displacement)
    "pluck_pull": (0.010, 0.150, "log", 1e3, "ms"),         # finger pull time
}
ALIAS = {"xp": "xp_frac", "gap": "target_gap"}


def _to_unit(name, v):
    lo, hi, sc, _, _ = FIT_SPACE[name]
    return (np.log(v / lo) / np.log(hi / lo)) if sc == "log" else (v - lo) / (hi - lo)


def _from_unit(name, u):
    lo, hi, sc, _, _ = FIT_SPACE[name]
    u = float(np.clip(u, 0.0, 1.0))
    return float(lo * (hi / lo) ** u) if sc == "log" else float(lo + u * (hi - lo))


def _defaults(string, names):
    import presets as P
    p = model_params(string, 1.0)
    out = {}
    for k in names:
        if k in ("target_gap", "thread_shift"):
            out[k] = float(P.KHARAJ_GEOM.get(k, 0.0))
        elif k == "xp_frac":
            out[k] = (p["xp"] / p["L"]) if p.get("xp") is not None else 1.0 / 3.0
        else:
            out[k] = float(p[k])
    return out


_W = {}


def _worker_init(A, string, dur, body, fixed, weights):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W.update(A=A, string=string, dur=dur, body=body, fixed=fixed, weights=weights)


GEOM_QUANT = dict(target_gap=0.05e-6, thread_shift=0.005e-3)   # statics re-solved per value: quantise (disk cache)


def _quantise(params):
    q = dict(params)
    for k, step in GEOM_QUANT.items():
        if k in q:
            q[k] = float(np.round(q[k] / step) * step)
    return q


def _worker_eval(params):
    t0 = time.time()
    A = _W["A"]
    try:
        over = dict(_W["fixed"])
        over.update(params)
        y, sr = render_model(_W["string"], _W["dur"], body=_W["body"], **over)
        st_m = dict(max_dur=min(DEFAULTS["max_dur"], A["dur"] + 0.1))
        M = analyse(y, sr, f0=A["nominal_f0"], string=A["string"], settings=st_m, name="model")
        items, _ = compare_features(A, M)
        loss, per = feature_distance(items, _W["weights"])
        return dict(params=params, loss=loss, per=per, ok=True, t=time.time() - t0, y=y.astype(np.float32),
                    summary=summary(M), items=[(i["key"], i["severity"], i["text"]) for i in items])
    except Exception as e:      # a failed geometry (no grazing position) etc. is a bad point, not a crash
        return dict(params=params, loss=np.inf, per={}, ok=False, err=f"{type(e).__name__}: {e}", t=time.time() - t0)


class _CMA:
    """Minimal (mu/mu_w, lambda)-CMA-ES (Hansen's standard parameters) in the unit box (mirrored at the bounds)."""

    def __init__(self, x0, sigma, lam, seed=0):
        self.n = n = len(x0)
        self.m = np.array(x0, float)
        self.sigma = sigma
        self.lam = lam
        self.mu = lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.rng = np.random.default_rng(seed)
        self.g = 0

    def ask(self):
        D2, Bm = np.linalg.eigh(self.C)
        D = np.sqrt(np.maximum(D2, 1e-20))
        self._B, self._D = Bm, D
        X = []
        for _ in range(self.lam):
            z = self.rng.standard_normal(self.n)
            x = self.m + self.sigma * (Bm @ (D * z))
            x = np.abs(x)                               # mirror at 0 and 1
            x = 1 - np.abs(1 - x)
            X.append(np.clip(x, 0, 1))
        return X

    def tell(self, X, f):
        idx = np.argsort(f)[:self.mu]
        Xs = np.array([X[i] for i in idx])
        old = self.m.copy()
        self.m = self.w @ Xs
        y = (self.m - old) / self.sigma
        Bm, D = self._B, self._D
        Cinv_half = Bm @ np.diag(1 / D) @ Bm.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (Cinv_half @ y)
        self.g += 1
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.g)) / self.chiN < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        art = (Xs - old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) +
                  (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * (art.T * self.w) @ art)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = float(np.clip(self.sigma, 1e-3, 0.5))


def fit(ref, string="kharaj", params=("target_gap", "delta_ve", "Yb", "xp_frac"), budget=40, jobs=3, dur=5.0,
        body="tanpura_male", sr=None, f0=None, init=None, fixed=None, n_init=None, sigma0=0.2, seed=0,
        weights=None, out=True, name=None, pluck=0, verbose=True, mirror_check=True, onsets=None, settings=None):
    """Fit a few interpretable model parameters to a reference pluck by matching the analyser's features.

    ref      path / array (+ sr) of the reference pluck, or an analysis dict.
    string   'kharaj' | 'pa' | 'sa' (presets.preset name; also the f0 hint).
    params   names from FIT_SPACE (aliases: xp -> xp_frac, gap -> target_gap).
    budget   total model renders; jobs = parallel processes; dur = rendered seconds (the reference is analysed
             over the same duration).
    body     body.apply_body preset for the model (None = dry bridge force); a smooth EQ ref/model is matched
             in closed form for every candidate, so body colouring does not bias the string parameters.
    fixed    other preset overrides held fixed (e.g. dict(pluck_disp=3e-3)).
    Search: the preset default + a Latin-hypercube design (n_init points), then a small CMA-ES from the best one,
    in the unit box of FIT_SPACE (log scale for the loss / time parameters).  target_gap / thread_shift are
    quantised to 0.05 um / 5 um (GEOM_QUANT) so the statics disk cache is reused.  Loss: feature_distance of
    compare_features (pitch, per-partial decay, jawari tracks, bloom, formant, comb, attack; capped squared
    severities).  Returns a dict (best params, loss history, best render, report text)."""
    t_start = time.time()
    params = [ALIAS.get(p_, p_) for p_ in params]
    for p_ in params:
        if p_ not in FIT_SPACE:
            raise ValueError(f"unknown fit parameter {p_!r}; choose from {sorted(FIT_SPACE)}")
    fixed = dict(fixed or {})
    if isinstance(ref, dict):
        A = ref
    else:
        # (verify3 fix) analysis settings (CLI --no-truncate / --max-dur) were ignored by fit
        st_ = dict(settings or {})
        st_["max_dur"] = min(st_.get("max_dur", dur), dur)
        A = analyse(ref, sr, f0=f0, string=string, pluck=pluck, settings=st_, onsets=onsets)
    if A["dur"] < dur - 0.2:
        dur = A["dur"] + 0.1
    d = len(params)
    defaults = _defaults(string, params)
    x_def = np.array([_to_unit(k, defaults[k]) for k in params])
    if init:
        for k, v in init.items():
            k = ALIAS.get(k, k)
            if k in params:
                x_def[params.index(k)] = _to_unit(k, v)
    n_init = n_init or jobs * int(np.ceil((2 * d + 1) / jobs)) - 1      # stage 1 = a multiple of jobs
    rng = np.random.default_rng(seed)
    # Latin hypercube
    L = (np.argsort(rng.random((n_init, d)), axis=0) + rng.random((n_init, d))) / n_init
    X0 = [np.clip(x_def, 0, 1)] + [L[i] for i in range(n_init)]
    hist = []
    best = dict(loss=np.inf)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ctx = mp.get_context("spawn")
    log = []

    def say(msg):
        log.append(msg)
        if verbose:
            print(msg, flush=True)

    def to_params(x):
        return _quantise({k: _from_unit(k, x[i]) for i, k in enumerate(params)})

    def fmt(pp):
        return ", ".join(f"{k}={pp[k] * FIT_SPACE[k][3]:.4g}{FIT_SPACE[k][4]}" for k in params)

    def run(ex, X):
        nonlocal best
        res = list(ex.map(_worker_eval, [to_params(x) for x in X]))
        f = []
        for x, r in zip(X, res):
            hist.append(dict(x=np.array(x), params=r["params"], loss=r["loss"], ok=r["ok"], t=r["t"],
                             err=r.get("err"), per=r.get("per")))
            f.append(r["loss"])
            say(f"  [{len(hist):3d}] loss {r['loss']:8.2f}  {fmt(r['params'])}  ({r['t']:.0f} s)"
                + ("" if r["ok"] else f"  FAILED {r.get('err')}"))
            if r["loss"] < best["loss"]:
                best = dict(r)
                best["x"] = np.array(x)
        return np.array(f)

    say(f"fit {A['name']} ({string}): params {params}, budget {budget}, jobs {jobs}, dur {dur:g} s, body {body}")
    say(f"  defaults: {fmt(defaults)}")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=_worker_init,
                             initargs=(_strip(A), string, dur, body, fixed, weights)) as ex:
        say(f"stage 1: preset default + {n_init} Latin-hypercube points")
        run(ex, X0[:budget])
        left = budget - len(hist)
        lam = jobs * int(np.ceil(max(6, 4 + int(3 * np.log(d))) / jobs))   # a multiple of jobs
        if left >= lam and np.isfinite(best["loss"]):
            say(f"stage 2: CMA-ES from the best point (lambda {lam}, sigma0 {sigma0})")
            es = _CMA(best["x"], sigma0, lam, seed=seed + 1)
            while budget - len(hist) >= lam:
                X = es.ask()
                fvals = run(ex, X)
                es.tell(X, np.where(np.isfinite(fvals), fvals, 1e9))
                say(f"    generation {es.g}: best so far {best['loss']:.2f}, sigma {es.sigma:.3f}, "
                    f"mean {fmt(to_params(es.m))}")
        mirror = None
        if mirror_check and "xp_frac" in params and best.get("ok"):
            pm = dict(best["params"])
            pm["xp_frac"] = 1.0 - pm["xp_frac"]
            r = ex.submit(_worker_eval, pm).result()
            mirror = dict(params=pm, loss=r["loss"])
            say(f"mirror check: pluck at {pm['xp_frac']:.3f} L instead of {best['params']['xp_frac']:.3f} L -> loss "
                f"{r['loss']:.2f} (best {best['loss']:.2f})")
    t_tot = time.time() - t_start
    res = dict(ref=A["name"], string=string, params=params, defaults=defaults, best=best["params"],
               best_loss=best["loss"], default_loss=hist[0]["loss"], n_evals=len(hist), runtime_s=t_tot,
               cpu_s=float(sum(h["t"] for h in hist)), history=hist, mirror=mirror, best_y=best.get("y"),
               best_items=best.get("items"), dur=dur, body=body, fixed=fixed, log=log)
    say(f"done: best loss {best['loss']:.2f} (default {hist[0]['loss']:.2f}) after {len(hist)} renders, "
        f"{t_tot / 60:.1f} min wall")
    say(f"  best: {fmt(best['params'])}")
    if out:
        dd = Path(out) if isinstance(out, (str, Path)) else OUT / _safe(f"fit_{name or A['name']}")
        dd.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        if best.get("y") is not None:
            yb = best["y"]
            sf.write(dd / "best_model.wav", 0.9 * yb / np.max(np.abs(yb)), 44100, subtype="PCM_24")
        rows = ["n,loss,ok,t_s," + ",".join(params)]
        for i, h in enumerate(hist):
            rows.append(f"{i},{h['loss']:.4f},{int(h['ok'])},{h['t']:.1f}," +
                        ",".join(f"{h['params'][k]:.6g}" for k in params))
        (dd / "history.csv").write_text("\n".join(rows) + "\n")
        (dd / "fit.json").write_text(json.dumps(_jsonable({k: v for k, v in res.items()
                                                           if k not in ("history", "best_y")}), indent=1))
        (dd / "fit_log.txt").write_text("\n".join(log) + "\n")
        if best.get("y") is not None:
            cmp_ = compare(A, best["y"].astype(float), model_sr=44100, string=string, out=dd / "best_vs_ref",
                           model_label="fitted model")
            res["compare_text"] = cmp_["text"]
        res["dir"] = dd
    return res


def _strip(A):
    """Analysis dict without the raw onset-detection arrays (smaller pickles for the worker processes)."""
    B = dict(A)
    B.pop("onset_det", None)
    return B


# ======================================================================================== CLI
def _onsets(sv):
    return [float(v) for v in sv.split(",")] if sv else None


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("analyse", help="analyse one pluck")
    a1.add_argument("wav")
    a2 = sub.add_parser("compare", help="compare a reference with a model render (WAV or --render)")
    a2.add_argument("ref")
    a2.add_argument("model", nargs="?")
    a2.add_argument("--render", metavar="PRESET", help="render this preset now (kharaj / pa / sa)")
    a2.add_argument("--set", default="", help="model overrides, e.g. 'target_gap=1e-6,delta_ve=6e-3,xp_frac=0.667'")
    a2.add_argument("--dur", type=float, default=8.0)
    a2.add_argument("--body", default="tanpura_male", help="body preset for --render ('none' = dry)")
    a3 = sub.add_parser("fit", help="fit model parameters to a reference")
    a3.add_argument("ref")
    a3.add_argument("--params", default="target_gap,delta_ve,Yb,xp_frac")
    a3.add_argument("--budget", type=int, default=40)
    a3.add_argument("--jobs", type=int, default=3)
    a3.add_argument("--dur", type=float, default=5.0)
    a3.add_argument("--body", default="tanpura_male")
    a3.add_argument("--set", default="", help="fixed model overrides")
    for a in (a1, a2, a3):
        a.add_argument("--string", choices=sorted(STRINGS), default=None)
        a.add_argument("--f0", type=float, default=None)
        a.add_argument("--pluck", type=int, default=0, help="which pluck in the file (0 = first)")
        a.add_argument("--name", default=None)
        a.add_argument("--no-truncate", action="store_true", help="do not stop at the next detected pluck")
        a.add_argument("--max-dur", type=float, default=None, help="seconds analysed after the onset (default 12)")
        a.add_argument("--onsets", default=None, help="pluck times in s, comma-separated (replaces detection)")
    args = ap.parse_args(argv)
    sett = {}
    if args.no_truncate:
        sett["truncate"] = False
    if args.max_dur:
        sett["max_dur"] = args.max_dur
    sett = sett or None

    def parse_set(sv):
        out = {}
        for kv in filter(None, sv.split(",")):
            k, v = kv.split("=")
            v = v.strip()
            out[ALIAS.get(k.strip(), k.strip())] = (v == "True") if v in ("True", "False") else float(v)
        return out
    if args.cmd == "analyse":
        A, d = analyse_report(args.wav, string=args.string, f0=args.f0, name=args.name, pluck=args.pluck,
                              settings=sett, onsets=_onsets(args.onsets))
        print(text_summary(A))
        print(f"-> {d}")
    elif args.cmd == "compare":
        if args.render:
            body = None if args.body == "none" else args.body
            y, sr = render_model(args.render, args.dur, body=body, **parse_set(args.set))
            model, msr, lab = y, sr, f"{args.render} model"
        else:
            if not args.model:
                ap.error("give a model WAV or --render PRESET")
            model, msr, lab = args.model, None, "model"
        r = compare(args.ref, model, model_sr=msr, string=args.string or args.render, f0=args.f0, name=args.name,
                    pluck=args.pluck, model_label=lab, settings=sett, onsets=_onsets(args.onsets))
        print(r["text"])
        print(f"-> {r['dir']}")
    elif args.cmd == "fit":
        body = None if args.body == "none" else args.body
        r = fit(args.ref, string=args.string or "kharaj", params=args.params.split(","), budget=args.budget,
                jobs=args.jobs, dur=args.dur, body=body, f0=args.f0, fixed=parse_set(args.set), name=args.name,
                pluck=args.pluck, onsets=_onsets(args.onsets), settings=sett)
        print(r.get("compare_text", ""))
        print(f"-> {r.get('dir')}")


if __name__ == "__main__":
    main()
