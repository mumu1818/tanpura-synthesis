"""Analysis-by-synthesis fit of the 2-D tanpura model (sideways finger pluck) to the user's REAL reference plucks.

    cd jawari
    python fit2d.py refs                    # reference features (read-only; the recordings are never copied)
    python fit2d.py fit kharaj --jobs 2     # grid -> screening (sensitivity) -> CMA-ES, per reference; resumable
    python fit2d.py fit pa ; python fit2d.py fit sa
    python fit2d.py pressed kharaj          # pressed-crest check at the best points (also run by `fit`)
    python fit2d.py final                   # 8 s best-fit renders, loudness-matched reference WAVs, PNGs, tables,
                                            # sensitivity, shared instrument chain, reference.py cross-check
    python fit2d.py refcheck                # only the reference.py cross-check of the final renders
    python fit2d.py all                     # everything above

Everything goes to out/fit2d/ (git-ignored).  See FIT2D.md for the method, the results and their limits.

Model of the measurement chain (what is fitted)
  string  : presets2d.preset2d(string) = the statics/tuned/finger-plucked/Valette-loss string of presets.py in the
            2-polarisation model tanpura2d, with these physically bounded parameters (FIT_SPACE):
              pluck_deg      finger stroke direction from the bridge normal towards the soundboard plane   45..90 deg
              theta_nut      nut principal-axis angle (polarisation coupling)                               0..45 deg
              detune_cents   axis-2 frequency split (polarisation coupling)                               0.2..10 cents
              target_gap     crest gap of the thread-only rest shape (statics re-solved; + = open)        -2..+6 um
              pluck_disp     static pad displacement along the stroke                                       1..3 mm
              delta_ve_x, Yb_x   visco-elastic loss and body-drain mobility, x the preset value         0.5..2
              kcz_x          lateral thread stiffness / vertical thread stiffness                        0.3..3
              xp_frac        pluck point (fraction of the length from the bridge): 1/3, mirror-checked at 2/3
            f1 is tuned so that the model SOUNDS at the reference pitch (contact sharpens it by ~+5 cents).
  radiation (cheap, fitted for every render in closed loop, per reference; POST_GRID):
              gz             weight of the in-plane bridge force F_z in the mono mix  mono = F_y + gz F_z   0..1
              hp_hz, hp_order  radiation/microphone high-pass after the body                  60..400 Hz, 1..3
              mode_gain_db   level of the body's explicit low modes (A0, T1, A1, plate modes)             -6..+6 dB
            body = body.body_ir('tanpura_male') (seed 0).  The body and high-pass are linear, so
            body(F_y + gz F_z) = body(F_y) + gz body(F_z) and the high-pass acts on each partial as |H(n f0)|^2:
            one string render serves every radiation setting (checked against body.apply_body in `final`).

Features (per signal, partial numbers from its own sequentially fitted f0 and B; quickfeat's per-partial STFT
envelopes, 180 ms Hann, 20 ms hop, partials 1..60 within +-0.3 f0; onset = recording: first sample above -30 dB of
the peak, minus 5 ms; model: the end of the finger release (t = pluck_pull + pluck_release) minus 5 ms):
  rise_db  level(max) - level(first frame, 0-0.18 s)        t_max, t_m3   time of the maximum / first time within 3 dB
  lev@t    total level of partials 1-60 re its maximum        share@t      energy share of partials 4-10
  oct b@t  level of partials 1, 2-3, 4-7, 8-15, 16-31, 32-60 re the total (the spectral shape over time)
Distance = sum of w * min((difference / tolerance)^2, 25) (FEATURE_SPEC); the bloom items (rise, t_max, t_m3, lev,
share) carry 20 of the 27.5 weight units.  rise/t_max/H4-10/H31-60 are the quickfeat / validate2d_parity G features.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal as ss

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
OUT_ROOT = Path(os.environ.get("TANPURA_OUT", PKG / "out"))
OUT = OUT_ROOT / "fit2d"
CACHE = OUT / "cache"
REF_DIR = Path(os.environ.get("TANPURA_REFS", "/home/user/tanpura-app/assets"))   # read in place, never copied
SR = 44100
FEAT_VERSION = "fit2d-feat-1"

# ------------------------------------------------------------------------------------------------ references
REFS = {   # name: (path relative to REF_DIR, string).  'own' = the user's male C# tanpura (PRIMARY target)
    "own_kharaj": ("Mumu_Tanpura/Male/tt_CS3_kharaj.mp3", "kharaj"),
    "itanpura_kharaj": ("iTanpura/Tanpura_CS3_Kharaj.mp3", "kharaj"),
    "own_pa": ("Mumu_Tanpura/Male/tt_CS3_pa.mp3", "pa"),
    "itanpura_pa": ("iTanpura/Tanpura_CS3_Pa.mp3", "pa"),
    "own_sa": ("Mumu_Tanpura/Male/tt_CS3_sa.mp3", "sa"),
    "itanpura_sa": ("iTanpura/Tanpura_CS3_Sa.mp3", "sa"),
}
STRING_REFS = {s: (f"own_{s}", f"itanpura_{s}") for s in ("kharaj", "pa", "sa")}
SHARPEN_CENTS = 5.0        # the jawari contact raises the sounding pitch ~ +5 cents above the contact-off f1


def trim_onset(y, rel_db=-30.0, pre=0.005):
    i0 = int(np.argmax(np.abs(y) > 10 ** (rel_db / 20) * np.abs(y).max()))
    return y[max(0, i0 - int(pre * SR)):]


def load_ref(name):
    """Reference pluck, mono, 44.1 kHz, trimmed at its onset.  Read-only."""
    import soundfile as sf
    rel, _ = REFS[name]
    y, sr = sf.read(str(REF_DIR / rel), always_2d=True)
    y = y.mean(1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        y = ss.resample_poly(y, SR // g, int(sr) // g)
    return trim_onset(y)


def fit_f0_B(y, f_guess, t0=0.3, t1=3.0, nmax=40, B0=0.0):
    """Sounding f0 and inharmonicity B by SEQUENTIAL partial tracking (validate2d_parity.fit_f0_B): Hann FFT of t0..t1,
    partial n searched within +-0.2 f_guess of the prediction from the robust fit to partials 1..n-1 (picking all
    partials at once around a guessed stretched series mis-numbers them above n ~ 15)."""
    from scipy.optimize import least_squares
    seg = y[int(t0 * SR):int(t1 * SR)]
    w = np.hanning(len(seg))
    nfft = 1 << 21
    Y = np.abs(np.fft.rfft(seg * w, nfft))
    f = np.fft.rfftfreq(nfft, 1 / SR)
    df = f[1]
    ns, fn = [], []
    f0, B = float(f_guess), float(B0)
    for n in range(1, nmax + 1):
        fg = n * f0 * np.sqrt(1 + B * n * n)
        sel = np.where(np.abs(f - fg) < 0.2 * f_guess)[0]
        k = sel[np.argmax(Y[sel])]
        a, b, c = np.log(Y[k - 1] + 1e-300), np.log(Y[k] + 1e-300), np.log(Y[k + 1] + 1e-300)
        den = a - 2 * b + c
        ns.append(n)
        fn.append((k + (0.5 * (a - c) / den if den != 0 else 0.0)) * df)
        if n >= 3:
            nn, ff = np.array(ns), np.array(fn)
            r = least_squares(lambda x: 1200 * np.log2(ff / (nn * x[0] * np.sqrt(1 + x[1] * nn ** 2))), [f0, B],
                              loss="soft_l1", f_scale=3.0)
            f0, B = float(r.x[0]), float(r.x[1])
    return f0, B


# ------------------------------------------------------------------------------------------------ features
WIN, HOP, NMAX = 0.18, 0.02, 60
NPER, NHOP = int(WIN * SR), int(HOP * SR)
T_LEV = (0.09, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0)   # 0.09 s = first frame (0-0.18 s) = quickfeat's "0.05"
T_SH = (0.09, 0.3, 1.0, 2.0, 3.0)
T_OCT = (0.09, 0.3, 1.0, 2.0, 3.0)
BANDS = ((1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 60))
FEATURE_SPEC = dict(   # item group: (tolerance, weight per item)
    rise_db=(1.5, 4.0), t_max=(np.log(1.25), 2.0), t_m3=(np.log(1.3), 2.0),
    lev=(2.0, 0.6), share=(0.07, 1.2), oct=(3.0, 0.25))
BLOOM_GROUPS = ("rise_db", "t_max", "t_m3", "lev", "share")
CAP = 25.0


def stft(y):
    f, t, Z = ss.stft(y, SR, nperseg=NPER, noverlap=NPER - NHOP, boundary=None)
    return f, t, Z


def partial_slices(f, f0, B, nmax=NMAX):
    """bin slices f in (c_n - 0.3 f0, c_n + 0.3 f0), c_n = n f0 sqrt(1 + B n^2) (quickfeat), and c_n."""
    n = np.arange(1, nmax + 1)
    c = n * f0 * np.sqrt(1 + B * n * n)
    df = f[1] - f[0]
    sl = []
    for cn in c:
        lo = int(np.floor((cn - 0.3 * f0) / df)) + 1
        hi = int(np.ceil((cn + 0.3 * f0) / df))
        sl.append((max(lo, 0), min(hi, len(f))))
    return sl, c


def penv(S, sl):
    return np.stack([S[lo:hi].max(0) if hi > lo else np.zeros(S.shape[1]) for lo, hi in sl])


def features(t, A, tcut):
    """Bloom features from a partial-envelope matrix A (NMAX x frames, power), frames up to tcut s."""
    keep = t <= tcut
    t, A = t[keep], A[:, keep]
    tot = A.sum(0) + 1e-30
    L = 10 * np.log10(tot)
    L = L - L.max()
    tl = float(t[-1])

    def i(x):
        return int(np.argmin(np.abs(t - x)))
    oct_ = {}
    for (b0, b1) in BANDS:
        for x in T_OCT:
            if x <= tl:
                oct_[f"{b0}-{b1}@{x:g}"] = float(10 * np.log10(A[b0 - 1:b1, i(x)].sum() / tot[i(x)] + 1e-30))
    return dict(
        rise_db=float(-L[0]), t_max=float(t[np.argmax(L)]), t_m3=float(t[np.argmax(L >= -3.0)]),
        lev={f"{x:g}": float(L[i(x)]) for x in T_LEV if x <= tl},
        share={f"{x:g}": float(A[3:10, i(x)].sum() / tot[i(x)]) for x in T_SH if x <= tl},
        oct=oct_,
        hf31_60={f"{x:g}": float(10 * np.log10(A[30:60, i(x)].sum() / tot[i(x)] + 1e-30)) for x in (1.0, 3.0)
                 if x <= tl},
        tcut=tl)


def distance(fr, fm, spec=FEATURE_SPEC):
    """Weighted, capped squared feature distance; returns (total, bloom part, per-group dict)."""
    per = {}
    for g, (tol, w) in spec.items():
        if g in ("rise_db",):
            d = [(fm[g] - fr[g]) / tol]
        elif g in ("t_max", "t_m3"):
            d = [np.log(max(fm[g], 0.05) / max(fr[g], 0.05)) / tol]
        else:
            d = [(fm[g][k] - fr[g][k]) / tol for k in fr[g] if k in fm[g]]
        per[g] = float(w * sum(min(x * x, CAP) for x in d))
    tot = float(sum(per.values()))
    bloom = float(sum(per[g] for g in BLOOM_GROUPS))
    return tot, bloom, per


_REF_FEAT = {}


def ref_features(name):
    """features + f0/B + envelope of a reference (cached in memory and in out/fit2d/cache/refs.json)."""
    if name in _REF_FEAT:
        return _REF_FEAT[name]
    y = load_ref(name)
    string = REFS[name][1]
    import presets as P1
    nominal = P1.STRING_SPECS[string]["geom"].get("f1", P1.KHARAJ_GEOM["f1"])
    f0, B = fit_f0_B(y, nominal)
    f, t, Z = stft(y)
    sl, c = partial_slices(f, f0, B)
    A = penv(np.abs(Z) ** 2, sl)
    tcut = min(4.0, len(y) / SR - 0.1)
    out = dict(name=name, string=string, f0=f0, B=B, dur=len(y) / SR, tcut=tcut, t=t, A=A,
               feat=features(t, A, tcut))
    _REF_FEAT[name] = out
    return out


# ------------------------------------------------------------------------------------------------ the model
FIT_SPACE = {   # name: (lo, hi, scale)  -- physically bounded
    "pluck_deg": (45.0, 90.0, "lin"),
    "theta_nut": (0.0, 45.0, "lin"),
    "detune_cents": (0.2, 10.0, "log"),
    "target_gap": (-2e-6, 6e-6, "lin"),
    "pluck_disp": (1e-3, 3e-3, "lin"),
    "delta_ve_x": (0.5, 2.0, "log"),
    "Yb_x": (0.5, 2.0, "log"),
    "kcz_x": (0.3, 3.0, "log"),
    "xp_frac": (0.2, 0.8, "lin"),
    "thread_shift": (-0.15e-3, 0.15e-3, "lin"),   # optional: pressed crest when no negative target_gap is feasible
}
QUANT = dict(target_gap=0.5e-6, pluck_deg=0.5, theta_nut=1.0, detune_cents=0.05, pluck_disp=0.05e-3,
             delta_ve_x=0.01, Yb_x=0.01, kcz_x=0.01, xp_frac=0.005)
DEFAULT_X = dict(pluck_deg=88.0, theta_nut=30.0, detune_cents=3.5, target_gap=0.0, pluck_disp=2e-3,
                 delta_ve_x=1.0, Yb_x=1.0, kcz_x=1.0, xp_frac=1.0 / 3.0)   # = presets2d.preset2d defaults
POST_GRID = dict(gz=(0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0),
                 hp_hz=tuple(float(v) for v in np.round(np.geomspace(60.0, 400.0, 9))),
                 hp_order=(1, 2, 3), mode_gain_db=(-6.0, 0.0, 6.0))
POST_DEFAULT = dict(gz=0.1, hp_hz=250.0, hp_order=2, mode_gain_db=0.0)   # presets2d.radiate + PRESET_2D gz


def quantise(x):
    q = dict(x)
    if "thread_shift" in q:      # optional key (not in DEFAULT_X, so the cache keys of the other runs stay valid)
        q["thread_shift"] = float(np.round(q["thread_shift"] / 5e-6) * 5e-6)
        if q["thread_shift"] == 0.0:
            del q["thread_shift"]
    for k, st in QUANT.items():
        if k in q and k != "xp_frac":
            q[k] = float(np.round(q[k] / st) * st)
    if "xp_frac" in q:
        q["xp_frac"] = float(np.round(q["xp_frac"] / QUANT["xp_frac"]) * QUANT["xp_frac"]) \
            if abs(q["xp_frac"] - 1 / 3) > 1e-9 and abs(q["xp_frac"] - 2 / 3) > 1e-9 else q["xp_frac"]
    return q


def target_f1(string):
    """contact-off f1 such that the model SOUNDS at the own reference's pitch (+SHARPEN_CENTS contact sharpening)."""
    f_ref = ref_features(STRING_REFS[string][0])["f0"]
    return round(f_ref * 2 ** (-SHARPEN_CENTS / 1200), 3)


def _h(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def base_preset(string, f1, target_gap, model="2d", thread_shift=0.0):
    """presets2d.preset2d (or presets.preset for the 1-D model) at f1, target_gap and thread_shift (thread moved from
    its target_gap position, + = towards the nut); statics cached on disk."""
    key = dict(string=string, f1=round(f1, 4), gap_nm=int(round(target_gap * 1e9)), model=model)
    if thread_shift:
        key["shift_um"] = int(round(thread_shift * 1e6))
    path = CACHE / "geom" / f"{model}_{string}_{_h(key)}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    fail = path.with_suffix(".fail")
    if fail.exists():                     # remembered: this geometry has no valid rest state
        raise ValueError(fail.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if model == "2d":
            import presets2d as P2
            p = P2.preset2d(string, f1=f1, target_gap=target_gap, thread_shift=thread_shift)
        else:
            import presets as P1
            p = P1.preset(string, f1=f1, target_gap=target_gap, thread_shift=thread_shift)
    except ValueError as e:
        fail.write_text(str(e))
        raise
    with open(path, "wb") as fh:
        pickle.dump(p, fh)
    return p


def make_params(string, x, dur, f1=None, model="2d"):
    x = dict(DEFAULT_X, **x)
    f1 = target_f1(string) if f1 is None else f1
    p = dict(base_preset(string, f1, x["target_gap"], model, x.get("thread_shift", 0.0)))
    p["dur"] = float(dur)
    p["pluck_disp"] = float(x["pluck_disp"])
    p["xp"] = float(x["xp_frac"]) * p["L"]
    p["delta_ve"] = p["delta_ve"] * x["delta_ve_x"]
    p["Yb"] = p["Yb"] * x["Yb_x"]
    if model == "2d":
        p["pluck_deg"] = float(x["pluck_deg"])
        p["theta_nut"] = float(x["theta_nut"])
        p["detune_cents"] = float(x["detune_cents"])
        p["kcz"] = None if abs(x["kcz_x"] - 1.0) < 1e-12 else float(p["kc"] * x["kcz_x"])
    else:
        p["x_out"] = p["xp"]
    return p


def run_key(string, x, dur, model="2d", seed_eps=0.0):
    return _h(dict(string=string, x=quantise(dict(DEFAULT_X, **x)), dur=dur, model=model, eps=seed_eps,
                   f1=target_f1(string)))


def render(string, x, dur=5.0, model="2d", seed_eps=0.0, f1=None):
    """Render (cached): 44.1 kHz F_y, F_z (F_z = 0 for 1-D), cpu, sounding f0/B, solver stats.  seed_eps perturbs
    pluck_disp by a relative 1e-9-ish amount for chaos twins."""
    x = quantise(dict(DEFAULT_X, **x))
    key = run_key(string, x, dur, model, seed_eps)
    path = CACHE / "runs" / f"{model}_{string}_{key}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files} | dict(key=key, x=x)
    path.parent.mkdir(parents=True, exist_ok=True)
    from common import to_audio
    p = make_params(string, x, dur, f1, model)
    if seed_eps:
        p["pluck_disp"] *= 1.0 + seed_eps
    t0 = time.time()
    if model == "2d":
        import presets2d as P2
        import tanpura2d as T2
        pp, s, r = T2.simulate(P2.sim_params(p), energy=False)
        Fy, Fz = to_audio(r["Fy"], pp["fs"]), to_audio(r["Fz"], pp["fs"])
        st = r["stats"]
        Ez = r["Ez"] / np.maximum(r["Ey"] + r["Ez"], 1e-30)
        ez = [float(Ez[np.argmin(np.abs(r["t_rec"] - tt))]) for tt in (0.05, 0.5, 1.0, 2.0, 3.0)]
        stats = [st["newton_fail"], st["fric_fail"], st["max_pen"], st["contact_frac"]]
    else:
        import tanpura1d as T1
        pp, s, r = T1.simulate({k: v for k, v in p.items() if k != "meta"})
        Fy = to_audio(r["Fbridge"], pp["fs"])
        Fz = np.zeros_like(Fy)
        ez = [0.0] * 5
        stats = [r.get("stats", {}).get("newton_fail", 0), 0, r.get("stats", {}).get("max_pen", 0.0), 0.0]
    cpu = time.time() - t0
    i0 = int(round((p["pluck_pull"] + p["pluck_release"] - 0.005) * SR))   # release end - 5 ms (~ the recordings'
    #                                                       -30 dB threshold - 5 ms: threshold is 0.8 ms after it)
    f0, B = fit_f0_B(Fy[i0:], p["f1"])
    np.savez(path, Fy=Fy.astype(np.float32), Fz=Fz.astype(np.float32), cpu=cpu, f0=f0, B=B, f1=p["f1"],
             i0=i0, ez=np.array(ez), stats=np.array(stats, float), T=float(s["T"]),
             x=json.dumps(x), string=string, model=model)
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files} | dict(key=key, x=x)


_IR = {}


def body_ir(mode_gain_db=0.0):
    import body
    if mode_gain_db not in _IR:
        h, _ = body.body_ir("tanpura_male", SR, 0, 1, hp_hz=0, mode_gain_db=float(mode_gain_db))
        _IR[mode_gain_db] = h[:, 0]
    return _IR[mode_gain_db]


def hp_gain(c, hp_hz, hp_order):
    return 1.0 / (1.0 + (hp_hz / np.maximum(c, 1e-9)) ** (2 * hp_order))


def model_envelopes(run, mgd_list=POST_GRID["mode_gain_db"], gz_list=POST_GRID["gz"]):
    """{(mode_gain_db, gz): A} partial envelopes (before the high-pass) of body(F_y) + gz body(F_z), and t, c_n."""
    Fy, Fz, i0 = run["Fy"].astype(float), run["Fz"].astype(float), int(run["i0"])
    out = {}
    t = c = None
    for mgd in mgd_list:
        h = body_ir(mgd)
        yb = ss.fftconvolve(Fy, h)[:len(Fy)][i0:]
        zb = ss.fftconvolve(Fz, h)[:len(Fz)][i0:]
        f, t, Zy = stft(yb)
        _, _, Zz = stft(zb)
        sl, c = partial_slices(f, float(run["f0"]), float(run["B"]))
        hi = max(h_ for _, h_ in sl)
        Zy, Zz = Zy[:hi], Zz[:hi]
        for gz in gz_list:
            out[(mgd, gz)] = penv(np.abs(Zy + gz * Zz) ** 2, sl)
    return t, c, out


def score_run(run, ref_names, post_grid=POST_GRID):
    """For each reference: the best radiation setting (closed-loop grid) and its distance, plus the distance with
    the default radiation chain (POST_DEFAULT).  Cached per run key and FEAT_VERSION."""
    path = CACHE / "scores" / f"{run['key']}_{_h([FEAT_VERSION, list(ref_names), post_grid])}.json"
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    t, c, envs = model_envelopes(run, post_grid["mode_gain_db"], post_grid["gz"])
    res = {}
    gz_list = post_grid["gz"] if float(np.abs(run["Fz"]).max()) > 0 else (0.0,)
    for rn in ref_names:
        R = ref_features(rn)
        best = None
        for (mgd, gz), A in envs.items():
            if gz not in gz_list:
                continue
            for o in post_grid["hp_order"]:
                for fc in post_grid["hp_hz"]:
                    fm = features(t, A * hp_gain(c, fc, o)[:, None], R["tcut"])
                    D, Db, per = distance(R["feat"], fm)
                    if best is None or D < best[0]:
                        best = (D, Db, per, dict(gz=gz, hp_hz=fc, hp_order=o, mode_gain_db=mgd), fm)
        A0 = envs.get((POST_DEFAULT["mode_gain_db"], POST_DEFAULT["gz"] if len(gz_list) > 1 else 0.0))
        if A0 is None:
            A0 = model_envelopes(run, (0.0,), (POST_DEFAULT["gz"] if len(gz_list) > 1 else 0.0,))[2].popitem()[1]
        fm0 = features(t, A0 * hp_gain(c, POST_DEFAULT["hp_hz"], POST_DEFAULT["hp_order"])[:, None], R["tcut"])
        D0, Db0, per0 = distance(R["feat"], fm0)
        res[rn] = dict(D=best[0], D_bloom=best[1], per=best[2], post=best[3], feat=best[4],
                       D_default=D0, D_bloom_default=Db0, feat_default=fm0)
    out = dict(key=run["key"], x=run["x"], f0=float(run["f0"]), B=float(run["B"]), f1=float(run["f1"]),
               cpu=float(run["cpu"]), ez=[float(v) for v in run["ez"]], stats=[float(v) for v in run["stats"]],
               refs=res)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh)
    return out


# ------------------------------------------------------------------------------------------------ evaluation pool
def _eval_one(args):
    string, x, dur, model, eps = args
    try:
        run = render(string, x, dur, model, eps)
        sc = score_run(run, STRING_REFS[string])
        sc["ok"] = True
        sc["eps"] = eps
        sc["model"] = model
        return sc
    except Exception as e:     # e.g. a geometry with no grazing thread position: a bad point, not a crash
        return dict(ok=False, x=quantise(dict(DEFAULT_X, **x)), err=f"{type(e).__name__}: {e}", eps=eps, model=model)


def evaluate(string, X, dur=5.0, jobs=2, model="2d", eps=0.0, log=None):
    """Render and score a list of parameter dicts (resumable: every render and score is cached)."""
    for rn in STRING_REFS[string]:
        ref_features(rn)                       # computed once here, inherited by the forked workers
    target_f1(string)
    # pre-solve the statics of every new geometry serially (avoids two workers solving the same one)
    geoms = {(q["target_gap"], q.get("thread_shift", 0.0)) for q in (quantise(dict(DEFAULT_X, **x)) for x in X)}
    for g, sh in sorted(geoms):
        try:
            base_preset(string, target_f1(string), g, model, sh)
        except Exception as e:
            if log:
                log(f"    statics failed for gap {g * 1e6:+.1f} um, thread shift {sh * 1e3:+.3f} mm: {e}")
    todo = [(string, x, dur, model, eps) for x in X]
    t0 = time.time()
    if jobs > 1 and len(todo) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(jobs, len(todo))) as pool:
            res = pool.map(_eval_one, todo, chunksize=1)
    else:
        res = [_eval_one(a) for a in todo]
    if log:
        log(f"    {len(todo)} evaluations, {time.time() - t0:.0f} s wall")
    return res


# ------------------------------------------------------------------------------------------------ search
def to_unit(name, v, space=None):
    lo, hi, sc = (space or FIT_SPACE)[name]
    return float(np.clip((np.log(v / lo) / np.log(hi / lo)) if sc == "log" else (v - lo) / (hi - lo), 0.0, 1.0))


def from_unit(name, u, space=None):
    lo, hi, sc = (space or FIT_SPACE)[name]
    u = float(np.clip(u, 0.0, 1.0))
    return float(lo * (hi / lo) ** u) if sc == "log" else float(lo + u * (hi - lo))


class CMA:
    """Small (mu/mu_w, lambda)-CMA-ES (Hansen's default constants) in the unit box, mirrored at the bounds."""

    def __init__(self, x0, sigma, lam, seed=0):
        self.n = n = len(x0)
        self.m = np.array(x0, float)
        self.sigma, self.lam, self.mu = sigma, lam, lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.pc, self.ps, self.C = np.zeros(n), np.zeros(n), np.eye(n)
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.rng = np.random.default_rng(seed)

    def ask(self):
        D2, Bm = np.linalg.eigh(self.C)
        D = np.sqrt(np.maximum(D2, 1e-20))
        X = []
        for _ in range(self.lam):
            x = self.m + self.sigma * (Bm @ (D * self.rng.standard_normal(self.n)))
            x = 1 - np.abs(1 - np.abs(x))          # mirror at 0 and 1
            X.append(np.clip(x, 0, 1))
        return X

    def tell(self, X, f):
        idx = np.argsort(f)
        Xs = np.array(X)[idx[:self.mu]]
        old = self.m.copy()
        self.m = self.w @ Xs
        y = (self.m - old) / self.sigma
        D2, Bm = np.linalg.eigh(self.C)
        Cinv_half = Bm @ np.diag(1 / np.sqrt(np.maximum(D2, 1e-20))) @ Bm.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (Cinv_half @ y)
        hs = np.linalg.norm(self.ps) / self.chiN < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        art = (Xs - old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * np.outer(self.pc, self.pc)
                  + self.cmu * (art.T * self.w) @ art)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = float(np.clip(self.sigma, 0.02, 0.5))


GRID = {   # stage 1: the three parameters expected to matter most (pluck direction x coupling x crest gap)
    "kharaj": dict(pluck_deg=(84.0, 87.0, 90.0), coupling=((30.0, 1.5), (30.0, 3.5), (45.0, 7.0)),
                   target_gap=(-1e-6, 0.0, 2e-6)),
    "pa": dict(pluck_deg=(87.0, 90.0), coupling=((30.0, 3.5), (45.0, 7.0)), target_gap=(0.0, 2e-6)),
    "sa": dict(pluck_deg=(87.0, 90.0), coupling=((30.0, 3.5), (45.0, 7.0)), target_gap=(0.0, 2e-6)),
}
SCREEN = {   # stage 2: one-at-a-time steps from the grid winner (sensitivity + screening for stage 3)
    "pluck_deg": (-4.0, +3.0), "theta_nut": (-15.0, +10.0), "detune_cents": (0.5, 2.0), "target_gap": (-1e-6, 2e-6),
    "pluck_disp": (-0.7e-3, +0.7e-3), "delta_ve_x": (0.5, 2.0), "Yb_x": (0.5, 2.0), "kcz_x": (0.4, 2.5),
}
MULT = {"detune_cents", "delta_ve_x", "Yb_x", "kcz_x"}      # screening steps applied as factors


def clamp(name, v):
    lo, hi, _ = FIT_SPACE[name]
    return float(np.clip(v, lo, hi))


class Log:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, s=""):
        print(s, flush=True)
        with open(self.path, "a") as fh:
            fh.write(s + "\n")


def _fmt_x(x):
    return (f"pluck {x['pluck_deg']:4.1f} deg, nut {x['theta_nut']:4.1f} deg/{x['detune_cents']:5.2f} c, "
            f"gap {x['target_gap'] * 1e6:+4.1f} um, disp {x['pluck_disp'] * 1e3:4.2f} mm, xp {x['xp_frac']:.3f}, "
            f"dve x{x['delta_ve_x']:.2f}, Yb x{x['Yb_x']:.2f}, kcz x{x['kcz_x']:.2f}"
            + (f", thread shift {x['thread_shift'] * 1e3:+.3f} mm" if x.get("thread_shift") else ""))


def _fmt_post(q):
    return f"gz {q['gz']:.2f}, HP {q['hp_hz']:.0f} Hz/{q['hp_order']}, modes {q['mode_gain_db']:+.0f} dB"


def history_path(string):
    return OUT / f"history_{string}.jsonl"


def append_history(string, stage, res):
    with open(history_path(string), "a") as fh:
        for r in res:
            fh.write(json.dumps(dict(r, stage=stage), default=float) + "\n")


def load_history(string):
    p = history_path(string)
    if not p.exists():
        return []
    return [json.loads(line) for line in open(p)]


def best_of(hist, ref, model="2d"):
    ok = [h for h in hist if h.get("ok") and h.get("model", "2d") == model and not h.get("eps")]
    return min(ok, key=lambda h: h["refs"][ref]["D"]) if ok else None


def fit_string(string, jobs=2, dur=5.0, cma_gens=4, lam=6, itan_gens=2):
    """grid -> screening -> CMA-ES on the most influential parameters (own reference), then a short CMA-ES for the
    iTanpura reference from its own best point.  Every stage is cached / resumable."""
    log = Log(OUT / f"fit_{string}.log")
    own, itan = STRING_REFS[string]
    log(f"\n==== fit2d {string}  {time.strftime('%Y-%m-%d %H:%M:%S')}  f1 = {target_f1(string)} Hz "
        f"(refs sound at {ref_features(own)['f0']:.3f} / {ref_features(itan)['f0']:.3f} Hz) ====")
    done = {(h["stage"], h.get("key", json.dumps(h["x"], sort_keys=True))) for h in load_history(string)}

    def run_stage(stage, X, eps=0.0, model="2d"):
        res = evaluate(string, X, dur, jobs, model, eps, log)
        new = [r for r in res if (stage, r.get("key", json.dumps(r["x"], sort_keys=True))) not in done]
        append_history(string, stage, new)
        for r in new:
            done.add((stage, r.get("key", json.dumps(r["x"], sort_keys=True))))
        for r in res:
            if r.get("ok"):
                a, b = r["refs"][own], r["refs"][itan]
                log(f"    {stage:7s} D own {a['D']:6.1f} (bloom {a['D_bloom']:5.1f}) | iT {b['D']:6.1f} "
                    f"(bloom {b['D_bloom']:5.1f}) | rise {a['feat']['rise_db']:4.1f} dB t_max "
                    f"{a['feat']['t_max']:.2f} s | {_fmt_x(r['x'])}")
            else:
                log(f"    {stage:7s} FAILED {r.get('err')} | {_fmt_x(r['x'])}")
        return res

    # stage 0: the current models (1-D preset, 2-D preset), same f1
    log("  stage 0: current 1-D preset and current 2-D preset (pluck 88 deg), tuned to the reference pitch")
    run_stage("1d", [dict()], model="1d")
    run_stage("preset", [dict()])
    # stage 1: grid
    G = GRID[string]
    X = [dict(pluck_deg=pd, theta_nut=th, detune_cents=dc, target_gap=g)
         for pd in G["pluck_deg"] for (th, dc) in G["coupling"] for g in G["target_gap"]]
    log(f"  stage 1: grid pluck_deg x coupling x gap ({len(X)} renders)")
    run_stage("grid", X)
    hist = load_history(string)
    b = best_of(hist, own)
    x0 = dict(DEFAULT_X, **b["x"])
    log(f"  grid best (own): D {b['refs'][own]['D']:.1f} -- {_fmt_x(x0)}")
    # chaos noise floor: two twins of the grid winner (pluck_disp x (1 + 1e-9))
    log("  chaos twins of the grid winner")
    run_stage("twin", [x0], eps=1e-9)
    run_stage("twin", [x0], eps=2e-9)
    # stage 2: screening (one at a time)
    X = []
    for k, steps in SCREEN.items():
        for st in steps:
            if st is None:
                continue
            v = x0[k] * st if k in MULT else x0[k] + st
            v = clamp(k, v)
            if abs(v - x0[k]) > 1e-12:
                X.append(dict(x0, **{k: v}))
    X.append(dict(x0, xp_frac=2.0 / 3.0))          # mirror check of the pluck point
    log(f"  stage 2: one-at-a-time screening around the grid winner ({len(X)} renders)")
    run_stage("screen", X)
    hist = load_history(string)
    # influence of each parameter on the own-reference distance
    D0 = b["refs"][own]["D"]
    infl = {}
    for h in hist:
        if h["stage"] != "screen" or not h.get("ok"):
            continue
        for k in SCREEN:
            if abs(h["x"][k] - x0[k]) > 1e-12:
                infl.setdefault(k, []).append(h["refs"][own]["D"] - D0)
    rank = sorted(infl, key=lambda k: -max(abs(v) for v in infl[k]))
    log("  screening: |dD| per parameter (own): " + ", ".join(f"{k} {max(abs(v) for v in infl[k]):.1f}" for k in rank))
    names = [k for k in rank if k in FIT_SPACE][:5]
    if "pluck_deg" not in names:
        names = ["pluck_deg"] + names[:4]
    b = best_of(hist, own)
    xs = dict(DEFAULT_X, **b["x"])
    log(f"  stage 3: CMA-ES (own reference) on {names} from D {b['refs'][own]['D']:.1f}")
    xs = _cma_stage(string, "cma", own, xs, names, cma_gens, lam, run_stage, seed=1)
    hist = load_history(string)
    bi = best_of(hist, itan)
    log(f"  stage 4: CMA-ES (iTanpura reference) from its best so far, D {bi['refs'][itan]['D']:.1f}")
    _cma_stage(string, "cma_it", itan, dict(DEFAULT_X, **bi["x"]), names, itan_gens, lam, run_stage, seed=2)
    hist = load_history(string)
    for rn in (own, itan):
        bb = best_of(hist, rn)
        a = bb["refs"][rn]
        log(f"  BEST {rn}: D {a['D']:.1f} (bloom {a['D_bloom']:.1f}) -- {_fmt_x(bb['x'])} -- {_fmt_post(a['post'])}")
    return hist


def pressed_check(string, jobs=2, dur=5.0, shifts=(-0.05e-3, -0.1e-3)):
    """A pressed crest (negative target_gap) has no valid rest state on this bridge at the reference pitch (statics:
    the string would leave the back of the bridge), so press the crest instead by moving the thread towards the
    bridge (thread_shift < 0) at each reference's best point."""
    log = Log(OUT / f"fit_{string}.log")
    hist = load_history(string)
    X = []
    for rn in STRING_REFS[string]:
        b = dict(DEFAULT_X, **best_of(hist, rn)["x"])
        X += [dict(b, thread_shift=sh) for sh in shifts if not b.get("thread_shift")]
    log(f"  pressed-crest check (thread moved {', '.join(f'{v * 1e3:+.2f}' for v in shifts)} mm) at the best points")
    res = evaluate(string, X, dur, jobs, "2d", 0.0, log)
    seen = {h.get("key") for h in hist if h["stage"] == "pressed"}
    append_history(string, "pressed", [r for r in res if r.get("key") not in seen])
    own, itan = STRING_REFS[string]
    for r in res:
        if r.get("ok"):
            a, b = r["refs"][own], r["refs"][itan]
            log(f"    pressed D own {a['D']:6.1f} (bloom {a['D_bloom']:5.1f}) | iT {b['D']:6.1f} "
                f"(bloom {b['D_bloom']:5.1f}) | rise {a['feat']['rise_db']:4.1f} dB t_max {a['feat']['t_max']:.2f} s"
                f" | {_fmt_x(r['x'])}")
        else:
            log(f"    pressed FAILED {r.get('err')} | {_fmt_x(r['x'])}")


CMA_NARROW = dict(pluck_deg=(80.0, 90.0, "lin"))   # stage 1-2 evidence: below ~86 deg the bloom vanishes


def cma_space(string):
    """FIT_SPACE narrowed for the CMA-ES: pluck_deg 80-90 deg; target_gap from 0 if a pressed crest (-1 um) has no
    valid rest state on this bridge at the reference pitch (the pressed side is then probed by pressed_check)."""
    sp = dict(FIT_SPACE, **CMA_NARROW)
    try:
        base_preset(string, target_f1(string), -1e-6, "2d")
    except ValueError:
        sp["target_gap"] = (0.0, FIT_SPACE["target_gap"][1], "lin")
    return sp


def _cma_stage(string, stage, ref, x0, names, gens, lam, run_stage, seed=0):
    sp = cma_space(string)
    u0 = [to_unit(k, x0[k], sp) for k in names]
    es = CMA(u0, 0.15, lam, seed)
    # resumable: the same seed and the cached scores replay every generation already evaluated
    for g in range(gens):
        U = es.ask()
        X = [dict(x0, **{k: from_unit(k, u[i], sp) for i, k in enumerate(names)}) for u in U]
        res = run_stage(f"{stage}{g}", X)
        f = [r["refs"][ref]["D"] if r.get("ok") else 1e9 for r in res]
        es.tell(U, f)
    return x0


# ------------------------------------------------------------------------------------------------ final renders
FINAL_DUR = 8.0
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
C_2D, C_1D, C_PRE = "#2a78d6", "#eb6834", "#1baf7a"      # categorical slots 1-3 (dataviz reference palette)
SEQ = ("#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")   # blue ramp


def k_weight(y, sr=SR):
    """ITU-R BS.1770 K-weighting (RBJ-cookbook realisation: +4 dB high shelf at 1.5 kHz, 38 Hz 2nd-order HP)."""
    A = 10 ** (4.0 / 40)
    w0 = 2 * np.pi * 1500.0 / sr
    al = np.sin(w0) / 2 * np.sqrt(2)
    cw = np.cos(w0)
    b = [A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * cw),
         A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * al)]
    a = [(A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * al, 2 * ((A - 1) - (A + 1) * cw),
         (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * al]
    y = ss.lfilter(b, a, y)
    return ss.sosfilt(ss.butter(2, 38.0, "highpass", fs=sr, output="sos"), y)


def loudness_db(y, t1=4.0):
    """K-weighted mean-square level (dB) of the first t1 seconds (an ungated BS.1770-style loudness)."""
    z = k_weight(np.asarray(y, float)[:int(t1 * SR)])
    return float(10 * np.log10(np.mean(z ** 2) + 1e-30))


def peak_norm(y, dbfs=-1.0):
    return y * (10 ** (dbfs / 20) / max(np.max(np.abs(y)), 1e-30))


def exact_features(y, f0, B, tcut):
    """features of a radiated signal that already starts at its onset (no closed-loop shortcut)."""
    f, t, Z = stft(y)
    sl, _ = partial_slices(f, f0, B)
    A = penv(np.abs(Z) ** 2, sl)
    return t, A, features(t, A, tcut)


def radiate_post(y, q):
    import body
    return body.apply_body(np.asarray(y, float), SR, "tanpura_male", hp_hz=q["hp_hz"], hp_order=int(q["hp_order"]),
                           mode_gain_db=q["mode_gain_db"])


def _row(label, fe, D=None, f0=None):
    sh = fe["share"]
    hf = fe["hf31_60"]
    dd = f"{D[0]:6.1f} {D[1]:6.1f}" if D is not None else f"{'':6s} {'':6s}"
    return (f"  {label:44s} {fe['rise_db']:5.1f} {fe['t_max']:5.2f} {fe['t_m3']:5.2f}  "
            f"{sh.get('0.09', np.nan):.2f}/{sh.get('1', np.nan):.2f}/{sh.get('3', np.nan):.2f}   "
            f"{hf.get('1', np.nan):6.1f}/{hf.get('3', np.nan):6.1f}  {dd}" + (f"  {f0:8.3f}" if f0 else ""))


def final_string(string, jobs=2):
    """Best fit per reference -> 8 s renders, WAVs, exact features, match table, sensitivity, PNG."""
    import soundfile as sf
    log = Log(OUT / f"final_{string}.txt")
    log(f"\n==== fit2d final {string}  {time.strftime('%Y-%m-%d %H:%M:%S')} ====")
    hist = load_history(string)
    if not hist:
        raise SystemExit(f"no fit history for {string}: run  python fit2d.py fit {string}")
    own, itan = STRING_REFS[string]
    wav = OUT / "wav"
    wav.mkdir(parents=True, exist_ok=True)
    png = OUT / "png"
    png.mkdir(parents=True, exist_ok=True)
    best = {rn: best_of(hist, rn) for rn in (own, itan)}
    # 8 s renders: best 2-D per reference, the current 1-D preset and the current 2-D preset (same f1)
    X = {f"best_{rn}": ("2d", dict(DEFAULT_X, **best[rn]["x"])) for rn in (own, itan)}
    X["1d"] = ("1d", dict(DEFAULT_X))
    X["preset2d"] = ("2d", dict(DEFAULT_X))
    import multiprocessing as mp
    todo = [(string, x, FINAL_DUR, m, 0.0) for (m, x) in X.values()]
    with mp.get_context("fork").Pool(min(jobs, len(todo))) as pool:
        pool.starmap(render, todo, chunksize=1)
    runs = {k: render(string, x, FINAL_DUR, m) for k, (m, x) in X.items()}
    table = {}
    curves = {}
    log(f"  {'signal':44s} {'rise':>5s} {'t_max':>5s} {'t_m3':>5s}  {'H4-10 .09/1/3 s':16s} "
        f"{'H31-60 dB 1/3 s':15s} {'D':>6s} {'Dbloom':>6s}  {'f0 Hz':>8s}")
    for rn in (own, itan):
        R = ref_features(rn)
        log(f"  -- reference {rn} ({'PRIMARY: the user’s own tanpura' if rn == own else 'secondary: iTanpura'}),"
            f" f0 {R['f0']:.3f} Hz, analysed 0-{R['tcut']:.2f} s")
        log(_row(f"{rn} (recording)", R["feat"], None, R["f0"]))
        curves[rn] = {"ref": (R["t"], R["A"])}
        table[rn] = dict(ref=dict(feat=R["feat"], f0=R["f0"]))
        b = best[rn]
        q = b["refs"][rn]["post"]
        entries = [(f"best_{rn}", f"2-D best fit [{_fmt_post(q)}]", q, "best2d"),
                   ("preset2d", "2-D preset, default chain (HP 250/2, gz 0.1)", POST_DEFAULT, "preset2d"),
                   ("1d", "1-D preset, default chain (HP 250/2)", dict(POST_DEFAULT, gz=0.0), "1d_default")]
        b1 = best_of(hist, rn, model="1d")
        if b1 is not None:
            entries.append(("1d", f"1-D preset, chain fitted [{_fmt_post(b1['refs'][rn]['post'])}]",
                            b1["refs"][rn]["post"], "1d_fitted"))
        for run_name, label, qq, tag in entries:
            run = runs[run_name]
            i0 = int(run["i0"])
            mono = run["Fy"].astype(float) + qq["gz"] * run["Fz"].astype(float)
            yr = radiate_post(mono, qq)[i0:]
            t, A, fe = exact_features(yr, float(run["f0"]), float(run["B"]), R["tcut"])
            D = distance(R["feat"], fe)
            log(_row(label, fe, D, float(run["f0"])))
            table[rn][tag] = dict(feat=fe, D=D[0], D_bloom=D[1], per=D[2], post=qq, f0=float(run["f0"]),
                                  x=run["x"] if run_name != "1d" else None, label=label)
            curves[rn][tag] = (t, A)
        # 5 s fit render vs 8 s final render (same parameters): same trajectory?
        r5 = render(string, best[rn]["x"], 5.0)
        n = min(len(r5["Fy"]), len(runs[f"best_{rn}"]["Fy"]))
        dmax = float(np.max(np.abs(r5["Fy"][:n] - runs[f"best_{rn}"]["Fy"][:n])))
        log(f"     5 s fit render vs 8 s final render, first 5 s: max |dF_y| = {dmax:.2e} "
            f"(peak {np.max(np.abs(r5['Fy'])):.2e})  |  fit-stage D was {b['refs'][rn]['D']:.1f}")
        x = best[rn]["x"]
        log(f"     best x: {_fmt_x(x)};  sounding f0 {float(runs[f'best_{rn}']['f0']):.3f} Hz "
            f"({1200 * np.log2(float(runs[f'best_{rn}']['f0']) / R['f0']):+.1f} cents re the recording); z share of "
            f"string energy at 0.05/0.5/1/2/3 s: {'/'.join(f'{v:.2f}' for v in runs[f'best_{rn}']['ez'])}; "
            f"solver fails (Newton, friction) {int(runs[f'best_{rn}']['stats'][0])}, "
            f"{int(runs[f'best_{rn}']['stats'][1])}")
        # WAVs: model dry F_y, dry mono, through the body (-1 dBFS); the recording loudness-matched to the body one
        run = runs[f"best_{rn}"]
        i0 = max(0, int(run["i0"]))
        Fy, Fz = run["Fy"].astype(float)[i0:], run["Fz"].astype(float)[i0:]
        mono = Fy + q["gz"] * Fz
        yb = radiate_post(mono, q)
        tag = f"{string}_fit_{rn}"
        sf.write(wav / f"{tag}_model_dryFy.wav", peak_norm(Fy), SR, subtype="PCM_24")
        sf.write(wav / f"{tag}_model_mono.wav", peak_norm(mono), SR, subtype="PCM_24")
        yb = peak_norm(yb)
        sf.write(wav / f"{tag}_model_body.wav", yb, SR, subtype="PCM_24")
        yref = load_ref(rn)
        g = 10 ** ((loudness_db(yb) - loudness_db(yref)) / 20)
        yref = yref * g
        pk = float(np.max(np.abs(yref)))
        if pk > 10 ** (-1 / 20):             # keep the recording unclipped: lower BOTH to its peak
            s_ = 10 ** (-1 / 20) / pk
            yref *= s_
            sf.write(wav / f"{tag}_model_body.wav", yb * s_, SR, subtype="PCM_24")
        sf.write(wav / f"{rn}_reference_loudness_matched.wav", yref, SR, subtype="PCM_24")
        yr1 = peak_norm(radiate_post(runs["1d"]["Fy"].astype(float)[int(runs["1d"]["i0"]):],
                                     dict(POST_DEFAULT, gz=0.0)))
        g1 = 10 ** ((loudness_db(yb) - loudness_db(yr1)) / 20)
        y1 = yr1 * g1
        pk1 = float(np.max(np.abs(y1)))
        if pk1 > 10 ** (-1 / 20):           # never clip: the 1-D tone is then quieter than the others (logged)
            y1 *= 10 ** (-1 / 20) / pk1
            log(f"     1-D comparison WAV peak-limited: {20 * np.log10(pk1) + 1:.1f} dB below loudness match")
        sf.write(wav / f"{string}_1d_preset_body_matched_to_{rn}.wav", y1, SR, subtype="PCM_24")
    # sensitivity (grid marginals and screening steps) for the own reference
    sens = sensitivity(string, hist, own, itan)
    for line in sens["text"]:
        log(line)
    _plot_partials(string, curves, table)
    with open(OUT / f"final_{string}.json", "w") as fh:
        json.dump(dict(table=table, best={rn: best[rn] for rn in best}, sens=sens["data"]), fh, default=float,
                  indent=1)
    return table


def sensitivity(string, hist, own, itan):
    """What creates the bloom: grid marginals (mean over the other grid axes) and one-at-a-time screening steps."""
    text, data = [], {}
    grid = [h for h in hist if h["stage"] == "grid" and h.get("ok")]
    text.append(f"  sensitivity ({string}): grid marginals, mean over the other two grid axes "
                f"[rise dB, t_max s, H4-10 share at 1 s, H31-60 dB at 1 s, D own, D iTanpura]")
    for name, key in (("pluck_deg", lambda x: f"{x['pluck_deg']:g} deg"),
                      ("coupling", lambda x: f"{x['theta_nut']:g} deg / {x['detune_cents']:g} c"),
                      ("target_gap", lambda x: f"{x['target_gap'] * 1e6:+g} um")):
        groups = {}
        for h in grid:
            groups.setdefault(key(h["x"]), []).append(h)
        for lev, hs in groups.items():
            f = [h["refs"][own]["feat"] for h in hs]
            row = dict(rise=np.mean([v["rise_db"] for v in f]), t_max=np.mean([v["t_max"] for v in f]),
                       share1=np.mean([v["share"]["1"] for v in f]), hf1=np.mean([v["hf31_60"]["1"] for v in f]),
                       D_own=np.mean([h["refs"][own]["D"] for h in hs]),
                       D_it=np.mean([h["refs"][itan]["D"] for h in hs]))
            data[f"grid {name} {lev}"] = row
            text.append(f"    {name:11s} {lev:16s} rise {row['rise']:5.1f}  t_max {row['t_max']:4.2f}  "
                        f"H4-10@1 {row['share1']:.2f}  H31-60@1 {row['hf1']:6.1f}  D own {row['D_own']:6.1f}  "
                        f"D iT {row['D_it']:6.1f}")
    scr = [h for h in hist if h["stage"] in ("screen", "pressed") and h.get("ok")]
    keys = list(DEFAULT_X) + ["thread_shift"]

    def ndiff(a, b):
        return [k for k in keys if abs(a.get(k, 0.0) - b.get(k, 0.0)) > 1e-12]
    if scr:
        g = best_of([h for h in hist if h["stage"] in ("preset", "grid")], own)   # = fit_string's screening centre
        # screening centres: the grid winner, then (when the fit was resumed after a better screening point was
        # found) that point, then each reference's overall best (the pressed-crest check is made there)
        r1 = [h for h in scr if len(ndiff(h["x"], g["x"])) == 1]
        centres = [g, best_of([g] + r1, own)] + [best_of(hist, rn) for rn in (own, itan)]
        uniq = []
        for c in centres:
            if c is not None and all(ndiff(c["x"], u["x"]) for u in uniq):
                uniq.append(c)
        tw = [h for h in hist if h["stage"] == "twin" and h.get("ok")]
        ok_ = [h for h in hist if h.get("ok") and not h.get("eps") and h.get("model", "2d") == "2d"]
        noise, noise_r, noise_it = [], [], []
        for h in tw:
            c = next((u for u in ok_ if not ndiff(u["x"], h["x"])), None)
            if c is not None:
                noise.append(abs(h["refs"][own]["D"] - c["refs"][own]["D"]))
                noise_it.append(abs(h["refs"][itan]["D"] - c["refs"][itan]["D"]))
                noise_r.append(abs(h["refs"][own]["feat"]["rise_db"] - c["refs"][own]["feat"]["rise_db"]))
        text.append(f"  chaos twins (pluck_disp x (1 + 1e-9 / 2e-9)): |dD own| {', '.join(f'{v:.1f}' for v in noise)}, "
                    f"|dD iT| {', '.join(f'{v:.1f}' for v in noise_it)}, |d rise| "
                    f"{', '.join(f'{v:.2f}' for v in noise_r)} dB")
        data["twin_noise_D"] = noise
        for ci, c in enumerate(uniq):
            x0, a0 = c["x"], c["refs"][own]
            rows, seen = [], set()
            for h in scr:
                if len(ndiff(h["x"], x0)) == 1 and h is not c and h["key"] not in seen \
                        and not any(len(ndiff(h["x"], u["x"])) == 1 for u in uniq[:ci]):
                    rows.append(h)
                    seen.add(h["key"])
            if not rows:
                continue
            text.append(f"  one-at-a-time steps around centre {ci + 1} ({_fmt_x(x0)}; D own {a0['D']:.1f}, D iT "
                        f"{c['refs'][itan]['D']:.1f}, rise {a0['feat']['rise_db']:.1f} dB, t_max "
                        f"{a0['feat']['t_max']:.2f} s)")
            for h in rows:
                k = ndiff(h["x"], x0)[0]
                a = h["refs"][own]
                v0, v1 = x0.get(k, 0.0), h["x"].get(k, 0.0)
                unit = 1e6 if k == "target_gap" else 1e3 if k in ("pluck_disp", "thread_shift") else 1.0
                row = dict(centre=ci + 1, param=k, v0=v0, v1=v1, dD=a["D"] - a0["D"],
                           drise=a["feat"]["rise_db"] - a0["feat"]["rise_db"],
                           dtmax=a["feat"]["t_max"] - a0["feat"]["t_max"],
                           dshare1=a["feat"]["share"]["1"] - a0["feat"]["share"]["1"],
                           dhf1=a["feat"]["hf31_60"]["1"] - a0["feat"]["hf31_60"]["1"],
                           dD_it=h["refs"][itan]["D"] - c["refs"][itan]["D"])
                data[f"screen c{ci + 1} {k} {v1:g}"] = row
                text.append(f"    {k:12s} {v0 * unit:7.3g} -> {v1 * unit:7.3g}:  dD own {row['dD']:+6.1f}  dD iT "
                            f"{row['dD_it']:+6.1f}  d rise {row['drise']:+5.1f} dB  d t_max {row['dtmax']:+5.2f} s  "
                            f"d H4-10@1 {row['dshare1']:+.2f}  d H31-60@1 {row['dhf1']:+5.1f} dB")
    return dict(text=text, data=data)


def _smooth(v, k=5):
    if len(v) < k:
        return v
    w = np.ones(k) / k
    vp = np.concatenate([np.full(k // 2, v[0]), v, np.full(k // 2, v[-1])])
    return np.convolve(vp, w, mode="valid")


def _plot_partials(string, curves, table):
    """Per reference (rows): partial-envelope images of the recording and of the 2-D best fit (radiated), and three
    feature tracks (level, share of partials 4-10, partials 31-60) for recording / 2-D best / 2-D preset / 1-D."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    refs = list(curves)
    fig = plt.figure(figsize=(19, 5.0 * len(refs) + 0.8))
    gs = fig.add_gridspec(len(refs), 7, width_ratios=(1.15, 1.15, 0.05, 0.13, 1, 1, 1), left=0.045, right=0.985,
                          top=1 - 0.95 / (5.0 * len(refs) + 0.8), bottom=0.11 / len(refs), wspace=0.32,
                          hspace=0.62)
    series = (("ref", "recording", INK, "-", 2.4), ("best2d", "2-D best fit (this fit)", C_2D, "-", 1.8),
              ("preset2d", "2-D preset, default radiation chain", C_PRE, "--", 1.4),
              ("1d_default", "1-D preset, default radiation chain", C_1D, "-", 1.3))
    for r_i, rn in enumerate(refs):
        tcut = table[rn]["ref"]["feat"]["tcut"]
        ax = [fig.add_subplot(gs[r_i, j]) for j in (0, 1, 2, 4, 5, 6)]
        for j, (tag, title) in enumerate((("ref", f"recording: {rn}"), ("best2d", "2-D best fit, radiated"))):
            t, A = curves[rn][tag]
            keep = t <= tcut
            Adb = 10 * np.log10(A[:, keep] + 1e-30)
            Adb -= Adb.max()
            im = ax[j].imshow(Adb, origin="lower", aspect="auto", cmap=cmap, vmin=-60, vmax=0,
                              extent=(t[keep][0], t[keep][-1], 0.5, NMAX + 0.5), interpolation="nearest")
            ax[j].set_title(title, fontsize=10, color=INK, loc="left")
            ax[j].set_xlabel("time after onset (s)")
            ax[j].set_ylabel("partial number")
        cb = fig.colorbar(im, cax=ax[2])
        cb.set_label("level re image max (dB)", color=INK2)
        cb.outline.set_edgecolor(MUTED)
        for tag, lab, col, ls, lw in series:
            if tag not in curves[rn]:
                continue
            t, A = curves[rn][tag]
            keep = t <= tcut
            t, A = t[keep], A[:, keep]
            tot = A.sum(0) + 1e-30
            L = 10 * np.log10(tot)
            ax[3].plot(t, L - L.max(), color=col, ls=ls, lw=lw)
            ax[4].plot(t, _smooth(A[3:10].sum(0) / tot), color=col, ls=ls, lw=lw)
            ax[5].plot(t, _smooth(10 * np.log10(A[30:60].sum(0) / tot + 1e-30)), color=col, ls=ls, lw=lw)
        ax[3].set_ylim(-30, 1.5)
        ax[3].set_title("level of partials 1-60 re max (dB)", fontsize=10, loc="left")
        ax[4].set_ylim(0, 1)
        ax[4].set_title("energy share of partials 4-10", fontsize=10, loc="left")
        ax[5].set_ylim(-50, -5)
        ax[5].set_title("partials 31-60 re total (dB)", fontsize=10, loc="left")
        for a in ax[3:]:
            a.set_xlabel("time after onset (s)")
            a.set_xlim(0, tcut)
            a.grid(alpha=0.25, lw=0.6)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        b = table[rn]["best2d"]
        ax[0].annotate(f"fit: {_fmt_x(b['x'])}\nradiation: {_fmt_post(b['post'])};   feature distance D {b['D']:.1f}"
                       f"  (2-D preset {table[rn]['preset2d']['D']:.1f}, 1-D preset {table[rn]['1d_default']['D']:.1f})",
                       xy=(0.0, -0.3), xycoords="axes fraction", fontsize=8.5, color=INK2, va="top")
    handles = [Line2D([], [], color=c, ls=ls, lw=lw) for _, _, c, ls, lw in series]
    fig.legend(handles, [s_[1] for s_ in series], loc="upper right", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.985, 1.0))
    fig.text(0.045, 1 - 0.25 / (5.0 * len(refs) + 0.8), f"{string}: partial envelopes, recording vs 2-D best fit "
             f"(sideways finger pluck); feature tracks smoothed over 100 ms", fontsize=12, color=INK, va="top")
    fig.savefig(OUT / "png" / f"{string}_partials_ref_vs_model.png", dpi=100)
    plt.close(fig)


def joint_own_chain(strings=("kharaj", "pa", "sa")):
    """One radiation chain for the user's instrument: the (gz, HP, mode gain) minimising the summed own-reference
    distance over the best renders of all strings (instrument + microphone properties should be shared)."""
    log = Log(OUT / "joint_chain.txt")
    tot, per_s = {}, {}
    for s in strings:
        hist = load_history(s)
        if not hist:
            continue
        rn = STRING_REFS[s][0]
        b = best_of(hist, rn)
        run = render(s, b["x"], 5.0)
        t, c, envs = model_envelopes(run)
        R = ref_features(rn)
        for (mgd, gz), A in envs.items():
            for o in POST_GRID["hp_order"]:
                for fc in POST_GRID["hp_hz"]:
                    D = distance(R["feat"], features(t, A * hp_gain(c, fc, o)[:, None], R["tcut"]))[0]
                    k = (gz, fc, o, mgd)
                    tot[k] = tot.get(k, 0.0) + D
                    per_s.setdefault(k, {})[s] = D
        per_s.setdefault("own", {})[s] = (b["refs"][rn]["D"], b["refs"][rn]["post"])
    if not tot:
        return None
    k = min(tot, key=tot.get)
    log(f"\n==== joint radiation chain for the own instrument {time.strftime('%Y-%m-%d %H:%M:%S')} ====")
    log(f"  best shared chain: gz {k[0]}, HP {k[1]:.0f} Hz / order {k[2]}, mode gain {k[3]:+.0f} dB; summed D "
        f"{tot[k]:.1f} vs {sum(v[0] for v in per_s['own'].values()):.1f} with per-string chains")
    for s, (D, q) in per_s["own"].items():
        log(f"    {s:7s} per-string chain D {D:6.1f} [{_fmt_post(q)}]  ->  shared chain D {per_s[k][s]:6.1f}")
    return dict(chain=dict(gz=k[0], hp_hz=k[1], hp_order=k[2], mode_gain_db=k[3]), D=tot[k])


def ref_check(strings=("kharaj", "pa", "sa")):
    """Independent cross-check with reference.py (the other workflow's analyser: heterodyne partial envelopes,
    per-partial decay, BI10 bloom, partial-octave tracks, after a closed-form colour EQ).  Read-only use."""
    log = Log(OUT / "reference_compare.txt")
    log(f"\n==== reference.py compare of the fit2d renders {time.strftime('%Y-%m-%d %H:%M:%S')} ====")
    try:
        import reference as RF
    except Exception as e:                  # the analyser belongs to another workflow; skip if it is unavailable
        log(f"  reference.py not usable: {type(e).__name__}: {e}")
        return
    wav = OUT / "wav"
    for s in strings:
        for rn in STRING_REFS[s]:
            for lab, f in (("2-D best fit", wav / f"{s}_fit_{rn}_model_body.wav"),
                           ("1-D preset", wav / f"{s}_1d_preset_body_matched_to_{rn}.wav")):
                if not f.exists():
                    continue
                try:
                    r = RF.compare(str(REF_DIR / REFS[rn][0]), str(f), string=s, model_label=lab,
                                   out=OUT / "reference_compare" / f"{rn}__{'2d_best' if lab[0] == '2' else '1d'}")
                    top = "; ".join(f"[{it['group']} {it['severity']:.1f}] {it['text']}" for it in r["items"][:4])
                    log(f"  {rn:16s} vs {lab:13s}: reference.py loss {r['loss']:7.1f} | top: {top}")
                except Exception as e:
                    log(f"  {rn:16s} vs {lab:13s}: compare failed: {type(e).__name__}: {e}")


def cmd_final(strings, jobs=2):
    for s in strings:
        final_string(s, jobs)
    joint_own_chain(strings)
    ref_check(strings)


# ------------------------------------------------------------------------------------------------ CLI
def cmd_refs():
    log = Log(OUT / "refs.txt")
    log(f"\n==== reference features {time.strftime('%Y-%m-%d %H:%M:%S')} (read in place from {REF_DIR}) ====")
    log(f"  {'reference':16s} {'f0 Hz':>8s} {'B':>8s} {'dur':>5s} {'rise':>5s} {'t_max':>5s} {'t_m3':>5s}  "
        f"{'H4-10 share 0.09/0.3/1/2/3 s':28s} {'H31-60 dB 1/3 s':15s}")
    for rn in REFS:
        R = ref_features(rn)
        f = R["feat"]
        log(f"  {rn:16s} {R['f0']:8.3f} {R['B']:8.1e} {R['dur']:5.2f} {f['rise_db']:5.1f} {f['t_max']:5.2f} "
            f"{f['t_m3']:5.2f}  {'/'.join(f'{v:.2f}' for v in f['share'].values()):28s} "
            f"{'/'.join(f'{v:.1f}' for v in f['hf31_60'].values()):15s}")


def main(argv):
    jobs = 2
    if "--jobs" in argv:
        jobs = min(2, int(argv[argv.index("--jobs") + 1]))      # at most 2 simulation processes
    args = [a for a in argv if not a.startswith("--") and a != str(jobs)]
    cmd = args[0] if args else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "refs":
        cmd_refs()
    elif cmd == "fit":
        for s in args[1:] or ("kharaj", "pa", "sa"):
            fit_string(s, jobs)
            pressed_check(s, jobs)
    elif cmd == "pressed":
        for s in args[1:] or ("kharaj", "pa", "sa"):
            pressed_check(s, jobs)
    elif cmd == "final":
        cmd_final(args[1:] or ("kharaj", "pa", "sa"), jobs)
    elif cmd == "refcheck":
        ref_check(args[1:] or ("kharaj", "pa", "sa"))
    elif cmd == "all":
        cmd_refs()
        for s in ("kharaj", "pa", "sa"):
            fit_string(s, jobs)
            pressed_check(s, jobs)
        cmd_final(("kharaj", "pa", "sa"), jobs)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
