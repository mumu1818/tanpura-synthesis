"""Round-3 refit of the 2-D tanpura model to the real plucks with a PHYSICAL output chain (FIT3.md).

    cd jawari
    python fit3.py refs                 # reference features (own, iTanpura, held-out iTanpura CS3Alt), read in place
    python fit3.py probe                # one-at-a-time physics probe on the kharaj (which knob closes which gap)
    python fit3.py fit kharaj [--jobs 2]    # grid -> CMA-ES -> geometry screening, shared string params; resumable
    python fit3.py fit pa sa
    python fit3.py ablate kharaj pa sa  # switch groove / body drain / pitch design / geometry off one at a time
    python fit3.py final kharaj pa sa   # pitch-corrected 6 s renders, WAVs, PNGs, match tables, held-out check
    python fit3.py all                  # everything

Everything goes to out/fit3/ (git-ignored).  Only fit2d.py's machinery is imported (features, distance, STFT, f0
fit, CMA-ES, loudness); fit2d.py itself is not changed.  The recordings are only READ from /home/user/tanpura-app.

What changed against fit2d (the judge's remaining gaps, out/fit2d -> results4/judge.json):
  * output chain CONSTRAINED to physics, fitted per recording in closed loop for every render (CHAIN_GRID):
      body.py 'tanpura_male' explicit modes with A0 in 55-80 Hz and T1 in 100-150 Hz, body-mode gain 0 dB (as
      designed), radiation high-pass <= 150 Hz and order <= 2 (or only the body's 20 Hz guard), in-plane weight
      gz 0-0.1.  No other EQ.  (fit2d allowed 400 Hz / 3rd order and -6 dB body modes.)
  * string parameters (SPACE) SHARED by the user's recording and the iTanpura fit sample of the same string; only
    the chain differs per recording.  Objective J = (2 D3_own + D3_itanpura) / 3.  Held out: iTanpura CS3Alt (chain
    re-fitted, string frozen).
  * new physics free (MODEL_2D_ROUND3.md): groove kappa_g / groove_z0 (string resting on the groove wall), body drain
    Yb_y (modal, bridge-normal) and Yb_z (in-plane), statics geometry (bridge curvature c, thread diameter d_t,
    crest gap) with design_f1 (psi re-designed so the thread grazes at x_c0 at the tension of f1), the finger
    release time (attack/click) and the pluck point.
  * distance D3 = fit2d's D + new terms (NEW_SPEC): partial-band shares at 0.09/0.5/1/2/3 s, attack (partials 16-31
    at 0.09 s, reference.py's > 4 kHz burst), level-curve RMS 0.1-3.8 s, late decay slope 1.5-3.8 s, second-bloom
    penalty (re-rise > 2 dB after 2 s).
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
import fit2d as F2  # noqa: E402  (features, distance, STFT, f0 fit, CMA-ES, loudness; not modified)

SR = F2.SR
OUT = F2.OUT_ROOT / "fit3"
CACHE = OUT / "cache"
REF_DIR = F2.REF_DIR
FEAT3_VERSION = "fit3-feat-1"

REFS3 = dict(F2.REFS)
REFS3.update({
    "alt_kharaj": ("iTanpura/Tanpura_CS3Alt_Kharaj.mp3", "kharaj"),     # HELD OUT: never used in the fit
    "alt_pa": ("iTanpura/Tanpura_CS3Alt_Pa.mp3", "pa"),
    "alt_sa": ("iTanpura/Tanpura_CS3Alt_Sa.mp3", "sa"),
})
FIT_REFS = {s: (f"own_{s}", f"itanpura_{s}") for s in ("kharaj", "pa", "sa")}
HELD_REF = {s: f"alt_{s}" for s in ("kharaj", "pa", "sa")}
REF_WEIGHT = {"own": 2.0, "itanpura": 1.0}
STRINGS = ("kharaj", "pa", "sa")


def _h(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ================================================================================================ references
def load_ref3(name):
    """Reference pluck, mono 44.1 kHz, trimmed at its onset (first sample > -30 dB of the peak, minus 5 ms).  Read-only."""
    import soundfile as sf
    rel, _ = REFS3[name]
    y, sr = sf.read(str(REF_DIR / rel), always_2d=True)
    y = y.mean(1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        y = ss.resample_poly(y, SR // g, int(sr) // g)
    return F2.trim_onset(y)


# ------------------------------------------------------------------------------------------------ new features
T_BAND = (0.09, 0.5, 1.0, 2.0, 3.0)
BANDS5 = ((1, 3), (4, 7), (8, 15), (16, 31), (31, 60))
T_LATE0, T_LATE1, T_RMS0, T_RMS1 = 1.5, 3.8, 0.1, 3.8
NEW_SPEC = dict(            # group: (tolerance, weight per item); capped squared like fit2d (CAP 25)
    band=(3.0, 0.3),        # 5 bands x 5 times, dB re total
    att16=(3.0, 2.0),       # partials 16-31 re total at 0.09 s (attack brightness)
    burst=(4.0, 1.5),       # reference.py HF burst: > 4 kHz energy, first 20 ms vs 20-200 ms (finger click)
    levrms=(1.5, 4.0),      # RMS of the level-curve difference 0.1-3.8 s (both re their own maximum)
    decay=(1.0, 2.0),       # late decay slope 1.5-3.8 s, dB/s
    rerise=(1.0, 2.0),      # second bloom: re-rise above the running minimum after 2 s, only the excess over
)                           #   max(2 dB, the recording's own re-rise) is penalised
CAP = F2.CAP


def hf_burst(y):
    """reference.py attack(): energy above max(4 kHz, 40 f0) (= 4 kHz here) in the first 20 ms (from 2 ms before the
    anchor) vs 20-200 ms, per unit time.  The signals start 5 ms before their onset (anchor at 5 ms)."""
    seg = np.asarray(y[:int(0.30 * SR)], float)
    h = ss.sosfiltfilt(ss.butter(4, 4000.0, "highpass", fs=SR, output="sos"), seg)
    a = int(0.005 * SR)
    ea = np.mean(h[max(0, a - int(0.002 * SR)):a + int(0.020 * SR)] ** 2)
    eb = np.mean(h[a + int(0.020 * SR):a + int(0.200 * SR)] ** 2)
    return float(10 * np.log10((ea + 1e-300) / (eb + 1e-300)))


def level_curve(t, A, tcut):
    keep = t <= tcut
    L = 10 * np.log10(A[:, keep].sum(0) + 1e-30)
    return t[keep], L - L.max()


def t_end_of(dur):
    """end of the late-time windows: 3.8 s, or 3.0 s for the 4 s files (they fade out from ~3.1 s: own/iTanpura Sa,
    iTanpura Pa, Alt Sa; the longer files fade from ~4.0-4.3 s)."""
    return 3.0 if dur < 4.05 else T_LATE1


def late_metrics(t, L, t1):
    sel = (t >= 2.0) & (t <= t1)
    Ls = L[sel]
    rr = float(np.max(Ls - np.minimum.accumulate(Ls))) if Ls.size else 0.0
    s2 = (t >= T_LATE0) & (t <= t1)
    slope = float(np.polyfit(t[s2], L[s2], 1)[0])
    return rr, slope


def new_features(t, A, tcut, burst, t_end):
    keep = t <= tcut
    tk, Ak = t[keep], A[:, keep]
    tot = Ak.sum(0) + 1e-30

    def i(x):
        return int(np.argmin(np.abs(tk - x)))
    band = {}
    for (a, b) in BANDS5:
        for x in T_BAND:
            if x <= tk[-1]:
                band[f"{a}-{b}@{x:g}"] = float(10 * np.log10(Ak[a - 1:b, i(x)].sum() / tot[i(x)] + 1e-30))
    tl, L = level_curve(t, A, tcut)
    rr, slope = late_metrics(tl, L, t_end)
    return dict(band=band, att16=band["16-31@0.09"], burst=float(burst), rerise=rr, decay=slope, t_end=t_end,
                Lt=tl.tolist(), L=L.tolist())


def new_distance(fr, fm):
    """New terms of D3 (NEW_SPEC); returns (total, per-group dict)."""
    per = {}
    tol, w = NEW_SPEC["band"]
    per["band"] = float(w * sum(min(((fm["band"][k] - fr["band"][k]) / tol) ** 2, CAP) for k in fr["band"]
                                if k in fm["band"]))
    for g in ("att16", "burst", "decay"):
        tol, w = NEW_SPEC[g]
        per[g] = float(w * min(((fm[g] - fr[g]) / tol) ** 2, CAP))
    tl = np.asarray(fr["Lt"])
    Lr = np.asarray(fr["L"])
    Lm = np.interp(tl, np.asarray(fm["Lt"]), np.asarray(fm["L"]))
    sel = (tl >= T_RMS0) & (tl <= fr["t_end"])
    rms = float(np.sqrt(np.mean((Lm[sel] - Lr[sel]) ** 2)))
    tol, w = NEW_SPEC["levrms"]
    per["levrms"] = float(w * min((rms / tol) ** 2, CAP))
    tol, w = NEW_SPEC["rerise"]
    ex = max(0.0, fm["rerise"] - max(2.0, fr["rerise"]))
    per["rerise"] = float(w * min((ex / tol) ** 2, CAP))
    per["_levrms_db"] = rms
    return float(sum(v for k, v in per.items() if not k.startswith("_"))), per


def distance3(R, fm2, fm3):
    """D3 = fit2d distance + new terms.  Returns (D3, D2, D2 bloom, Dnew, per)."""
    D2, Db, per2 = F2.distance(R["feat"], fm2)
    Dn, pern = new_distance(R["feat3"], fm3)
    return D2 + Dn, D2, Db, Dn, dict(per2, **{f"new_{k}": v for k, v in pern.items()})


_REF = {}


def ref3(name):
    """Reference analysis (in memory): f0 (sequential partial fit), B = 0 (the recordings are harmonic), partial
    envelopes, fit2d features, fit3 features."""
    if name in _REF:
        return _REF[name]
    y = load_ref3(name)
    string = REFS3[name][1]
    import presets as P1
    nominal = P1.STRING_SPECS[string]["geom"].get("f1", P1.KHARAJ_GEOM["f1"])
    f0, B_fit = F2.fit_f0_B(y, nominal)
    f, t, Z = F2.stft(y)
    sl, c = F2.partial_slices(f, f0, 0.0)
    A = F2.penv(np.abs(Z) ** 2, sl)
    tcut = min(4.0, len(y) / SR - 0.1)
    t_end = t_end_of(len(y) / SR)
    out = dict(name=name, string=string, f0=f0, B_fit=B_fit, dur=len(y) / SR, tcut=tcut, t_end=t_end, t=t, A=A,
               feat=F2.features(t, A, tcut), feat3=new_features(t, A, tcut, hf_burst(y), t_end))
    _REF[name] = out
    return out


# ================================================================================================ model
SPACE = {   # name: (lo, hi, scale) -- physically bounded string parameters (shared per string by both recordings)
    "pluck_deg": (82.0, 90.0, "lin"),        # stroke direction from the bridge normal towards the soundboard plane
    "pluck_release": (0.1e-3, 3e-3, "log"),  # finger roll-off (cos^2 release) time: attack brightness / click
    "pluck_disp": (1.5e-3, 3e-3, "lin"),     # static pad displacement along the stroke
    "xp_frac": (0.30, 0.36, "lin"),          # pluck point / L from the bridge
    "theta_nut": (0.0, 45.0, "lin"),         # nut principal-axis angle (y-z coupling)
    "detune_cents": (0.5, 8.0, "log"),       # nut axis-2 split (y-z coupling, sets the exchange period)
    "kcz_x": (0.5, 3.0, "log"),              # lateral / vertical thread stiffness
    "groove_kappa": (0.0, 30.0, "lin"),      # transverse groove curvature kappa_g [1/m] (0 = off)
    "groove_z0": (0.0, 1e-3, "lin"),         # groove bottom offset [m]: string rests on the groove wall
    "Yb_y": (0.0, 0.3, "lin"),               # bridge-normal MODAL body drain, peak Re Y of T(0,1) [s/kg] (0 = off)
    "Ybz_x": (0.2, 5.0, "log"),              # in-plane smooth drain Yb_z = Ybz_x * Yb (1 = isotropic = off)
    "drain_a0": (55.0, 80.0, "lin"),         # A0 (air/neck mode) of the DRAIN body [Hz] (same range as the chain body)
    "drain_t1": (100.0, 150.0, "lin"),       # T1 (tabli breathing mode) of the drain body [Hz]
    "c": (1.5, 6.0, "log"),                  # bridge (jawari) curvature [1/m], statics
    "d_t": (0.1e-3, 0.4e-3, "log"),          # thread diameter [m], statics
    "target_gap": (-1e-6, 3e-6, "lin"),      # crest gap of the thread-only rest shape [m], statics
    "delta_ve_x": (0.5, 2.0, "log"),         # visco-elastic loss x preset
}
QUANT = dict(pluck_deg=0.25, pluck_release=None, pluck_disp=0.05e-3, xp_frac=0.0025, theta_nut=0.5,
             detune_cents=None, kcz_x=None, groove_kappa=1.0, groove_z0=0.02e-3, Yb_y=0.005, Ybz_x=None,
             c=None, d_t=None, target_gap=0.25e-6, delta_ve_x=None, drain_a0=0.5, drain_t1=0.5)
LOGQ = 0.01     # relative quantisation step of the log-scaled parameters
START = dict(   # FIT2D best own-kharaj string (the round-3 demo baseline) with the new physics off
    pluck_deg=90.0, pluck_release=2e-3, pluck_disp=2.7e-3, xp_frac=1 / 3, theta_nut=45.0, detune_cents=7.0,
    kcz_x=2.5, groove_kappa=0.0, groove_z0=0.0, Yb_y=0.0, Ybz_x=1.0, c=3.0, d_t=0.2e-3, target_gap=0.0,
    delta_ve_x=1.0, drain_a0=65.0, drain_t1=120.0, drain_modes="gain")
PRESET_X = dict(START, pluck_deg=88.0, theta_nut=30.0, detune_cents=3.5, kcz_x=1.0, pluck_disp=2e-3)
SHARPEN0 = dict(kharaj=8.0, pa=4.0, sa=4.0)   # initial guess of the jawari pitch sharpening (cents above f1)
GEOM_X = ("c", "d_t", "target_gap")


def quantise(x):
    q = dict(START, **x)
    if q["drain_modes"] not in DRAIN_MODES:
        raise ValueError(f"drain_modes {q['drain_modes']!r} not in {DRAIN_MODES}")
    for k, st in QUANT.items():
        v = float(q[k])
        if st is None:
            q[k] = float(np.exp(np.round(np.log(v) / LOGQ) * LOGQ)) if v > 0 else v
        elif k == "xp_frac" and abs(v - 1 / 3) < 1e-9:
            q[k] = 1 / 3
        else:
            q[k] = float(np.round(v / st) * st)
    if q["groove_kappa"] <= 0:
        q["groove_kappa"], q["groove_z0"] = 0.0, 0.0
    if q["Yb_y"] <= 0:      # no modal drain: its body is irrelevant (canonical key)
        q["drain_a0"], q["drain_t1"], q["drain_modes"] = START["drain_a0"], START["drain_t1"], START["drain_modes"]
    for k in ("c", "d_t", "detune_cents", "kcz_x", "Ybz_x", "delta_ve_x", "pluck_release"):
        for ref in (START[k], 1.0):   # snap to the start value / unity so that "off" is exactly off
            if ref and abs(q[k] / ref - 1) < 0.5 * LOGQ:
                q[k] = ref
    return q


DRAIN_MODES = ("gain", "axisym", "low")


def drain_body(x):
    """Body whose explicit modes shape the bridge-normal modal drain (tanpura2d drain_body / drain_weights).
    'gain'   : all body.py explicit modes with body.py's relative gains (MODEL_2D_ROUND3: the plate modes at 320-860 Hz
               also drain partials 4-12);
    'axisym' : A0 + the axisymmetric T(0,n) modes;
    'low'    : only the two lowest body modes, A0 (air/neck) and T(0,1) (tabli breathing) -- ASSUMPTION: the
               bridge-normal drive-point mobility below ~200 Hz is dominated by these two (whole-body) modes.
    A0/T1 are moved to drain_a0 / drain_t1 (the plate modes scale with T1 as in body.explicit_modes)."""
    import body
    bp = dict(body.PRESETS["tanpura_male"], a0_hz=float(x["drain_a0"]), t1_hz=float(x["drain_t1"]))
    if x["drain_modes"] == "low":
        bp["modes"] = [m for m in body.explicit_modes(bp) if m[0].startswith("A0") or m[0] == "T(0,1)"]
        return bp, "gain"
    return bp, x["drain_modes"]


def clamp(name, v):
    lo, hi, _ = SPACE[name]
    return float(np.clip(v, lo, hi))


def to_unit(name, v):
    return F2.to_unit(name, v, SPACE)


def from_unit(name, u):
    return F2.from_unit(name, u, SPACE)


_F1 = {}


def target_f1(string, sharpen=None):
    """contact-off f1 such that the model SOUNDS near the own recording's pitch (jawari sharpening `sharpen` cents)."""
    s = SHARPEN0[string] if sharpen is None else sharpen
    f_ref = ref3(FIT_REFS[string][0])["f0"]
    return round(f_ref * 2 ** (-s / 1200), 3)


def base_preset(string, f1, c, d_t, gap, design=True):
    """presets2d.preset2d at f1 with the statics geometry (c, d_t, crest gap) and, with design=True, psi re-designed
    (design_f1) so that the thread grazes at x_c0 at the tension of f1.  Cached on disk (statics ~20-60 s)."""
    key = dict(string=string, f1=round(f1, 4), c=round(c, 5), d_t_um=round(d_t * 1e6, 3),
               gap_nm=int(round(gap * 1e9)), design=bool(design))
    path = CACHE / "geom" / f"{string}_{_h(key)}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    fail = path.with_suffix(".fail")
    if fail.exists():
        raise ValueError(fail.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    import presets2d as P2
    kw = dict(f1=f1, target_gap=gap)
    if abs(c - 3.0) > 1e-12:
        kw["c"] = float(c)
    if abs(d_t - 0.2e-3) > 1e-15:
        kw["d_t"] = float(d_t)
    if design:
        kw["design_f1"] = True
    try:
        p = P2.preset2d(string, **kw)
    except ValueError as e:
        fail.write_text(str(e))
        raise
    with open(path, "wb") as fh:
        pickle.dump(p, fh)
    return p


def make_params(string, x, dur, f1=None, design=True):
    x = quantise(x)
    f1 = target_f1(string) if f1 is None else f1
    p = dict(base_preset(string, f1, x["c"], x["d_t"], x["target_gap"], design))
    p["dur"] = float(dur)
    p["pluck_disp"] = float(x["pluck_disp"])
    p["pluck_release"] = float(x["pluck_release"])
    p["xp"] = float(x["xp_frac"]) * p["L"]
    p["delta_ve"] = p["delta_ve"] * x["delta_ve_x"]
    p["pluck_deg"] = float(x["pluck_deg"])
    p["theta_nut"] = float(x["theta_nut"])
    p["detune_cents"] = float(x["detune_cents"])
    p["kcz"] = None if x["kcz_x"] == 1.0 else float(p["kc"] * x["kcz_x"])
    if x["groove_kappa"] > 0:
        p["groove_kappa"] = float(x["groove_kappa"])
        p["groove_z0"] = float(x["groove_z0"])
    if x["Yb_y"] > 0:
        p["Yb_y"] = float(x["Yb_y"])
        p["drain_body"], p["drain_weights"] = drain_body(x)
    if x["Ybz_x"] != 1.0:
        p["Yb_z"] = float(p["Yb"] * x["Ybz_x"])
    return p


def run_key(string, x, dur, f1, design=True, extra=None):
    xq = quantise(x)
    if xq["Yb_y"] <= 0:          # keys of renders without a modal drain do not depend on the drain body
        for k in ("drain_a0", "drain_t1", "drain_modes"):
            xq.pop(k)
    return _h(dict(string=string, x=xq, dur=dur, f1=round(f1, 4), design=bool(design), extra=extra,
                   v="fit3-run-1"))


def render(string, x, dur=4.2, f1=None, design=True, rec_modes=0):
    """Render (cached): 44.1 kHz F_y, F_z, onset index i0, sounding f0, solver stats, y/z energy shares.
    rec_modes > 0 also stores the per-mode energies (round-3 kernel; same trajectory as the original kernel)."""
    x = quantise(x)
    f1 = target_f1(string) if f1 is None else float(f1)
    key = run_key(string, x, dur, f1, design, extra=dict(rm=rec_modes) if rec_modes else None)
    path = CACHE / "runs" / f"{string}_{key}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files} | dict(key=key, x=x)
    path.parent.mkdir(parents=True, exist_ok=True)
    from common import to_audio
    import presets2d as P2
    import tanpura2d as T2
    p = make_params(string, x, dur, f1, design)
    sp = P2.sim_params(p)
    if rec_modes:
        sp["rec_modes"] = int(rec_modes)
    t0 = time.time()
    pp, s, r = T2.simulate(sp, energy=False)
    cpu = time.time() - t0
    Fy, Fz = to_audio(r["Fy"], pp["fs"]), to_audio(r["Fz"], pp["fs"])
    st = r["stats"]
    Ez = r["Ez"] / np.maximum(r["Ey"] + r["Ez"], 1e-30)
    tt = (0.05, 0.5, 1.0, 2.0, 3.0)
    ez = [float(Ez[np.argmin(np.abs(r["t_rec"] - v))]) for v in tt]
    i0 = int(round((p["pluck_pull"] + p["pluck_release"] - 0.005) * SR))
    f0, B = F2.fit_f0_B(Fy[i0:], p["f1"])
    extra = {}
    if rec_modes:
        mE = r["mEy"] + r["mEz"] + r["mEx"]
        extra = dict(mE13=mE[:, :3].sum(1) / np.maximum(mE.sum(1), 1e-300), mE_t=r["t_modes"],
                     mE=mE.astype(np.float32))
    np.savez(path, Fy=Fy.astype(np.float32), Fz=Fz.astype(np.float32), cpu=cpu, f0=f0, B=B, f1=p["f1"], i0=i0,
             ez=np.array(ez), stats=np.array([st["newton_fail"], st["fric_fail"], st["max_pen"], st["contact_frac"]],
                                             float),
             T=float(s["T"]), psi=float(p["meta"].get("psi_deg", np.nan)), x_c=float(p["meta"].get("x_c", np.nan)),
             x=json.dumps(x), string=string, kernel=str(st.get("kernel", "_run")), **extra)
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files} | dict(key=key, x=x)


# ------------------------------------------------------------------------------------------------ output chain
BODY_A0 = (55.0, 65.0, 80.0)
BODY_T1 = (100.0, 120.0, 150.0)
CHAIN_GRID = dict(gz=(0.0, 0.02, 0.05, 0.1), hp=((0.0, 0), (60.0, 1), (60.0, 2), (80.0, 1), (80.0, 2),
                                                   (100.0, 1), (100.0, 2), (125.0, 1), (125.0, 2), (150.0, 1),
                                                   (150.0, 2)))
CHAIN_PHYS = dict(a0=65.0, t1=120.0, gz=0.1, hp_hz=150.0, hp_order=2)     # round-3 demo's fixed physical chain
_IR = {}


def body_params(a0, t1):
    import body
    return dict(body.PRESETS["tanpura_male"], a0_hz=float(a0), t1_hz=float(t1))


def body_ir(a0, t1):
    """body.body_ir of tanpura_male with A0/T1 moved (seed 0, body-mode gain 0 dB), without its 20 Hz guard (the
    radiation high-pass is applied per partial: |H(n f0)|^2; hp (0, 0) = the body's own 20 Hz 2nd-order guard)."""
    import body
    k = (a0, t1)
    if k not in _IR:
        h, _ = body.body_ir(body_params(a0, t1), SR, 0, 1, hp_hz=0)
        _IR[k] = h[:, 0]
    return _IR[k]


def hp_gain(c, hp_hz, order):
    if hp_hz <= 0:
        return F2.hp_gain(c, 20.0, 2)
    return F2.hp_gain(c, hp_hz, order)


def chain_signal(run, q, start_at_onset=True):
    """Radiated time signal of a render through chain q (exact body.apply_body; un-normalised): body(F_y + gz F_z)
    then the high-pass; trimmed at the onset after filtering (as scored)."""
    import body
    y = np.asarray(run["Fy"], float) + q["gz"] * np.asarray(run["Fz"], float)
    hp = (20.0, 2) if q["hp_hz"] <= 0 else (q["hp_hz"], int(q["hp_order"]))
    out = body.apply_body(y, SR, body_params(q["a0"], q["t1"]), hp_hz=hp[0], hp_order=hp[1], normalize=False)
    return out[int(run["i0"]):] if start_at_onset else out


def model_variants(run):
    """{(a0, t1, gz): (A, burst)} partial envelopes before the high-pass, and t, c_n.  Linear: body(F_y + gz F_z) =
    body(F_y) + gz body(F_z); the high-pass acts per partial (|H|^2) and does not touch the > 4 kHz burst."""
    Fy, Fz, i0 = run["Fy"].astype(float), run["Fz"].astype(float), int(run["i0"])
    out = {}
    t = c = None
    for a0 in BODY_A0:
        for t1 in BODY_T1:
            h = body_ir(a0, t1)
            yb = ss.fftconvolve(Fy, h)[:len(Fy)][i0:]
            zb = ss.fftconvolve(Fz, h)[:len(Fz)][i0:]
            f, t, Zy = F2.stft(yb)
            _, _, Zz = F2.stft(zb)
            sl, c = F2.partial_slices(f, float(run["f0"]), 0.0)
            hi = max(h_ for _, h_ in sl)
            Zy, Zz = Zy[:hi], Zz[:hi]
            for gz in CHAIN_GRID["gz"]:
                out[(a0, t1, gz)] = (F2.penv(np.abs(Zy + gz * Zz) ** 2, sl), hf_burst(yb + gz * zb))
    return t, c, out


def score_chain(t, c, A, burst, R, q):
    Ah = A * hp_gain(c, q["hp_hz"], q["hp_order"])[:, None]
    fm2 = F2.features(t, Ah, R["tcut"])
    fm3 = new_features(t, Ah, R["tcut"], burst, R["t_end"])
    D3, D2, Db, Dn, per = distance3(R, fm2, fm3)
    return D3, dict(D3=D3, D2=D2, D2_bloom=Db, Dnew=Dn, per=per, post=q, feat=fm2, feat3=fm3)


def score_run(run, ref_names):
    """Best physical chain per reference (closed-loop grid) and its D3; also the fixed physical chain CHAIN_PHYS.
    Cached per run key and feature version."""
    path = CACHE / "scores" / f"{run['key']}_{_h([FEAT3_VERSION, sorted(ref_names), CHAIN_GRID, BODY_A0, BODY_T1])}.json"
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    t, c, var = model_variants(run)
    res = {}
    for rn in ref_names:
        R = ref3(rn)
        best = None
        for (a0, t1, gz), (A, burst) in var.items():
            for (fc, o) in CHAIN_GRID["hp"]:
                q = dict(a0=a0, t1=t1, gz=gz, hp_hz=fc, hp_order=o)
                D3, info = score_chain(t, c, A, burst, R, q)
                if best is None or D3 < best[0]:
                    best = (D3, info)
        A, burst = var[(CHAIN_PHYS["a0"], CHAIN_PHYS["t1"], CHAIN_PHYS["gz"])]
        _, phys = score_chain(t, c, A, burst, R, CHAIN_PHYS)
        res[rn] = dict(best[1], phys=dict(D3=phys["D3"], D2=phys["D2"], Dnew=phys["Dnew"]))
    out = dict(key=run["key"], x=run["x"], f0=float(run["f0"]), f1=float(run["f1"]), cpu=float(run["cpu"]),
               ez=[float(v) for v in run["ez"]], stats=[float(v) for v in run["stats"]], refs=res,
               psi=float(run["psi"]), x_c=float(run["x_c"]), kernel=str(run["kernel"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh)
    return out


def objective(sc, string):
    """J = (2 D3_own + D3_itanpura) / 3 (shared string, chain per recording)."""
    own, it = FIT_REFS[string]
    return (REF_WEIGHT["own"] * sc["refs"][own]["D3"] + REF_WEIGHT["itanpura"] * sc["refs"][it]["D3"]) / \
        (REF_WEIGHT["own"] + REF_WEIGHT["itanpura"])


# ------------------------------------------------------------------------------------------------ evaluation pool
def _eval_one(args):
    string, x, dur, f1, design = args
    try:
        run = render(string, x, dur, f1, design)
        sc = score_run(run, FIT_REFS[string] + (HELD_REF[string],))
        sc["J"] = objective(sc, string)
        sc["ok"] = True
        sc["design"] = design
        return sc
    except Exception as e:   # e.g. a geometry with no valid rest state: a bad point, not a crash
        import traceback
        return dict(ok=False, x=quantise(x), err=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-800:],
                    J=1e9, design=design)


def presolve(string, X, f1=None, design=True, log=print):
    f1 = target_f1(string) if f1 is None else f1
    geoms = sorted({(q["c"], q["d_t"], q["target_gap"]) for q in (quantise(x) for x in X)})
    for (c, dt, g) in geoms:
        t0 = time.time()
        try:
            base_preset(string, f1, c, dt, g, design)
        except Exception as e:
            log(f"    statics failed: c {c:.2f}, d_t {dt * 1e3:.3f} mm, gap {g * 1e6:+.2f} um: {e}")
            continue
        if time.time() - t0 > 1:
            log(f"    statics c {c:.2f} /m, d_t {dt * 1e3:.3f} mm, gap {g * 1e6:+.2f} um, f1 {f1}: {time.time() - t0:.0f} s")


def evaluate(string, X, dur=4.2, jobs=2, f1=None, design=True, log=print):
    for rn in FIT_REFS[string] + (HELD_REF[string],):
        ref3(rn)                                  # computed once, inherited by the forked workers
    f1 = target_f1(string) if f1 is None else f1
    presolve(string, X, f1, design, log)
    todo = [(string, x, dur, f1, design) for x in X]
    t0 = time.time()
    if jobs > 1 and len(todo) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(jobs, len(todo))) as pool:
            res = pool.map(_eval_one, todo, chunksize=1)
    else:
        res = [_eval_one(a) for a in todo]
    log(f"    {len(todo)} evaluations, {time.time() - t0:.0f} s wall")
    return res


# ================================================================================================ reporting helpers
class Log(F2.Log):
    pass


def fmt_x(x):
    x = quantise(x)
    s = (f"pluck {x['pluck_deg']:.2f} deg rel {x['pluck_release'] * 1e3:.2f} ms disp {x['pluck_disp'] * 1e3:.2f} mm "
         f"xp {x['xp_frac']:.4f} | nut {x['theta_nut']:.1f} deg/{x['detune_cents']:.2f} c kcz x{x['kcz_x']:.2f}")
    if x["groove_kappa"] > 0:
        s += f" | groove {x['groove_kappa']:.0f}/m z0 {x['groove_z0'] * 1e3:.2f} mm"
    if x["Yb_y"] > 0:
        s += f" | Yb_y {x['Yb_y']:.3f} ({x['drain_modes']}, A0 {x['drain_a0']:.1f} T1 {x['drain_t1']:.1f} Hz)"
    if x["Ybz_x"] != 1.0:
        s += f" | Yb_z x{x['Ybz_x']:.2f}"
    if x["c"] != 3.0 or x["d_t"] != 0.2e-3 or x["target_gap"] != 0.0:
        s += f" | c {x['c']:.2f}/m d_t {x['d_t'] * 1e3:.3f} mm gap {x['target_gap'] * 1e6:+.2f} um"
    if x["delta_ve_x"] != 1.0:
        s += f" | dve x{x['delta_ve_x']:.2f}"
    return s


def fmt_post(q):
    hp = "HP none" if q["hp_hz"] <= 0 else f"HP {q['hp_hz']:.0f}/{q['hp_order']}"
    return f"A0 {q['a0']:.0f} T1 {q['t1']:.0f} gz {q['gz']:.2f} {hp}"


def feat_line(f2, f3):
    b = f3["band"]
    return (f"rise {f2['rise_db']:5.1f} tmax {f2['t_max']:4.2f} | 1-3/4-7/8-15/16-31/31-60 @1s "
            + "/".join(f"{b[f'{a}-{c}@1']:5.1f}" for a, c in BANDS5)
            + f" | att16 {f3['att16']:5.1f} burst {f3['burst']:+5.1f} | rerise {f3['rerise']:4.1f} decay {f3['decay']:+5.1f}")


def score_line(sc, rn):
    r = sc["refs"][rn]
    return (f"{rn:16s} D3 {r['D3']:6.1f} (D2 {r['D2']:5.1f} new {r['Dnew']:5.1f}, levrms {r['per']['new__levrms_db']:4.1f} dB)"
            f" phys {r['phys']['D3']:6.1f} | {feat_line(r['feat'], r['feat3'])} | {fmt_post(r['post'])}")


def ref_line(rn):
    R = ref3(rn)
    return f"{rn:16s} {'recording':>56s} | {feat_line(R['feat'], R['feat3'])}"


# ================================================================================================ probe
PROBE = {   # one-at-a-time from START (FIT2D best own-kharaj string, physics off), pitch design on
    "start (FIT2D best string)": {},
    "split 2 c": dict(detune_cents=2.0),
    "split 1 c": dict(detune_cents=1.0),
    "nut 20 deg": dict(theta_nut=20.0),
    "release 0.3 ms": dict(pluck_release=0.3e-3),
    "pluck 85 deg": dict(pluck_deg=85.0),
    "groove wall 30/m z0 1 mm": dict(groove_kappa=30.0, groove_z0=1e-3),
    "drain Yb_y 0.1": dict(Yb_y=0.1),
    "in-plane drain Yb_z x4": dict(Ybz_x=4.0),
    "kcz x1": dict(kcz_x=1.0),
    "c 1.5 /m": dict(c=1.5),
    "c 6 /m": dict(c=6.0),
    "d_t 0.1 mm": dict(d_t=0.1e-3),
    "d_t 0.4 mm": dict(d_t=0.4e-3),
    "gap -1 um": dict(target_gap=-1e-6),
}


PROBE2 = {
    "low drain 0.1 (A0 65 T1 120)": dict(Yb_y=0.1, drain_modes="low"),
    "low drain 0.1 (A0 69 T1 139)": dict(Yb_y=0.1, drain_modes="low", drain_a0=69.5, drain_t1=139.0),
    "low drain 0.3 (A0 69 T1 139)": dict(Yb_y=0.3, drain_modes="low", drain_a0=69.5, drain_t1=139.0),
    "gain drain 0.1 (A0 69 T1 139)": dict(Yb_y=0.1, drain_modes="gain", drain_a0=69.5, drain_t1=139.0),
    "low drain 0.3 + split 3 c": dict(Yb_y=0.3, drain_modes="low", drain_a0=69.5, drain_t1=139.0, detune_cents=3.0),
    "low drain 0.3 + rel 0.5 ms + groove wall": dict(Yb_y=0.3, drain_modes="low", drain_a0=69.5, drain_t1=139.0,
                                                     pluck_release=0.5e-3, groove_kappa=30.0, groove_z0=1e-3),
    "rel 0.5 ms + groove wall": dict(pluck_release=0.5e-3, groove_kappa=30.0, groove_z0=1e-3),
    "low drain 0.3 + pluck 87 + rel 0.5": dict(Yb_y=0.3, drain_modes="low", drain_a0=69.5, drain_t1=139.0,
                                               pluck_deg=87.0, pluck_release=0.5e-3),
}


def cmd_probe(string="kharaj", jobs=2, cases=None):
    log = Log(OUT / f"probe_{string}.txt")
    log(f"\n==== fit3 probe {string}  {time.strftime('%Y-%m-%d %H:%M:%S')}  f1 {target_f1(string)} (design_f1) ====")
    cases = cases or PROBE
    X = [dict(START, **v) for v in cases.values()]
    res = evaluate(string, X, jobs=jobs, log=log)
    own, it = FIT_REFS[string]
    for rn in (own, it, HELD_REF[string]):
        log(ref_line(rn))
    for lab, sc in zip(cases, res):
        if not sc["ok"]:
            log(f"  {lab:28s} FAILED {sc['err']}")
            continue
        log(f"  {lab:28s} J {sc['J']:6.1f}  f0 {sc['f0']:.3f} ({1200 * np.log2(sc['f0'] / ref3(own)['f0']):+.1f} c) "
            f"cpu {sc['cpu']:.0f} s ez {'/'.join(f'{v:.2f}' for v in sc['ez'])} {sc['kernel']}")
        for rn in (own, it):
            log("      " + score_line(sc, rn))
    return res



# ================================================================================================ search
STRING_START = {   # FIT2D's best own-recording string per string (FIT2D.md), new physics off
    "kharaj": dict(START),
    "pa": dict(START, pluck_deg=88.0, theta_nut=15.0, detune_cents=3.5, kcz_x=1.0, pluck_disp=2.0e-3),
    "sa": dict(START, pluck_deg=89.0, theta_nut=30.0, detune_cents=3.3, kcz_x=2.03, pluck_disp=2.05e-3,
               target_gap=1e-6),
}
FIT2D_BEST_KEY = {"kharaj": "13e81d618d1a19f0", "pa": "efc8015e5cbf3160", "sa": "c785be2238be83c1"}   # 5 s fit renders
# stage-1 grid (from the probes, FIT3.md): coupling split x nut angle x string on the groove wall x finger release
GRID3 = {
    "kharaj": {"detune_cents": (2.0, 3.5, 7.0), "theta_nut": (30.0, 45.0),
               ("groove_kappa", "groove_z0"): ((0.0, 0.0), (30.0, 1e-3)), "pluck_release": (0.7e-3, 2e-3)},
    "pa": {"detune_cents": (2.0, 3.5, 7.0), "theta_nut": (15.0, 35.0),
           ("groove_kappa", "groove_z0"): ((0.0, 0.0), (30.0, 1e-3)), "pluck_release": (0.7e-3, 2e-3)},
    "sa": {"detune_cents": (2.0, 3.3, 7.0), "theta_nut": (30.0, 45.0),
           ("groove_kappa", "groove_z0"): ((0.0, 0.0), (30.0, 1e-3)), "pluck_release": (0.7e-3, 2e-3)},
}
CMA_NAMES = ("pluck_deg", "pluck_release", "pluck_disp", "theta_nut", "detune_cents", "kcz_x", "Yb_y",
             "groove_z0")
GEOM_SCREEN = dict(c=(1.5, 6.0), d_t=(0.15e-3, 0.3e-3), target_gap=(-1e-6, 2e-6), xp_frac=(0.30, 0.36),
                   delta_ve_x=(0.5, 2.0), Ybz_x=(0.25, 4.0))   # d_t 0.1 / 0.4 mm: no valid rest state (probe)
FIT_DUR = 4.2


def history_path(string):
    return OUT / f"history_{string}.jsonl"


def append_history(string, stage, res):
    history_path(string).parent.mkdir(parents=True, exist_ok=True)
    with open(history_path(string), "a") as fh:
        for r in res:
            fh.write(json.dumps(dict({k: v for k, v in r.items() if k != "tb"}, stage=stage), default=float) + "\n")


def load_history(string):
    p = history_path(string)
    return [json.loads(line) for line in open(p)] if p.exists() else []


def best_of(hist, key="J", stages=None):
    ok = [h for h in hist if h.get("ok") and h.get("design", True) and (stages is None or h["stage"] in stages)]
    if not ok:
        return None
    if key == "J":
        return min(ok, key=lambda h: h["J"])
    return min(ok, key=lambda h: h["refs"][key]["D3"])


def grid_points(string, x0):
    g = GRID3[string]
    names = list(g)
    pts = [dict(x0)]
    import itertools
    for combo in itertools.product(*(g[n] for n in names)):
        x = dict(x0)
        for n, v in zip(names, combo):
            if isinstance(n, tuple):
                x.update(dict(zip(n, v)))
            else:
                x[n] = v
        pts.append(x)
    return pts


def run_stage(string, stage, X, jobs, log, top=6):
    res = evaluate(string, X, FIT_DUR, jobs, log=log)
    append_history(string, stage, res)
    ok = sorted([r for r in res if r["ok"]], key=lambda r: r["J"])
    own, it = FIT_REFS[string]
    for r in ok[:top]:
        log(f"    J {r['J']:6.1f} own {r['refs'][own]['D3']:6.1f} itan {r['refs'][it]['D3']:6.1f} "
            f"alt {r['refs'][HELD_REF[string]]['D3']:6.1f} | {fmt_x(r['x'])}")
    nbad = sum(not r["ok"] for r in res)
    if nbad:
        log(f"    {nbad} failed: " + "; ".join(sorted({r['err'][:120] for r in res if not r['ok']})))
    return res


def cma_stage(string, stage, x0, names, gens, lam, jobs, log, sigma=0.18, seed=0):
    """CMA-ES in the unit box of SPACE over `names` from x0 (the other parameters fixed at x0)."""
    names = [n for n in names if n in SPACE]
    xs = dict(x0)
    if "groove_z0" in names and xs["groove_kappa"] <= 0:   # z0 only acts with a groove: CMA explores a 30 /m groove
        xs["groove_kappa"] = 30.0
    es = F2.CMA([to_unit(n, xs[n]) for n in names], sigma, lam, seed)
    best = None
    for g in range(gens):
        U = es.ask()
        X = []
        for u in U:
            x = dict(xs)
            for n, ui in zip(names, u):
                x[n] = from_unit(n, ui)
            if x.get("groove_z0", 0.0) < 0.05e-3 and "groove_z0" in names:   # z0 ~ 0: groove off (plain kappa z^2
                x["groove_kappa"], x["groove_z0"] = 0.0, 0.0                 #   is inaudible, MODEL_2D_ROUND3 s.6)
            X.append(x)
        log(f"  {stage} generation {g + 1}/{gens} (sigma {es.sigma:.3f})")
        res = run_stage(string, f"{stage}-g{g + 1}", X, jobs, log, top=3)
        f = [r["J"] if r["ok"] else 1e9 for r in res]
        es.tell(U, f)
        for r in res:
            if r["ok"] and (best is None or r["J"] < best["J"]):
                best = r
    return best


def fit_string(string, jobs=2, gens=5, lam=8, log=None):
    log = log or Log(OUT / f"fit_{string}.log")
    log(f"\n==== fit3 fit {string}  {time.strftime('%Y-%m-%d %H:%M:%S')}  f1 {target_f1(string)} (design_f1) ====")
    for rn in FIT_REFS[string] + (HELD_REF[string],):
        log(ref_line(rn))
    x0 = STRING_START[string]
    log("  stage 0: FIT2D best string and the 2-D preset string (new physics off, pitch design on)")
    run_stage(string, "s0", [x0, dict(PRESET_X)], jobs, log)
    log(f"  stage 1: grid {GRID3[string]}")
    run_stage(string, "s1", grid_points(string, x0), jobs, log)
    hist = load_history(string)
    b = best_of(hist)
    log(f"  stage 2: CMA-ES lam {lam}, {gens} generations on {CMA_NAMES} from J {b['J']:.1f}")
    cma_stage(string, "s2", b["x"], CMA_NAMES, gens, lam, jobs, log)
    hist = load_history(string)
    b = best_of(hist)
    log(f"  stage 3: geometry / pluck point / loss / in-plane drain screening from J {b['J']:.1f}: {fmt_x(b['x'])}")
    X = []
    for n, vals in GEOM_SCREEN.items():
        for v in vals:
            X.append(dict(b["x"], **{n: v}))
    res = run_stage(string, "s3", X, jobs, log, top=12)
    # combine the improving screening steps (one value per parameter) and re-evaluate
    imp = {}
    for x, r in zip(X, res):
        if r["ok"] and r["J"] < b["J"]:
            n = next(k for k in GEOM_SCREEN if quantise(x)[k] != quantise(b["x"])[k])
            if n not in imp or r["J"] < imp[n][1]:
                imp[n] = (x[n], r["J"])
    if len(imp) >= 2:
        xc = dict(b["x"], **{n: v for n, (v, _) in imp.items()})
        log(f"  stage 3b: combine improving steps {imp}")
        run_stage(string, "s3b", [xc], jobs, log)
    hist = load_history(string)
    b = best_of(hist)
    log(f"  stage 4: CMA-ES polish (2 generations) from J {b['J']:.1f}")
    cma_stage(string, "s4", b["x"], CMA_NAMES, 2, lam, jobs, log, sigma=0.08, seed=1)
    b = best_of(load_history(string))
    log(f"  BEST J {b['J']:.1f}: {fmt_x(b['x'])}")
    for rn in FIT_REFS[string] + (HELD_REF[string],):
        log("    " + score_line(b, rn))
    return b



# ================================================================================================ ablation
F1_FIT2D = dict(kharaj=69.127, pa=103.663, sa=138.142)   # FIT2D: f0(own) x 2^(-5/1200) with the fixed preset psi


def ablation_points(string, xb):
    """One switch at a time, from the best point."""
    pts = {"best": dict(xb)}
    if xb["groove_kappa"] > 0:
        pts["groove off"] = dict(xb, groove_kappa=0.0, groove_z0=0.0)
    if xb["Yb_y"] > 0:
        pts["modal drain Yb_y off"] = dict(xb, Yb_y=0.0)
    if xb["Ybz_x"] != 1.0:
        pts["in-plane drain Yb_z off"] = dict(xb, Ybz_x=1.0)
    if (xb["c"], xb["d_t"], xb["target_gap"]) != (3.0, 0.2e-3, 0.0):
        pts["geometry back to c 3, d_t 0.2 mm, gap 0"] = dict(xb, c=3.0, d_t=0.2e-3, target_gap=0.0)
    if xb["pluck_release"] != 2e-3:
        pts["release back to 2 ms"] = dict(xb, pluck_release=2e-3)
    x0 = STRING_START[string]
    cpl = {k: x0[k] for k in ("theta_nut", "detune_cents", "kcz_x")}
    if any(xb[k] != v for k, v in cpl.items()):
        pts["coupling back to FIT2D"] = dict(xb, **cpl)
    if xb["pluck_deg"] != x0["pluck_deg"]:
        pts["pluck angle back to FIT2D"] = dict(xb, pluck_deg=x0["pluck_deg"])
    return pts


def cmd_ablate(string, jobs=2, log=None):
    log = log or Log(OUT / f"ablation_{string}.txt")
    b = best_of(load_history(string))
    log(f"\n==== fit3 ablation {string}  {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n  best: {fmt_x(b['x'])}")
    pts = ablation_points(string, b["x"])
    res = evaluate(string, list(pts.values()), FIT_DUR, jobs, log=log)
    rd = evaluate(string, [b["x"]], FIT_DUR, 1, f1=F1_FIT2D[string], design=False, log=log)[0]   # pitch design off
    out = {}
    own, it = FIT_REFS[string]
    al = HELD_REF[string]
    rows = list(zip(pts, res)) + [(f"pitch design off (fixed psi, f1 {F1_FIT2D[string]})", rd)]
    for lab, r in rows:
        if not r["ok"]:
            log(f"  {lab:44s} FAILED {r['err']}")
            out[lab] = dict(ok=False, err=r["err"])
            continue
        cents = 1200 * np.log2(r["f0"] / ref3(own)["f0"])
        log(f"  {lab:44s} J {r['J']:6.1f} | own {r['refs'][own]['D3']:6.1f} itan {r['refs'][it]['D3']:6.1f} "
            f"alt {r['refs'][al]['D3']:6.1f} | pitch {cents:+5.1f} c")
        for rn in (own, it):
            log("      " + score_line(r, rn))
        out[lab] = dict(ok=True, J=r["J"], cents=cents, x=r["x"],
                        refs={rn: dict(D3=r["refs"][rn]["D3"], D2=r["refs"][rn]["D2"], Dnew=r["refs"][rn]["Dnew"],
                                       per=r["refs"][rn]["per"], feat=r["refs"][rn]["feat"],
                                       feat3={k: v for k, v in r["refs"][rn]["feat3"].items() if k not in ("L", "Lt")},
                                       post=r["refs"][rn]["post"]) for rn in (own, it, al)})
    with open(OUT / f"ablation_{string}.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    return out



# ================================================================================================ finals
FINAL_DUR = 6.0
BANDS_J = ((1, 3), (4, 10), (16, 31), (31, 60))     # the judge's bands
FIT2D_ITAN_KEY = {"kharaj": "06e7339fc387a9f2", "pa": "ab129bfa9718bffb", "sa": "c785be2238be83c1"}
INK, INK2, MUTED, GRIDC, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
C_NEW, C_OLD = "#2a78d6", "#eb6834"       # categorical slots 1-2 of the dataviz reference palette (validated)


def analyse_signal(y, f0, R):
    """Exact analysis of a radiated signal that starts 5 ms before its onset (what the closed loop approximates)."""
    f, t, Z = F2.stft(y)
    sl, _ = F2.partial_slices(f, f0, 0.0)
    A = F2.penv(np.abs(Z) ** 2, sl)
    fm2 = F2.features(t, A, R["tcut"])
    fm3 = new_features(t, A, R["tcut"], hf_burst(y), R["t_end"])
    D3, D2, Db, Dn, per = distance3(R, fm2, fm3)
    return dict(t=t, A=A, feat=fm2, feat3=fm3, D3=D3, D2=D2, D2_bloom=Db, Dnew=Dn, per=per, jb=judge_bands(t, A))


def judge_bands(t, A, times=(0.09, 1.0, 3.0)):
    tot = A.sum(0) + 1e-30
    out = {}
    for x in times:
        i = int(np.argmin(np.abs(t - x)))
        for a, b in BANDS_J + ((4, 7), (8, 15)):
            out[f"{a}-{b}@{x:g}"] = float(10 * np.log10(A[a - 1:b, i].sum() / tot[i] + 1e-30))
    return out


def ref_analysis(rn):
    R = ref3(rn)
    y = load_ref3(rn)
    return dict(t=R["t"], A=R["A"], feat=R["feat"], feat3=R["feat3"], jb=judge_bands(R["t"], R["A"]), y=y, f0=R["f0"])


def fit2d_run(string, key):
    path = F2.CACHE / "runs" / f"2d_{string}_{key}.npz"
    d = np.load(path, allow_pickle=False)
    run = {k: d[k] for k in d.files}
    run.update(key=f"fit2d_{key}", x=json.loads(str(d["x"])), psi=np.nan, x_c=np.nan, kernel="_run")
    return run


def fit2d_chain_signal(run, post):
    """FIT2D's own (unconstrained) chain: tanpura_male, its HP (up to 400 Hz / 3rd order) and body-mode gain."""
    import body
    y = np.asarray(run["Fy"], float) + post["gz"] * np.asarray(run["Fz"], float)
    out = body.apply_body(y, SR, "tanpura_male", hp_hz=post["hp_hz"], hp_order=int(post["hp_order"]),
                          mode_gain_db=post["mode_gain_db"], normalize=False)
    return out[int(run["i0"]):]


def _row3(label, a, extra=""):
    f2, f3, jb = a["feat"], a["feat3"], a["jb"]
    b = " | ".join("/".join(f"{jb[f'{p}-{q}@{x:g}']:5.1f}" for p, q in BANDS_J) for x in (0.09, 1.0, 3.0))
    d = (f" D3 {a['D3']:6.1f} (D2 {a['D2']:5.1f}, new {a['Dnew']:5.1f}, level RMS {a['per']['new__levrms_db']:4.1f} dB)"
         if "D3" in a else "")
    return (f"  {label:40s} rise {f2['rise_db']:5.1f} t_max {f2['t_max']:4.2f} | {b} | 4-7/8-15@1 "
            f"{jb['4-7@1']:5.1f}/{jb['8-15@1']:5.1f} | att16 {f3['att16']:5.1f} burst {f3['burst']:+5.1f} | decay "
            f"{f3['decay']:+5.1f} 2nd bloom {f3['rerise']:4.1f}{d}{extra}")


def k_match(y, target_db):
    g = 10 ** ((target_db - F2.loudness_db(y)) / 20)
    return y * g


def write_wav(path, y):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(y, np.float32), SR, subtype="PCM_24")


def final_string(string, jobs=2, log=None):
    log = log or Log(OUT / f"final_{string}.txt")
    log(f"\n==== fit3 final {string}  {time.strftime('%Y-%m-%d %H:%M:%S')} ====")
    own, it = FIT_REFS[string]
    al = HELD_REF[string]
    hist = load_history(string)
    b = best_of(hist)
    # 1. pitch correction: one step, so the model SOUNDS at the own recording's pitch
    sharp = 1200 * np.log2(b["f0"] / b["f1"])
    f1c = round(ref3(own)["f0"] * 2 ** (-sharp / 1200), 3)
    log(f"  best (fit render, f1 {b['f1']}): J {b['J']:.1f}, sounds {1200 * np.log2(b['f0'] / ref3(own)['f0']):+.1f} c "
        f"re own; jawari sharpening {sharp:+.1f} c -> f1 {f1c}\n  {fmt_x(b['x'])}")
    presolve(string, [b["x"]], f1c, True, log)
    runF = render(string, b["x"], FINAL_DUR, f1c, True, rec_modes=441)
    scF = score_run(runF, FIT_REFS[string] + (al,))
    scF["J"] = objective(scF, string)
    cents = 1200 * np.log2(float(runF["f0"]) / ref3(own)["f0"])
    log(f"  final 6 s render at f1 {f1c}: sounds at {float(runF['f0']):.3f} Hz ({cents:+.1f} c re own), J {scF['J']:.1f}, "
        f"cpu {float(runF['cpu']):.0f} s, kernel {runF['kernel']}, solver fails {runF['stats'][:2]}")
    i0s = int(runF["i0"]) / SR
    tm = np.asarray(runF["mE_t"]) - i0s
    E13 = {x: float(np.interp(x, tm, runF["mE13"])) for x in (0.09, 0.5, 1.0, 1.5, 3.0)}
    log("  string energy share of partials 1-3: " + ", ".join(f"{v:.2f} @ {k:g} s" for k, v in E13.items()))
    # 2. exact analyses (time-domain body.apply_body) of the final render and of FIT2D's previous fits
    res = dict(string=string, best_x=b["x"], f1_fit=b["f1"], J_fit=b["J"], f1_final=f1c, f0_final=float(runF["f0"]),
               cents_final=cents, J_final=scF["J"], E13=E13, sharpen=sharp, refs={})
    prev_own = fit2d_run(string, FIT2D_BEST_KEY[string])
    prev_it = fit2d_run(string, FIT2D_ITAN_KEY[string])
    fj = {r: json.load(open(F2.OUT / f"final_{string}.json"))["best"][r]["refs"][r]["post"] for r in (own, it)}
    sc_prev_phys = score_run(prev_own, FIT_REFS[string] + (al,))
    curves = {}
    for rn in (own, it, al):
        R = ref3(rn)
        ra = ref_analysis(rn)
        q = scF["refs"][rn]["post"]
        y = chain_signal(runF, q)
        a_new = analyse_signal(y, float(runF["f0"]), R)
        prev = prev_own if rn != it else prev_it
        if rn in fj:
            yp = fit2d_chain_signal(prev, fj[rn])
            a_old = analyse_signal(yp, float(prev["f0"]), R)
            old_lab = f"FIT2D fit ({fj[rn]['hp_hz']:.0f} Hz/{fj[rn]['hp_order']} HP, modes {fj[rn]['mode_gain_db']:+.0f} dB)"
        else:   # held out: FIT2D's own-recording string, its own-recording chain
            yp = fit2d_chain_signal(prev, fj[own])
            a_old = analyse_signal(yp, float(prev["f0"]), R)
            old_lab = "FIT2D own-fit string + its own chain"
        qp = sc_prev_phys["refs"][rn]["post"]
        a_old_phys = analyse_signal(chain_signal(prev, qp), float(prev["f0"]), R)
        log(f"\n  -- {rn} (t_end {R['t_end']} s) --")
        log(_row3("recording", ra))
        log(_row3("NEW fit3 (physical chain)", a_new, f" | {fmt_post(q)}"))
        log(_row3(old_lab, a_old))
        log(_row3("FIT2D string, physical chain", a_old_phys, f" | {fmt_post(qp)}"))
        res["refs"][rn] = {lab: dict(feat=a["feat"], feat3={k: v for k, v in a["feat3"].items() if k not in ("L", "Lt")},
                                     jb=a["jb"], **({k: a[k] for k in ("D3", "D2", "Dnew")} if "D3" in a else {}),
                                     levrms=a["per"]["new__levrms_db"] if "per" in a else None)
                           for lab, a in (("recording", ra), ("new", a_new), ("fit2d", a_old),
                                          ("fit2d_string_phys_chain", a_old_phys))}
        res["refs"][rn]["new_post"] = q
        res["refs"][rn]["fit2d_phys_post"] = qp
        curves[rn] = dict(rec=ra, new=a_new, old=a_old, y_new=y, y_old=yp)
        # WAVs: model through the chain (-1 dBFS), the recording K-loudness matched to it (local copy, git-ignored)
        tag = rn.split("_")[0]
        ym = F2.peak_norm(y[:int(FINAL_DUR * SR) - int(runF["i0"])])
        write_wav(OUT / "wav" / f"{string}_model_chain_{tag}.wav", ym)
        yr = k_match(ra["y"], F2.loudness_db(ym))
        pk = np.max(np.abs(yr))
        if pk > 0.891:
            yr *= 0.891 / pk
            log(f"    reference copy peak-limited by {20 * np.log10(pk / 0.891):.1f} dB")
        write_wav(OUT / "wav" / f"{string}_reference_{tag}_loudness_matched.wav", yr)
        if rn == own:
            gap = np.zeros(int(0.6 * SR))
            write_wav(OUT / "wav" / f"{string}_ab_own_then_model.wav",
                      np.concatenate([yr[:int(4.5 * SR)], gap, ym[:int(4.5 * SR)]]))
            dry = (np.asarray(runF["Fy"], float) + q["gz"] * np.asarray(runF["Fz"], float))[int(runF["i0"]):]
            write_wav(OUT / "wav" / f"{string}_model_dry_bridge_force.wav", F2.peak_norm(dry))
            yo = k_match(F2.peak_norm(yp), F2.loudness_db(ym))
            write_wav(OUT / "wav" / f"{string}_previous_fit2d_loudness_matched.wav", yo / max(1.0, np.max(np.abs(yo)) / 0.891))
    res["D3"] = {rn: res["refs"][rn]["new"]["D3"] for rn in (own, it, al)}
    plot_final(string, curves, res)
    with open(OUT / f"final_{string}.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    return res


def _sm(v, k=5):
    v = np.asarray(v, float)
    return np.convolve(np.pad(v, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def plot_final(string, curves, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    own = FIT_REFS[string][0]
    c = curves[own]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6})
    fig, ax = plt.subplot_mosaic("AAAABBBBCCCC;DDDEEEFFFGGG", figsize=(15, 8.2), constrained_layout=True)
    seq = matplotlib.colors.LinearSegmentedColormap.from_list("seq", F2.SEQ)
    for k, (lab, a) in zip("ABC", (("your recording", c["rec"]), ("new fit3 (physical chain)", c["new"]),
                                    ("previous FIT2D fit (400 Hz-class HP)", c["old"]))):
        A = a["A"][:40]
        t = a["t"]
        keep = t <= 4.0
        Z = 10 * np.log10(A[:, keep] + 1e-30)
        Z -= Z.max()
        im = ax[k].imshow(np.maximum(Z, -60), aspect="auto", origin="lower", cmap=seq, vmin=-60, vmax=0,
                          extent=(t[keep][0], t[keep][-1], 0.5, 40.5), interpolation="nearest")
        ax[k].set_title(f"{lab}: partial envelopes", loc="left")
        ax[k].set_xlabel("time after onset (s)")
        ax[k].set_ylabel("partial number")
        ax[k].grid(False)
    fig.colorbar(im, ax=ax["C"], label="dB re maximum", shrink=0.8)

    def band(a, lo, hi):
        A = a["A"]
        return 10 * np.log10(A[lo - 1:hi].sum(0) / (A.sum(0) + 1e-30) + 1e-30)
    t_end = ref3(own)["t_end"]
    for k, title, fn in (("D", "total level re maximum (dB)", lambda a: 10 * np.log10(a["A"].sum(0) + 1e-30)),
                         ("E", "partials 1-3 re total (dB)", lambda a: band(a, 1, 3)),
                         ("F", "partials 4-10 re total (dB)", lambda a: band(a, 4, 10)),
                         ("G", "partials 16-31 re total (dB)", lambda a: band(a, 16, 31))):
        for lab, a, col, lw in (("your recording", c["rec"], INK, 2.0), ("new fit3", c["new"], C_NEW, 2.0),
                                ("previous FIT2D", c["old"], C_OLD, 2.0)):
            t = a["t"]
            keep = t <= 4.0
            v = _sm(fn(a))[keep]
            if k == "D":
                v = v - v.max()
            ax[k].plot(t[keep], v, color=col, lw=lw, label=lab)
        ax[k].axvline(t_end, color=MUTED, lw=1, ls=":")
        ax[k].set_title(title, loc="left")
        ax[k].set_xlabel("time after onset (s)")
    ax["D"].legend(frameon=False, loc="lower right")
    ax["D"].set_ylim(-30, 2)
    D = res["D3"]
    fig.suptitle(f"{string}: your recording vs the fit3 string (D3 own {D[own]:.0f}, iTanpura "
                 f"{D[FIT_REFS[string][1]]:.0f}, held-out Alt {D[HELD_REF[string]]:.0f}); dotted line = end of the "
                 f"late-time window (recording fade)", x=0.01, ha="left", color=INK)
    (OUT / "png").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "png" / f"{string}_partials_ref_vs_model.png", dpi=110)
    plt.close(fig)


def main(argv):
    jobs = 2
    if "--jobs" in argv:
        k = argv.index("--jobs")
        jobs = int(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]
    cmd = argv[0] if argv else "refs"
    args = argv[1:]
    if cmd == "refs":
        for rn in REFS3:
            print(ref_line(rn))
    elif cmd == "probe":
        for s in (args or ["kharaj"]):
            cmd_probe(s, jobs)
    elif cmd == "probe2":
        for s in (args or ["kharaj"]):
            cmd_probe(s, jobs, PROBE2)
    elif cmd == "final":
        for s in (args or list(STRINGS)):
            final_string(s, jobs)
    elif cmd == "ablate":
        for s in (args or list(STRINGS)):
            cmd_ablate(s, jobs)
    elif cmd == "fit":
        for s in (args or list(STRINGS)):
            fit_string(s, jobs)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main(sys.argv[1:])
