"""Named parameter sets for tanpura1d.simulate.

    from presets import preset
    import tanpura1d as T1
    p = preset("kharaj", dur=12.0)                 # the user's C#2 kharaj (phosphor bronze), finger pluck at L/3
    p, s, r = T1.simulate(p)
    p = preset("kharaj", thread_shift=+0.1e-3)     # thread moved 0.1 mm towards the nut from its grazing position
    p = preset("kharaj_steel")                     # same instrument design rule, steel string (shared/params.py)
    p = preset("dafx16_c3")                        # exactly tanpura1d.DEFAULT (DAFx-16 C3 reference)
    p = preset("sa", dur=12.0)                     # the two middle (jodi) strings, C#3, 0.50 mm steel (ASSUMED gauge)
    p = preset("pa", dur=12.0)                     # the first string (pancham), G#2, 0.55 mm bronze (ASSUMED gauge)
    p = preset("kharaj", f1=68.95, design_f1=True) # psi re-designed so the thread grazes at x_c0 at this pitch

Presets
-------
'dafx16_c3'     exactly tanpura1d.DEFAULT (DAFx-16 C3 steel string, parabola grazing at the thread, whole-string pulse,
                Woodhouse loss defaults).
'kharaj'        the user's kharaj: speaking length 1.05 m, 0.7 mm, C#2 = 69.4 Hz, PHOSPHOR BRONZE (rho 8800, E 1.1e11).
                The material and the pluck position (L/3 from the bridge) are the user's own statements recorded in
                docs/EVALUATION.md section 6 / README.md ("Instrument facts from the user"); the steel values of
                shared/params.py are superseded there.  'kharaj_bronze' is an alias.
'kharaj_steel'  the same string in steel (rho 7800, E 2e11; the values of shared/params.py), as the alternative.
'sa'            the two Sa (jodi) strings: C#3 = 2 x 69.4 = 138.8 Hz, 0.50 mm STEEL.
'pa'            the pancham string: G#2 = 3/2 x 69.4 = 104.1 Hz, 0.55 mm PHOSPHOR BRONZE.
'pa_steel'      the same pancham in steel (0.55 mm), as the alternative.

Pa and Sa strings (see STRINGS.md).  Same instrument as the kharaj: speaking length 1.05 m, the same bridge (parabola
c = 3 /m, crest, length, edge radius), thread diameter, break angle, tailpiece, pluck (finger at L/3) and loss model.
* Gauges/materials are ASSUMED (the user's own Pa/Sa gauges are not known; replace with a micrometer measurement):
  two independent male-tanpura string-set listings agree with each other and with the user's 0.7 mm kharaj:
  Pyramid 706/4 "Tanpura strings gents" (Thomann listing): 0.55 / 0.50 / 0.50 / 0.70 mm;
  sangtar.com (2007), a 60" male tanpura: 1 bronze .022" (0.56 mm), 2 & 3 steel .020" (0.51 mm), 4 brass .028"
  (0.71 mm).  Materials: the jodi (Sa) strings are steel in every source; the pancham is brass/bronze in the
  sangtar set, in a Pyramid-set buyer's review and in sikhiwiki (one other page there says steel), hence 'pa' is
  bronze (the user's kharaj material; brass, rho ~8500, E ~1.0e11, is within the assumption) and 'pa_steel' is
  provided.  Tensions: Sa 129.5 N (660 MPa, ~30 % of music-wire strength), Pa 99.4 N (420 MPa).
* Tuning: pure (beat-free) intervals on the user's kharaj, as a tanpura is tuned by ear: Sa = 2 x 69.4 Hz, Pa = 3/2 x
  69.4 Hz (4 Pa = 3 Sa, 2 Pa = 3 kharaj).  The A440 equal-tempered values (138.59, 103.83 Hz) would be -2.6 and
  -4.5 cents off these and beat against the kharaj's harmonics.  Override with f1=...
* Bridge/thread: each string has its own jawari setting.  As for the kharaj, the speaking-length angle psi of each
  string's track (equivalently the tilt the jawari is filed to under that string) is designed so that its thread
  grazes at x_c0 = 4 mm at the tuned tension (PSI_STRING, from design_instrument()); the thread is then auto-placed at
  the grazing root (target_gap=0).  With the kharaj's psi (5.79 deg) the Sa/Pa strings are pressed ~260 um into the
  bridge at x_c = 4 mm; they graze only with the thread at 8.82 mm (Sa) / 8.77 mm (Pa) (common-bridge alternative:
  preset('sa', psi_deg=PSI_DESIGN['bronze'], x_c0=8.8e-3); not the default, see STRINGS.md).

Kharaj model (all presets except dafx16_c3)
------------------------------------------
* Rest geometry from statics.py: rest_state -> profile_for_dynamics -> static_init.  The thread is auto-placed so that
  the crest gap of the thread-only rest shape is `target_gap` (default 0: just grazing); `thread_shift` then moves it.
* Instrument (ASSUMED, to be replaced by measurements): parabolic bridge c = 3 /m (DAFx-16 value), crest 2 mm from the
  back face, 40 mm long (bridge width from shared/params.py), back-edge radius 0.5 mm; tailpiece 0.15 m behind the
  bridge at a break angle phi = 10 deg; thread = cotton/sewing thread of effective (compressed) diameter 0.2 mm; the
  speaking-length angle psi (equivalently the tilt the jawari is filed to) is designed per string so that the thread
  grazes at x_c0 = 4 mm (PSI_DESIGN; recompute with design_instrument()).  Why: with a 0.2 mm thread, phi = 10 deg is
  the only one of the tested break angles (2, 5, 10 deg; STATICS.md) at which the kharaj still rests on the back of the
  bridge when grazing (smaller angles lift it off the back and leave it hanging from the thread, which the
  pinned-termination dynamic model cannot represent); 0.2 mm is a typical sewing/cotton thread (players use the
  thinnest thread that works; no published diameters were found); c = 3 /m (DAFx-16) and x_c0 = 4 mm keep the
  instrument of STATICS.md so its results carry over.  The bridge filed for one string does not graze with the other
  material: see PSI_DESIGN.
* Tension: tune='speaking' (lowest linear eigenfrequency with the thread spring = f1); the statics are re-solved at the
  tuned tension until statics and dynamics use the same T (fixed point, |dT|/T < 1e-8).
* Modes: all below 20 kHz (fmax), integrated at 176.4 kHz.
* Excitation: pluck='finger' at x_p = L/3 (the user's own plucking point; no published tanpura plucking position was
  found, and Raman reported that the plucking point hardly changes the overtone strengths of a tanpura), 12 mm
  raised-cosine pad, 40 ms sin^2 pull, 2 ms cos^2 release (ASSUMED; harpsichord plucks last ~8-12 ms with a slip phase
  of 1/4-1/3 of that, harp sticking phases ~300 ms; tanpura fingers "roll off" the string), amplitude for a 2 mm static
  pad displacement (pluck_disp).  The displacement output x_out is the pluck point.  1-D model: the pluck acts in the
  bridge-normal polarisation only.
* Losses: loss='valette' (see tanpura1d.loss_sigma and KHARAJ_LOSS below for values and sources).
"""
from __future__ import annotations

import functools

import numpy as np
from scipy.optimize import brentq

import statics as S
import tanpura1d as T1

MATERIALS = {
    "bronze": dict(rho=8800.0, E=1.1e11, label="phosphor bronze"),
    "steel": dict(rho=7800.0, E=2.0e11, label="steel"),
}

# Designed speaking-length angle (deg) per material: thread at x_c0 = 4 mm just grazes at the TUNED tension
# (from design_instrument(); grazing positions drift by < 1 um if recomputed).  Bronze psi on steel leaves the
# string ~160 um above the bridge everywhere; steel psi on bronze presses it ~260 um into the bridge everywhere.
PSI_DESIGN = {"bronze": 5.7911, "steel": 6.9362}

# The other strings of the user's male tanpura in C# (standard tuning Pa Sa Sa kharaj); gauges/materials ASSUMED
# (module docstring).  geom overrides KHARAJ_GEOM; psi_deg from design_instrument(material, 4e-3, geom) (PSI_STRING).
PSI_STRING = {"sa": 4.5052, "pa": 4.5361, "pa_steel": 5.1236}
STRING_SPECS = {
    "kharaj": dict(material="bronze", role="kharaj", geom={}),
    "kharaj_steel": dict(material="steel", role="kharaj", geom={}),
    "sa": dict(material="steel", role="sa", geom=dict(d=0.50e-3, f1=138.8, psi_deg=PSI_STRING["sa"])),
    "pa": dict(material="bronze", role="pa", geom=dict(d=0.55e-3, f1=104.1, psi_deg=PSI_STRING["pa"])),
    "pa_steel": dict(material="steel", role="pa", geom=dict(d=0.55e-3, f1=104.1, psi_deg=PSI_STRING["pa_steel"])),
}

# Instrument and string geometry (statics), all assumptions except d, L_speak, f1, bridge_len (shared/params.py).
KHARAJ_GEOM = dict(
    d=0.7e-3, L_speak=1.05, f1=69.4,
    c=3.0, x_crest=2e-3, bridge_len=40e-3, edge_radius=0.5e-3,
    d_t=0.2e-3, phi_deg=10.0, L_tail=0.15, psi_deg=None,
    x_c0=4e-3, target_gap=0.0, thread_shift=0.0,
    h_floor=-0.3e-3, statics_h=10e-6,
)

# Losses (loss='valette' in tanpura1d.loss_sigma: Valette & Cuesta string losses + a smooth body drain).
#   eta_air, rho_air  physical constants of air at ~20 C.
#   delta_ve, Qte_inv visco-elastic and frequency-independent internal loss factors: the values used with the Valette
#                     model for steel strings by Pate, Le Carrou & Fabre (JASA 2014) and Issanchou et al. (JSV 2017),
#                     4.5e-3 and 2.03e-4, quoted from memory because neither full text was readable here (UNVERIFIED);
#                     applied to bronze too (no bronze value found).  Consistency check: a piano-string study
#                     (arXiv:1101.4511) attributes decay rates > 20 /s above 10 kHz to intrinsic loss, i.e. delta_ve >~ 2e-3
#                     for ~1 mm wire.
#   Yb, mb, fb        body drain sigma_b = T Re Y(f) / L.  ASSUMED.  The bare-plate mobility of body.py's tabli
#                     (4 mm toon, 1/(8 sqrt(D rho h)) = 0.0227 s/kg) gives T60 of only 4-8 s at 140-500 Hz and a tone that
#                     decays FASTER than the old Woodhouse defaults, contrary to the long ringing of a real tanpura;
#                     Yb = 5e-3 s/kg is the bare-plate value of an ~8.5 mm plate (Y ~ h^-2), standing in for the thicker
#                     /convex tabli and broad bone bridge of a tanpura.  mb = 15 g heavy bridge (matters only > 2 kHz),
#                     fb = 145 Hz = body.py's first tabli mode (stiffness-controlled roll-off below it).
#                     body.py itself is a one-way output filter and adds no string loss, so nothing is counted twice.
#                     This is the least certain loss term: fit it to a recording (EVALUATION section 6, step 3).
KHARAJ_LOSS = dict(loss="valette", eta_air=1.8e-5, rho_air=1.2, delta_ve=4.5e-3, Qte_inv=2.03e-4,
                   Yb=5.0e-3, mb=0.015, fb=145.0)

KHARAJ_DYN = dict(
    f1=69.4, fmax=20000.0, fs=176400.0, M=None,
    tune="speaking", static_init=True,
    pluck="finger", xp=None, wp=0.012, pluck_pull=0.040, pluck_release=0.002, pluck_disp=2e-3,
    **KHARAJ_LOSS,
)

ALIASES = {"kharaj_bronze": "kharaj"}
NAMES = ("dafx16_c3", "kharaj", "kharaj_bronze", "kharaj_steel", "sa", "pa", "pa_steel")


# ----------------------------------------------------------------------------------------------- statics pipeline
def _profile(g):
    return S.BridgeProfile.parabolic(c=g["c"], x_crest=g["x_crest"], length=g["bridge_len"],
                                     edge_radius=g["edge_radius"])


def _string(material, g, T=None, role="kharaj"):
    m = MATERIALS[material]
    st = S.String.kharaj(d=g["d"], L=g["L_speak"], f1=g["f1"], rho=m["rho"], E=m["E"])   # round solid wire
    st.name = f"{role} ({g['d'] * 1e3:g} mm {m['label']})"
    if T is not None:
        st.T = float(T)
    return st


def _setup(material, g, T=None, psi=None, x_c=None, role="kharaj"):
    st = _string(material, g, T, role)
    psi = PSI_DESIGN[material] if psi is None else psi
    return S.Setup(_profile(g), st, S.Thread(g["x_c0"] if x_c is None else x_c, g["d_t"]), phi_deg=g["phi_deg"],
                   psi_deg=psi, L_tail=g["L_tail"], L_speak=g["L_speak"])


def graze_positions(su, target_gap=0.0, x_lo=None, x_hi=None, step=0.1e-3, h=10e-6):
    """All thread positions x_c in [x_lo, x_hi] with crest gap = target_gap at which the string still rests on the
    back of the bridge (pinned-termination model valid).  Returns a sorted list (may be empty)."""
    x0 = su.thread.x_c
    x_lo = max(x0 - 1.5e-3, 0.5 * su.thread.d_t + 0.2e-3) if x_lo is None else x_lo
    x_hi = x0 + 1.5e-3 if x_hi is None else x_hi

    def ev(xc):
        sol = S.solve(su.replace(x_c=xc), h=h, front=False)
        return sol.crest_gap()[0] - target_gap, np.isfinite(sol.x_term())

    xs = np.arange(x_lo, x_hi + 0.5 * step, step)
    vals = [ev(x) for x in xs]
    roots = []
    for k in range(len(xs) - 1):
        (fa, ba), (fb, bb) = vals[k], vals[k + 1]
        if not (np.isfinite(fa) and np.isfinite(fb)) or np.sign(fa) == np.sign(fb):
            continue
        r = brentq(lambda x: ev(x)[0], xs[k], xs[k + 1], xtol=1e-9)
        if ev(r)[1]:
            roots.append(float(r))
    return roots


def _tuned_T(t1, g, dyn):
    """Tension that tanpura1d.build(tune='speaking') will use for these overrides."""
    p = dict(T1.DEFAULT)
    p.update(t1)
    p.update({k: dyn[k] for k in ("fs", "fmax", "M", "kc", "wc", "thread") if k in dyn})
    p.update(f1=g["f1"], tune="speaking")
    mu = p["rho"] * p["A"]
    EI = p["E"] * p["I"] * p["EI_scale"]
    L = p["L"]
    beta = np.arange(1, 5000) * np.pi / L
    T0 = mu * (2 * L * p["f1"]) ** 2 - EI * (np.pi / L) ** 2
    return T1.tune_tension(p, T0, mu, EI, L, beta)


@functools.lru_cache(maxsize=64)
def _geometry_cached(material, gkey, dkey, role="kharaj"):
    g = dict(gkey)
    dyn = dict(dkey)
    T = _string(material, g).T
    hist = []
    roots0 = None
    xg = None
    for it in range(12):
        su = _setup(material, g, T=T, psi=g["psi_deg"], role=role)
        if xg is None:
            roots = graze_positions(su, g["target_gap"], h=g["statics_h"])
            roots0 = roots
        else:   # later iterations: T changes by < 1 %, the root moves by < 0.2 mm
            roots = graze_positions(su, g["target_gap"], x_lo=xg - 0.25e-3, x_hi=xg + 0.25e-3, step=0.05e-3,
                                    h=g["statics_h"])
        if not roots:
            raise ValueError(f"{material}: no thread position near x_c0 = {g['x_c0'] * 1e3:.2f} mm gives crest gap "
                             f"{g['target_gap'] * 1e6:.2f} um with the string resting on the back of the bridge "
                             f"(psi = {su.psi_deg:.4f} deg)")
        xg = min(roots, key=lambda r: abs(r - (g["x_c0"] if it == 0 else xg)))
        xc = xg + g["thread_shift"]
        rs = S.rest_state(su.replace(x_c=xc), h=g["statics_h"])
        d = S.profile_for_dynamics(rs, h_floor=g["h_floor"])
        T_new = _tuned_T(d["t1"], g, dyn) if dyn.get("tune") == "speaking" else T
        hist.append((T, T_new, xg))
        if abs(T_new - T) <= 1e-8 * T:
            break
        # T -> T_new is a contraction (factor ~0.03 per iteration): plain fixed-point iteration
        T = T_new
    # all grazing positions at the converged tension (reported; the first scan was at the untuned T)
    roots = graze_positions(_setup(material, g, T=T, psi=g["psi_deg"], role=role), g["target_gap"],
                            h=g["statics_h"]) \
        or roots0
    info = dict(material=material, label=MATERIALS[material]["label"], psi_deg=su.psi_deg, x_c_graze=xg,
                graze_roots=roots, x_c=xc, thread_shift=g["thread_shift"], crest_gap=rs.crest_gap,
                x_graze=rs.x_graze, x_term=rs.x_term, F_thread_preload=d["F_thread"],
                F_thread_full=d["F_thread_full"], T_statics=T, T_hist=hist, l=float(np.sqrt(su.string.EI / T)),
                summary=rs.summary(), n_profile=len(d["x_rel"]), L_rel=d["L_rel"], xc_rel=d["xc_rel"])
    return d["t1"], info


def kharaj_geometry(material="bronze", geom=None, dyn=None, role="kharaj"):
    """Statics-derived tanpura1d overrides (L, xc, profile, rho, A, E, I, T, xe, we) and an info dict.
    Any string of the instrument: geom overrides KHARAJ_GEOM (d, f1, psi_deg, ... ; see STRING_SPECS)."""
    g = dict(KHARAJ_GEOM)
    g.update(geom or {})
    dd = dict(KHARAJ_DYN)
    dd.update(dyn or {})
    dkey = tuple(sorted((k, dd.get(k, T1.DEFAULT.get(k))) for k in ("fs", "fmax", "M", "kc", "wc", "thread", "tune")))
    t1, info = _geometry_cached(material, tuple(sorted(g.items())), dkey, role)
    return dict(t1, profile=(t1["profile"][0].copy(), t1["profile"][1].copy())), dict(info)


def design_instrument(material, x_c0=4e-3, geom=None, dyn=None, iters=6):
    """Speaking-length angle psi (deg) for which the thread at x_c0 just grazes, at the tuned tension.
    (How PSI_DESIGN was obtained; returns (psi, T).)"""
    g = dict(KHARAJ_GEOM)
    g.update(geom or {})
    dd = dict(KHARAJ_DYN)
    dd.update(dyn or {})
    T = _string(material, g).T
    psi = None
    for _ in range(iters):
        su = _setup(material, g, T=T, psi=PSI_DESIGN[material], x_c=x_c0)
        psi = S.design_psi(su, x_c0, target_gap=g["target_gap"], h=g["statics_h"])
        rs = S.rest_state(su.replace(psi_deg=psi), h=g["statics_h"])
        d = S.profile_for_dynamics(rs, h_floor=g["h_floor"])
        T_new = _tuned_T(d["t1"], g, dd)
        if abs(T_new - T) <= 1e-10 * T:
            break
        T = T_new
    return psi, T


@functools.lru_cache(maxsize=32)
def _designed_psi_cached(material, f_design, gkey, dkey):
    g = dict(gkey)
    g["f1"] = float(f_design)
    psi, T = design_instrument(material, g["x_c0"], g, dict(dkey))
    return float(psi), float(T)


def designed_psi(material, f_design, geom=None, dyn=None):
    """psi (deg) for which the thread at x_c0 just grazes (target_gap) at the tension tuned to f_design
    (design_instrument at f1 = f_design), cached.  Used by preset(..., design_f1=...).  Returns (psi, T)."""
    g = dict(KHARAJ_GEOM)
    g.update(geom or {})
    g.pop("psi_deg", None)
    dd = dict(KHARAJ_DYN)
    dd.update(dyn or {})
    dkey = tuple(sorted((k, dd.get(k, T1.DEFAULT.get(k))) for k in ("fs", "fmax", "M", "kc", "wc", "thread", "tune")))
    return _designed_psi_cached(material, float(f_design), tuple(sorted(g.items())), dkey)


# ----------------------------------------------------------------------------------------------- presets
# design_f1 (optional, default None = off): re-design the speaking-length angle psi of the string's track for grazing at
# x_c0 at the tension of f1 = design_f1 (True: at the preset's own f1) instead of using PSI_DESIGN / PSI_STRING.  This is
# how an instrument is set up for a different pitch: the jawari is re-filed (equivalently psi re-chosen) so the thread
# grazes at its usual place.  Needed to tune the bronze kharaj below ~69 Hz, where the fixed psi has no grazing thread
# position (MODEL_2D_ROUND3.md).
DESIGN_KEYS = {"design_f1"}
GEOM_KEYS = set(KHARAJ_GEOM) | DESIGN_KEYS


def preset(name, **overrides):
    """Parameter dict for tanpura1d.simulate.  Geometry keys (KHARAJ_GEOM: thread_shift, target_gap, d_t, phi_deg,
    psi_deg, c, ...) are consumed by the statics; every other key overrides tanpura1d parameters.
    The returned dict carries the statics/tuning results under 'meta' (ignored by tanpura1d)."""
    name = ALIASES.get(name, name)
    if name == "dafx16_c3":
        bad = GEOM_KEYS & set(overrides)
        if bad:
            raise ValueError(f"geometry keys {sorted(bad)} do not apply to dafx16_c3")
        p = dict(T1.DEFAULT)
        p.update(overrides)
        return p
    if name not in STRING_SPECS:
        raise ValueError(f"unknown preset {name!r}; choose from {NAMES}")
    spec = STRING_SPECS[name]
    material = spec["material"]
    geom = dict(spec["geom"])                     # the string's own gauge, pitch and designed psi ({} for kharaj)
    user_geom = {k: overrides.pop(k) for k in list(overrides) if k in GEOM_KEYS}
    geom.update(user_geom)
    design_f1 = geom.pop("design_f1", None)
    dyn = dict(KHARAJ_DYN)
    dyn.update(overrides)
    design = None
    if design_f1 is not None:
        if user_geom.get("psi_deg") is not None:
            raise ValueError("give either psi_deg or design_f1, not both")
        f_d = float(geom.get("f1", KHARAJ_GEOM["f1"])) if design_f1 is True else float(design_f1)
        psi_d, T_d = designed_psi(material, f_d, geom, dyn)
        geom["psi_deg"] = psi_d
        design = dict(design_f1=f_d, psi_designed=psi_d, T_design=T_d)
    t1, info = kharaj_geometry(material, geom, dyn, role=spec["role"])
    if design is not None:
        info.update(design)
    p = dict(T1.DEFAULT)
    p.update(t1)
    p.update(KHARAJ_DYN)
    p["f1"] = geom.get("f1", KHARAJ_GEOM["f1"])     # f1 is a statics key too: keep statics and dynamics consistent
    if dyn.get("pluck") == "finger":
        xp = p["L"] / 3.0 if dyn.get("xp") is None else dyn["xp"]
        p["x_out"] = xp
    p.update(overrides)
    info["preset"] = name
    p["meta"] = info
    return p


def modal_info(p):
    """Mode count, highest modal frequency and the physics Nyquist check for a parameter dict."""
    q = {k: v for k, v in p.items() if k != "meta"}
    s = T1.build(dict(T1.DEFAULT, **q))
    fM = float(s["om"][-1] / (2 * np.pi))
    return dict(M=s["M"], f_M=fM, nyquist=0.5 * s["fs"], T=s["T"], Nb=s["Nb"],
                T60_at=lambda f: float(6.9078 / np.interp(f, s["om"] / (2 * np.pi), s["sig"])))
