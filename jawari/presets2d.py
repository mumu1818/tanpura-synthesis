"""Two-polarisation presets: the presets.py strings (statics geometry, tuning, finger pluck, Valette loss) translated
to tanpura2d, plus the 2-D keys (pluck direction, nut coupling, friction law, tension modulation, mono mix).

    from presets2d import preset2d, render2d, radiate
    p = preset2d("kharaj", dur=5.0)          # the user's bronze kharaj, SIDEWAYS finger pluck (pluck_deg 88)
    sig = render2d(p)                        # dict of 44.1 kHz signals: Fy, Fz, Fny, Fnz, mono = Fy + gz Fz, ...
    y = radiate(sig["mono"])                 # body.apply_body('tanpura_male') with a 250 Hz 2nd-order radiation HP
    p = preset2d("kharaj", pluck_deg=0.0, pol2=False, tm=False)   # the 1-D limit: == presets.preset('kharaj')
    p = preset2d("kharaj", thread_shift=+0.05e-3, detune_cents=3.0)   # statics keys work as in presets.py

How the 1-D preset is translated (preset2d):
  * presets.preset(name, **shared) builds the 1-D dict.  `shared` = statics keys (presets.GEOM_KEYS: thread_shift,
    target_gap, f1, d_t, phi_deg, psi_deg, ...) and every key that tanpura1d and tanpura2d share (fs, fmax, kc, wc,
    pluck_disp, delta_ve, Yb, ...), so the statics, the thread auto-placement and tune='speaking' see them.
  * every key of the 1-D dict is copied to the 2-D dict (same meaning in both models), except DROP_1D (x_out: the 1-D
    displacement output point; tanpura2d has none).  A 1-D key that tanpura2d does not know raises ValueError
    (e.g. a future 1-D option), so nothing essential is dropped silently.
  * then the 2-D keys: PRESET_2D below, then the caller's 2-D-only overrides.
  * 'dafx16_c3' is tanpura2d.DEFAULT (the C3 2-D model of MODEL_2D.md) + overrides.

2-D keys (PRESET_2D; the evidence is in MODEL_2D_PLUCK.md, validate2d_parity.py test G):
  pluck_deg 88     finger pluck direction from the bridge normal (y) towards the soundboard plane (z).  A tanpura is
                   plucked by the finger pad roughly PARALLEL to the soundboard.  This is what produces the real 'bloom'
                   (the radiated level rises by 14-18 dB to a peak at 0.7-1.3 s in every reference): the z motion cannot
                   hit the bridge, and the jawari builds up only as energy migrates into y through the nut coupling.
                   On the kharaj model the F_y rise is 3-7 dB at 80 deg, 8-13 dB at 86, 15-16 dB at 88, 16-18 dB at 90,
                   so the references need >~ 87 deg.  ASSUMED (no measurement of the real pluck direction).
  theta_nut 30, detune_cents 3.5
                   nut principal-axis angle and frequency split (axis 2 sharper): the y-z coupling strength.  Unmeasured:
                   a FIT parameter (natural sources: different y/z effective boundary stiffness at thread/bridge, bridge
                   or soundboard anisotropy).  The split sets the bloom time (partial n exchanges y <-> z with period
                   1/(n f1 (2^(c/1200) - 1))); with a parallel pluck every tested value 2-5 cents, 30-45 deg blooms.
  friction True, mu_c1 0.39, mu_c2 2.84     VT-2015 kinetic friction law along the bridge face.
  tm True          Kirchhoff-Carrier tension modulation (both polarisations).
  kcz, rcz None    lateral thread = vertical thread (isotropic: keeps tune='speaking' exact on the nut's axis 1).
  tune_ref 'axis1' tune='speaking' tunes the nut's axis-1 series to f1; axis 2 is detune_cents sharper.
  gz 0.1           mono mix weight of the horizontal (in-plane) bridge force.  ASSUMED, -20 dB: the in-plane force
                   drives the soundboard only through bridge rocking.  The 2-D C3 work used 0.3 (common.GZ_MONO); with
                   0.3 the strong initial in-plane force masks the bloom (mono rise 0-8 dB), the references' 14-18 dB
                   rise needs gz <~ 0.1 in this model.
"""
from __future__ import annotations

import numpy as np

import presets as P1
import tanpura1d as T1
import tanpura2d as T2
from common import GZ_MONO, to_audio

DROP_1D = {"x_out"}            # 1-D only and not essential (displacement output point)
EXTRA_2D = {"gz"}              # keys consumed here (not tanpura2d parameters)
KEYS_2D_ONLY = (set(T2.DEFAULT) - set(T1.DEFAULT)) | EXTRA_2D

PRESET_2D = dict(
    pol2=True, pluck_deg=88.0, theta_nut=30.0, detune_cents=3.5, tune_ref="axis1",
    friction=True, mu_c1=0.39, mu_c2=2.84, tm=True, kcz=None, rcz=None,
    gz=0.1,
)


def preset2d(name, **overrides):
    """Parameter dict for tanpura2d.simulate (plus 'gz' and 'meta', which tanpura2d ignores).  See the module doc."""
    name = P1.ALIASES.get(name, name)
    o2 = {k: overrides.pop(k) for k in list(overrides) if k in KEYS_2D_ONLY}
    unknown = set(overrides) - set(T1.DEFAULT) - P1.GEOM_KEYS
    if unknown:
        raise ValueError(f"unknown parameter(s) {sorted(unknown)} (not a tanpura1d/tanpura2d or statics key)")
    if name == "dafx16_c3":
        bad = P1.GEOM_KEYS & set(overrides)
        if bad:
            raise ValueError(f"geometry keys {sorted(bad)} do not apply to dafx16_c3")
        p = dict(T2.DEFAULT)
        p.update(overrides)
        p.update(o2)
        p.setdefault("gz", GZ_MONO)
        p["meta"] = dict(preset="dafx16_c3", preset2d="dafx16_c3")
        return p
    p1 = P1.preset(name, **overrides)
    p = dict(T2.DEFAULT)
    dropped = []
    for k, v in p1.items():
        if k == "meta":
            continue
        if k in DROP_1D:
            dropped.append(k)
            continue
        if k not in T2.DEFAULT:
            raise ValueError(f"preset {name!r}: 1-D key {k!r} has no tanpura2d equivalent; add it to tanpura2d or to "
                             f"presets2d.DROP_1D if it is not essential")
        p[k] = v
    p.update(PRESET_2D)
    p.update(o2)
    for k in ("profile", "T", "tune", "static_init", "pluck", "loss"):   # what makes it the statics kharaj
        if p1.get(k) is not None and p.get(k) is None:
            raise ValueError(f"preset {name!r}: essential key {k!r} lost in the 2-D translation")
    p["meta"] = dict(p1.get("meta", {}), preset2d=name, dropped_1d=dropped)
    return p


def to_1d(p):
    """The tanpura1d dict a 2-D preset was translated from (2-D keys removed; x_out = pluck point)."""
    q = {k: v for k, v in p.items() if k in T1.DEFAULT}
    if q.get("pluck") == "finger":
        q["x_out"] = q["L"] / 3.0 if q.get("xp") is None else q["xp"]
    return q


def sim_params(p):
    """Strip the keys tanpura2d does not take (meta, gz)."""
    return {k: v for k, v in p.items() if k not in ("meta",) and k not in EXTRA_2D}


def render2d(p, energy=False, sr=44100):
    """Simulate a 2-D preset and return 44.1 kHz signals (common.to_audio: 20 Hz DC block + decimation):
    Fy, Fz (force on the bridge, bridge-normal / along the bridge face), Fny, Fnz (nut), mono = Fy + gz Fz,
    plus sr, gz, cpu (s), the parameter/set-up/result dicts p, s, r of tanpura2d.simulate."""
    pp, s, r = T2.simulate(sim_params(p), energy=energy)
    out = {k: to_audio(r[k], pp["fs"], sr) for k in ("Fy", "Fz", "Fny", "Fnz")}
    gz = float(p.get("gz", GZ_MONO))
    out["mono"] = out["Fy"] + gz * out["Fz"]
    out.update(sr=sr, gz=gz, cpu=r["cpu"], p=pp, s=s, r=r)
    return out


def radiate(y, sr=44100, preset="tanpura_male", hp_hz=250.0, hp_order=2, **body_overrides):
    """Radiated sound for comparisons with recordings: body.apply_body(y, sr, preset, hp_hz=..., hp_order=...).
    hp_hz is the body's output high-pass (here used as a crude radiation-efficiency high-pass of a small soundboard
    below its coincidence region; body.py's own default is a 20 Hz DC guard)."""
    import body
    return body.apply_body(np.asarray(y, float), sr, preset, hp_hz=hp_hz, hp_order=hp_order, **body_overrides)
