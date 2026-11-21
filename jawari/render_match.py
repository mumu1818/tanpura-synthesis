"""Final A/B listening set: the user's own recordings vs the old 1-D model vs the fitted 2-D model, plus a 4-string drone.

    cd jawari && python render_match.py                 # everything, ~5 min wall with 2 worker processes
    python render_match.py --only singles               # single plucks only
    python render_match.py --only drone                 # drone only
    python render_match.py --jobs 1                     # serial

Writes to jawari/out/match/ (git-ignored; the reference copies there are the user's private / copyrighted recordings,
loudness-matched for listening: NEVER commit them, never move them into the repository).

Single plucks, per string (kharaj, pa, sa), 7 s each, all loudness-matched to one K-weighted level (first 4 s):
  <s>_1_reference_own.wav       the user's own recording (Mumu_Tanpura/Male/tt_CS3_<s>.mp3), read in place, onset-trimmed
  <s>_2_old_1d.wav              the current 1-D model: presets.preset(<s>) -> tanpura1d -> body.apply_body('tanpura_male')
                                with its own defaults (the chain render_strings.py uses), at the 2-D fit's contact-off f1
                                so that all three files sound at the same pitch (within a few cents)
  <s>_3_new_2d_fit.wav          the 2-D fit to the own recording (FIT2D.md): presets2d.preset2d + fitted string parameters
                                (sideways finger pluck, nut coupling, lateral thread) -> tanpura2d -> F_y + gz F_z ->
                                body.apply_body('tanpura_male', hp_hz, hp_order, mode_gain_db) with the FITTED chain
  <s>_4_control_1d_2d_chain.wav the 1-D model through the 2-D's fitted chain: what the filter alone does (no bloom)
  <s>_ab.wav                    1 | 2 | 3 in sequence with 0.7 s gaps, for quick listening
Drone (30 s): Pa - Sa - Sa - kharaj (gaps 0.8 / 0.8 / 0.8 / 1.0 s, cycle 3.4 s as render_strings.py, +-15 ms and
  +-0.75 dB seeded jitter), each of the four strings its OWN 2-D simulation with the fitted parameters of its string
  (the two Sa share parameters, differ in pluck times/levels, hence in their chaotic trajectories), re-plucked by
  adding the finger-pluck force history at every pluck time (string memory kept; the finger does not damp the string).
  The strings' bridge forces are summed and passed through ONE body filter with the joint chain fitted to all three
  own recordings (out/fit2d/joint_chain.txt: gz 0.05, HP 316 Hz / 2nd order, body modes -6 dB):
  drone_2d_mono.wav, drone_2d_stereo_room.wav (body.apply_body stereo=True, room=0.3).
  Tuning, by ear to the kharaj: the kharaj keeps the fit's contact-off f1 (69.127 Hz; it sounds ~ +4 cents above its
  recording, and the bronze bridge has no grazing thread position at the lower tension that would fix that), Pa and Sa
  are set to sound a pure 3:2 and 2:1 above the kharaj's sounding pitch: contact-off f1 = target minus the contact
  sharpening measured on each fitted single pluck (kharaj ~ +9 c, Pa / Sa ~ +4 c).  The statics re-place the thread
  to graze at that f1, as in the fit.
Also: match_results.txt / .json (features of every file vs the recording, pitch, drone balance and level per cycle,
  reproducibility of the fit's cached renders) and png/match_levels.png (level vs time).

The fitted parameters are copied below from out/fit2d/final_<s>.json ('best', own recording) so that this script
runs without the fit's outputs; if those files exist they are checked against the copy.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
OUT_ROOT = Path(os.environ.get("TANPURA_OUT", PKG / "out"))
OUT = OUT_ROOT / "match"
SR = 44100
STRINGS = ("kharaj", "pa", "sa")

# ---- the fit (FIT2D.md "Best parameters", own recording); f1 = contact-off f1 of the fit (fit2d.target_f1)
FIT = {
    "kharaj": dict(ref="own_kharaj", f1=69.127,
                   x=dict(pluck_deg=90.0, theta_nut=45.0, detune_cents=7.0, target_gap=0.0, pluck_disp=2.7e-3,
                          delta_ve_x=1.0, Yb_x=1.0, kcz_x=2.5, xp_frac=1.0 / 3.0),
                   chain=dict(gz=0.10, hp_hz=400.0, hp_order=2, mode_gain_db=-6.0)),
    "pa": dict(ref="own_pa", f1=103.663,
               x=dict(pluck_deg=88.0, theta_nut=15.0, detune_cents=3.5, target_gap=0.0, pluck_disp=2.0e-3,
                      delta_ve_x=1.0, Yb_x=1.0, kcz_x=1.0, xp_frac=1.0 / 3.0),
               chain=dict(gz=0.05, hp_hz=400.0, hp_order=1, mode_gain_db=-6.0)),
    "sa": dict(ref="own_sa", f1=138.142,
               x=dict(pluck_deg=89.0, theta_nut=30.0, detune_cents=3.3000000000000003, target_gap=1e-06,
                      pluck_disp=0.00205, delta_ve_x=1.0, Yb_x=1.0, kcz_x=2.0300000000000002, xp_frac=1.0 / 3.0),
               chain=dict(gz=0.02, hp_hz=249.0, hp_order=3, mode_gain_db=-6.0)),
}
JOINT_CHAIN = dict(gz=0.05, hp_hz=316.0, hp_order=2, mode_gain_db=-6.0)   # out/fit2d/joint_chain.txt
OLD_BODY = "tanpura_male"            # render_strings.py's chain for the 1-D model (body defaults, no radiation HP)
DUR_SIM = 8.0                        # single-pluck simulation [s]
DUR_OUT = 7.0                        # single-pluck files [s] (0.3 s fade-out at the end; references padded)
FADE_OUT = 0.3
LEAD = 0.1                           # silence before the onset in every single-pluck file [s]
GAP_AB = 0.7                         # silence between the parts of <s>_ab.wav [s]
PEAK_DBFS = -1.0

# drone (render_strings.py conventions)
VOICES = (("pa", "pa"), ("sa1", "sa"), ("sa2", "sa"), ("kharaj", "kharaj"))   # (voice, string), plucking order
GAPS = (0.8, 0.8, 0.8, 1.0)          # Pa->Sa, Sa->Sa, Sa->kharaj, kharaj->Pa [s]; cycle 3.4 s
N_CYCLES = 8
T_START = 0.05
DUR_DRONE = 30.0
FADE_DRONE = 1.0
JITTER_T, JITTER_DB, SEED = 0.015, 0.75, 7
ROOM = 0.3


def say(lines, s=""):
    print(s, flush=True)
    lines.append(s)


# ----------------------------------------------------------------------------------------------- workers
def _patched_excitation(T2, schedule, n_total):
    """Context manager: tanpura2d.simulate then uses copies of the finger-pluck force history at the scheduled times
    (amplitude x gain) instead of one pluck at t = 0.  tanpura2d.py itself is not modified."""
    import contextlib
    orig = T2.excitation

    def exc(p, s=None):
        fe1 = orig(p, s)
        Fe = np.zeros(n_total)
        for t, g in schedule:
            n0 = int(round(t * p["fs"]))
            n1 = min(n_total, n0 + len(fe1))
            Fe[n0:n1] += g * fe1[:n1 - n0]
        return Fe

    @contextlib.contextmanager
    def cm():
        T2.excitation = exc
        try:
            yield
        finally:
            T2.excitation = orig
    return cm()


def job(task):
    """('2d' | '1d', string, x, f1, dur, schedule or None) -> dict of 44.1 kHz signals and diagnostics."""
    kind, string, x, f1, dur, schedule = task
    import fit2d
    from common import to_audio
    t0 = time.time()
    p = fit2d.make_params(string, x, dur, f1=f1, model=kind)
    if kind == "2d":
        import presets2d as P2
        import tanpura2d as T2
        sp = P2.sim_params(p)
        if schedule is None:
            pp, s, r = T2.simulate(sp, energy=False)
        else:
            with _patched_excitation(T2, schedule, int(round(dur * p["fs"]))):
                pp, s, r = T2.simulate(sp, energy=False)
        Fy, Fz = to_audio(r["Fy"], pp["fs"]), to_audio(r["Fz"], pp["fs"])
        st = r["stats"]
        stats = dict(newton_fail=int(st["newton_fail"]), fric_fail=int(st["fric_fail"]), max_pen=float(st["max_pen"]),
                     contact_frac=float(st["contact_frac"]))
    else:
        import tanpura1d as T1
        pp, s, r = T1.simulate({k: v for k, v in p.items() if k != "meta"})
        Fy = to_audio(r["Fbridge"], pp["fs"])
        Fz = np.zeros_like(Fy)
        stats = dict(newton_fail=int(r.get("stats", {}).get("newton_fail", 0)))
    i0 = int(round((p["pluck_pull"] + p["pluck_release"] - 0.005) * SR))   # release end - 5 ms (as fit2d)
    return dict(kind=kind, string=string, Fy=Fy, Fz=Fz, i0=i0, f1=float(pp["f1"]), T=float(s["T"]), stats=stats,
                cpu=time.time() - t0, schedule=schedule, gz=p.get("gz"))


def run_jobs(pool, tasks):
    if pool is None:
        return [job(t) for t in tasks]
    return list(pool.map(job, tasks))


# ----------------------------------------------------------------------------------------------- helpers
def radiate(y, chain, stereo=False, room=0.0, normalize=True):
    """body.apply_body('tanpura_male') with a fitted chain.  normalize=False keeps absolute levels (needed when the
    first 2 s are silent, e.g. one drone voice alone: apply_body normalises to the input RMS of the first 2 s)."""
    import body
    return body.apply_body(np.asarray(y, float), SR, "tanpura_male", stereo=stereo, room=room, hp_hz=chain["hp_hz"],
                           hp_order=int(chain["hp_order"]), mode_gain_db=chain["mode_gain_db"], normalize=normalize)


def fit_out(y, i0):
    """trim at the onset (i0), DUR_OUT s, 0.3 s fade-out, LEAD s of silence in front."""
    y = np.asarray(y, float)[i0:i0 + int(DUR_OUT * SR)]
    y = np.concatenate([y, np.zeros(int(DUR_OUT * SR) - len(y))])
    nf = int(FADE_OUT * SR)
    y[-nf:] *= np.cos(0.5 * np.pi * np.arange(nf) / nf) ** 2
    return np.concatenate([np.zeros(int(LEAD * SR)), y])


def sounding_f0(y, f_guess, t0=0.3, t1=3.0):
    import fit2d
    return fit2d.fit_f0_B(y, f_guess, t0=t0, t1=t1)[0]


def cents(a, b):
    return 1200.0 * np.log2(a / b)


def drone_schedule():
    rng = np.random.default_rng(SEED)
    offs = np.concatenate([[0.0], np.cumsum(GAPS[:-1])])
    cyc = float(np.sum(GAPS))
    sch = {v: [] for v, _ in VOICES}
    for c in range(N_CYCLES):
        for (v, _), o in zip(VOICES, offs):
            t = T_START + c * cyc + o + rng.uniform(-JITTER_T, JITTER_T)
            g = 10 ** (rng.uniform(-JITTER_DB, JITTER_DB) / 20)
            sch[v].append((max(t, 0.0), g))
    return sch


def check_fit_copy(lines):
    """compare FIT with out/fit2d/final_<s>.json if present."""
    for s in STRINGS:
        path = OUT_ROOT / "fit2d" / f"final_{s}.json"
        if not path.exists():
            say(lines, f"  [info] {path} not found: using the parameters copied into render_match.FIT")
            continue
        b = json.load(open(path))["best"][FIT[s]["ref"]]
        ok_x = all(abs(float(b["x"][k]) - float(v)) < 1e-12 for k, v in FIT[s]["x"].items())
        ok_c = all(abs(float(b["refs"][FIT[s]["ref"]]["post"][k]) - float(v)) < 1e-12 for k, v in FIT[s]["chain"].items())
        ok_f = abs(float(b["f1"]) - FIT[s]["f1"]) < 1e-9
        say(lines, f"  {s}: fitted parameters, chain and f1 identical to {path.name}: {ok_x and ok_c and ok_f}")
        if not (ok_x and ok_c and ok_f):
            raise SystemExit(f"render_match.FIT[{s!r}] differs from {path}: update the copy")


# ----------------------------------------------------------------------------------------------- analysis
def analyse(y, f0, R):
    """fit2d features + distance to the recording's features R (the fit's own metric)."""
    import fit2d
    t, A, fe = fit2d.exact_features(y, f0, 0.0, R["tcut"])
    D, Db, _ = fit2d.distance(R["feat"], fe)
    return fe, D, Db


def level_curve(y, f0, tmax=6.5):
    """partial-free broadband level (50 ms frames, 40 Hz HP), dB re max."""
    import scipy.signal as ss
    z = ss.sosfilt(ss.butter(2, 40.0, "highpass", fs=SR, output="sos"), np.asarray(y, float)[:int(tmax * SR)])
    m = int(0.05 * SR)
    k = len(z) // m
    e = 10 * np.log10((z[:k * m].reshape(k, m) ** 2).mean(1) + 1e-30)
    return (np.arange(k) + 0.5) * 0.05, e - e.max()


# ----------------------------------------------------------------------------------------------- singles
def singles(pool, lines, res):
    import soundfile as sf
    import body
    import fit2d
    say(lines, "\n==== single plucks: recording | old 1-D | fitted 2-D ====")
    tasks = []
    for s in STRINGS:
        tasks.append(("2d", s, dict(FIT[s]["x"]), FIT[s]["f1"], DUR_SIM, None))
        tasks.append(("1d", s, dict(fit2d.DEFAULT_X), FIT[s]["f1"], DUR_SIM, None))
    t0 = time.time()
    out = run_jobs(pool, tasks)
    say(lines, f"  6 renders ({DUR_SIM:g} s each) in {time.time() - t0:.0f} s wall, "
               f"{sum(o['cpu'] for o in out):.0f} s CPU")
    files, curves = {}, {}
    for s in STRINGS:
        r2 = next(o for o in out if o["string"] == s and o["kind"] == "2d")
        r1 = next(o for o in out if o["string"] == s and o["kind"] == "1d")
        R = fit2d.ref_features(FIT[s]["ref"])
        yref = fit2d.load_ref(FIT[s]["ref"])
        # reproducibility: the fit's cached 8 s render with the same parameters (float32)
        key = fit2d.run_key(s, fit2d.quantise(dict(fit2d.DEFAULT_X, **FIT[s]["x"])), DUR_SIM, "2d")
        cpath = fit2d.CACHE / "runs" / f"2d_{s}_{key}.npz"
        if cpath.exists():                   # informational: the fit's cache is optional
            c2 = np.load(cpath)
            n = min(len(c2["Fy"]), len(r2["Fy"]))
            rep = float(np.max(np.abs(c2["Fy"][:n] - r2["Fy"][:n].astype(np.float32))) / np.max(np.abs(r2["Fy"])))
        else:
            rep = "n/a (no cached fit render)"
        ch = FIT[s]["chain"]
        mono2 = r2["Fy"] + ch["gz"] * r2["Fz"]
        y2 = radiate(mono2, ch)
        y1 = body.apply_body(r1["Fy"], SR, OLD_BODY)
        y1c = radiate(r1["Fy"], dict(ch, gz=0.0))
        f2 = sounding_f0(r2["Fy"][r2["i0"]:], FIT[s]["f1"])
        f1d = sounding_f0(r1["Fy"][r1["i0"]:], FIT[s]["f1"])
        sig = {"1_reference_own": (fit_out(yref, 0), R["f0"]),
               "2_old_1d": (fit_out(y1, r1["i0"]), f1d),
               "3_new_2d_fit": (fit_out(y2, r2["i0"]), f2),
               "4_control_1d_2d_chain": (fit_out(y1c, r1["i0"]), f1d)}
        files[s] = sig
        res[s] = dict(ref_f0=R["f0"], f0_2d=f2, f0_1d=f1d, sharpen_2d_cents=cents(f2, FIT[s]["f1"]),
                      sharpen_1d_cents=cents(f1d, FIT[s]["f1"]), reproducible_vs_fit_cache=rep,
                      stats_2d=r2["stats"], cpu_2d=r2["cpu"], cpu_1d=r1["cpu"])
        say(lines, f"\n  -- {s}: recording {FIT[s]['ref']} f0 {R['f0']:.3f} Hz; contact-off f1 {FIT[s]['f1']:.3f} Hz")
        say(lines, f"     2-D fit sounds {f2:.3f} Hz ({cents(f2, R['f0']):+.1f} c re recording, contact sharpening "
                   f"{cents(f2, FIT[s]['f1']):+.1f} c); old 1-D {f1d:.3f} Hz ({cents(f1d, R['f0']):+.1f} c)")
        say(lines, f"     2-D solver: {r2['stats']}; reproduces the fit's cached render: max |dF_y|/max F_y = {rep}")
        say(lines, f"     {'file':28s} {'rise':>5s} {'t_max':>5s} {'t_m3':>5s}  H4-10 .09/1/3   H31-60 1/3 s   "
                   f"{'D':>6s} {'Dbloom':>6s}")
        curves[s] = {}
        for k, (y, f0) in sig.items():
            yy = y[int(LEAD * SR):]
            fe, D, Db = analyse(yy, f0, R)
            res[s][k] = dict(rise_db=fe["rise_db"], t_max=fe["t_max"], t_m3=fe["t_m3"], share=fe["share"],
                             hf31_60=fe["hf31_60"], D=D, D_bloom=Db)
            sh = fe["share"]
            hf = fe["hf31_60"]
            say(lines, f"     {k:28s} {fe['rise_db']:5.1f} {fe['t_max']:5.2f} {fe['t_m3']:5.2f}  "
                       f"{sh['0.09']:.2f}/{sh['1']:.2f}/{sh['3']:.2f}   {hf.get('1', np.nan):5.1f}/{hf.get('3', np.nan):5.1f}"
                       f"   {D:6.1f} {Db:6.1f}")
            curves[s][k] = level_curve(yy, f0)
    # one loudness for all single-pluck files, the loudest that keeps every file at or below PEAK_DBFS
    allf = [(s, k) for s in STRINGS for k in files[s]]
    loud = {sk: fit2d.loudness_db(files[sk[0]][sk[1]][0][int(LEAD * SR):]) for sk in allf}
    head = {sk: PEAK_DBFS - 20 * np.log10(np.max(np.abs(files[sk[0]][sk[1]][0]))) + loud[sk] for sk in allf}
    target = min(head.values())
    OUT.mkdir(parents=True, exist_ok=True)
    for s in STRINGS:
        parts = []
        for k, (y, _) in files[s].items():
            yy = y * 10 ** ((target - loud[(s, k)]) / 20)
            sf.write(OUT / f"{s}_{k}.wav", yy, SR, subtype="PCM_24")
            if k != "4_control_1d_2d_chain":
                parts += [yy, np.zeros(int(GAP_AB * SR))]
        sf.write(OUT / f"{s}_ab.wav", np.concatenate(parts[:-1]), SR, subtype="PCM_24")
    say(lines, f"\n  loudness: every single-pluck file at {target:.1f} dB K-weighted (first 4 s), peaks <= {PEAK_DBFS:g} dBFS")
    _plot(curves)
    return files


def _plot(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (OUT / "png").mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    style = {"1_reference_own": ("#222222", "-", "recording (own)"), "2_old_1d": ("#d1495b", "-", "old 1-D"),
             "3_new_2d_fit": ("#2e86de", "-", "new 2-D fit"),
             "4_control_1d_2d_chain": ("#d1495b", ":", "1-D through the 2-D chain")}
    for ax, s in zip(axs, STRINGS):
        for k, (t, L) in curves[s].items():
            c, ls, lab = style[k]
            ax.plot(t, L, color=c, ls=ls, lw=1.6 if k != "4_control_1d_2d_chain" else 1.2, label=lab)
        ax.set_title(f"{s}: broadband level (50 ms frames)")
        ax.set_xlabel("time after onset (s)")
        ax.set_xlim(0, 6.5)
        ax.set_ylim(-45, 2)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("dB re maximum")
    axs[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "png" / "match_levels.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- drone
def drone(pool, lines, res):
    import soundfile as sf
    import fit2d
    say(lines, "\n==== 30 s drone: Pa - Sa - Sa - kharaj, each string its own fitted 2-D simulation, one body ====")
    # tuning, by ear to the kharaj: the kharaj keeps the fit's f1 (the bronze bridge has no grazing thread position
    # at a lower tension: f1 = 68.95 Hz, which would make it sound at its recording's 69.33 Hz, raises in presets.py),
    # Pa and Sa are set to sound a pure 3:2 and 2:1 above the kharaj's SOUNDING pitch, using the contact sharpening
    # measured on each fitted single pluck.  The whole drone then sounds ~ +4 cents above the user's recordings.
    sh = {}
    missing = [s for s in STRINGS if not (s in res and "sharpen_2d_cents" in res[s])]
    if missing:             # drone-only run: measure it (one 8 s single pluck per string)
        for s, r in zip(missing, run_jobs(pool, [("2d", s, dict(FIT[s]["x"]), FIT[s]["f1"], DUR_SIM, None)
                                                 for s in missing])):
            res.setdefault(s, {})["sharpen_2d_cents"] = cents(sounding_f0(r["Fy"][r["i0"]:], FIT[s]["f1"]), FIT[s]["f1"])
    for s in STRINGS:
        sh[s] = res[s]["sharpen_2d_cents"]
    f_k = FIT["kharaj"]["f1"] * 2 ** (sh["kharaj"] / 1200)          # sounding kharaj
    target = {"kharaj": f_k, "pa": 1.5 * f_k, "sa": 2.0 * f_k}
    f1d = {"kharaj": FIT["kharaj"]["f1"]}
    for s in ("pa", "sa"):
        f1d[s] = round(target[s] * 2 ** (-sh[s] / 1200), 3)
    for s in STRINGS:
        f_ref = fit2d.ref_features(FIT[s]["ref"])["f0"]
        say(lines, f"  {s}: single-pluck sharpening {sh[s]:+.2f} c; target sounding {target[s]:.3f} Hz "
                   f"({cents(target[s], f_ref):+.2f} c re its recording) -> drone contact-off f1 {f1d[s]:.3f} Hz "
                   f"({cents(f1d[s], FIT[s]['f1']):+.2f} c re the fit's f1)")
    sch = drone_schedule()
    cyc = float(np.sum(GAPS))
    tasks = [("2d", s, dict(FIT[s]["x"]), f1d[s], DUR_DRONE, sch[v]) for v, s in VOICES]
    order = [3, 0, 1, 2]                      # kharaj, Pa (most modes) first
    t0 = time.time()
    out_o = run_jobs(pool, [tasks[i] for i in order])
    out = [None] * 4
    for i, r in zip(order, out_o):
        out[i] = r
    say(lines, f"  4 voices x {DUR_DRONE:g} s simulated in {time.time() - t0:.0f} s wall, "
               f"{sum(o['cpu'] for o in out):.0f} s CPU; solver: " +
        ", ".join(f"{v} {o['stats']['newton_fail']}/{o['stats']['fric_fail']} fails" for (v, _), o in zip(VOICES, out)))
    gz = JOINT_CHAIN["gz"]
    n = min(len(o["Fy"]) for o in out)
    xsum = sum(o["Fy"][:n] + gz * o["Fz"][:n] for o in out)
    # per-voice pitch (each voice's own bridge force, every interval between its own plucks, 0.3 s after a pluck
    # to 0.05 s before the next) and level (body-radiated, alone)
    fv = {}
    for (v, s), o in zip(VOICES, out):
        tt = [t for t, _ in sch[v]]
        vals = []
        for a, b in zip(tt[:-1], tt[1:]):
            seg = o["Fy"][int(a * SR):int((b - 0.05) * SR)]
            vals.append(sounding_f0(seg, f1d[s], 0.3, (b - 0.05 - a)))
        fv[v] = (float(np.median(vals)), float(np.min(vals)), float(np.max(vals)))
    yv = {v: radiate(o["Fy"][:n] + gz * o["Fz"][:n], JOINT_CHAIN, normalize=False) for (v, _), o in zip(VOICES, out)}
    lv = {v: 10 * np.log10(np.mean(y[int(3 * SR):int(26.5 * SR)] ** 2)) for v, y in yv.items()}
    lk = lv["kharaj"]
    say(lines, "  sounding pitch per voice (median and range over its pluck intervals): " + ", ".join(
        f"{v} {f[0]:.3f} Hz ({f[1]:.3f}-{f[2]:.3f})" for v, f in fv.items()))
    k = fv["kharaj"][0]
    say(lines, f"  intervals: Pa/kharaj {cents(fv['pa'][0] / k, 1.5):+.2f} c re 3:2, Sa1/kharaj "
               f"{cents(fv['sa1'][0] / k, 2.0):+.2f} c re 2:1, Sa2/kharaj {cents(fv['sa2'][0] / k, 2.0):+.2f} c, "
               f"Sa1/Sa2 {cents(fv['sa1'][0], fv['sa2'][0]):+.2f} c, Sa1/Pa {cents(fv['sa1'][0] / fv['pa'][0], 4 / 3):+.2f} c "
               f"re 4:3  (the recordings: Pa/kharaj {cents(res_ref('pa') / res_ref('kharaj'), 1.5):+.2f} c, "
               f"Sa/kharaj {cents(res_ref('sa') / res_ref('kharaj'), 2.0):+.2f} c)")
    say(lines, "  balance (each voice alone through the drone chain, 3-26.5 s, re kharaj): " +
        ", ".join(f"{v} {lv[v] - lk:+.1f} dB" for v in lv))
    ym = radiate(xsum, JOINT_CHAIN)
    ys = radiate(xsum, JOINT_CHAIN, stereo=True, room=ROOM)
    nf = int(FADE_DRONE * SR)
    w = np.cos(0.5 * np.pi * np.arange(nf) / nf) ** 2
    ym[-nf:] *= w
    ys[-nf:] *= w[:, None]
    # level per cycle (does the re-plucked drone build up or sag?)
    cyc_db = []
    for c in range(N_CYCLES):
        a = int((T_START + c * cyc) * SR)
        cyc_db.append(10 * np.log10(np.mean(ym[a:a + int(cyc * SR)] ** 2)))
    cyc_db = np.array(cyc_db) - cyc_db[1]
    say(lines, "  mono level per 3.4 s cycle re cycle 2: " + " ".join(f"{v:+.1f}" for v in cyc_db) + " dB")
    lr = float(np.corrcoef(ys[:, 0], ys[:, 1])[0, 1])
    ym *= 10 ** (PEAK_DBFS / 20) / np.max(np.abs(ym))
    ys *= 10 ** (PEAK_DBFS / 20) / np.max(np.abs(ys))
    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(OUT / "drone_2d_mono.wav", ym, SR, subtype="PCM_24")
    sf.write(OUT / "drone_2d_stereo_room.wav", ys, SR, subtype="PCM_24")
    say(lines, f"  wrote drone_2d_mono.wav, drone_2d_stereo_room.wav ({len(ym) / SR:.1f} s, {FADE_DRONE:g} s fade-out, "
               f"stereo L/R correlation {lr:.2f}); chain {JOINT_CHAIN}")
    res["drone"] = dict(f1=f1d, pitch=fv, balance_db={v: lv[v] - lk for v in lv}, cycle_db=cyc_db.tolist(),
                        lr_corr=lr, cpu=sum(o["cpu"] for o in out),
                        stats={v: o["stats"] for (v, _), o in zip(VOICES, out)})


def res_ref(s):
    import fit2d
    return fit2d.ref_features(FIT[s]["ref"])["f0"]


# ----------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=("singles", "drone"), default=None)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args(argv)
    lines = []
    say(lines, f"==== render_match.py  {time.strftime('%Y-%m-%d %H:%M:%S')}  ({a.jobs} worker process(es)) ====")
    check_fit_copy(lines)
    pool = ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) if a.jobs > 1 else None
    res = {}
    t0 = time.time()
    try:
        if a.only in (None, "singles"):
            singles(pool, lines, res)
        if a.only in (None, "drone"):
            drone(pool, lines, res)
    finally:
        if pool is not None:
            pool.shutdown()
    say(lines, f"\n  total {time.time() - t0:.0f} s wall.  Outputs in {OUT} (git-ignored; the reference copies are "
               f"private recordings: do not commit or redistribute them)")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if a.only is None else f"_{a.only}"
    (OUT / f"match_results{tag}.txt").write_text("\n".join(lines) + "\n")
    with open(OUT / f"match_results{tag}.json", "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))


if __name__ == "__main__":
    main()
