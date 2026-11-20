"""Pa, Sa and kharaj of the user's male tanpura in C#: single plucks and a first four-string drone.

    cd jawari && python render_strings.py              # ~6 min wall with 2 worker processes (no cache)
    python render_strings.py --stage single            # only the single plucks (and the preset/statics table)
    python render_strings.py --jobs 1 --no-cache       # serial, recompute everything

Chain per string (every stage is another file's model, unchanged): presets.preset('pa' | 'sa' | 'kharaj') (statics
rest geometry, thread auto-placed to graze, static_init, tune='speaking', finger pluck at L/3, loss='valette') ->
tanpura1d (bridge force) -> common.to_audio -> body.apply_body('tanpura_male').

Single plucks: 12 s, the SAME finger pluck on every string (12 mm pad at L/3, 40 ms pull, 2 ms release, 2 mm static
pad displacement), nominal tuning (contact-off f1 = 104.1 / 138.8 / 69.4 Hz).

Drone (traditional pattern Pa - Sa - Sa - kharaj, repeating): four separate simulations (Pa, Sa 1, Sa 2, kharaj), each
a running string re-plucked at its own times (render_demo.py's method: copies of the finger-pluck force history in one
excitation signal; string memory is kept, the finger does not damp the string).  Their bridge forces are summed at the
model rate and passed through ONE body filter.  The strings are NOT coupled through the bridge (each sees a rigid
bridge; no sympathetic resonance, no energy exchange between strings).
  * timing: plucks 0.8 s apart, 1.0 s pause after the kharaj (cycle 3.4 s), small human timing jitter (+-15 ms) and
    level jitter (+-0.75 dB), seeded;
  * balance (choice, see BALANCE): equal finger work per pluck on every string, i.e. the static pad displacement is
    scaled as sqrt(compliance): 2 mm on the kharaj;
  * tuning (TUNING='by_ear'): a tanpura is tuned by ear with the jawari on, and jawari contact sharpens each string
    by a different amount (single plucks, 0.5-3 s: Pa +3.7, Sa +2.9, kharaj +6.0 cents).  The kharaj stays the
    validated preset (contact-off 69.4 Hz) and is the reference.  Pass 1 renders the drone at nominal tuning; pass 2
    re-renders Pa, Sa1 and Sa2 with their contact-off f1 corrected by the error measured in pass 1, so that they sound
    a pure 3:2 and 2:1 above the kharaj's sounding pitch (statics re-solved, thread re-set to graze).  'nominal' stops
    after pass 1.

Writes to jawari/out/strings/ (git-ignored): {pa,sa,kharaj}_single_{dry,body,body_stereo_room}.wav,
drone_dry.wav, drone_body.wav, drone_body_stereo_room.wav, a spectrogram PNG per file, strings_results.txt.
Simulation results are cached in out/strings/cache/ (keyed by parameters + model source); --no-cache recomputes.
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from common import OUT_ROOT, to_audio
from metrics import analyze, report, spectrogram_png
import body
import presets as P
import tanpura1d as T1
from render_demo import (simulate_fe, peak_freq, ltas, harmonics_db, band_shares, rolloff_track, mean_in, level_at,
                         corr, mono, finish)

SR = 44100
OUT = OUT_ROOT / "strings"
CACHE = OUT / "cache"
PKG = Path(__file__).resolve().parent
BODY = "tanpura_male"
ROOM = 0.3
DUR_SINGLE = 12.0
DUR_REF = 6.0                          # contact-off reference run (pitch with the jawari off)
STRINGS = ("pa", "sa", "kharaj")       # presets.py names
TARGET = {"pa": 104.1, "sa": 138.8, "kharaj": 69.4}     # = preset f1 (pure intervals on the user's 69.4 Hz kharaj)
LABEL = {"pa": "Pa (G#2)", "sa": "Sa (C#3)", "kharaj": "kharaj (C#2)"}

# drone
VOICES = (("pa", "pa"), ("sa1", "sa"), ("sa2", "sa"), ("kharaj", "kharaj"))   # (voice, preset), in plucking order
GAPS = (0.8, 0.8, 0.8, 1.0)            # Pa->Sa, Sa->Sa, Sa->kharaj, kharaj->(next) Pa  [s]; cycle 3.4 s
N_CYCLES = 8                           # last kharaj pluck at ~26.2 s
T_START = 0.05                         # first pluck [s] (leaves room for negative timing jitter)
DUR_DRONE = 30.0                       # simulated [s]; 1 s fade-out at the end, then a 1.5 s silent tail
JITTER_T = 0.015                       # +- uniform timing jitter [s]
JITTER_DB = 0.75                       # +- uniform level jitter [dB]
SEED = 7
BALANCE = "equal_work"                 # 'equal_work' | 'equal_disp' (2 mm on every string)
KHARAJ_DISP = 2e-3                     # static pad displacement on the kharaj [m] (= render_demo / single plucks)
TUNING = "by_ear"                      # 'by_ear' (Pa, Sa tuned to the sounding kharaj) | 'nominal'

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


# ----------------------------------------------------------------------------------------------- caching + workers
def _src_hash():
    h = hashlib.sha1()
    for f in ("tanpura1d.py", "presets.py", "statics.py"):
        h.update((PKG / f).read_bytes())
    return h.hexdigest()[:12]


def _canon(v):
    if isinstance(v, np.ndarray):
        return ("nd", v.dtype.str, v.shape, v.tobytes())
    if isinstance(v, dict):
        return tuple((k, _canon(v[k])) for k in sorted(v))
    if isinstance(v, (tuple, list)):
        return tuple(_canon(x) for x in v)
    return repr(v)


def _key(kind, *args):
    return f"{kind}_" + hashlib.sha1(pickle.dumps((_src_hash(), kind, _canon(args)))).hexdigest()[:20]


def excitation_from_schedule(p, s, n_total, schedule):
    """Sum of copies of the preset's finger-pluck force history at (time [s], gain) pairs."""
    fe1 = T1.excitation(p, s)
    Fe = np.zeros(n_total)
    for t, g in schedule:
        n0 = int(round(t * p["fs"]))
        n1 = min(n_total, n0 + fe1.size)
        Fe[n0:n1] += g * fe1[:n1 - n0]
    return Fe


def job(task):
    """Worker: ('preset', name, overrides) -> preset dict; ('sim', p, schedule|None) -> run results."""
    kind = task[0]
    key = _key(*task)
    path = CACHE / f"{key}.pkl"
    if USE_CACHE and path.exists():
        with open(path, "rb") as fh:
            res = pickle.load(fh)
        os.utime(path)                        # mark as used (stale entries can be deleted by age)
        res["cached"] = True
        return res
    t0 = time.time()
    if kind == "preset":
        _, name, ov = task
        res = dict(p=P.preset(name, **ov))
    else:
        _, p, schedule = task
        if schedule is None:
            _, s, r = T1.simulate(p)
        else:
            s = T1.build(p)
            N = int(round(p["dur"] * p["fs"]))
            s, r = simulate_fe(p, excitation_from_schedule(p, s, N, schedule))
        res = dict(Fbridge=r["Fbridge"], Fnut=r["Fnut"], stats=r["stats"], cpu=r["cpu"], N=r["N"],
                   f_lin=s["om"] / (2 * np.pi), M=s["M"], Nb=s["Nb"], T=s["T"],
                   amp=T1.excitation_amp(p, s), compliance=T1.static_compliance(s))
    res["wall"] = time.time() - t0
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
    res["cached"] = False
    return res


USE_CACHE = os.environ.get("RS_NO_CACHE", "") == ""


def run_jobs(pool, tasks):
    if pool is None:
        return [job(t) for t in tasks]
    return list(pool.map(job, tasks))


# ----------------------------------------------------------------------------------------------- analysis helpers
def cents(f, fref):
    return 1200 * np.log2(f / fref)


def demod_f(x, f0, t0, t1, bw=8.0):
    """Mean frequency (Hz) of the partial near f0 in [t0, t1] (44.1 kHz signal): complex demodulation, zero-phase
    low-pass (bw Hz), linear fit of the unwrapped phase (as study_kharaj.measure_f)."""
    import scipy.signal as ss
    n = np.arange(len(x))
    z = x * np.exp(-2j * np.pi * f0 * n / SR)
    sos = ss.butter(4, bw, fs=SR, output="sos")
    z = ss.sosfiltfilt(sos, z.real) + 1j * ss.sosfiltfilt(sos, z.imag)
    i0, i1 = int(t0 * SR), int(t1 * SR)
    return f0 + np.polyfit(n[i0:i1] / SR, np.unwrap(np.angle(z[i0:i1])), 1)[0] / (2 * np.pi)


def seg_pitch(x, f0, plucks, lo=3.0, hi=26.5):
    """Sounding pitch of a re-plucked string: demod_f in every interval between its own plucks (from 0.3 s after a
    pluck to 0.05 s before the next, inside [lo, hi]), length-weighted mean; also the per-interval values.
    (One long-window FFT peak is biased: periodic re-plucking turns the partial into a comb of lines 1/3.4 s apart.)"""
    tt = [t for t, _ in plucks] + [hi + 10.0]
    fv, w = [], []
    for a, b in zip(tt[:-1], tt[1:]):
        a0, b0 = a + 0.3, min(b - 0.05, hi)
        if a0 >= lo and b0 - a0 >= 1.0:
            fv.append(demod_f(x, f0, a0, b0))
            w.append(b0 - a0)
    fv, w = np.array(fv), np.array(w)
    return float(np.sum(fv * w) / np.sum(w)), fv


def a_weight_db(f):
    """IEC 61672 A-weighting (dB) at frequency f [Hz]."""
    f2 = np.asarray(f, float) ** 2
    ra = (12194.0 ** 2 * f2 ** 2) / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) *
                                     (f2 + 12194.0 ** 2))
    return 20 * np.log10(np.maximum(ra, 1e-30)) + 2.0


def level_db(y, t0, t1, aw=False):
    """mean-square level (dB, arbitrary reference) of y over [t0, t1]; aw: A-weighted (from the Welch PSD)."""
    f, Pw = ltas(y, t0, t1, nper=1 << 13)
    w = 10 ** (a_weight_db(np.maximum(f, 1.0)) / 10) if aw else 1.0
    return float(10 * np.log10(np.sum(Pw * w) * (f[1] - f[0]) + 1e-300))


def onset_check(y, label, t0=0.0):
    import scipy.signal as ss
    w = mono(y)[int(t0 * SR):]
    n20 = int(0.020 * SR)
    pk = np.max(np.abs(w))
    seg = w[:n20]
    f, Pw = ss.periodogram(seg * np.hanning(n20), SR)
    hf = Pw[f > 2000].sum() / max(Pw.sum(), 1e-300)
    say(f"  {label:32s} first 20 ms: peak {20 * np.log10(np.max(np.abs(seg)) / pk + 1e-300):6.1f} dB re file peak, "
        f"first 1 ms {20 * np.log10(np.max(np.abs(w[:int(1e-3 * SR)])) / pk + 1e-300):7.1f} dB, "
        f"energy share > 2 kHz {hf:.1e}")


def write_group(sig, tag):
    pk = max(np.max(np.abs(v)) for v in sig.values())
    gain = 10 ** (-1 / 20) / pk
    say(f"\n[{tag}] common write gain {20 * np.log10(gain):+.1f} dB (loudest file -> -1 dBFS)")
    say(f"  {'file':32s} {'peak dBFS':>9s} {'RMS 0-2 s':>9s} {'A-wt 0-2 s':>10s} {'finite':>6s} {'clip':>5s}")
    for k, v in sig.items():
        w = v * gain
        ok = bool(np.all(np.isfinite(w)))
        clip = bool(np.max(np.abs(w)) > 1.0)
        say(f"  {k:32s} {20 * np.log10(np.max(np.abs(w))):9.2f} {level_db(w, 0, 2):9.2f} "
            f"{level_db(w, 0, 2, aw=True):10.2f} {str(ok):>6s} {str(clip):>5s}")
        assert ok and not clip
        sf.write(str(OUT / f"{k}.wav"), w, SR, subtype="PCM_24")


def describe(sig, t_windows, fpart, label, what):
    """spectral balance + jawari indicators + metrics.report; spectrogram PNG per file."""
    (a0, a1), (b0, b1) = t_windows
    say(f"\n---- {label}: spectral balance (stereo: channel-averaged spectra) ----")
    say(f"  {'file':32s} {'<100':>6s} {'100-300':>7s} {'300-1k':>6s} {'1-4k':>6s} {'>4k':>7s} {'centroid':>8s} | "
        f"harmonics 2..6 re H1 (dB), {a0:g}-{a1:g} s")
    ms = {}
    for k, v in sig.items():
        sh, cen = band_shares(v, a0, a1)
        hl = harmonics_db(v, a0, a1, fpart[k][:6])
        say(f"  {k:32s} {sh['<100']:6.3f} {sh['100-300']:7.3f} {sh['300-1k']:6.3f} {sh['1-4k']:6.3f} {sh['>4k']:7.1e} "
            f"{cen:7.0f}Hz | " + " ".join(f"{x:+5.1f}" for x in hl[1:] - hl[0]))
    say(f"  ---- jawari indicators (metrics.analyze; stereo files mono-averaged) ----")
    say(f"  {'file':32s} {'buzz ' + f'{a0:g}-{a1:g}s':>12s} {'buzz ' + f'{b0:g}-{b1:g}s':>12s} {'hf_frac':>8s} "
        f"{'99% rolloff @0.5/1/2/4/8 s (Hz)':>34s} {'level @1/2/4/8 s (dB)':>26s}")
    for k, v in sig.items():
        m = analyze(v, SR)
        ms[k] = m
        ro = rolloff_track(m)
        ts = [t for t in (0.5, 1.0, 2.0, 4.0, 8.0) if t < m["dur"] - 0.1]
        say(f"  {k:32s} {mean_in(m, 'buzz_db', a0, a1):12.1f} {mean_in(m, 'buzz_db', b0, b1):12.1f} "
            f"{mean_in(m, 'hf_frac', a0, a1):8.3f} "
            f"{' / '.join(f'{ro[int(np.argmin(np.abs(m['t'] - t)))]:.0f}' for t in ts):>34s} "
            f"{' / '.join(f'{level_at(m, t):.1f}' for t in (1, 2, 4, 8) if t < m['dur'] - 0.1):>26s}")
        spectrogram_png(m, str(OUT / f"{k}.png"), title=f"{k} ({what})")
    say("  ---- metrics.report ----")
    for k in sig:
        say(report(ms[k], k))
    return ms


def run_stats_line(r, dur, preload):
    st = r["stats"]
    return (f"cpu {r['cpu']:.1f} s for {dur:g} s ({r['cpu'] / dur:.2f} s/s){' [cached]' if r['cached'] else ''}; "
            f"contact {100 * st['contact_frac']:.1f}% of samples, max penetration {st['max_pen'] * 1e9:.1f} nm "
            f"(edge {st['edge_pen'] * 1e9:.1f} nm), Newton mean {st['newton_mean']:.2f} / max {st['newton_max']}, "
            f"failures {st['newton_fail']}; max |thread force| {st['max_Fc']:.3f} N vs preload {preload:.3f} N -> "
            f"bilateral thread {'HOLDS' if st['max_Fc'] < preload else 'VIOLATED'}")


def drone_schedule():
    """{voice: [(t, gain), ...]} for the Pa-Sa-Sa-kharaj cycle, with seeded timing/level jitter."""
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


# ----------------------------------------------------------------------------------------------- main
def main():
    global USE_CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("single", "all"), default="all")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    if a.no_cache:
        USE_CACHE = False
        os.environ["RS_NO_CACHE"] = "1"      # inherited by the worker processes
    os.makedirs(OUT, exist_ok=True)
    t_all = time.time()
    timings = {}
    say(f"==== render_strings.py  {time.strftime('%Y-%m-%d %H:%M:%S')}  (stage {a.stage}, {a.jobs} worker(s)) ====")
    pool = ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) if a.jobs > 1 else None

    # ---- 1. presets (statics + tuning) ------------------------------------------------------------------
    t0 = time.time()
    res = run_jobs(pool, [("preset", nm, dict(dur=DUR_SINGLE)) for nm in STRINGS])
    base = {nm: r["p"] for nm, r in zip(STRINGS, res)}
    meta = {nm: base[nm].pop("meta") for nm in STRINGS}
    timings["presets (statics + tuning), 3 strings"] = time.time() - t0
    say("\n---- presets (all four strings share L = 1.05 m, bridge parabola c = 3/m, thread 0.2 mm, phi = 10 deg) ----")
    say(f"  {'string':14s} {'material':16s} {'d mm':>5s} {'f1 Hz':>7s} {'T N':>9s} {'stress MPa':>10s} {'l mm':>6s} "
        f"{'psi deg':>8s} {'x_c mm':>7s} {'graze roots mm':>16s} {'x_graze':>7s} {'preload N':>9s} {'M':>4s} "
        f"{'f_M Hz':>7s} {'Nb':>4s}")
    mi = {}
    for nm in STRINGS:
        m, p = meta[nm], base[nm]
        mi[nm] = P.modal_info(p)
        d = np.sqrt(4 * p["A"] / np.pi)
        say(f"  {LABEL[nm]:14s} {m['label']:16s} {d * 1e3:5.2f} {p['f1']:7.2f} {m['T_statics']:9.4f} "
            f"{m['T_statics'] / p['A'] / 1e6:10.0f} {m['l'] * 1e3:6.3f} {m['psi_deg']:8.4f} {m['x_c'] * 1e3:7.4f} "
            f"{str([round(x * 1e3, 3) for x in m['graze_roots']]):>16s} {m['x_graze'] * 1e3:7.2f} "
            f"{m['F_thread_preload']:9.3f} {mi[nm]['M']:4d} {mi[nm]['f_M']:7.0f} {mi[nm]['Nb']:4d}")
        say(f"  {'':14s} crest gap {m['crest_gap'] * 1e6:+.4f} um, x_term {m['x_term'] * 1e6:.1f} um, "
            f"L_rel {m['L_rel'] * 1e3:.3f} mm, tanpura1d tuned T {mi[nm]['T']:.4f} N, modal T60 [s] at f1/2f1/1k/2k/4k: "
            + " / ".join(f"{mi[nm]['T60_at'](f):.2f}" for f in (p['f1'], 2 * p['f1'], 1000, 2000, 4000)))

    # ---- 2. single plucks + contact-off references -------------------------------------------------------
    t0 = time.time()
    tasks = []
    for nm in STRINGS:
        tasks.append(("sim", base[nm], None))
        tasks.append(("sim", dict(base[nm], bridge=False, dur=DUR_REF), None))
    res = run_jobs(pool, tasks)
    single = {nm: res[2 * i] for i, nm in enumerate(STRINGS)}
    ref = {nm: res[2 * i + 1] for i, nm in enumerate(STRINGS)}
    timings["single plucks 3 x 12 s + contact-off refs 3 x 6 s"] = time.time() - t0

    say("\n---- single plucks: the same finger pluck on every string (2 mm static pad displacement at L/3) ----")
    f1_single = {}
    for nm in STRINGS:
        r, rr, p = single[nm], ref[nm], base[nm]
        f0 = TARGET[nm]
        fn_ref = to_audio(rr["Fnut"], p["fs"])
        fn = to_audio(r["Fnut"], p["fs"])
        fr = [peak_freq(fn_ref, SR, f0, a_, b_) for a_, b_ in ((0.5, 3.0), (3.0, 6.0))]
        fj = {(a_, b_): peak_freq(fn, SR, f0, a_, b_) for a_, b_ in ((0.5, 3.0), (3.0, 6.0), (6.0, 11.5))}
        f1_single[nm] = fj
        W = 0.5 * r["amp"] * p["pluck_disp"]
        say(f"  {LABEL[nm]}: pluck force amplitude {r['amp']:.4f} N, finger work {W * 1e3:.3f} mJ")
        say(f"    {run_stats_line(r, DUR_SINGLE, meta[nm]['F_thread_preload'])}")
        say(f"    f1 contact off: " + ", ".join(f"{v:.4f} Hz ({cents(v, f0):+.3f} c)" for v in fr)
            + f"  [contact-off run cpu {rr['cpu']:.1f} s for {DUR_REF:g} s]")
        say(f"    f1 with jawari (nut force): " + ", ".join(f"{a_:g}-{b_:g} s {v:.4f} Hz ({cents(v, f0):+.2f} c)"
                                                       for (a_, b_), v in fj.items()))
        sinfo = r["stats"].get("static", {})
        say(f"    static_init: {sinfo.get('n_active', 0)} active bridge points, static bridge force "
            f"{1e3 * sinfo.get('static_force', 0.0):.3f} mN")

    t0 = time.time()
    sig = {}
    fpart = {}
    for nm in STRINGS:
        x = finish(to_audio(single[nm]["Fbridge"], base[nm]["fs"]))
        sig[f"{nm}_single_dry"] = x
        sig[f"{nm}_single_body"] = body.apply_body(x, SR, BODY)
        sig[f"{nm}_single_body_stereo_room"] = body.apply_body(x, SR, BODY, stereo=True, room=ROOM)
        for k in ("dry", "body", "body_stereo_room"):
            fpart[f"{nm}_single_{k}"] = single[nm]["f_lin"]
    timings["body (9 single-pluck versions)"] = time.time() - t0
    say("\n---- onset (no start-up click) ----")
    for nm in STRINGS:
        onset_check(sig[f"{nm}_single_dry"], f"{nm}_single_dry")
    write_group(sig, "single plucks (one gain for all 9 files: relative string levels are kept)")
    say("  single-pluck levels re kharaj (body, mono, 0-2 s): " + ", ".join(
        f"{nm} RMS {level_db(sig[f'{nm}_single_body'], 0, 2) - level_db(sig['kharaj_single_body'], 0, 2):+.1f} dB / "
        f"A-wt {level_db(sig[f'{nm}_single_body'], 0, 2, True) - level_db(sig['kharaj_single_body'], 0, 2, True):+.1f} dB"
        for nm in STRINGS))
    t0 = time.time()
    describe(sig, ((0.3, 3.0), (3.0, 8.0)), fpart, "single plucks", "finger pluck at L/3, statics geometry, body.py")
    timings["metrics + spectrograms (single)"] = time.time() - t0

    if a.stage == "all":
        drone(pool, base, meta, single, timings)

    if pool is not None:
        pool.shutdown()
    say("\n---- timings (wall) ----")
    for k, v in timings.items():
        say(f"  {k:58s} {v:6.1f} s")
    say(f"  {'total':58s} {time.time() - t_all:6.1f} s")
    say(f"\nwrote {OUT}/")
    with open(OUT / ("strings_results.txt" if a.stage == "all" else "strings_results_single.txt"), "w") as fh:
        fh.write("\n".join(LINES) + "\n")


def drone(pool, base, meta, single, timings):
    say("\n==== four-string drone: Pa - Sa - Sa - kharaj ====")
    cyc = float(np.sum(GAPS))
    say(f"  pattern: gaps {' / '.join(f'{g:g}' for g in GAPS)} s (cycle {cyc:g} s, mean pluck interval "
        f"{cyc / 4:.2f} s), {N_CYCLES} cycles, jitter +-{JITTER_T * 1e3:g} ms / +-{JITTER_DB:g} dB (seed {SEED}); "
        f"{DUR_DRONE:g} s simulated per string")
    # balance
    disp = {}
    for nm in STRINGS:
        if BALANCE == "equal_work":
            disp[nm] = KHARAJ_DISP * np.sqrt(single[nm]["compliance"] / single["kharaj"]["compliance"])
        else:
            disp[nm] = KHARAJ_DISP
    say(f"  balance '{BALANCE}': static pad displacement " + ", ".join(
        f"{nm} {disp[nm] * 1e3:.3f} mm (finger work {0.5 * disp[nm] ** 2 / single[nm]['compliance'] * 1e3:.3f} mJ)"
        for nm in STRINGS))
    sch = drone_schedule()

    # tuning.  The kharaj is the validated preset at its nominal tuning (contact-off 69.4 Hz) and is the reference;
    # by ear, Pa and Sa are tuned to its SOUNDING pitch in the drone (pure 3:2 and 2:1).  Pass 1 = the whole drone at
    # nominal tuning; pass 2 re-runs Pa, Sa1, Sa2 with the contact-off f1 corrected by the pitch error measured in
    # pass 1 (Sa: mean of Sa1 and Sa2, which share one preset).  Pitch = seg_pitch (per pluck interval).  The kharaj
    # is not retuned: its grazing thread position is very sensitive to tension (lowering it by 8 c re-grazes the
    # thread at 4.26 instead of 4.00 mm), so retuning it would change its validated jawari setting.
    fs = base["kharaj"]["fs"]
    for nm in STRINGS:
        meta[nm + "_drone"] = meta[nm]

    def voice_tasks(pre, which):
        return [("sim", dict(pre[VOICES[i][1]], pluck_disp=disp[VOICES[i][1]], dur=DUR_DRONE), sch[VOICES[i][0]])
                for i in which]

    def pitches(res):
        return {v: seg_pitch(to_audio(r["Fnut"], fs), TARGET[nm], sch[v]) for (v, nm), r in zip(VOICES, res)}

    def intervals(fv):
        return (f"Sa1/kharaj {cents(fv['sa1'] / fv['kharaj'], 2.0):+.2f} c re 2:1, Sa2/kharaj "
                f"{cents(fv['sa2'] / fv['kharaj'], 2.0):+.2f} c, Sa1/Sa2 {cents(fv['sa1'], fv['sa2']):+.2f} c, Pa/kharaj "
                f"{cents(fv['pa'] / fv['kharaj'], 1.5):+.2f} c re 3:2, Sa1/Pa {cents(fv['sa1'] / fv['pa'], 4 / 3):+.2f} c re 4:3")

    t0 = time.time()
    order = [3, 0, 1, 2]                      # kharaj, pa (140 modes) first
    res_o = run_jobs(pool, voice_tasks(base, order))
    res = [None] * 4
    for i, r in zip(order, res_o):
        res[i] = r
    timings["drone: 4 voices x 30 s, nominal tuning"] = time.time() - t0
    pf = pitches(res)
    f_nom = {v: pf[v][0] for v in pf}
    say(f"  pass 1, nominal tuning (contact-off f1 = 104.1 / 138.8 / 69.4 Hz): sounding (per pluck interval, "
        f"3-26.5 s) " + ", ".join(f"{v} {f_nom[v]:.4f} Hz ({cents(f_nom[v], base[nm]['f1']):+.2f} c, intervals "
                                  f"{np.min(cents(pf[v][1], base[nm]['f1'])):+.2f}..{np.max(cents(pf[v][1], base[nm]['f1'])):+.2f})"
                                  for v, nm in VOICES))
    say(f"    intervals: {intervals(f_nom)}")
    pre = dict(base)
    if TUNING == "by_ear":
        f_k = f_nom["kharaj"]
        err = {"pa": cents(f_nom["pa"], 1.5 * f_k),
               "sa": 0.5 * (cents(f_nom["sa1"], 2 * f_k) + cents(f_nom["sa2"], 2 * f_k))}
        f1_use = {nm: base[nm]["f1"] * 2 ** (-err[nm] / 1200) for nm in ("pa", "sa")}
        say(f"  pass 2, Pa and Sa tuned by ear to the sounding kharaj ({f_k:.4f} Hz): error " + ", ".join(
            f"{nm} {err[nm]:+.2f} c -> contact-off f1 {f1_use[nm]:.4f} Hz" for nm in ("pa", "sa")))
        t0 = time.time()
        rp = run_jobs(pool, [("preset", nm, dict(f1=f1_use[nm], dur=DUR_DRONE)) for nm in ("pa", "sa")])
        for nm, r in zip(("pa", "sa"), rp):
            pre[nm] = r["p"]
            m = pre[nm].pop("meta")
            meta[nm + "_drone"] = m
            say(f"    {nm}: re-solved statics at f1 {f1_use[nm]:.4f} Hz: T {m['T_statics']:.4f} N, thread re-set to "
                f"graze at x_c {m['x_c'] * 1e3:.4f} mm (was {meta[nm]['x_c'] * 1e3:.4f}), crest gap "
                f"{m['crest_gap'] * 1e6:+.4f} um, preload {m['F_thread_preload']:.3f} N")
        timings["drone: retuned Pa/Sa presets (statics)"] = time.time() - t0
        t0 = time.time()
        res[:3] = run_jobs(pool, voice_tasks(pre, [0, 1, 2]))
        timings["drone: Pa, Sa1, Sa2 voices x 30 s, tuned by ear"] = time.time() - t0
    Fsum = np.zeros_like(res[0]["Fbridge"])
    say("  voices:")
    for (v, nm), r in zip(VOICES, res):
        Fsum += r["Fbridge"]
        say(f"    {v:7s} {len(sch[v])} plucks at {', '.join(f'{t:.2f}' for t, _ in sch[v][:3])}, ... s: "
            + run_stats_line(r, DUR_DRONE, meta[nm + "_drone"]["F_thread_preload"]))
    # pitch of each voice in the drone (own nut force) and the intervals
    say("  sounding pitch in the drone (each voice's own nut force, per pluck interval in 3-26.5 s):")
    pf = pitches(res)
    fv = {v: pf[v][0] for v in pf}
    for v, nm in VOICES:
        say(f"    {v:7s} {fv[v]:.4f} Hz ({cents(fv[v], TARGET[nm]):+.2f} c re {TARGET[nm]:g} Hz; "
            f"{cents(fv[v], pre[nm]['f1']):+.2f} c re its contact-off f1 {pre[nm]['f1']:.4f} Hz; per interval "
            f"{np.min(cents(pf[v][1], pre[nm]['f1'])):+.2f}..{np.max(cents(pf[v][1], pre[nm]['f1'])):+.2f} c)")
    say(f"    intervals: {intervals(fv)}")

    # ONE body filter on the summed bridge force
    t0 = time.time()
    xd = finish(to_audio(Fsum, fs), fade=1.0)
    dsig = {"drone_dry": xd.copy()}
    dsig["drone_body"] = body.apply_body(xd, SR, BODY)
    dsig["drone_body_stereo_room"] = body.apply_body(xd, SR, BODY, stereo=True, room=ROOM)
    # per-voice contributions (linear: body(sum) = sum of body(voice) with normalize=False) -> balance report
    say("  per-voice level in the drone (through the body, mono, 3-26 s; dB re kharaj):")
    lv = {}
    for (v, nm), r in zip(VOICES, res):
        yb = body.apply_body(finish(to_audio(r["Fbridge"], fs), fade=1.0), SR, BODY, normalize=False)
        lv[v] = (level_db(yb, 3, 26), level_db(yb, 3, 26, aw=True))
    say("    " + ", ".join(f"{v} RMS {lv[v][0] - lv['kharaj'][0]:+.1f} / A-wt {lv[v][1] - lv['kharaj'][1]:+.1f}"
                           for v, _ in VOICES))
    timings["drone: body + per-voice levels"] = time.time() - t0
    say("\n---- onset ----")
    onset_check(dsig["drone_body"], "drone_body (from the first pluck)", T_START)
    write_group(dsig, "drone")
    v = dsig["drone_body_stereo_room"]
    say(f"  drone_body_stereo_room: L/R correlation broadband {corr(v[:, 0], v[:, 1]):.2f}, <300 Hz "
        f"{corr(v[:, 0], v[:, 1], hi=300):.2f}, >2 kHz {corr(v[:, 0], v[:, 1], lo=2000):.2f}")
    t0 = time.time()
    fk = res[3]["f_lin"]
    describe(dsig, ((3.0, 26.0), (26.3, 29.0)), {k: fk for k in dsig}, "drone (harmonics of the kharaj)",
             "Pa-Sa-Sa-kharaj, 4 uncoupled strings, one body")
    timings["metrics + spectrograms (drone)"] = time.time() - t0


if __name__ == "__main__":
    main()
