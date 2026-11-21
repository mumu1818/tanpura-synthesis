"""Round-3 listening set: the recordings vs the previous round (FIT2D) vs round 3 (fit3), plus a 4-string drone.

Status (2026-11-21): the A/B part (--only ab) has been run and checked (out/match3/match3_results_ab.txt). The drone
part has NOT been run yet: its author was stopped before it, so treat it as untested.

    cd jawari && OMP_NUM_THREADS=1 python render_match3.py      # everything (~10 min, ONE simulation process)
    python render_match3.py --only ab                          # A/B files only (no simulation: fit caches)
    python render_match3.py --only drone                       # drone only

Writes to jawari/out/match3/ (git-ignored).  The recording segments in there are copies of the user's private
recordings (Mumu_Tanpura) and of copyrighted iTanpura app samples, loudness-matched for listening: NEVER commit,
upload or move them into the repository.  render_match.py (round 2) is not changed; fit3.py / fit2d.py are only
imported.  No simulation is run for the A/B files: they reuse fit3's cached final renders (out/fit3/cache) and
FIT2D's cached fit renders (out/fit2d/cache).

README: what each file is
-------------------------
Single plucks, for each string <s> in kharaj, pa, sa and each reference <ref> in own, itanpura:
  <s>_<ref>_ab.wav             [recording | 0.5 s | round 2 | 0.5 s | round 3], 44.1 kHz 16-bit mono, the whole
                               file peak-normalised to -1 dBFS.  Each segment starts 5 ms before the pluck onset and
                               lasts 4 s, or 3 s where the recording is a 4 s file that fades out from ~3.1 s (own and
                               iTanpura Sa, iTanpura Pa; fit3.t_end_of).  2 ms fade-in (ends before the onset), 0.15 s
                               cos^2 fade-out.  The two model segments are matched to the recording's ITU-R BS.1770-4
                               integrated (gated) loudness over the segment.
  <s>_<ref>_1_recording.wav    the recording: own = Mumu_Tanpura/Male/tt_CS3_<s>.mp3 (the user's instrument),
                               itanpura = iTanpura/Tanpura_CS3_<Kharaj|Pa|Sa>.mp3; read in place, mono, onset-trimmed
                               (fit3.load_ref3)
  <s>_<ref>_2_round2_fit2d.wav previous round: FIT2D's best string for that recording (fit2d cache, 5 s fit render at
                               fit2d's contact-off f1) through FIT2D's own fitted chain for that recording
                               (fit3.fit2d_chain_signal: tanpura_male, up to 400 Hz / 3rd-order high-pass, body modes
                               -6 dB).  For iTanpura this is FIT2D's iTanpura-fitted string (fit3.FIT2D_ITAN_KEY; for
                               Sa the same render as own).
  <s>_<ref>_3_round3_fit3.wav  round 3: fit3's best string (best_of(load_history), ONE string shared by own and
                               iTanpura), the final 6 s render pitch-corrected to the own recording (f1 69.075 /
                               103.783 / 138.264 Hz), through the PHYSICAL chain fitted to that recording
                               (fit3.chain_signal; body.py tanpura_male explicit modes with A0/T1 moved, HP <= 150 Hz,
                               order <= 2).  The iTanpura segment is the same render through the iTanpura chain.
  The individual segment files are exactly the samples of the A/B file (same gain), so they can be looped or
  compared in an editor.
Drone (30 s):
  drone_fit3_mono.wav          Pa - Sa - Sa - kharaj, gaps 1.1 / 1.1 / 1.1 / 1.45 s (cycle 4.75 s, 6 cycles, last
                               kharaj at ~27.1 s), +-15 ms / +-0.75 dB seeded jitter (seed 7, as round 2).  EVERY
                               PLUCK IS SIMULATED: each of the four strings (two Sa strings = two independent
                               simulations of the same fit3 Sa model) is ONE continuous 30 s fit3 simulation at its
                               reference pitch (its final f1), re-plucked at each scheduled time by adding the finger-
                               pluck force history (render_match._patched_excitation; force amplitude x the jittered
                               gain, so every pluck also differs through the nonlinear contact; the string keeps
                               ringing, the finger does not damp it).  The four bridge forces (F_y + gz F_z) are summed
                               and passed through ONE physical body chain: the chain of fit3's grid that minimises the
                               mean D3 to the three own recordings (joint fit over the final renders).
                               44.1 kHz 16-bit, 1 s fade-out, peak -1 dBFS.
  drone_fit3_stereo_room.wav   the same sum through the same chain, body.apply_body(stereo=True, room=0.3): two mic
                               positions of the same body plus a light small room.  16-bit stereo, peak -1 dBFS.
  The round-2 drone (render_match.py) plucked every 0.8 s (kharaj gap 1.0 s); this one uses the requested classic
  rhythm (one pluck every ~1.1 s, longer pause after the kharaj), with round 2's schedule code, order and jitter.
Also: match3_results.txt / .json (per segment: BS.1770 integrated loudness, ungated K-weighted level, sample and
  true peak, sounding pitch, D3 from fit3's final analysis; the joint drone chain; per-voice drone pitch, balance,
  level per cycle, solver stats) and cache/ (the drone voice simulations).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal as ss

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
import fit2d as F2  # noqa: E402
import fit3 as F3  # noqa: E402

SR = 44100
OUT = F2.OUT_ROOT / "match3"
STRINGS = ("kharaj", "pa", "sa")
REFS = ("own", "itanpura")
GAP_AB = 0.5
FADE_IN, FADE_OUT = 0.002, 0.15
PEAK_DBFS = -1.0
LABELS = ("1_recording", "2_round2_fit2d", "3_round3_fit3")

# drone
VOICES = (("pa", "pa"), ("sa1", "sa"), ("sa2", "sa"), ("kharaj", "kharaj"))   # (voice, string), plucking order
GAPS = (1.1, 1.1, 1.1, 1.45)          # Pa->Sa, Sa->Sa, Sa->kharaj, kharaj->Pa [s]; cycle 4.75 s
N_CYCLES = 6
T_START = 0.05
DUR_DRONE = 30.0
FADE_DRONE = 1.0
JITTER_T, JITTER_DB, SEED = 0.015, 0.75, 7      # render_match.py's values
ROOM = 0.3
DRONE_VERSION = "match3-drone-1"


def say(lines, s=""):
    print(s, flush=True)
    lines.append(s)


def _h(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------- loudness / peaks
def _biquad_shelf(fs):
    """ITU-R BS.1770-4 K-weighting, stage 1 (high shelf), designed at fs as libebur128 does (reproduces the
    standard's 48 kHz coefficients to ~1e-9)."""
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = np.tan(np.pi * f0 / fs)
    Vh = 10 ** (G / 20)
    Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K * K
    b = [(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]
    a = [1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    return np.array(b), np.array(a)


def _biquad_hp(fs):
    """stage 2 (RLB high-pass), libebur128's design (numerator 1, -2, 1 as in the standard)."""
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = np.tan(np.pi * f0 / fs)
    a0 = 1 + K / Q + K * K
    return np.array([1.0, -2.0, 1.0]), np.array([1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0])


def lufs(y, fs=SR):
    """ITU-R BS.1770-4 integrated loudness (LUFS) of a mono or (N, 2) signal: K-weighting, 400 ms blocks with 75 %
    overlap, absolute gate -70 LUFS, relative gate -10 LU."""
    y = np.asarray(y, float)
    y = y[:, None] if y.ndim == 1 else y
    b1, a1 = _biquad_shelf(fs)
    b2, a2 = _biquad_hp(fs)
    z = ss.lfilter(b2, a2, ss.lfilter(b1, a1, y, axis=0), axis=0)
    nb, step = int(0.4 * fs), int(0.1 * fs)
    if len(z) < nb:
        return float("-inf")
    starts = np.arange(0, len(z) - nb + 1, step)
    ms = np.array([np.mean(z[i:i + nb] ** 2, axis=0).sum() for i in starts])
    lk = -0.691 + 10 * np.log10(ms + 1e-30)
    g = lk > -70.0
    if not g.any():
        return float("-inf")
    rel = -0.691 + 10 * np.log10(ms[g].mean()) - 10.0
    g &= lk > rel
    return float(-0.691 + 10 * np.log10(ms[g].mean()))


def peak_dbfs(y):
    return float(20 * np.log10(np.max(np.abs(y)) + 1e-30))


def true_peak_dbtp(y):
    """BS.1770-4 style true peak: 4x oversampled (polyphase) sample peak, per channel max."""
    y = np.asarray(y, float)
    y = y[:, None] if y.ndim == 1 else y
    return float(20 * np.log10(max(np.max(np.abs(ss.resample_poly(y[:, c], 4, 1))) for c in range(y.shape[1])) + 1e-30))


def write16(path, y):
    """16-bit PCM (rounded, clipped at full scale; the signals are peak-normalised below it).  Returns what was
    written, as float, for measuring."""
    import soundfile as sf
    q = np.clip(np.round(np.asarray(y, float) * 32767.0), -32768, 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), q, SR, subtype="PCM_16")
    return q.astype(float) / 32767.0


def segment(y, dur):
    """first dur s (zero-padded), 2 ms half-Hann fade-in (ends before the onset at 5 ms), 0.15 s cos^2 fade-out."""
    n = int(round(dur * SR))
    y = np.asarray(y, float)[:n]
    y = np.concatenate([y, np.zeros(n - len(y))])
    ni, no = int(FADE_IN * SR), int(FADE_OUT * SR)
    y[:ni] *= np.sin(0.5 * np.pi * np.arange(ni) / ni) ** 2
    y[-no:] *= np.cos(0.5 * np.pi * np.arange(no) / no) ** 2
    return y


def seg_dur(rn):
    """4 s, or 3 s for the 4 s recordings that fade out from ~3.1 s (fit3.t_end_of)."""
    return 4.0 if F3.ref3(rn)["t_end"] > 3.5 else 3.0


def cents(a, b):
    return float(1200.0 * np.log2(a / b))


def sounding_f0(y, f_guess, t1):
    return float(F2.fit_f0_B(np.asarray(y, float), f_guess, t0=0.3, t1=t1)[0])


# ----------------------------------------------------------------------------------------------- the fits
def fit3_final(string, lines):
    """fit3's final render (cache) and its per-recording physical chains, re-derived as fit3.final_string does and
    checked against out/fit3/final_<s>.json."""
    own, it = F3.FIT_REFS[string]
    al = F3.HELD_REF[string]
    b = F3.best_of(F3.load_history(string))
    sharp = 1200 * np.log2(b["f0"] / b["f1"])
    f1c = round(F3.ref3(own)["f0"] * 2 ** (-sharp / 1200), 3)
    fj = json.load(open(F3.OUT / f"final_{string}.json"))
    key = F3.run_key(string, b["x"], F3.FINAL_DUR, f1c, True, extra=dict(rm=441))
    cached = (F3.CACHE / "runs" / f"{string}_{key}.npz").exists()
    if not cached:
        say(lines, f"  [warn] {string}: fit3 final render not cached: re-simulating it (one process)")
    run = F3.render(string, b["x"], F3.FINAL_DUR, f1c, True, rec_modes=441)
    sc = F3.score_run(run, F3.FIT_REFS[string] + (al,))
    same_x = F3.quantise(b["x"]) == F3.quantise(fj["best_x"])
    same_f1 = abs(f1c - fj["f1_final"]) < 1e-9
    same_q = all(sc["refs"][rn]["post"] == fj["refs"][rn]["new_post"] for rn in (own, it))
    say(lines, f"  {string}: fit3 best string {F3.fmt_x(b['x'])}; final f1 {f1c} Hz (cache hit {cached}); string, f1 "
               f"and chains identical to final_{string}.json: {same_x and same_f1 and same_q}")
    if not (same_x and same_f1 and same_q):
        raise SystemExit(f"fit3 history / scores disagree with out/fit3/final_{string}.json")
    return run, {rn: sc["refs"][rn]["post"] for rn in (own, it, al)}, fj


def fit2d_prev(string, rn):
    """FIT2D's best string for recording rn and its own fitted chain (as fit3.final_string uses them)."""
    own, it = F3.FIT_REFS[string]
    key = F3.FIT2D_ITAN_KEY[string] if rn == it else F3.FIT2D_BEST_KEY[string]
    run = F3.fit2d_run(string, key)
    post = json.load(open(F2.OUT / f"final_{string}.json"))["best"][rn]["refs"][rn]["post"]
    return run, post, key


# ----------------------------------------------------------------------------------------------- A/B
def ab(lines, res):
    say(lines, "\n==== A/B: recording | round 2 (FIT2D) | round 3 (fit3), loudness-matched (BS.1770-4 integrated) ====")
    finals = {}
    for s in STRINGS:
        run, posts, fj = fit3_final(s, lines)
        finals[s] = (run, posts)
        for tag in REFS:
            rn = f"{tag}_{s}"
            R = F3.ref3(rn)
            T = seg_dur(rn)
            prev, post, pkey = fit2d_prev(s, rn)
            sig = {"1_recording": (F3.load_ref3(rn), R["f0"]),
                   "2_round2_fit2d": (F3.fit2d_chain_signal(prev, post), float(prev["f0"])),
                   "3_round3_fit3": (F3.chain_signal(run, posts[rn]), float(run["f0"]))}
            seg = {k: segment(y, T) for k, (y, _) in sig.items()}
            L0 = {k: lufs(v) for k, v in seg.items()}
            target = L0["1_recording"]
            seg = {k: v * 10 ** ((target - L0[k]) / 20) for k, v in seg.items()}
            gap = np.zeros(int(GAP_AB * SR))
            whole = np.concatenate([seg[LABELS[0]], gap, seg[LABELS[1]], gap, seg[LABELS[2]]])
            g = 10 ** (PEAK_DBFS / 20) / np.max(np.abs(whole))
            w = write16(OUT / f"{s}_{tag}_ab.wav", whole * g)
            n, ng = int(round(T * SR)), len(gap)
            d3 = {"2_round2_fit2d": fj["refs"][rn]["fit2d"].get("D3"), "3_round3_fit3": fj["refs"][rn]["new"].get("D3")}
            chain = {"2_round2_fit2d": F2._fmt_post(post) + f" (FIT2D run {pkey})",
                     "3_round3_fit3": F3.fmt_post(posts[rn]) + f" (fit3 run {run['key']})"}
            say(lines, f"\n  -- {s} / {tag} ({F3.REFS3[rn][0]}): segments {T:g} s; written gain {20 * np.log10(g):+.1f} dB "
                       f"-> {s}_{tag}_ab.wav")
            say(lines, f"     {'segment':16s} {'LUFS':>6s} {'Kungated':>8s} {'peak':>6s} {'TP':>6s} {'f0 Hz':>8s} "
                       f"{'cents':>6s} {'D3':>6s}  chain")
            entry = {}
            for i, k in enumerate(LABELS):
                a = i * (n + ng)
                y = w[a:a + n]
                y_ = write16(OUT / f"{s}_{tag}_{k}.wav", y)
                assert np.array_equal(y_, y)
                f0 = R["f0"] if k == "1_recording" else sounding_f0(y, sig[k][1], min(3.0, T - 0.15))
                e = dict(lufs=lufs(y), k_ungated_db=F2.loudness_db(y, T), peak_dbfs=peak_dbfs(y),
                         true_peak_dbtp=true_peak_dbtp(y), f0=f0, cents_re_recording=cents(f0, R["f0"]),
                         D3=d3.get(k), chain=chain.get(k), seconds=T, gain_db=20 * np.log10(g) + target - L0[k])
                entry[k] = e
                d3s = "     -" if e["D3"] is None else f"{e['D3']:6.1f}"
                say(lines, f"     {k:16s} {e['lufs']:6.1f} {e['k_ungated_db']:8.1f} {e['peak_dbfs']:6.1f} "
                           f"{e['true_peak_dbtp']:6.1f} {f0:8.3f} {e['cents_re_recording']:+6.1f} {d3s}  "
                           f"{chain.get(k, '')}")
            entry["whole"] = dict(lufs=lufs(w), peak_dbfs=peak_dbfs(w), true_peak_dbtp=true_peak_dbtp(w),
                                  seconds=len(w) / SR)
            say(lines, f"     whole file: {entry['whole']['lufs']:.1f} LUFS, peak {entry['whole']['peak_dbfs']:.2f} dBFS, "
                       f"true peak {entry['whole']['true_peak_dbtp']:.2f} dBTP, {len(w) / SR:.2f} s")
            res.setdefault("ab", {})[f"{s}_{tag}"] = entry
    say(lines, "\n  LUFS = ITU-R BS.1770-4 integrated (gated) loudness of the segment as written; Kungated = fit2d's "
               "ungated K-weighted mean-square level (dB, no -0.691 offset) of the same samples; peak = sample peak "
               "dBFS; TP = 4x-oversampled true peak dBTP; cents = sounding pitch re the recording (0.3-3 s partial "
               "fit); D3 = fit3 feature distance to that recording (out/fit3/final_<s>.json; lower is closer).")
    return finals


# ----------------------------------------------------------------------------------------------- drone
def joint_chain(finals, lines, res):
    """One physical body chain for all strings: the grid point of fit3 (CHAIN_GRID x BODY_A0 x BODY_T1) with the
    lowest mean D3 to the three OWN recordings, scored on the final renders (fit3's closed loop).  Cached."""
    keys = {s: str(finals[s][0]["key"]) for s in STRINGS}
    path = OUT / "cache" / f"joint_chain_{_h([keys, F3.FEAT3_VERSION, F3.CHAIN_GRID, F3.BODY_A0, F3.BODY_T1])}.json"
    if path.exists():
        J = json.load(open(path))
    else:
        acc = {}
        for s in STRINGS:
            R = F3.ref3(F3.FIT_REFS[s][0])
            t, c, var = F3.model_variants(finals[s][0])
            for (a0, t1, gz), (A, burst) in var.items():
                for (fc, o) in F3.CHAIN_GRID["hp"]:
                    q = dict(a0=a0, t1=t1, gz=gz, hp_hz=fc, hp_order=o)
                    D3, _ = F3.score_chain(t, c, A, burst, R, q)
                    acc.setdefault(json.dumps(q, sort_keys=True), {})[s] = D3
        best = min(acc, key=lambda k: np.mean(list(acc[k].values())))
        phys = json.dumps(F3.CHAIN_PHYS, sort_keys=True)
        J = dict(chain=json.loads(best), D3=acc[best], D3_chain_phys=acc.get(phys))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(J, fh, indent=1)
    own_d3 = {s: json.load(open(F3.OUT / f"final_{s}.json"))["refs"][f"own_{s}"]["new"]["D3"] for s in STRINGS}
    say(lines, f"  joint physical chain: {F3.fmt_post(J['chain'])}; D3 to the own recordings (closed loop) " +
        ", ".join(f"{s} {J['D3'][s]:.1f} (own chain {own_d3[s]:.1f})" for s in STRINGS) +
        (f"; the fixed round-3 demo chain {F3.fmt_post(F3.CHAIN_PHYS)} would give " +
         ", ".join(f"{s} {J['D3_chain_phys'][s]:.1f}" for s in STRINGS) if J.get("D3_chain_phys") else ""))
    res["drone_chain"] = J
    return J["chain"]


def drone_schedule():
    """render_match.drone_schedule with this round's gaps and cycle count (same seed, jitter, order)."""
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


def drone_voice(string, x, f1, schedule, lines, voice):
    """One continuous fit3 simulation of DUR_DRONE s, re-plucked at the scheduled times (cached)."""
    key = _h(dict(string=string, x=F3.quantise(x), f1=round(f1, 4), dur=DUR_DRONE, sch=schedule, v=DRONE_VERSION))
    path = OUT / "cache" / f"drone_{voice}_{key}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=False)
        say(lines, f"    {voice}: cached ({path.name})")
        return dict(Fy=d["Fy"].astype(float), Fz=d["Fz"].astype(float), cpu=float(d["cpu"]),
                    stats=json.loads(str(d["stats"])))
    from common import to_audio
    import presets2d as P2
    import tanpura2d as T2
    from render_match import _patched_excitation
    p = F3.make_params(string, x, DUR_DRONE, f1, True)
    sp = P2.sim_params(p)
    t0 = time.time()
    with _patched_excitation(T2, schedule, int(round(DUR_DRONE * p["fs"]))):
        pp, s, r = T2.simulate(sp, energy=False)
    cpu = time.time() - t0
    Fy, Fz = to_audio(r["Fy"], pp["fs"]), to_audio(r["Fz"], pp["fs"])
    st = r["stats"]
    stats = dict(newton_fail=int(st["newton_fail"]), fric_fail=int(st["fric_fail"]), max_pen=float(st["max_pen"]),
                 contact_frac=float(st["contact_frac"]), kernel=str(st.get("kernel", "_run")))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, Fy=Fy.astype(np.float32), Fz=Fz.astype(np.float32), cpu=cpu, stats=json.dumps(stats))
    say(lines, f"    {voice}: simulated {DUR_DRONE:g} s in {cpu:.0f} s CPU, {stats}")
    return dict(Fy=Fy, Fz=Fz, cpu=cpu, stats=stats)


def drone(finals, lines, res):
    import body
    say(lines, "\n==== 30 s drone: Pa - Sa - Sa - kharaj, every pluck simulated (fit3 strings), one physical body ====")
    if finals is None:
        finals = {s: fit3_final(s, lines)[:2] for s in STRINGS}
    q = joint_chain(finals, lines, res)
    sch = drone_schedule()
    cyc = float(np.sum(GAPS))
    xs = {s: dict(finals[s][0]["x"]) if not isinstance(finals[s][0]["x"], str) else json.loads(finals[s][0]["x"])
          for s in STRINGS}
    f1 = {s: float(finals[s][0]["f1"]) for s in STRINGS}
    say(lines, f"  schedule: gaps {GAPS} s (cycle {cyc:g} s), {N_CYCLES} cycles, jitter +-{JITTER_T * 1e3:.0f} ms / "
               f"+-{JITTER_DB} dB (seed {SEED}); plucks at " + "; ".join(
                   f"{v} " + " ".join(f"{t:.2f}" for t, _ in sch[v]) for v, _ in VOICES))
    say(lines, "  reference pitch (contact-off f1 of fit3's final render): " +
        ", ".join(f"{s} f1 {f1[s]:.3f} Hz (single pluck sounds {float(finals[s][0]['f0']):.3f})" for s in STRINGS))
    order = ("kharaj", "pa", "sa1", "sa2")
    vmap = dict(VOICES)
    t0 = time.time()
    out = {v: drone_voice(vmap[v], xs[vmap[v]], f1[vmap[v]], sch[v], lines, v) for v in order}
    say(lines, f"  4 voices in {time.time() - t0:.0f} s wall, {sum(o['cpu'] for o in out.values()):.0f} s CPU (one process)")
    gz = q["gz"]
    hp = (20.0, 2) if q["hp_hz"] <= 0 else (q["hp_hz"], int(q["hp_order"]))
    bp = F3.body_params(q["a0"], q["t1"])

    def radiate(y, stereo=False, room=0.0):
        return body.apply_body(np.asarray(y, float), SR, bp, stereo=stereo, room=room, hp_hz=hp[0], hp_order=hp[1],
                               normalize=False)
    n = min(len(o["Fy"]) for o in out.values())
    xv = {v: out[v]["Fy"][:n] + gz * out[v]["Fz"][:n] for v, _ in VOICES}
    xsum = sum(xv.values())
    # pitch per voice: each voice's own bridge force over every interval between its plucks (0.3 s after a pluck to
    # 0.05 s before the next); level per voice: radiated alone, 3-26 s, re kharaj
    fv = {}
    for v, s in VOICES:
        tt = [t for t, _ in sch[v]]
        vals = [sounding_f0(out[v]["Fy"][int(a * SR):int((b - 0.05) * SR)], f1[s], b - 0.05 - a)
                for a, b in zip(tt[:-1], tt[1:])]
        fv[v] = (float(np.median(vals)), float(np.min(vals)), float(np.max(vals)))
    lv = {v: 10 * np.log10(np.mean(radiate(xv[v])[int(3 * SR):int(26 * SR)] ** 2)) for v, _ in VOICES}
    ref_f0 = {s: F3.ref3(f"own_{s}")["f0"] for s in STRINGS}
    say(lines, "  sounding pitch per voice (median, range over its pluck intervals): " + ", ".join(
        f"{v} {f[0]:.3f} Hz ({f[1]:.3f}-{f[2]:.3f}; {cents(f[0], ref_f0[vmap[v]]):+.1f} c re its recording)"
        for v, f in fv.items()))
    k = fv["kharaj"][0]
    say(lines, f"  intervals: Pa/kharaj {cents(fv['pa'][0] / k, 1.5):+.2f} c re 3:2, Sa1/kharaj "
               f"{cents(fv['sa1'][0] / k, 2.0):+.2f} c re 2:1, Sa2/kharaj {cents(fv['sa2'][0] / k, 2.0):+.2f} c, "
               f"Sa1/Sa2 {cents(fv['sa1'][0], fv['sa2'][0]):+.2f} c  (the own recordings: Pa/kharaj "
               f"{cents(ref_f0['pa'] / ref_f0['kharaj'], 1.5):+.2f} c, Sa/kharaj "
               f"{cents(ref_f0['sa'] / ref_f0['kharaj'], 2.0):+.2f} c)")
    say(lines, "  balance (each voice alone through the drone chain, 3-26 s, re kharaj): " +
        ", ".join(f"{v} {lv[v] - lv['kharaj']:+.1f} dB" for v in lv))
    ym = radiate(xsum)
    ys = radiate(xsum, stereo=True, room=ROOM)
    nf = int(FADE_DRONE * SR)
    w = np.cos(0.5 * np.pi * np.arange(nf) / nf) ** 2
    ym[-nf:] *= w
    ys[-nf:] *= w[:, None]
    cyc_db = []
    for c in range(N_CYCLES):
        a = int((T_START + c * cyc) * SR)
        cyc_db.append(10 * np.log10(np.mean(ym[a:a + int(cyc * SR)] ** 2)))
    cyc_db = np.array(cyc_db) - cyc_db[1]
    say(lines, f"  mono level per {cyc:g} s cycle re cycle 2: " + " ".join(f"{v:+.1f}" for v in cyc_db) + " dB")
    lr = float(np.corrcoef(ys[:, 0], ys[:, 1])[0, 1])
    wm = write16(OUT / "drone_fit3_mono.wav", ym * 10 ** (PEAK_DBFS / 20) / np.max(np.abs(ym)))
    wst = write16(OUT / "drone_fit3_stereo_room.wav", ys * 10 ** (PEAK_DBFS / 20) / np.max(np.abs(ys)))
    lvl = {"mono": dict(lufs=lufs(wm), peak_dbfs=peak_dbfs(wm), true_peak_dbtp=true_peak_dbtp(wm)),
           "stereo_room": dict(lufs=lufs(wst), peak_dbfs=peak_dbfs(wst), true_peak_dbtp=true_peak_dbtp(wst))}
    say(lines, f"  wrote drone_fit3_mono.wav ({lvl['mono']['lufs']:.1f} LUFS, peak {lvl['mono']['peak_dbfs']:.2f} dBFS, "
               f"TP {lvl['mono']['true_peak_dbtp']:.2f} dBTP) and drone_fit3_stereo_room.wav "
               f"({lvl['stereo_room']['lufs']:.1f} LUFS, peak {lvl['stereo_room']['peak_dbfs']:.2f} dBFS, TP "
               f"{lvl['stereo_room']['true_peak_dbtp']:.2f} dBTP, L/R correlation {lr:.2f}); {len(wm) / SR:.1f} s, "
               f"{FADE_DRONE:g} s fade-out, chain {F3.fmt_post(q)}, room {ROOM}")
    res["drone"] = dict(gaps=GAPS, n_cycles=N_CYCLES, schedule=sch, f1=f1, pitch=fv,
                        balance_db={v: lv[v] - lv["kharaj"] for v in lv}, cycle_db=cyc_db.tolist(), lr_corr=lr,
                        levels=lvl, cpu=sum(o["cpu"] for o in out.values()),
                        stats={v: o["stats"] for v, o in out.items()}, chain=q, room=ROOM)


# ----------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=("ab", "drone"), default=None)
    a = ap.parse_args(argv)
    lines = []
    say(lines, f"==== render_match3.py  {time.strftime('%Y-%m-%d %H:%M:%S')}  (one simulation process) ====")
    res = {}
    t0 = time.time()
    finals = None
    if a.only in (None, "ab"):
        finals = ab(lines, res)
    if a.only in (None, "drone"):
        drone(finals, lines, res)
    say(lines, f"\n  total {time.time() - t0:.0f} s wall.  Outputs in {OUT} (git-ignored; the recording segments are "
               f"private / copyrighted recordings: do not commit or redistribute them)")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if a.only is None else f"_{a.only}"
    (OUT / f"match3_results{tag}.txt").write_text("\n".join(lines) + "\n")
    with open(OUT / f"match3_results{tag}.json", "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
