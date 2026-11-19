"""Integrated kharaj demo: statics -> kharaj preset -> 1-D jawari string -> body/radiation filter.

    cd jawari && python render_demo.py            # ~2-3 min single-threaded (statics ~20 s, 12 s + 26 s of string)

Chain (every stage is another file's model, unchanged):
  1. presets.preset('kharaj')   phosphor-bronze C#2 kharaj (0.7 mm, 1.05 m, 69.4 Hz); rest geometry from statics.py
                                (thread auto-placed so the crest just grazes; profile_for_dynamics; static_init),
                                tune='speaking', finger pluck at L/3, loss='valette' (see presets.py for assumptions)
  2. tanpura1d                  1-D modal string with bridge + thread contact; output = bridge force
  3. common.to_audio            20 Hz DC block + band-limited decimation 176.4 -> 44.1 kHz
  4. body.apply_body            one-way bridge-force -> radiated-sound filter ('tanpura_male' body: A0 65 Hz,
                                T1 120 Hz, tuned for a male tanpura in C#; seed 0)

Writes to jawari/out/demo/:
  kharaj_jawari_dry.wav               bridge force, no body (A/B reference)
  kharaj_jawari_body.wav              + body, mono
  kharaj_jawari_body_stereo_room.wav  + body, two mic positions, light room (room = 0.3)
  kharaj_jawari_body_genericpreset.wav  A/B variant with body.py's generic 'tanpura' preset (A0 85 Hz, T1 145 Hz,
                                      placed for a 131 Hz string): thins the kharaj fundamental, octave booms
  kharaj_drone_dry.wav / kharaj_drone_body.wav / kharaj_drone_body_stereo_room.wav
                                      ~28 s drone: the SAME running string re-plucked every 2.8 s (the excitation
                                      force history of tanpura1d contains 9 finger plucks; string memory is kept).
                                      The finger is a prescribed force: it does not damp the vibrating string when
                                      it touches it again (no finger-string contact model).
  <name>.png                          metrics.spectrogram_png of each file
  demo_results.txt                    all numbers (timings, checks, metrics)

Levels: the single-pluck files share one write gain and the drone files another (loudest file of each group peaks at
-1 dBFS).  apply_body() RMS-matches its output to its input over the first 2 s, so files within a group are
loudness-matched (RMS).
"""
from __future__ import annotations

import os
import time

import numpy as np
import scipy.signal as ss
import soundfile as sf

from common import OUT_ROOT, to_audio
from metrics import analyze, report, spectrogram_png
import body
import presets as P
import tanpura1d as T1

SR = 44100
OUT = str(OUT_ROOT / "demo")
os.makedirs(OUT, exist_ok=True)
F1 = 69.4
DUR_SINGLE = 12.0          # simulated seconds, single pluck
PLUCK_PERIOD = 2.8         # drone: re-pluck interval [s]
N_PLUCKS = 9               # drone: plucks at 0, 2.8, ..., 22.4 s
DUR_DRONE = (N_PLUCKS - 1) * PLUCK_PERIOD + 4.0   # 26.4 s simulated, then fade + body/room tail
FADE = 0.5                 # fade-out of the simulated force [s] (the simulation is cut while the string rings)
TAIL = 1.5                 # silence appended so body and room tails ring out [s]
BODY = "tanpura_male"          # body preset for the kharaj (see body.py)
BODY_GENERIC = "tanpura"       # A/B comparison
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


# ----------------------------------------------------------------------------------------------- simulation
def simulate_fe(p, Fe):
    """tanpura1d.simulate with a caller-supplied excitation force history Fe (same positional _run call)."""
    s = T1.build(p)
    N = int(round(p["dur"] * p["fs"]))
    q_init, stat_info = T1.initial_state(p, s)
    t0 = time.time()
    out = T1._run(N, s["a"], s["b"], s["c"], s["ge"], np.ascontiguousarray(Fe, dtype=np.float64), s["gc"], s["kap"],
                  s["lam"], s["thcc"], s["kc"], s["rc"], bool(p["bridge"]), s["Phi"], s["hj"], s["dxb"], s["v"],
                  s["Qp"], float(p["kb"]), float(p["alpha"]), s["w0"], s["wn"], s["phio"], float(p["tol"]),
                  int(p["maxit"]), False, s["m"], s["om2s"], s["sigs"], s["dt"], bool(p["guess"]), q_init)
    cpu = time.time() - t0
    Fbr, Fnut, Fbs, Fcs, F0s, yo, H, Dcum, Wcum, pen, st = out
    stats = dict(max_pen=st[0], contact_frac=st[2] / N, newton_mean=(st[3] / st[2] if st[2] > 0 else 0.0),
                 newton_max=int(st[4]), newton_fail=int(st[5]), max_res=st[6], max_Fc=st[7], edge_pen=st[8],
                 max_active=int(st[9]))
    if stat_info is not None:
        stats["static"] = stat_info
    return s, dict(Fbridge=Fbr, Fnut=Fnut, Fc=Fcs, y=yo, stats=stats, cpu=cpu, N=N)


def drone_excitation(p, s, n_total):
    """N_PLUCKS copies of the preset's single finger-pluck force history, PLUCK_PERIOD apart."""
    fe1 = T1.excitation(p, s)
    Fe = np.zeros(n_total)
    for k in range(N_PLUCKS):
        n0 = int(round(k * PLUCK_PERIOD * p["fs"]))
        n1 = min(n_total, n0 + fe1.size)
        Fe[n0:n1] += fe1[:n1 - n0]
    return Fe


# ----------------------------------------------------------------------------------------------- analysis helpers
def peak_freq(x, fs, f0, t0, t1, search=2.0, pad=16):
    """Frequency of the spectral peak near f0 in [t0, t1]: Hann window, zero-padded FFT, parabolic interpolation of
    the log magnitude (independent of study_kharaj.measure_f, which uses complex demodulation)."""
    seg = x[int(t0 * fs):int(t1 * fs)]
    seg = (seg - seg.mean()) * np.hanning(seg.size)
    nfft = 1 << int(np.ceil(np.log2(seg.size * pad)))
    X = np.abs(np.fft.rfft(seg, nfft))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    band = np.where((f > f0 - search) & (f < f0 + search))[0]
    k = band[np.argmax(X[band])]
    a, b, c = np.log(X[k - 1:k + 2])
    d = 0.5 * (a - c) / (a - 2 * b + c)
    return float(f[k] + d * (f[1] - f[0]))


def cents(f, fref=F1):
    return 1200 * np.log2(f / fref)


def mono(y):
    return y if y.ndim == 1 else y.mean(axis=1)


def ltas(y, t0, t1, nper=1 << 15):
    """Welch PSD; stereo: channel PSDs averaged (a mono sum would cancel decorrelated body modes)."""
    seg = y[int(t0 * SR):int(t1 * SR)]
    if seg.ndim == 1:
        return ss.welch(seg, SR, nperseg=nper)
    f, Pw = ss.welch(seg, SR, nperseg=nper, axis=0)
    return f, Pw.mean(axis=1)


def harmonics_db(y, t0, t1, freqs):
    """level (dB) of the partials at `freqs` (sum of +-2 Welch bins)."""
    f, Pw = ltas(y, t0, t1)
    lv = []
    for fk in freqs:
        i = int(np.argmin(np.abs(f - fk)))
        lv.append(10 * np.log10(Pw[i - 2:i + 3].sum()))
    return np.array(lv)


def band_shares(y, t0, t1):
    f, Pw = ltas(y, t0, t1)
    tot = Pw.sum()
    return {k: float(Pw[(f >= lo) & (f < hi)].sum() / tot)
            for k, (lo, hi) in {"<100": (0, 100), "100-300": (100, 300), "300-1k": (300, 1000),
                                "1-4k": (1000, 4000), ">4k": (4000, SR / 2)}.items()}, float((f * Pw).sum() / tot)


def rolloff_track(m, frac=0.99):
    """per-frame frequency below which `frac` of the frame energy lies (upper edge of the jawari band)."""
    c = np.cumsum(m["S"], axis=0)
    return m["f"][np.argmax(c >= frac * c[-1:], axis=0)]


def mean_in(m, key_or_arr, t0, t1):
    v = m[key_or_arr] if isinstance(key_or_arr, str) else key_or_arr
    sel = (m["t"] >= t0) & (m["t"] <= t1)
    return float(np.mean(v[sel]))


def level_at(m, t):
    return float(m["frame_db"][int(np.argmin(np.abs(m["t"] - t)))])


def corr(a, b, lo=None, hi=None):
    if lo or hi:
        btype, wn = ("bandpass", [lo, hi]) if (lo and hi) else (("highpass", lo) if lo else ("lowpass", hi))
        sos = ss.butter(4, wn, btype, fs=SR, output="sos")
        a, b = ss.sosfilt(sos, a), ss.sosfilt(sos, b)
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def onset_check(y, label):
    """No click at the start: level and >2 kHz energy share of the first 20 ms vs the whole file."""
    w = mono(y)
    n20 = int(0.020 * SR)
    pk = np.max(np.abs(w))
    seg = w[:n20]
    f, Pw = ss.periodogram(seg * np.hanning(n20), SR)
    hf = Pw[f > 2000].sum() / max(Pw.sum(), 1e-300)
    say(f"  {label:34s} first 20 ms: peak {20 * np.log10(np.max(np.abs(seg)) / pk + 1e-300):6.1f} dB re file peak, "
        f"first 1 ms {20 * np.log10(np.max(np.abs(w[:int(1e-3 * SR)])) / pk + 1e-300):7.1f} dB, "
        f"energy share > 2 kHz {hf:.1e}")


def finish(y_force, fade=FADE, tail=TAIL):
    y = np.array(y_force, float)
    nf = int(fade * SR)
    y[-nf:] *= np.cos(0.5 * np.pi * np.arange(nf) / nf) ** 2
    return np.concatenate([y, np.zeros(int(tail * SR))])


def write_group(sig, tag):
    pk = max(np.max(np.abs(v)) for v in sig.values())
    gain = 10 ** (-1 / 20) / pk
    say(f"\n[{tag}] common write gain {20 * np.log10(gain):+.1f} dB (loudest file -> -1 dBFS)")
    say(f"  {'file':34s} {'peak dBFS':>9s} {'RMS 0-2 s dBFS':>14s} {'DC':>9s} {'finite':>6s} {'clip':>5s}")
    for k, v in sig.items():
        w = v * gain
        ok = bool(np.all(np.isfinite(w)))
        clip = bool(np.max(np.abs(w)) > 1.0)
        say(f"  {k:34s} {20 * np.log10(np.max(np.abs(w))):9.2f} {20 * np.log10(np.sqrt(np.mean(mono(w)[:2 * SR] ** 2))):14.2f} "
            f"{np.mean(w):9.1e} {str(ok):>6s} {str(clip):>5s}")
        assert ok and not clip
        sf.write(f"{OUT}/{k}.wav", w, SR, subtype="PCM_24")


def describe(sig, t_windows, fpart, label=""):
    """metrics + spectral-balance table for a group of signals; spectrogram PNG per file."""
    rows = {}
    (a0, a1), (b0, b1) = t_windows
    say(f"\n---- {label}: spectral balance (stereo: channel-averaged spectra) ----")
    say(f"  {'file':34s} {'<100':>6s} {'100-300':>7s} {'300-1k':>6s} {'1-4k':>6s} {'>4k':>7s} {'centroid':>8s} | "
        f"harmonics 2..6 re H1 (dB), {a0:g}-{a1:g} s")
    for k, v in sig.items():
        sh, cen = band_shares(v, a0, a1)
        hl = harmonics_db(v, a0, a1, fpart[:6])
        rows[k] = dict(shares=sh, cen=cen, harm=hl)
        say(f"  {k:34s} {sh['<100']:6.3f} {sh['100-300']:7.3f} {sh['300-1k']:6.3f} {sh['1-4k']:6.3f} {sh['>4k']:7.1e} "
            f"{cen:7.0f}Hz | " + " ".join(f"{x:+5.1f}" for x in hl[1:] - hl[0]))
    say(f"  ---- jawari indicators (metrics.analyze; stereo files mono-averaged as in metrics.py) ----")
    say(f"  {'file':34s} {'buzz ' + f'{a0:g}-{a1:g}s':>12s} {'buzz ' + f'{b0:g}-{b1:g}s':>12s} {'hf_frac':>8s} "
        f"{'99% rolloff @0.5/1/2/4/8 s (Hz)':>34s} {'level @1/2/4/8 s (dB)':>24s}")
    for k, v in sig.items():
        m = analyze(v, SR)
        ro = rolloff_track(m)
        rows[k]["m"] = m
        ts = [t for t in (0.5, 1.0, 2.0, 4.0, 8.0) if t < m["dur"] - 0.1]
        say(f"  {k:34s} {mean_in(m, 'buzz_db', a0, a1):12.1f} {mean_in(m, 'buzz_db', b0, b1):12.1f} "
            f"{mean_in(m, 'hf_frac', a0, a1):8.3f} "
            f"{' / '.join(f'{ro[int(np.argmin(np.abs(m['t'] - t)))]:.0f}' for t in ts):>34s} "
            f"{' / '.join(f'{level_at(m, t):.1f}' for t in (1, 2, 4, 8) if t < m['dur'] - 0.1):>24s}")
        spectrogram_png(m, f"{OUT}/{k}.png", title=f"{k} (kharaj, statics geometry, finger pluck, body.py)")
    say("  ---- metrics.report ----")
    for k in sig:
        say(report(rows[k]["m"], k))
    return rows


# ----------------------------------------------------------------------------------------------- main
def main():
    t_all = time.time()
    say(f"==== render_demo.py  {time.strftime('%Y-%m-%d %H:%M:%S')} ====")
    timings = {}

    # ---- 1. preset (statics + tuning) -------------------------------------------------------------------
    t0 = time.time()
    p = P.preset("kharaj", dur=DUR_SINGLE)
    meta = p.pop("meta")
    timings["statics + tuning (preset)"] = time.time() - t0
    mi = P.modal_info(p)
    say(f"preset 'kharaj': {meta['label']}, T = {meta['T_statics']:.4f} N, l = {meta['l'] * 1e3:.2f} mm, "
        f"psi = {meta['psi_deg']:.4f} deg; thread x_c = {meta['x_c'] * 1e3:.4f} mm (grazing roots "
        f"{[round(r * 1e3, 4) for r in meta['graze_roots']]} mm), crest gap {meta['crest_gap'] * 1e6:+.4f} um at "
        f"x = {meta['x_graze'] * 1e3:.2f} mm, thread preload {meta['F_thread_preload']:.3f} N, x_term "
        f"{meta['x_term'] * 1e6:.1f} um, L_rel = {meta['L_rel'] * 1e3:.3f} mm")
    say(f"  modes M = {mi['M']} (highest {mi['f_M']:.0f} Hz; physics Nyquist {mi['nyquist']:.0f} Hz), bridge points "
        f"{mi['Nb']}, tuned T in tanpura1d {mi['T']:.4f} N; pluck '{p['pluck']}' at x_p = {p['L'] / 3 * 1e3:.1f} mm "
        f"(L/3), pad {p['wp'] * 1e3:g} mm, pull {p['pluck_pull'] * 1e3:g} ms, release {p['pluck_release'] * 1e3:g} ms, "
        f"static pad displacement {p['pluck_disp'] * 1e3:g} mm; loss '{p['loss']}' (Yb {p['Yb']:g} s/kg)")
    say("  modal T60 [s] at 69.4/139/278/555/1k/2k/4k/8k Hz: " +
        " / ".join(f"{mi['T60_at'](f):.2f}" for f in (69.4, 138.8, 277.6, 555.2, 1000, 2000, 4000, 8000)))

    # even-mode content of the finger pluck (static modal deflection under the pad force, thread spring ignored)
    s0 = T1.build(p)
    qs = s0["ge"] / (s0["m"] * s0["om2s"])
    E = 0.5 * s0["m"] * s0["om2s"] * qs ** 2
    n = np.arange(1, s0["M"] + 1)
    say(f"  finger pluck: even-mode share of the static modal energy {100 * E[n % 2 == 0].sum() / E.sum():.1f}% "
        f"(modes 3, 6, 9: {E[2] / E.sum():.1e}, {E[5] / E.sum():.1e}, {E[8] / E.sum():.1e} -- nodes at L/3); "
        f"force amplitude {T1.excitation_amp(p, s0):.4f} N")

    # ---- 2. reference: contact off (tuning + linear jawari baseline) ------------------------------------
    t0 = time.time()
    pr_ = dict(p, bridge=False, dur=6.0)
    pr_, sr_, rr = T1.simulate(pr_)
    timings["reference run (contact off, 6 s)"] = time.time() - t0
    fn_ref = to_audio(rr["Fnut"], p["fs"])
    say(f"\ncontact-off reference (thread spring on, bridge off): f1 = "
        + ", ".join(f"{peak_freq(fn_ref, SR, F1, a, b):.4f} Hz ({cents(peak_freq(fn_ref, SR, F1, a, b)):+.3f} c) in {a:g}-{b:g} s"
                    for a, b in ((0.5, 3.0), (3.0, 6.0))))

    # ---- 3. single pluck --------------------------------------------------------------------------------
    t0 = time.time()
    p1, s1, r1 = T1.simulate(p)
    timings["single pluck 12 s (tanpura1d)"] = time.time() - t0
    st = r1["stats"]
    sinfo = st.get("static", {})
    say(f"\nsingle pluck: cpu {r1['cpu']:.1f} s for {DUR_SINGLE:g} s ({r1['cpu'] / DUR_SINGLE:.2f} s per s); contact "
        f"{100 * st['contact_frac']:.1f}% of samples, max penetration {st['max_pen'] * 1e9:.1f} nm (edge "
        f"{st['edge_pen'] * 1e9:.1f} nm), Newton mean {st['newton_mean']:.2f} / max {st['newton_max']}, failures "
        f"{st['newton_fail']}, max residual {st['max_res']:.1e}")
    say(f"  bilateral thread: max |dynamic thread force| {st['max_Fc']:.3f} N vs static preload "
        f"{meta['F_thread_preload']:.3f} N -> {'HOLDS' if st['max_Fc'] < meta['F_thread_preload'] else 'VIOLATED'}")
    say(f"  static_init: {sinfo.get('n_active', 0)} active bridge points, static bridge force "
        f"{1e3 * sinfo.get('static_force', 0.0):.3f} mN (crest grazes: nothing to pre-load)")
    fnut = to_audio(r1["Fnut"], p["fs"])
    fbr = to_audio(r1["Fbridge"], p["fs"])
    f1m = {(a, b): peak_freq(fnut, SR, F1, a, b) for a, b in ((0.5, 3.0), (3.0, 6.0), (6.0, 11.5))}
    say("  f1 (nut force, FFT peak): " + ", ".join(f"{a:g}-{b:g} s {v:.4f} Hz ({cents(v):+.2f} c)" for (a, b), v in f1m.items()))
    f1b = peak_freq(fbr, SR, F1, 0.5, 3.0)
    say(f"  f1 (bridge force, 0.5-3 s) {f1b:.4f} Hz ({cents(f1b):+.2f} c)")

    t0 = time.time()
    x = finish(fbr)
    single = {"kharaj_jawari_dry": x.copy()}
    single["kharaj_jawari_body"] = body.apply_body(x, SR, BODY)
    single["kharaj_jawari_body_stereo_room"] = body.apply_body(x, SR, BODY, stereo=True, room=0.3)
    single["kharaj_jawari_body_genericpreset"] = body.apply_body(x, SR, BODY_GENERIC)
    timings["body (4 single-pluck versions)"] = time.time() - t0

    # ---- 4. drone: same running string re-plucked ----------------------------------------------------------
    t0 = time.time()
    pd_ = dict(p, dur=DUR_DRONE)
    sd = T1.build(pd_)
    Nd = int(round(DUR_DRONE * pd_["fs"]))
    Fe = drone_excitation(pd_, sd, Nd)
    sd, rd = simulate_fe(pd_, Fe)
    timings[f"drone {DUR_DRONE:g} s (tanpura1d, {N_PLUCKS} plucks)"] = time.time() - t0
    std = rd["stats"]
    say(f"\ndrone: {N_PLUCKS} finger plucks every {PLUCK_PERIOD:g} s on the same running string, {DUR_DRONE:g} s "
        f"simulated, cpu {rd['cpu']:.1f} s ({rd['cpu'] / DUR_DRONE:.2f} s per s); contact {100 * std['contact_frac']:.1f}%,"
        f" max penetration {std['max_pen'] * 1e9:.1f} nm, Newton failures {std['newton_fail']}, max |thread force| "
        f"{std['max_Fc']:.3f} N (preload {meta['F_thread_preload']:.3f} N)")
    # consistency: the first 2.8 s of the drone equal the single-pluck run (same p, same first pluck)
    nchk = int(round(PLUCK_PERIOD * p["fs"])) - 1
    dchk = np.max(np.abs(rd["Fbridge"][:nchk] - r1["Fbridge"][:nchk]))
    say(f"  drone vs single-pluck run over the first {PLUCK_PERIOD:g} s: max |dF| = {dchk:.1e} N (identical scheme)")
    t0 = time.time()
    xd = finish(to_audio(rd["Fbridge"], p["fs"]), fade=1.0)
    drone = {"kharaj_drone_dry": xd.copy()}
    drone["kharaj_drone_body"] = body.apply_body(xd, SR, BODY)
    drone["kharaj_drone_body_stereo_room"] = body.apply_body(xd, SR, BODY, stereo=True, room=0.3)
    timings["body (3 drone versions)"] = time.time() - t0
    fd = to_audio(rd["Fnut"], p["fs"])
    say("  drone f1 (nut force): " + ", ".join(
        f"{a:g}-{a + 2.3:g} s {cents(peak_freq(fd, SR, F1, a, a + 2.3)):+.2f} c" for a in (0.3, 11.5, 22.7)))

    # ---- 5. checks, files, metrics -------------------------------------------------------------------------
    say("\n---- onset (no start-up click; static_init + sin^2 finger pull) ----")
    for k in ("kharaj_jawari_dry", "kharaj_jawari_body", "kharaj_drone_body"):
        onset_check({**single, **drone}[k], k)
    write_group(single, "single pluck")
    write_group(drone, "drone")
    for k in ("kharaj_jawari_body_stereo_room", "kharaj_drone_body_stereo_room"):
        v = {**single, **drone}[k]
        say(f"  {k}: L/R correlation broadband {corr(v[:, 0], v[:, 1]):.2f}, <300 Hz {corr(v[:, 0], v[:, 1], hi=300):.2f}, "
            f">2 kHz {corr(v[:, 0], v[:, 1], lo=2000):.2f}")
    t0 = time.time()
    fpart = s1["om"] / (2 * np.pi)   # linear partial frequencies (stiff string + thread ignored)
    describe(single, ((0.3, 3.0), (3.0, 8.0)), fpart, label="single pluck")
    describe(drone, ((3.0, 22.0), (22.4, 26.0)), fpart, label="drone")
    timings["metrics + spectrograms"] = time.time() - t0

    # body-response facts at the kharaj harmonics
    h, md = body.body_ir(BODY, SR, 0, 1)
    Hf = np.fft.rfft(h[:, 0], 1 << 20)
    ff = np.fft.rfftfreq(1 << 20, 1 / SR)
    HdB = 20 * np.log10(np.abs(Hf) / (md["scale"] * SR) + 1e-30)
    say(f"\nbody '{BODY}' |H| at kharaj partials 1..12 (dB re mean level at 1 kHz): " +
        " ".join(f"{k}:{np.interp(fpart[k - 1], ff, HdB):+.1f}" for k in range(1, 13)))
    say(f"  explicit modes: " + ", ".join(f"{nm} {fq:.0f} Hz" for nm, fq in zip(md["explicit_names"], md["f"][md["explicit"]])))
    say(f"  min modal decay rate {md['sigma'].min():.2f} 1/s (all > 0: stable); IR {h.shape[0] / SR:g} s, causal by construction")

    say("\n---- timings (wall, single-threaded) ----")
    for k, v in timings.items():
        say(f"  {k:42s} {v:6.1f} s")
    say(f"  {'total':42s} {time.time() - t_all:6.1f} s")
    say(f"\nwrote {OUT}/")
    with open(f"{OUT}/demo_results.txt", "w") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
