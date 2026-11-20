"""Validation of reference.py (the recorded-pluck analyser, comparison and fit).

    cd jawari
    python validate_reference.py synthetic        # A: known synthetic plucks (~1 min)
    python validate_reference.py roundtrip        # B: kharaj model renders, dry and through the body (~3 min, 3 jobs)
    python validate_reference.py fit              # C: fit a pseudo-reference with known parameters (~10-15 min)
    python validate_reference.py all

Writes out/reference/validation/{synthetic,roundtrip,fit}.txt plus figures and compare reports.

A  synthetic: sums of damped, inharmonic partials with a known f0, B, T60(n), pluck comb (beta), pitch glide and
   drift, a known moving 'formant' with a bloom, overlapping plucks (iTanpura-style) -> recovery errors.
B  model round trip: tanpura1d kharaj renders (linear and jawari; pluck at L/3, L/4, 2L/3; thread +-0.1 mm;
   delta_ve x 2) through common.to_audio and body.apply_body.  Linear renders are checked against the exact
   damped eigenvalues of the modal string + thread spring-damper; jawari renders against a demodulation of the
   raw bridge force, the string's B = pi^2 EI / (T L^2), and the known parameter changes.
C  fitting: a pseudo-reference with known non-default parameters (target_gap +1 um, delta_ve x 1.5, pluck at 2L/3)
   is fitted from the defaults.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import scipy.signal as ss

import reference as R

OUT = R.OUT / "validation"
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(str(s))


def flush(name):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.txt").write_text("\n".join(LINES) + "\n")
    LINES.clear()


def check(label, ok, msg):
    say(f"  [{'PASS' if ok else 'FAIL'}] {label}: {msg}")
    return bool(ok)


# ============================================================================================ A. synthetic
def synth(sr=44100, dur=8.0, t_on=0.4, f0=69.4 * 2 ** (3 / 1200), B=1.6e-4, beta=0.27, T60=None, glide=8.0,
          tau=0.15, drift=-0.5, noise_db=-75.0, fmax=10000.0, formant=None, amp=1.0, seed=0, harmonic=False):
    """Damped partials n = 1.. (f_n < fmax): amplitude |sin(n pi beta)| / n, T60(n), common pitch glide
    c(t) = glide exp(-t/tau) + drift t (cents), random phases, 1 ms raised-cosine onset at t_on (exact sample
    t_on * sr may be fractional: the onset ramp is evaluated on the continuous time axis).  formant: dict(G, w,
    nc0, nc_rate, t_bloom, t_decay) adds a moving bump G g(t) exp(-0.5 ((n - nc(t)) / w)^2) to the amplitudes.
    Returns (y, truth dict)."""
    rng = np.random.default_rng(seed)
    T60 = T60 or (lambda n: 30.0 / (1.0 + (n / 8.0) ** 1.5))
    t = np.arange(int(dur * sr)) / sr
    tt = t - t_on
    c = np.where(tt >= 0, glide * np.exp(-np.maximum(tt, 0) / tau) + drift * np.maximum(tt, 0), glide)
    ratio = 2 ** (c / 1200)
    phi_base = 2 * np.pi * np.cumsum(ratio) / sr                        # integral of the relative frequency
    ramp = np.clip(tt / 1e-3, 0, 1)
    ramp = np.sin(0.5 * np.pi * ramp) ** 2
    y = np.zeros_like(t)
    n_all = []
    amps = []
    nmax = 1
    while (nmax + 1) * f0 * np.sqrt(1 + (0 if harmonic else B) * (nmax + 1) ** 2) < fmax:
        nmax += 1
    for n in range(1, nmax + 1):
        fn = n * f0 * (1.0 if harmonic else np.sqrt(1 + B * n * n))
        a = amp * max(abs(np.sin(n * np.pi * beta)), 1e-3) / n
        env = a * np.exp(-6.9078 * np.maximum(tt, 0) / T60(n))
        if formant:
            F = formant
            nc = F["nc0"] - F["nc_rate"] * np.maximum(tt, 0)
            g = (1 - np.exp(-np.maximum(tt, 0) / F["t_bloom"])) * np.exp(-np.maximum(tt, 0) / F["t_decay"])
            env = env + amp * F["G"] * g * np.exp(-0.5 * ((n - nc) / F["w"]) ** 2)
        amps.append(env)
        y += ramp * env * np.sin(fn * phi_base + rng.uniform(0, 2 * np.pi))
        n_all.append(n)
    y += 10 ** (noise_db / 20) * np.sqrt(np.mean(y[int(t_on * sr):int((t_on + 1) * sr)] ** 2)) * rng.standard_normal(t.size)
    truth = dict(f0=f0, B=B, beta=beta, T60=np.array([T60(n) for n in n_all]), glide=glide, tau=tau, drift=drift,
                 t_on=t_on, n=np.array(n_all), amps=np.array(amps), t=t, c=c)
    return y, truth


def truth_tracks(tr, A):
    """BI_k and partial-octave levels of the known amplitudes on the analyser's time grid (power = a^2 / 2)."""
    tg = A["t"]
    t_abs = tg + tr["t_on"]
    P = np.stack([np.interp(t_abs, tr["t"], a) ** 2 / 2 for a in tr["amps"]])
    k = A["jawari"]["BI_main"]
    BI = 10 * np.log10(P[k - 1:].sum(0) / P[:k - 1].sum(0))
    oc = []
    for a, b in A["jawari"]["oct_bands"]:
        oc.append(10 * np.log10(P[a - 1:min(b, P.shape[0])].sum(0) / P.sum(0)))
    return BI, np.array(oc), P


def part_a():
    say(f"==== A. synthetic signals ({time.strftime('%Y-%m-%d %H:%M')}) ====")
    sr = 44100
    allok = []
    # ---- A1: inharmonic damped partials, comb at beta = 0.27, glide + drift, noise
    y, tr = synth()
    A = R.analyse(y, sr, string="kharaj", name="synthA1")
    say(R.text_summary(A))
    on = A["onset"]
    say(f"\nA1 truth: f0 {tr['f0']:.4f} Hz (+3.0 c), B {tr['B']:.2e}, beta {tr['beta']}, glide {tr['glide']} c "
        f"(tau {tr['tau']} s), drift {tr['drift']} c/s, onset {tr['t_on']} s, noise -75 dB")
    allok.append(check("onset start", abs(on["t_start"] - tr["t_on"]) < 2e-3,
                       f"start {1e3 * (on['t_start'] - tr['t_on']):+.2f} ms re the true onset, anchor "
                       f"{1e3 * (on['t_anchor'] - tr['t_on']):+.2f} ms (anchor = -6 dB of the attack peak)"))
    # shift invariance of the anchor (alignment precision)
    sh = []
    for k in (1, 37, 1001):
        A2 = R.analyse(np.concatenate([np.zeros(k), y])[:y.size], sr, string="kharaj")
        sh.append(A2["onset"]["anchor"] - on["anchor"] - k)
    allok.append(check("anchor shift invariance", max(abs(np.array(sh))) < 0.05,
                       f"anchor moves with an integer shift of 1/37/1001 samples to within {max(np.abs(sh)):.3f} samples"))
    # f0: the analyser measures the mean over its spectrum window (0.1-2.5 s after the anchor)
    t0, t1 = A["partials"]["window"]
    sel = (tr["t"] >= tr["t_on"] + t0 + A["seg_t0"] + 0.05) & (tr["t"] < tr["t_on"] + t1 + A["seg_t0"] + 0.05)
    c_true = 3.0 + np.mean(tr["c"][sel])
    d = A["f0"]["cents_fit"] - c_true
    allok.append(check("f0", abs(d) < 0.3, f"{A['f0']['cents_fit']:+.2f} c vs {c_true:+.2f} c expected "
                                             f"(window mean incl. glide/drift): error {d:+.2f} c"))
    eB = A["partials"]["B"] / tr["B"] - 1
    eBa = A["partials"]["B_all"] / tr["B"] - 1
    allok.append(check("B", abs(eBa) < 0.02 and abs(eB) < 0.1,
                       f"all partials {A['partials']['B_all']:.3e} ({100 * eBa:+.1f} %), partials <= 12 "
                       f"{A['partials']['B']:.3e} ({100 * eB:+.1f} %)"))
    g_true = (np.mean(tr["glide"] * np.exp(-np.linspace(0.03, 0.15, 200) / tr["tau"]) + tr["drift"] * np.linspace(0.03, 0.15, 200))
              - np.mean(tr["glide"] * np.exp(-np.linspace(0.5, 1.5, 200) / tr["tau"]) + tr["drift"] * np.linspace(0.5, 1.5, 200)))
    allok.append(check("glide", abs(A["f0"]["glide_cents"] - g_true) < 1.0,
                       f"{A['f0']['glide_cents']:+.2f} c vs {g_true:+.2f} c (mean 30-150 ms minus mean 0.5-1.5 s)"))
    allok.append(check("drift", abs(A["f0"]["drift_cents_per_s"] - tr["drift"]) < 0.1,
                       f"{A['f0']['drift_cents_per_s']:+.3f} c/s vs {tr['drift']:+.3f}"))
    dc = A["decay"]
    N = min(dc["T60_late"].size, tr["T60"].size)
    snr = (A["Pdb"].max(1) - A["floor_db"])[:N]
    okp = np.isfinite(dc["T60_late"][:N]) & (snr > 30)
    rel = dc["T60_late"][:N][okp] / tr["T60"][:N][okp] - 1
    say(f"  T60 late: {okp.sum()} partials with SNR > 30 dB; median |error| {100 * np.median(np.abs(rel)):.1f} %, "
        f"90th pct {100 * np.percentile(np.abs(rel), 90):.1f} %, worst {100 * np.max(np.abs(rel)):.1f} % (n = "
        f"{A['n'][:N][okp][int(np.argmax(np.abs(rel)))]})")
    for (a, b) in R.partial_groups(N):
        s_ = okp & (A["n"][:N] >= a) & (A["n"][:N] <= b)
        if s_.any():
            say(f"     partials {a:3d}-{b:3d}: measured {np.median(dc['T60_late'][:N][s_]):6.2f} s, true "
                f"{np.median(tr['T60'][:N][s_]):6.2f} s")
    allok.append(check("T60 late", np.median(np.abs(rel)) < 0.05 and np.percentile(np.abs(rel), 90) < 0.15,
                       "median < 5 %, 90th percentile < 15 %"))
    okq = np.isfinite(dc["T60_early"][:N]) & (snr > 30)
    rel_e = dc["T60_early"][:N][okq] / tr["T60"][:N][okq] - 1
    say(f"  T60 early (EDT-like, first 0.5 s): median |error| {100 * np.median(np.abs(rel_e)):.1f} % over {okq.sum()} partials")
    c = A["comb"]
    allok.append(check("pluck comb beta", abs(c["beta"] - tr["beta"]) < 0.01,
                       f"{c['beta']:.4f} vs {tr['beta']} (evidence {c['evidence_db']:.1f} dB, depth {c['depth_db']:.0f} dB, "
                       f"notches {c['notches']})"))
    # comb at beta = 0.4 (notches 5, 10, 15) and 1/4 (4, 8, 12)
    for b_ in (0.4, 0.25):
        yb, trb = synth(beta=b_, seed=3)
        Ab = R.analyse(yb, sr, string="kharaj")
        allok.append(check(f"pluck comb beta={b_}", abs(Ab["comb"]["beta"] - b_) < 0.01,
                           f"{Ab['comb']['beta']:.4f} (notches {Ab['comb']['notches']})"))
    # ---- A2: jawari-like moving formant with a bloom (harmonic partials)
    F = dict(G=0.06, w=4.0, nc0=30.0, nc_rate=4.0, t_bloom=0.25, t_decay=2.0)
    y2, tr2 = synth(formant=F, harmonic=True, beta=1 / 3, seed=5, glide=0.0, drift=0.0)
    A2 = R.analyse(y2, sr, string="kharaj", name="synthA2_formant")
    BI_t, oct_t, P_t = truth_tracks(tr2, A2)
    j = A2["jawari"]
    k = j["BI_main"]
    m = (A2["t"] > 0.05) & (A2["t"] < 6.0) & np.isfinite(j[f"BI{k}"])
    err = j[f"BI{k}"][m] - BI_t[m]
    say(f"\nA2 moving formant: bump G {F['G']} at partial nc(t) = {F['nc0']} - {F['nc_rate']} t, width {F['w']}, "
        f"bloom tau {F['t_bloom']} s, decay {F['t_decay']} s; harmonic partials, comb at 1/3")
    allok.append(check(f"BI{k} track", np.sqrt(np.mean(err ** 2)) < 1.0,
                       f"rms error {np.sqrt(np.mean(err ** 2)):.2f} dB over 0.05-6 s (bias {np.mean(err):+.2f} dB)"))
    eo = []
    for i, (a, b) in enumerate(j["oct_bands"]):
        mm = m & np.isfinite(j["octaves"][i]) & (oct_t[i] > -50)
        if mm.sum() > 10:
            eo.append(np.sqrt(np.mean((j["octaves"][i][mm] - oct_t[i][mm]) ** 2)))
    allok.append(check("partial-octave levels", max(eo) < 1.0, "rms errors " + ", ".join(f"{v:.2f}" for v in eo) + " dB"))
    t_pk_true = A2["t"][m][int(np.argmax(R._nan_smooth(BI_t, 20)[m]))]
    allok.append(check("bloom peak time", abs(j["t_bloom_peak"] - t_pk_true) < 0.1,
                       f"{j['t_bloom_peak']:.2f} s vs {t_pk_true:.2f} s (truth BI{k}, 100 ms smoothing)"))
    nc = F["nc0"] - F["nc_rate"] * A2["t"]
    mp_ = (A2["t"] > 0.3) & (A2["t"] < 2.0)
    epk = np.nanmedian(np.abs(j["peak"][mp_] - nc[mp_]))
    found = np.mean(np.isfinite(j["peak"][mp_]))
    allok.append(check("formant peak tracking", epk < 2.0 and found > 0.8,
                       f"median |peak - nc(t)| = {epk:.1f} partials over 0.3-2 s (nc from {nc[mp_][0]:.0f} to "
                       f"{nc[mp_][-1]:.0f}); hump found in {100 * found:.0f} % of the frames"))
    say(f"  bloom: onset {j['t_bloom_onset']:.2f} s, peak {j['t_bloom_peak']:.2f} s, gain {j['bloom_gain_db']:+.1f} dB, "
        f"persistence {j['persistence_s']:.2f} s")
    # ---- A3: overlapping plucks
    yk, trk = synth(dur=6.0, t_on=0.4, seed=7)
    yp, trp = synth(dur=6.0, t_on=2.0, f0=69.4 * 1.5, B=1.0e-4, beta=0.3, seed=8, amp=0.8, glide=4.0, drift=0.0)
    A3 = R.analyse(yk + yp, sr, string="kharaj", name="synthA3_overlap")
    say(f"\nA3 kharaj at 0.4 s + Pa at 2.0 s: onsets {np.round(A3['onsets'], 3)}, analysed {A3['dur']:.2f} s")
    allok.append(check("next-pluck detection", A3["t_next"] is not None and abs(A3["t_next"] - 2.0) < 0.03,
                       f"next pluck at {A3['t_next']} s; analysis stops at {A3['t_end']:.3f} s"))
    allok.append(check("f0/B with truncation", abs(A3["f0"]["cents_fit"] - A["f0"]["cents_fit"]) < 0.5 and
                       abs(A3["partials"]["B_all"] / trk["B"] - 1) < 0.03,
                       f"f0 {A3['f0']['cents_fit']:+.2f} c, B {A3['partials']['B_all']:.3e}"))
    d3 = A3["decay"]["T60_late"][:20]
    okd = np.isfinite(d3)
    rel3 = d3[okd] / trk["T60"][:20][okd] - 1
    allok.append(check("T60 with truncation (partials <= 20)", np.median(np.abs(rel3)) < 0.1,
                       f"median |error| {100 * np.median(np.abs(rel3)):.1f} % over {okd.sum()} partials (1.6 s of data)"))
    # ---- A4: iTanpura-like 4-string cycle, analyse the kharaj (4th pluck) while the others ring on
    seq = [(0.3, 69.4 * 1.5, 1.0e-4, 0.8), (1.5, 138.8, 6e-5, 0.6), (2.7, 138.8, 6e-5, 0.6), (3.9, 69.4, 1.6e-4, 1.0)]
    ys = np.zeros(int(10.0 * sr))
    for i, (t_, f_, b_, a_) in enumerate(seq):
        yy, _ = synth(dur=10.0, t_on=t_, f0=f_, B=b_, amp=a_, beta=0.3, seed=20 + i, glide=3.0, drift=0.0,
                      noise_db=-200)
        ys += yy
    A4 = R.analyse(ys, sr, string="kharaj", pluck=3, name="synthA4_cycle")
    say(f"\nA4 Pa/Sa/Sa/kharaj cycle (1.2 s apart), analysing pluck 3: onsets {np.round(A4['onsets'], 3)}, "
        f"onset {A4['t_onset']:.3f} s, {len(A4['contaminated'])} partials flagged as contaminated "
        f"(first {[int(k) for k in A4['contaminated'][:10]]}); f0/B refit without "
        f"{len(A4['partials'].get('refit_excluded', []))} partials")
    allok.append(check("4-pluck cycle onsets", len(A4["onsets"]) == 4 and abs(A4["onset"]["t_start"] - 3.9) < 0.005,
                       f"{len(A4['onsets'])} onsets; kharaj start {A4['onset']['t_start']:.4f} s, anchor "
                       f"{A4['t_onset']:.4f} s (true 3.9 s)"))
    # Sa partials coincide with even kharaj partials and Pa's even ones with multiples of 3 (within ~1 cent:
    # different B), Pa's odd partials sit on the kharaj band edges: ~1 c of f0 bias is inherent here
    allok.append(check("kharaj f0 / B under overlap", abs(A4["f0"]["cents_fit"]) < 2.0 and
                       abs(A4["partials"]["B"] / 1.6e-4 - 1) < 0.15,
                       f"f0 {A4['f0']['cents_fit']:+.2f} c (true +0 c + glide mean), B (<= 12) {A4['partials']['B']:.3e}"))
    # ---- A5: stereo + room: the A1 pluck damped by hand at 4 s, in a decorrelated exponential room (T60 0.6 s)
    rng = np.random.default_rng(11)
    yd = y.copy()
    i_d = int(4.0 * sr)
    nfd = int(0.01 * sr)
    yd[i_d:i_d + nfd] *= np.cos(0.5 * np.pi * np.arange(nfd) / nfd) ** 2
    yd[i_d + nfd:] = 0.0
    tt = np.arange(int(2.0 * sr)) / sr
    h = [rng.standard_normal(tt.size) * np.exp(-6.9078 * tt / 0.6) for _ in range(2)]
    g = np.sqrt(np.sum(yd ** 2)) / np.sqrt(np.sum(ss.fftconvolve(yd, h[0])[:yd.size] ** 2)) * 10 ** (-6 / 20)
    st = np.stack([yd + g * ss.fftconvolve(yd, h[0])[:yd.size], 0.9 * yd + g * ss.fftconvolve(yd, h[1])[:yd.size]], 1)
    st += 1e-5 * rng.standard_normal(st.shape)
    A5 = R.analyse(st, sr, string="kharaj", name="synthA5_stereo_room")
    S5 = A5["stereo"]
    say(f"\nA5 stereo pluck in a decorrelated room (T60 0.6 s, reverb -6 dB re direct), damped at 4 s: L/R corr "
        f"{S5['corr_broad']:.2f}, side/mid {S5['side_mid_db']:.1f} dB, coherence per octave "
        + ", ".join(f"{k:.0f}: {v:.2f}" for k, v in S5["coh_oct"].items()))
    allok.append(check("stereo features", 0.3 < S5["corr_broad"] < 0.99 and S5["side_mid_db"] < -3,
                       "a common direct sound + independent reverbs: correlated, mid-dominated"))
    rm = A5["room"]
    allok.append(check("room T60 after damping", rm is not None and abs(rm["T60"] - 0.6) < 0.1,
                       f"damping detected at {rm['t_damp'] if rm else None} s after the onset, room T60 "
                       f"{rm['T60'] if rm else None} s (true 0.6 s)"))
    say(f"\nA: {sum(allok)}/{len(allok)} checks pass")
    plot_a(A, tr, A2, BI_t, oct_t)
    flush("synthetic")
    return allok


def plot_a(A, tr, A2, BI_t, oct_t):
    plt, seq, div = R._mpl()
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    N = min(A["decay"]["T60_late"].size, tr["T60"].size)
    axs[0].semilogy(tr["n"][:N], tr["T60"][:N], "-", color=R.INK2, label="true T60(n)")
    axs[0].semilogy(A["n"][:N], A["decay"]["T60_late"][:N], "o", ms=3, color=R.C_REF, label="measured (late)")
    axs[0].set_xlabel("partial number")
    axs[0].set_ylabel("s")
    axs[0].set_title("A1: per-partial T60", color=R.INK)
    axs[0].legend()
    j = A2["jawari"]
    k = j["BI_main"]
    axs[1].plot(A2["t"], BI_t, color=R.INK2, label="truth")
    axs[1].plot(A2["t"], j[f"BI{k}"], color=R.C_REF, lw=1, label="measured")
    axs[1].set_xlabel("s")
    axs[1].set_title(f"A2: BI{k} (dB)", color=R.INK)
    axs[1].legend()
    F_nc = 30.0 - 4.0 * A2["t"]
    axs[2].plot(A2["t"], F_nc, color=R.INK2, label="true formant centre nc(t)")
    axs[2].plot(A2["t"], j["peak"], ".", ms=2, color=R.C_REF, label="measured peak")
    axs[2].set_ylim(0, 40)
    axs[2].set_xlabel("s")
    axs[2].set_ylabel("partial number")
    axs[2].set_title("A2: formant tracking", color=R.INK)
    axs[2].legend()
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "synthetic.png", dpi=110)
    plt.close(fig)


# ============================================================================================ B. model round trip
CASES = {
    "lin_L3": dict(bridge=False),
    "lin_L4": dict(bridge=False, xp_frac=0.25),
    "lin_2L3": dict(bridge=False, xp_frac=2 / 3),
    "lin_loss2": dict(bridge=False, delta_ve=9.0e-3),
    "jaw_L3": dict(),
    "jaw_2L3": dict(xp_frac=2 / 3),
    "jaw_L4": dict(xp_frac=0.25),
    "jaw_gap": dict(thread_shift=+0.1e-3),
    "jaw_pressed": dict(thread_shift=-0.1e-3),
    "jaw_loss2": dict(delta_ve=9.0e-3),
}
DUR_B = 6.0


def _render_case(name):
    import tanpura1d as T1
    from common import to_audio
    import body as BD
    p = R.model_params("kharaj", DUR_B, **CASES[name])
    p.pop("meta")
    p, s, r = T1.simulate(p)
    fs = p["fs"]
    y = to_audio(r["Fbridge"], fs)
    yb = BD.apply_body(y, 44100, "tanpura_male")
    # independent f1: complex demodulation of the RAW bridge force (176.4 kHz) at 69.4 Hz, phase slope
    return dict(name=name, y=y, yb=yb, p={k: v for k, v in p.items() if k != "profile"}, cpu=r["cpu"],
                om=s["om"], sig=s["sig"], gc=s["gc"], kc=s["kc"], rc=s["rc"], m=s["m"], T=s["T"], EI=s["EI"],
                L=s["L"], mu=s["mu"], Fraw=r["Fbridge"][::4].copy(), fs_raw=fs / 4, contact=r["stats"]["contact_frac"])


def exact_linear(rd):
    """Damped eigenfrequencies / T60 of the modal string + thread spring-damper (continuous time)."""
    om, sig, gc, kc, rc, m = rd["om"], rd["sig"], rd["gc"], rd["kc"], rd["rc"], rd["m"]
    M = om.size
    K = np.diag(m * om ** 2) + kc * np.outer(gc, gc)
    C = np.diag(2 * m * sig) + rc * np.outer(gc, gc)
    Asys = np.block([[np.zeros((M, M)), np.eye(M)], [-K / m, -C / m]])
    lam = np.linalg.eigvals(Asys)
    lam = lam[lam.imag > 0]
    lam = lam[np.argsort(lam.imag)]
    return lam.imag / (2 * np.pi), 6.9078 / (-lam.real)


def demod_f(x, fs, f0, t0, t1, bw=8.0):
    n = np.arange(x.size)
    z = x * np.exp(-2j * np.pi * f0 * n / fs)
    sos = ss.butter(4, bw, fs=fs, output="sos")
    z = ss.sosfiltfilt(sos, z.real) + 1j * ss.sosfiltfilt(sos, z.imag)
    i0, i1 = int(t0 * fs), int(t1 * fs)
    ph = np.unwrap(np.angle(z[i0:i1]))
    return f0 + np.polyfit(n[i0:i1] / fs, ph, 1)[0] / (2 * np.pi)


def part_b(jobs=3):
    say(f"==== B. model round trip ({time.strftime('%Y-%m-%d %H:%M')}) ====")
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn")) as ex:
        res = {r["name"]: r for r in ex.map(_render_case, list(CASES))}
    say(f"rendered {len(res)} cases x {DUR_B:g} s in {time.time() - t0:.0f} s wall ({jobs} processes); "
        f"string CPU {sum(r['cpu'] for r in res.values()):.0f} s")
    allok = []
    An = {}
    for nm, r in res.items():
        for tag, y in (("dry", r["y"]), ("body", r["yb"])):
            An[(nm, tag)] = R.analyse(y, 44100, string="kharaj", name=f"{nm}_{tag}")
    # ---- linear: exact eigenvalues
    r = res["lin_L3"]
    f_ex, T_ex = exact_linear(r)
    B_th = np.pi ** 2 * r["EI"] / (r["T"] * r["L"] ** 2)
    say(f"\nlinear kharaj (bridge contact off), pluck at L/3; T {r['T']:.3f} N, B_theory = pi^2 EI/(T L^2) = {B_th:.4e}")
    nn = np.arange(1, f_ex.size + 1)
    f0_ex, B_ex, _ = R.fit_inharmonicity(nn[:12], f_ex[:12])
    for tag in ("dry", "body"):
        A = An[("lin_L3", tag)]
        tp = A["partials"]
        g = tp["good"]
        N = min(tp["n"].size, f_ex.size)
        ce = 1200 * np.log2(tp["f"][:N][g[:N]] / f_ex[:N][g[:N]])
        say(f"  [{tag}] f0 {A['f0']['f0_fit']:.4f} Hz (exact eigenfrequency 1: {f_ex[0]:.4f} Hz; f0 of the exact "
            f"modes' n<=12 fit {f0_ex:.4f}); B (n<=12) {tp['B']:.4e}, all {tp['B_all']:.4e} (exact modes' fit "
            f"{B_ex:.4e}, theory {B_th:.4e})")
        allok.append(check(f"partial frequencies [{tag}]", np.percentile(np.abs(ce), 90) < 0.5,
                           f"{g[:N].sum()} partials: median |error| {np.median(np.abs(ce)):.3f} c, 90th pct "
                           f"{np.percentile(np.abs(ce), 90):.3f} c vs the exact eigenfrequencies"))
        allok.append(check(f"B [{tag}]", abs(tp["B_all"] / B_th - 1) < 0.03,
                           f"{100 * (tp['B_all'] / B_th - 1):+.1f} % vs theory"))
        dT = A["decay"]["T60_late"][:N]
        snr = (A["Pdb"].max(1) - A["floor_db"])[:N]
        ok = np.isfinite(dT) & (snr > 30)
        rel = dT[ok] / T_ex[:N][ok] - 1
        allok.append(check(f"T60 late [{tag}]", np.median(np.abs(rel)) < (0.05 if tag == "dry" else 0.10),
                           f"{ok.sum()} partials: median |error| {100 * np.median(np.abs(rel)):.1f} %, 90th pct "
                           f"{100 * np.percentile(np.abs(rel), 90):.1f} % vs the exact damped eigenvalues "
                           f"(T60 {T_ex[0]:.1f} s at n=1, {T_ex[9]:.1f} s at n=10)"))
    say(f"  note: the uncoupled modal T60 6.91/sigma differs from the exact one by "
        f"{100 * np.median(np.abs(6.9078 / r['sig'][:20] / T_ex[:20] - 1)):.1f} % (median, n <= 20): the thread "
        "damper r_c adds loss")
    # pluck position from the comb (linear)
    for nm, b_ in (("lin_L3", 1 / 3), ("lin_L4", 0.25), ("lin_2L3", 1 / 3)):
        for tag in ("dry", "body"):
            c = An[(nm, tag)]["comb"]
            allok.append(check(f"comb {nm} [{tag}]", abs(c["beta"] - b_) < 0.01,
                               f"beta {c['beta']:.4f} / {1 - c['beta']:.4f} (true {res[nm]['p'].get('xp') or res[nm]['p']['L'] / 3:.4f} m "
                               f"= {(res[nm]['p'].get('xp') or res[nm]['p']['L'] / 3) / res[nm]['p']['L']:.3f} L), "
                               f"evidence {c['evidence_db']:.1f} dB, notches {c['notches']}"))
    # loss change, linear: expected ratio from the exact eigenvalues
    f2, T2 = exact_linear(res["lin_loss2"])
    for tag in ("dry", "body"):
        A1, A2 = An[("lin_L3", tag)], An[("lin_loss2", tag)]
        # per-partial ratio on the partials measured in BOTH renders, then the group median (a ratio of group
        # medians would compare different partial sets)
        G = R.partial_groups(40)
        rm = A2["decay"]["T60_late"][:40] / A1["decay"]["T60_late"][:40]
        rx = T2[:40] / T_ex[:40]
        gm = R._group_median(rm, np.arange(1, 41), G)
        gx = R._group_median(np.where(np.isfinite(rm), rx, np.nan), np.arange(1, 41), G)
        rows = [f"{a}-{b}: {m_:.3f} (exact {x_:.3f})" for (a, b), m_, x_ in zip(G, gm, gx) if np.isfinite(m_)]
        errs = [abs(m_ / x_ - 1) for m_, x_ in zip(gm, gx) if np.isfinite(m_)]
        allok.append(check(f"delta_ve x2 decay ratio [{tag}]", max(errs) < 0.03,
                           "T60 ratio by partial group " + "; ".join(rows)))
    # ---- jawari renders
    say("\njawari renders (kharaj preset, statics geometry, finger pluck), dry / through body.apply_body:")
    fr = res["jaw_L3"]
    say(f"{'case':12s} {'tag':4s} {'f0 c':>6s} {'f1 c':>6s} {'raw f1 c':>8s} {'B<=12':>9s} {'B>12':>9s} {'beta':>6s} "
        f"{'ev':>4s} {'BI0':>6s} {'BImax':>6s} {'bloom':>6s} {'t_pk':>5s} {'persist':>7s} {'oct8-15@.5':>10s} "
        f"{'oct16-31@2':>10s} {'centre':>6s} {'e95':>5s} {'T60 late 1|4-6|11-16|26-40':>28s}")
    for nm in ("lin_L3", "jaw_L3", "jaw_2L3", "jaw_L4", "jaw_gap", "jaw_pressed", "jaw_loss2"):
        r = res[nm]
        A0 = An[(nm, "dry")]
        t_on = A0["t_onset"]
        f_raw = demod_f(r["Fraw"], r["fs_raw"], 69.6, t_on + A0["partials"]["window"][0], t_on + A0["partials"]["window"][1])
        for tag in ("dry", "body"):
            A = An[(nm, tag)]
            s = R.summary(A)
            j = A["jawari"]

            def oc(i, tt):
                return R._at(A["t"], j["octaves"][i], np.array([tt]))[0]
            G = s["T60_late_groups"]
            say(f"{nm:12s} {tag:4s} {s['cents_fit']:+6.1f} {s['cents_f1'] if s['cents_f1'] is not None else np.nan:+6.1f} "
                f"{R.cents(f_raw, 69.4):+8.2f} {s['B']:9.2e} {s['B_hi'] if s['B_hi'] is not None else np.nan:9.1e} "
                f"{s['pluck_beta'] or np.nan:6.3f} {s['comb_evidence_db'] or np.nan:4.1f} {s['BI_start']:6.1f} "
                f"{s['BI_max']:6.1f} {s['bloom_gain_db']:+6.1f} {s['t_bloom_peak']:5.2f} {s['persistence_s']:7.2f} "
                f"{oc(3, 0.5):10.1f} {oc(4, 2.0):10.1f} {R._at(A['t'], j['centre'], np.array([1.0]))[0]:6.1f} "
                f"{R._at(A['t'], j['e95'], np.array([1.0]))[0]:5.0f} "
                f"{'|'.join(_f2(G.get(k)) for k in ('1-1', '4-6', '11-16', '26-40')):>28s}")
            if tag == "dry" and nm.startswith("jaw"):
                An[(nm, "raw_f1")] = f_raw
    A = An[("jaw_L3", "dry")]
    w0, w1 = A["partials"]["window"]
    w0, w1 = w0 + A["seg_t0"], w1 + A["seg_t0"]
    sel = (A["t"] >= w0) & (A["t"] < w1)
    if_1 = float(np.mean(A["IF"][0][sel]))          # uniform time weighting, like the phase-slope demodulation
    f_raw = An[("jaw_L3", "raw_f1")]
    allok.append(check("jawari partial-1 frequency vs raw bridge force", abs(R.cents(if_1, f_raw)) < 0.3,
                       f"analyser band-1 instantaneous frequency (time mean over its spectrum window) {if_1:.4f} Hz vs "
                       f"phase slope of the raw 176.4 kHz bridge force {f_raw:.4f} Hz ({R.cents(if_1, f_raw):+.2f} c); "
                       f"contact sharpening +{R.cents(f_raw, 69.4):.1f} c"))
    say(f"  (the long-window PEAK of partial 1 is {A['f0']['f1']:.4f} Hz = {R.cents(A['f0']['f1'], f_raw):+.1f} c re "
        "the raw phase slope: the peak weights the louder, sharper start of a drifting pitch; f0 of partials <= 12 "
        f"{A['f0']['cents_fit']:+.1f} c: partials 2-12 are not exactly harmonic multiples of partial 1 under contact)")
    B_th = np.pi ** 2 * fr["EI"] / (fr["T"] * fr["L"] ** 2)
    say(f"  jawari harmonicity: B (partials <= 12) {A['partials']['B']:.2e} = {100 * A['partials']['B'] / B_th:.0f} % of "
        f"the string's {B_th:.2e}; B (> 12) {A['partials']['B_hi']:.1e}: the contact locks the upper partials harmonic")
    # distinguishability: compare() of variants vs the nominal, body versions
    say("\ncompare(): nominal (L/3, grazing) as 'reference' vs each variant, through the body; top 4 mismatches:")
    for nm in ("jaw_2L3", "jaw_gap", "jaw_pressed", "jaw_loss2", "lin_L3"):
        cr = R.compare(An[("jaw_L3", "body")], An[(nm, "body")], model_label=nm,
                       out=R.OUT / "validation" / f"cmp_jaw_L3_vs_{nm}", plots=(nm in ("jaw_gap", "jaw_2L3")))
        say(f"  {nm}: loss {cr['loss']:.1f}")
        for it in cr["items"][:4]:
            say(f"     [{it['group']:7s} {it['severity']:5.1f}] {it['text']}")
    tw = R.compare(An[("jaw_L3", "body")], An[("jaw_L3", "body")], out=False)
    say(f"  (self-comparison loss {tw['loss']:.2f})")
    # descriptors must separate grazing / gap / pressed
    g = {k: An[(k, "body")]["jawari"] for k in ("jaw_L3", "jaw_gap", "jaw_pressed", "lin_L3")}
    say("\njawari descriptors (body): grazing vs gap (+0.1 mm) vs pressed (-0.1 mm) vs linear")
    for k, j in g.items():
        say(f"  {k:12s} BI{j['BI_main']} start {j['BI_start']:6.1f} max {j['BI_max']:6.1f} bloom {j['bloom_gain_db']:+5.1f} dB "
            f"peak {j['t_bloom_peak']:.2f} s persistence {j['persistence_s']:.2f} s; mean BI 0.1-1 s "
            f"{np.nanmean(j['BI_smooth'][(j['t'] > 0.1) & (j['t'] < 1)]):6.1f}, 3-5 s "
            f"{np.nanmean(j['BI_smooth'][(j['t'] > 3) & (j['t'] < 5)]):6.1f}")
    plot_b(An, res)
    say(f"\nB: {sum(allok)}/{len(allok)} checks pass")
    flush("roundtrip")
    return allok, An, res


def _f2(v):
    return "nan" if v is None else f"{v:.1f}"


def plot_b(An, res):
    plt, seq, div = R._mpl()
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    cols = {"jaw_L3": R.C_REF, "jaw_gap": R.C_MOD, "jaw_pressed": "#1baf7a", "lin_L3": R.INK2}
    for k, c in cols.items():
        A = An[(k, "body")]
        j = A["jawari"]
        axs[0].plot(A["t"], j["BI_smooth"], color=c, label=k)
        axs[1].plot(A["t"], j["octaves"][4], color=c, label=k)
        axs[2].semilogy(A["n"], A["decay"]["T60_late"], ".", color=c, ms=3, label=k)
    axs[0].set_title("BI10 (dB), body", color=R.INK)
    axs[1].set_title("partials 16-31 re total (dB), body", color=R.INK)
    axs[1].set_ylim(-80, 0)
    axs[2].set_title("late T60 (s) vs partial", color=R.INK)
    axs[2].set_xscale("log")
    for ax in axs[:2]:
        ax.set_xscale("symlog", linthresh=0.5)
        ax.set_xlabel("s")
        ax.legend(fontsize=7)
    axs[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "roundtrip_jawari.png", dpi=110)
    plt.close(fig)


# ============================================================================================ C. fitting
def part_c(jobs=3, budget=int(os.environ.get("FIT_BUDGET", "36")), dur=5.0):
    say(f"==== C. fit recovery ({time.strftime('%Y-%m-%d %H:%M')}) ====")
    true = dict(target_gap=1.0e-6, delta_ve=4.5e-3 * 1.5, xp_frac=2.0 / 3.0)
    t0 = time.time()
    y, sr = R.render_model("kharaj", dur, body="tanpura_male", **true)
    say(f"pseudo-reference: kharaj, target_gap +1 um, delta_ve 6.75e-3 (x1.5), pluck at 2L/3, body tanpura_male, "
        f"{dur:g} s ({time.time() - t0:.0f} s)")
    r = R.fit(y, sr=sr, string="kharaj", params=["target_gap", "delta_ve", "xp_frac"], budget=budget, jobs=jobs,
              dur=dur, body="tanpura_male", out=R.OUT / "validation" / "fit_pseudo_reference", verbose=False)
    for line in r["log"]:
        say(line)
    b = r["best"]
    say("\nrecovery:")
    say(f"  target_gap {b['target_gap'] * 1e6:+.2f} um (true +1.00, default 0)")
    say(f"  delta_ve   {b['delta_ve']:.2e} (true 6.75e-03, default 4.50e-03): {100 * (b['delta_ve'] / 6.75e-3 - 1):+.0f} %")
    say(f"  xp_frac    {b['xp_frac']:.3f} (true 0.667, default 0.333)")
    if r["mirror"]:
        say(f"  mirror (pluck at 1 - xp): loss {r['mirror']['loss']:.1f} vs best {r['best_loss']:.1f}")
    say(f"  loss: default {r['default_loss']:.1f} -> best {r['best_loss']:.1f}; runtime {r['runtime_s'] / 60:.1f} min wall, "
        f"{r['cpu_s'] / 60:.1f} min of worker time for {r['n_evals']} renders ({jobs} processes)")
    # loss at the true parameters (twin noise floor: tiny perturbation)
    tw = dict(true)
    tw["delta_ve"] *= 1 + 1e-9
    yt, _ = R.render_model("kharaj", dur, body="tanpura_male", **tw)
    cr = R.compare(R.analyse(y, sr, string="kharaj", settings=dict(max_dur=dur)), yt, model_sr=sr,
                   string="kharaj", out=False)
    say(f"  loss at the true parameters (x(1+1e-9) twin: chaotic-contact noise floor): {cr['loss']:.1f}")
    ok = [check("xp recovered (side of the string)", abs(b["xp_frac"] - 2 / 3) < 0.05, f"{b['xp_frac']:.3f}"),
          check("target_gap recovered", abs(b["target_gap"] - 1e-6) < 0.5e-6, f"{b['target_gap'] * 1e6:+.2f} um"),
          check("delta_ve recovered", abs(np.log(b["delta_ve"] / 6.75e-3)) < np.log(1.25), f"{b['delta_ve']:.2e}")]
    flush("fit")
    return ok


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = int(os.environ.get("JOBS", "3"))
    if what in ("synthetic", "all"):
        part_a()
    if what in ("roundtrip", "all"):
        part_b(jobs)
    if what in ("fit", "all"):
        part_c(jobs)
