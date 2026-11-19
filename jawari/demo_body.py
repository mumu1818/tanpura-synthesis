"""Demo of the body/radiation model (body.py) on the nominal 1-D jawari render.

    cd jawari && python demo_body.py            # ~40 s (12 s simulation + body + metrics + T1 sweep)

Writes to jawari/out/body/:
    dry.wav                 bridge force (as render1d.py, but loudness-matched to the others)
    body.wav                mono, default male-tanpura body ('tanpura', seed 0, mic 1)
    body_stereo.wav         two mic positions (same body poles, different residues)
    body_stereo_room.wav    + light small room (room = 0.3)
    body_small.wav          mono, 'tanpura_small' (female / smaller instrument), for comparison
    body_deep.wav           mono, 'tanpura' with t1_hz=135, a0_hz=78 (T1 close to the C3 fundamental -> boomy)
    body_light.wav          mono, 'tanpura' with t1_hz=160, a0_hz=92 (fundamental in the A0-T1 dip -> lighter)
    NB 'tanpura_small' is a female instrument (Sa ~ F3-A3); on this C3 test string its fundamental falls below
    T1, into the A0-T1 antiresonance, so it sounds thin -- a pitch mismatch, not a model failure.
    body_ir_tanpura.wav     the stereo body IR used (32-bit float) -- an example input for body.apply_ir
    body_response.png       body magnitude response (both mics), SEA envelope, IR, per-band body T60
    body_response_small.png same for 'tanpura_small'
    ltas.png                long-term average spectra dry vs body
    <name>.png              metrics.spectrogram_png of each WAV
    body_results.txt        everything printed below

All WAVs share ONE gain, chosen so that the loudest file peaks at -1 dBFS. apply_body() RMS-matches its output
to the input over the first 2 s, so the files are loudness-matched (RMS) against each other.
The 12 s simulated force gets a 0.3 s fade-out at its end (the simulation is cut while the string still rings)
and 1.5 s of silence appended, identically for every file, so body and room tails decay naturally.
"""
import os
import time

import numpy as np
import scipy.signal as ss
import soundfile as sf

from common import OUT_ROOT, to_audio
from metrics import analyze, report, spectrogram_png
import body
from tanpura1d import simulate

SR = 44100
OUT = str(OUT_ROOT / "body")
os.makedirs(OUT, exist_ok=True)
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def mono(y):
    return y if y.ndim == 1 else y.mean(axis=1)


def ltas(y, t0=0.0, t1=None):
    """Welch PSD; for stereo the channel PSDs are AVERAGED (a mono sum would cancel the decorrelated
    high-frequency body modes and bias every indicator towards low frequencies)."""
    seg = slice(int(t0 * SR), (int(t1 * SR) if t1 else None))
    if y.ndim == 1:
        return ss.welch(y[seg], SR, nperseg=8192)
    f, P = ss.welch(y[seg], SR, nperseg=8192, axis=0)
    return f, P.mean(axis=1)


def lf_share(y, fcut=500.0, t0=0.0, t1=None):
    f, P = ltas(y, t0, t1)
    return float(P[f < fcut].sum() / P.sum())


def centroid(y, t0=0.0, t1=None):
    f, P = ltas(y, t0, t1)
    return float((f * P).sum() / P.sum())


def _edc_t30(x):
    """T30 from the Schroeder curve of x (1-D) or of the summed channel energies (2-D)."""
    e = x ** 2 if x.ndim == 1 else np.sum(x ** 2, axis=1)
    return 2.0 * body.edc_decay(np.sqrt(e), SR, -5.0, -35.0)


def t30(y):
    return _edc_t30(y)


def band_t30(y, centres=(125, 250, 500, 1000, 2000, 4000)):
    out = {}
    for fc in centres:
        sos = ss.butter(3, [fc / np.sqrt(2), fc * np.sqrt(2)], "bandpass", fs=SR, output="sos")
        out[fc] = _edc_t30(ss.sosfilt(sos, y, axis=0))
    return out


def corr(a, b, lo=None, hi=None):
    if lo or hi:
        if lo and hi:
            sos = ss.butter(4, [lo, hi], "bandpass", fs=SR, output="sos")
        elif lo:
            sos = ss.butter(4, lo, "highpass", fs=SR, output="sos")
        else:
            sos = ss.butter(4, hi, "lowpass", fs=SR, output="sos")
        a, b = ss.sosfilt(sos, a), ss.sosfilt(sos, b)
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def main():
    t_start = time.time()
    say(f"==== demo_body.py  {time.strftime('%Y-%m-%d %H:%M:%S')} ====")
    p, s, r = simulate(dict(dur=12.0))
    y = to_audio(r["Fbridge"], p["fs"])
    say(f"simulated nominal 1-D jawari case, 12 s, cpu {r['cpu']:.1f} s; contact {100 * r['stats']['contact_frac']:.1f}% "
        f"of samples, Newton failures {r['stats']['newton_fail']}")
    nf = int(0.3 * SR)
    y[-nf:] *= np.cos(0.5 * np.pi * np.arange(nf) / nf) ** 2
    x = np.concatenate([y, np.zeros(int(1.5 * SR))])

    t0 = time.time()
    sig = {"dry": x.copy()}
    sig["body"], ir_m = body.apply_body(x, SR, "tanpura", stereo=False, return_ir=True)
    sig["body_stereo"], ir_s = body.apply_body(x, SR, "tanpura", stereo=True, return_ir=True)
    sig["body_stereo_room"] = body.apply_body(x, SR, "tanpura", stereo=True, room=0.3)
    sig["body_small"] = body.apply_body(x, SR, "tanpura_small", stereo=False)
    sig["body_deep"] = body.apply_body(x, SR, "tanpura", t1_hz=135.0, a0_hz=78.0)
    sig["body_light"] = body.apply_body(x, SR, "tanpura", t1_hz=160.0, a0_hz=92.0)
    say(f"body processing of {len(sig) - 1} versions: {time.time() - t0:.1f} s")

    # ---- sanity checks -----------------------------------------------------------------------------------
    pk = max(np.max(np.abs(v)) for v in sig.values())
    gain = 10 ** (-1 / 20) / pk
    onset = int(np.argmax(np.abs(x) > 1e-4 * np.abs(x).max()))
    say(f"\ncommon write gain {20 * np.log10(gain):+.1f} dB (loudest file -> -1 dBFS); input onset at sample {onset}")
    say(f"{'file':18s} {'peak dBFS':>9s} {'RMS0-2s dBFS':>12s} {'DC (mean)':>10s} {'finite':>6s} "
        f"{'pre-onset dB re peak':>20s} {'min |x|>0':>9s}")
    for k, v in sig.items():
        w = v * gain
        pre = np.max(np.abs(w[:onset])) if onset > 0 else 0.0
        nz = np.abs(w[w != 0])
        say(f"{k:18s} {20 * np.log10(np.max(np.abs(w))):9.2f} {20 * np.log10(np.sqrt(np.mean(w[:2 * SR] ** 2))):12.2f} "
            f"{np.mean(w):10.1e} {str(bool(np.all(np.isfinite(w)))):>6s} "
            f"{20 * np.log10(pre / np.max(np.abs(w)) + 1e-300):20.0f} {nz.min():9.1e}")
        assert np.all(np.isfinite(w)) and np.max(np.abs(w)) <= 10 ** (-1 / 20) + 1e-9
    for k, v in sig.items():
        sf.write(f"{OUT}/{k}.wav", v * gain, SR, subtype="PCM_24")
    sf.write(f"{OUT}/body_ir_tanpura.wav", ir_s / np.max(np.abs(ir_s)), SR, subtype="FLOAT")

    # round trip: apply_ir with the saved IR reproduces body_stereo
    ir_read, _ = sf.read(f"{OUT}/body_ir_tanpura.wav")
    y_ir = body.apply_ir(x, ir_read, SR, prepare=False)
    err = np.max(np.abs(y_ir - sig["body_stereo"])) / np.max(np.abs(sig["body_stereo"]))
    say(f"apply_ir(saved body_ir_tanpura.wav) vs apply_body(stereo): max abs difference {20 * np.log10(err + 1e-300):.0f} dB re peak")

    # ---- metrics -----------------------------------------------------------------------------------------
    say("\n---- objective 'depth' indicators (stereo: channel-averaged spectra / energies) ----")
    say(f"{'file':18s} {'LF<500Hz 0-12s':>14s} {'LF<500 0-2s':>11s} {'centroid':>9s} {'centroid0.3-3s':>14s} "
        f"{'T30 bb (s)':>10s}  T30 per octave band (s) 125/250/500/1k/2k/4k")
    rows = {}
    for k, v in sig.items():
        m = analyze(v, SR)
        bt = band_t30(v)
        rows[k] = dict(lf=lf_share(v, t1=12.0), lf2=lf_share(v, t1=2.0), cen=centroid(v, t1=12.0),
                       cen3=centroid(v, 0.3, 3.0), t30=t30(v), bt=bt, m=m)
        q = rows[k]
        say(f"{k:18s} {100 * q['lf']:13.1f}% {100 * q['lf2']:10.1f}% {q['cen']:8.0f}Hz {q['cen3']:13.0f}Hz "
            f"{q['t30']:10.2f}  " + " / ".join(f"{bt[c]:.2f}" for c in bt))
    sel = (rows["dry"]["m"]["t"] >= 0.3) & (rows["dry"]["m"]["t"] <= 3.0)
    say("(metrics.analyze / report / buzz / hf_frac use the mono mean of stereo files, as in metrics.py)")
    say("mean buzz dB (0.3-3 s): " + ", ".join(f"{k} {np.mean(q['m']['buzz_db'][sel]):.1f}" for k, q in rows.items()))
    say("mean hf_frac (0.3-3 s): " + ", ".join(f"{k} {np.mean(q['m']['hf_frac'][sel]):.3f}" for k, q in rows.items()))
    for k in ("body_stereo", "body_stereo_room"):
        v = sig[k]
        say(f"{k}: L/R correlation broadband {corr(v[:, 0], v[:, 1]):.2f}, <300 Hz {corr(v[:, 0], v[:, 1], hi=300):.2f}, "
            f">2 kHz {corr(v[:, 0], v[:, 1], lo=2000):.2f}")
    # partial-by-partial level change (first 2 s) caused by the mono bodies
    f, Pd = ss.welch(sig["dry"][:2 * SR], SR, nperseg=32768)
    f1 = p["f1"]
    say("level change of partials 1..12 vs dry (RMS-matched, first 2 s), dB:")
    for k in ("body", "body_small", "body_deep", "body_light"):
        _, Pb = ss.welch(mono(sig[k])[:2 * SR], SR, nperseg=32768)
        ch = []
        for kk in range(1, 13):
            i = np.argmin(np.abs(f - kk * f1))
            ch.append(10 * np.log10(Pb[i - 2:i + 3].max() / Pd[i - 2:i + 3].max()))
        say(f"  {k:12s} " + " ".join(f"{c:+5.1f}" for c in ch))

    # sensitivity of the depth indicators to the tabli breathing-mode frequency T1 (A0 fixed at 88 Hz)
    say("\n---- sensitivity to T1 (male body, A0 = 85 Hz, mono, mean over seeds 0-3; dry: "
        f"LF {100 * rows['dry']['lf']:.1f}%, centroid0.3-3s {rows['dry']['cen3']:.0f} Hz) ----")
    say(f"{'t1_hz':>6s} {'LF<500 0-12s':>12s} {'centroid0.3-3s':>14s} {'buzz0.3-3s':>10s} {'P1 dB':>6s} {'P2 dB':>6s}")
    for t1 in range(120, 205, 5):
        acc = []
        for sd in range(4):
            v = body.apply_body(x, SR, "tanpura", seed=sd, t1_hz=float(t1))
            mm = analyze(v, SR)
            _, Pb = ss.welch(v[:2 * SR], SR, nperseg=32768)
            pp = [10 * np.log10(Pb[np.argmin(np.abs(f - kk * f1)) - 2: np.argmin(np.abs(f - kk * f1)) + 3].max()
                                / Pd[np.argmin(np.abs(f - kk * f1)) - 2: np.argmin(np.abs(f - kk * f1)) + 3].max())
                  for kk in (1, 2)]
            acc.append((lf_share(v, t1=12.0), centroid(v, 0.3, 3.0), np.mean(mm["buzz_db"][sel]), pp[0], pp[1]))
        a = np.mean(acc, axis=0)
        say(f"{t1:6d} {100 * a[0]:11.1f}% {a[1]:13.0f}Hz {a[2]:10.1f} {a[3]:+6.1f} {a[4]:+6.1f}")

    say("\n---- metrics.report ----")
    for k, v in sig.items():
        say(report(rows[k]["m"], k))
        spectrogram_png(rows[k]["m"], f"{OUT}/{k}.png", title=f"{k} (nominal 1-D jawari, body.py)")

    # ---- figures -----------------------------------------------------------------------------------------
    hb, md = body.plot_body(f"{OUT}/body_response.png", "tanpura", SR, seed=0, stereo=True)
    body.plot_body(f"{OUT}/body_response_small.png", "tanpura_small", SR, seed=0, stereo=True)
    d = md["derived"]
    say("\n---- body 'tanpura' (seed 0) ----")
    say(f"modes: {md['f'].size} total = " + ", ".join(f"{fam} {int((md['family'] == fam).sum())}"
                                                       for fam in ("explicit", "plate", "shell", "cav")))
    say(f"tabli coincidence (along/cross grain) {d['fcL']:.0f}/{d['fcR']:.0f} Hz, plate impedance {d['Zp']:.1f} N s/m, "
        f"bridge-mass corner {d['f_mass']:.0f} Hz, gourd ring frequency {d['f_ring']:.0f} Hz, cavity V {1e3 * d['V']:.1f} L, "
        f"first cavity mode {d['f_cav1']:.0f} Hz, plate modal density {d['n_plate']:.4f} /Hz")
    say("body T60 per octave band (ms): " + ", ".join(f"{c}: {1e3 * v:.0f}" for c, v in body.band_t60(hb[:, 0], SR).items()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor="#fcfcfb")
    for k, col in (("dry", "#8a8984"), ("body", "#2a78d6"), ("body_deep", "#6250d6"), ("body_light", "#eb6834"),
                   ("body_small", "#1baf7a")):
        ff, P = ltas(sig[k], 0.0, 12.0)
        sm = body.frac_oct_smooth(ff[1:], P[1:], 6)
        ax.semilogx(ff[1:], 10 * np.log10(sm / Pd.max() + 1e-20), color=col, lw=2, label=k)
    ax.set_xlim(40, 20000); ax.set_ylim(-110, 0)
    ax.set_xlabel("Hz"); ax.set_ylabel("dB (1/6-oct smoothed LTAS, 0-12 s)")
    ax.set_title("long-term average spectrum (RMS-matched): dry bridge force vs body variants",
                 fontsize=10)
    ax.grid(alpha=0.25, which="both"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/ltas.png", dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)
    say(f"\nwrote {OUT}/  (total {time.time() - t_start:.1f} s)")
    with open(f"{OUT}/body_results.txt", "w") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
