"""The user's C#2 kharaj with a finger pluck: 12 s renders, spectrograms, jawari metrics and model checks.

    cd jawari && python study_kharaj.py [--jobs 2] [case ...]     (~6 min with 2 processes)
    -> out/kharaj/<case>.wav (bridge force, 44.1 kHz, peak -1 dBFS), <case>.png (spectrogram + metrics),
       study_kharaj.txt (all numbers), study_kharaj.json, decay_compare.png

Cases (presets.py; 'kharaj' = phosphor bronze, the user's material):
  nominal        statics geometry, thread auto-placed to graze, tuned, finger pluck at L/3, Valette+body losses
  pulse          same, but the old symmetric whole-string pulse (10 ms rise / 1 ms fall) instead of the finger pluck
  thread_m0p3 .. thread_p0p3   thread moved -0.3/-0.1/+0.1/+0.3 mm (+ = towards the nut) from grazing, statics re-solved
  steel          'kharaj_steel' (steel string, its own designed bridge), nominal
  linear         no bridge, no thread (linear string reference, f1 = 69.4 Hz by construction)
  loss_woodhouse nominal with the old Woodhouse loss defaults (damping sensitivity)
  loss_bareplate nominal with the bare 4 mm-tabli body drain of body.py (Yb 0.0227 s/kg, bridge 2.5 g)
  tuning_check   thread spring on, bridge contact off: checks tune='speaking' (6 s, no WAV)
  conv_*         convergence checks: k_b x 10 (nominal and thread +0.1 mm), modes up to 40 kHz
Also partials.png: per-partial envelopes (bridge force) of nominal / pulse / linear.
Optional extra: nominal_body.wav = nominal through body.apply_body (another agent's one-way body filter), if importable.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import scipy.signal as ss
import soundfile as sf

from common import OUT_ROOT, analyze, report, spectrogram_png, to_audio, mean_metric
import presets as P
import tanpura1d as T1

OUT = str(OUT_ROOT / "kharaj")
os.makedirs(OUT, exist_ok=True)
DUR = 12.0
F1 = 69.4

CASES = {
    "nominal": ("kharaj", dict(), "kharaj (bronze) nominal: finger pluck at L/3, thread grazing, tuned"),
    "pulse": ("kharaj", dict(pluck=None), "kharaj (bronze), old whole-string pulse instead of the finger pluck"),
    "thread_m0p3": ("kharaj", dict(thread_shift=-0.3e-3), "kharaj, thread moved 0.3 mm towards the back"),
    "thread_m0p1": ("kharaj", dict(thread_shift=-0.1e-3), "kharaj, thread moved 0.1 mm towards the back"),
    "thread_p0p1": ("kharaj", dict(thread_shift=+0.1e-3), "kharaj, thread moved 0.1 mm towards the nut"),
    "thread_p0p3": ("kharaj", dict(thread_shift=+0.3e-3), "kharaj, thread moved 0.3 mm towards the nut"),
    "steel": ("kharaj_steel", dict(), "kharaj_steel nominal (steel string, its own designed bridge)"),
    "linear": ("kharaj", dict(bridge=False, thread=False, T=None, static_init=False),
               "linear string reference: no bridge, no thread"),
    "loss_woodhouse": ("kharaj", dict(loss=None), "kharaj nominal with the old Woodhouse loss defaults"),
    "loss_bareplate": ("kharaj", dict(Yb=0.0227, mb=2.5e-3), "kharaj nominal, body drain of body.py's bare 4 mm tabli"),
    "tuning_check": ("kharaj", dict(bridge=False, dur=6.0), "tuning check: thread spring on, no bridge contact"),
    "conv_kb1e12": ("kharaj", dict(kb=1e12), "convergence: nominal with 10x stiffer bridge penalty (k_b = 1e12)"),
    "conv_modes40k": ("kharaj", dict(fmax=40000.0), "convergence: nominal with all modes below 40 kHz"),
    "conv_p0p1_kb1e12": ("kharaj", dict(thread_shift=+0.1e-3, kb=1e12), "convergence: thread +0.1 mm, k_b = 1e12"),
}


def measure_f(x, fs, f0, t0, t1, bw=8.0):
    """Mean frequency (Hz) of the partial near f0 in [t0, t1]: complex demodulation at f0, zero-phase low-pass
    (bw Hz), linear fit of the unwrapped phase."""
    n = np.arange(len(x))
    z = x * np.exp(-2j * np.pi * f0 * n / fs)
    sos = ss.butter(4, bw, fs=fs, output="sos")
    z = ss.sosfiltfilt(sos, z.real) + 1j * ss.sosfiltfilt(sos, z.imag)
    i0, i1 = int(t0 * fs), int(t1 * fs)
    ph = np.unwrap(np.angle(z[i0:i1]))
    t = n[i0:i1] / fs
    slope = np.polyfit(t, ph, 1)[0]
    return f0 + slope / (2 * np.pi)


def cents(f, fref=F1):
    return 1200 * np.log2(f / fref)


def run_case(name):
    pre, over, desc = CASES[name]
    over = dict(over)
    dur = over.pop("dur", DUR)
    t0 = time.time()
    p = P.preset(pre, dur=dur, **over)
    t_statics = time.time() - t0
    meta = p.pop("meta")
    p, s, r = T1.simulate(p)
    st = r["stats"]
    fs = p["fs"]
    res = dict(name=name, preset=pre, desc=desc, dur=dur, t_statics=t_statics, cpu=r["cpu"], cpu_per_s=r["cpu"] / dur,
               M=int(s["M"]), f_M=float(s["om"][-1] / 2 / np.pi), nyquist=0.5 * fs, Nb=int(s["Nb"]), T=float(s["T"]),
               T_statics=float(meta["T_statics"]), L=float(p["L"]), xc_rel=float(p["xc"]), x_c=meta["x_c"],
               x_c_graze=meta["x_c_graze"], graze_roots=meta["graze_roots"], crest_gap_um=meta["crest_gap"] * 1e6,
               x_graze=meta["x_graze"], psi_deg=meta["psi_deg"], F_thread_preload=meta["F_thread_preload"],
               material=meta["label"], l_mm=meta["l"] * 1e3,
               max_pen_nm=float(st["max_pen"]) * 1e9, edge_pen_nm=float(st["edge_pen"]) * 1e9,
               contact_frac=float(st["contact_frac"]), newton_mean=float(st["newton_mean"]),
               newton_max=int(st["newton_max"]), newton_fail=int(st["newton_fail"]), max_res=float(st["max_res"]),
               max_Fc=float(st["max_Fc"]), max_active=int(st["max_active"]))
    if "static" in st:
        si = st["static"]
        res.update(static_force=float(si["static_force"]), static_thread_force=float(si["thread_force"]),
                   static_n_active=int(si["n_active"]), static_pen_nm=float(si["max_pen"]) * 1e9,
                   static_iters=int(si["iters"]))
    # bilateral thread: the thread model is a bilateral spring around the static preload; valid while the dynamic
    # change of the thread force never exceeds the preload (the string never lifts off the thread)
    res["thread"] = bool(p["thread"])
    res["bilateral_ok"] = res["thread"] and res["max_Fc"] < meta["F_thread_preload"]
    # excitation and displacement at the output point (= pluck point for the finger pluck)
    Fe = T1.excitation(p, s)
    n_rel = len(Fe) - 1
    res.update(Fe_amp=float(Fe.max()), x_out=float(p["x_out"]), y_peak_mm=float(np.abs(r["y"]).max() * 1e3),
               y_peak_pull_mm=float(np.abs(r["y"][:n_rel]).max() * 1e3),
               y_peak_free_mm=float(np.abs(r["y"][n_rel:]).max() * 1e3), static_compliance=T1.static_compliance(s))
    # share of the initial (static) modal energy in the even modes: 0 for the symmetric whole-string pulse
    qs = s["ge"] / (s["m"] * s["om"] ** 2)
    Es = s["om"] ** 2 * qs ** 2
    res["even_mode_share"] = float(Es[1::2].sum() / Es.sum())
    # T60 of the linear modes
    f = s["om"] / 2 / np.pi
    res["T60"] = {str(ff): float(6.9078 / np.interp(ff, f, s["sig"])) for ff in (69.4, 139, 278, 500, 1000, 2000, 4000, 8000)}
    # achieved fundamental (nut force: cleaner low partials than the bridge force)
    xn = r["Fnut"]
    wins = [(0.5, 3.0), (3.0, 6.0)] + ([(6.0, 11.5)] if dur > 11.9 else [])
    res["f1"] = {f"{a:g}-{b:g}s": float(measure_f(xn, fs, F1, a, b)) for a, b in wins}
    res["f1_cents"] = {k: float(cents(v)) for k, v in res["f1"].items()}
    res["f1_linear_theory"] = float(T1.lowest_linear_freq(s["T"], s["mu"], s["EI"], s["L"], s["beta"], s["gc"], s["kc"],
                                                           s["M"], s["dt"]))
    if name == "tuning_check":
        return res, None
    y = to_audio(r["Fbridge"], fs)
    y = 10 ** (-1 / 20) * y / np.max(np.abs(y))
    sf.write(f"{OUT}/{name}.wav", y, 44100, subtype="PCM_24")
    m = analyze(y, 44100)
    mn = analyze(to_audio(r["Fnut"], fs), 44100)
    spectrogram_png(m, f"{OUT}/{name}.png", title=f"{name}: {desc} (bridge force)")
    res["report"] = report(m, f"{name} / bridge force (WAV)")
    res["report_nut"] = report(mn, f"{name} / nut force")
    tt = m["t"]
    res["level_db"] = {str(t): float(m["frame_db"][int(np.argmin(np.abs(tt - t)))]) for t in (1, 2, 4, 6, 8, 11)}
    res["buzz_mean"] = {"0.3-3s": mean_metric(m, "buzz_db", 0.3, 3.0), "3-8s": mean_metric(m, "buzz_db", 3.0, 8.0),
                        "8-11.5s": mean_metric(m, "buzz_db", 8.0, 11.5)}
    res["buzz_mean_nut"] = {"0.3-3s": mean_metric(mn, "buzz_db", 0.3, 3.0), "3-8s": mean_metric(mn, "buzz_db", 3.0, 8.0)}
    res["hf_frac_mean"] = {"0.3-3s": mean_metric(m, "hf_frac", 0.3, 3.0), "3-8s": mean_metric(m, "hf_frac", 3.0, 8.0)}
    # level decay of the whole tone: time to -20 / -40 / -60 dB (frame level, smoothed)
    lev = ss.medfilt(m["frame_db"], 9)
    res["t_to_db"] = {str(d): (float(tt[np.argmax(lev < d)]) if np.any(lev < d) else None) for d in (-20, -40, -60)}
    np.savez_compressed(f"{OUT}/{name}_traces.npz", bridge_44k=y.astype(np.float32),
                        t=tt, buzz_db=m["buzz_db"], frame_db=m["frame_db"], peak_hz=m["peak_hz"],
                        B=np.pi ** 2 * s["EI"] / (s["T"] * s["L"] ** 2), f1=res["f1"][f"{wins[0][0]:g}-{wins[0][1]:g}s"])
    track = dict(t=tt.tolist(), frame_db=m["frame_db"].tolist(), buzz_db=m["buzz_db"].tolist())
    return res, track


def body_extra():
    """nominal through body.apply_body (one-way body/radiation filter by another agent), if available."""
    try:
        import body
    except Exception as e:  # noqa: BLE001
        return f"body.apply_body not available ({e!r}); no nominal_body.wav"
    y, sr = sf.read(f"{OUT}/nominal.wav")
    a = body.apply_body(y, sr, tail=1.0)
    a = 10 ** (-1 / 20) * a / np.max(np.abs(a))
    sf.write(f"{OUT}/nominal_body.wav", a, sr, subtype="PCM_24")
    return "wrote nominal_body.wav (nominal bridge force through body.apply_body(preset='tanpura'), peak -1 dBFS)"


def fmt(res):
    L = [f"\n##### {res['name']} -- {res['desc']}",
         f"string {res['material']}, l = sqrt(EI/T) = {res['l_mm']:.2f} mm, psi = {res['psi_deg']:.4f} deg; "
         f"T = {res['T']:.4f} N (statics at {res['T_statics']:.4f} N), L = {res['L'] * 1e3:.3f} mm",
         f"thread x_c = {res['x_c'] * 1e3:.4f} mm (grazing at {res['x_c_graze'] * 1e3:.4f} mm; all grazing roots "
         f"{[round(x * 1e3, 4) for x in res['graze_roots']]}), crest gap {res['crest_gap_um']:+.2f} um "
         f"(- = pressed), graze at x = {res['x_graze'] * 1e3:.2f} mm, thread preload {res['F_thread_preload']:.3f} N",
         f"modes M = {res['M']} (f_M = {res['f_M']:.0f} Hz < 20 kHz; physics Nyquist {res['nyquist']:.0f} Hz), "
         f"bridge points {res['Nb']}, cpu {res['cpu']:.1f} s for {res['dur']:.0f} s ({res['cpu_per_s']:.2f} s per s), "
         f"statics+tuning {res['t_statics']:.1f} s",
         f"excitation amplitude {res['Fe_amp']:.4f} N, even-mode share of the initial modal energy "
         f"{100 * res['even_mode_share']:.1f}%; |y| at x_out = {res['x_out'] * 1e3:.1f} mm: peak "
         f"{res['y_peak_mm']:.3f} mm (during pull {res['y_peak_pull_mm']:.3f}, after release {res['y_peak_free_mm']:.3f})",
         f"contact {100 * res['contact_frac']:.1f}% of samples, max penetration {res['max_pen_nm']:.1f} nm "
         f"(edge points {res['edge_pen_nm']:.1f} nm), Newton mean {res['newton_mean']:.2f} it, max {res['newton_max']}, "
         f"failures {res['newton_fail']}, max residual {res['max_res']:.1e}, max active points {res['max_active']}",
         f"thread: max |dynamic force change| {res['max_Fc']:.3f} N vs static preload {res['F_thread_preload']:.3f} N "
         f"-> bilateral assumption {('HOLDS' if res['bilateral_ok'] else 'VIOLATED') if res['thread'] else 'n/a (no thread)'}"]
    if "static_force" in res:
        L.append(f"static init: bridge force {res['static_force'] * 1e3:.3f} mN on {res['static_n_active']} points, "
                 f"thread force change {res['static_thread_force'] * 1e3:.3f} mN, penetration {res['static_pen_nm']:.2f} nm, "
                 f"{res['static_iters']} Newton its")
    L.append("f1 (nut force, demodulation): " + ", ".join(
        f"{k}: {v:.4f} Hz ({res['f1_cents'][k]:+.3f} cents)" for k, v in res["f1"].items())
        + f"; linear theory (discrete scheme + thread, no contact) {res['f1_linear_theory']:.5f} Hz "
          f"({cents(res['f1_linear_theory']):+.4f} cents)")
    L.append("modal T60 [s] at " + ", ".join(f"{k} Hz: {v:.2f}" for k, v in res["T60"].items()))
    if "report" in res:
        L += [res["report"], res["report_nut"],
              "level re peak [dB] at " + ", ".join(f"{k} s: {v:.1f}" for k, v in res["level_db"].items()),
              "time to -20/-40/-60 dB: " + ", ".join(f"{v:.2f} s" if v is not None else "> dur" for v in res["t_to_db"].values()),
              "mean buzz (bridge) " + ", ".join(f"{k} {v:.1f} dB" for k, v in res["buzz_mean"].items())
              + " | (nut) " + ", ".join(f"{k} {v:.1f} dB" for k, v in res["buzz_mean_nut"].items()),
              "mean hf_frac " + ", ".join(f"{k} {v:.3f}" for k, v in res["hf_frac_mean"].items())]
    return "\n".join(L)


def partials_png(names, path, nmax=80):
    """Per-partial envelopes (dB re max) of the bridge force: the jawari spreads energy to high partials."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jlib import partial_envelope
    names = [n for n in names if os.path.exists(f"{OUT}/{n}_traces.npz")]
    fig, ax = plt.subplots(1, len(names), figsize=(5 * len(names), 5), sharey=True, squeeze=False)
    for k, nm in enumerate(names):
        d = np.load(f"{OUT}/{nm}_traces.npz")
        t, A = partial_envelope(d["bridge_44k"].astype(float), 44100, float(d["f1"]), nmax=nmax, inharm_B=float(d["B"]))
        im = ax[0, k].pcolormesh(t, np.arange(1, nmax + 1), A - A.max(), vmin=-80, vmax=0, shading="auto",
                                 cmap="magma")
        ax[0, k].set_title(f"{nm}: partial envelopes (bridge force)"); ax[0, k].set_xlabel("s")
    ax[0, 0].set_ylabel("partial number (peak search near n f1)")
    fig.colorbar(im, ax=ax[0, -1], label="dB re max")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def decay_png(tracks, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for nm in ("nominal", "loss_woodhouse", "loss_bareplate", "linear", "pulse", "steel"):
        if nm in tracks and tracks[nm] is not None:
            tr = tracks[nm]
            ax[0].plot(tr["t"], ss.medfilt(np.array(tr["frame_db"]), 9), label=nm)
            ax[1].plot(tr["t"], ss.medfilt(np.array(tr["buzz_db"]), 9), label=nm)
    ax[0].set_ylabel("frame level re peak [dB]"); ax[0].set_ylim(-110, 3); ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].set_ylabel("buzz [dB] (2-8 kHz vs <1 kHz)"); ax[1].set_xlabel("s"); ax[1].grid(alpha=.3)
    ax[0].set_title("kharaj: decay of the bridge force and buzz (median-smoothed)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:]
    jobs = 2
    if "--jobs" in args:
        i = args.index("--jobs"); jobs = int(args[i + 1]); del args[i:i + 2]
    names = args or list(CASES)
    t0 = time.time()
    if jobs > 1:
        with get_context("forkserver").Pool(jobs) as pool:
            out = pool.map(run_case, names, chunksize=1)
    else:
        out = [run_case(n) for n in names]
    results = {r_["name"]: r_ for r_, _ in out}
    tracks = {r_["name"]: tr for r_, tr in out}
    txt = [f"==== study_kharaj {time.strftime('%Y-%m-%d %H:%M:%S')}, {len(names)} cases, wall {time.time() - t0:.0f} s ===="]
    txt += [fmt(results[n]) for n in names]
    if "nominal" in results:
        txt.append("\n" + body_extra())
        decay_png(tracks, f"{OUT}/decay_compare.png")
        partials_png(["nominal", "pulse", "linear"], f"{OUT}/partials.png")
    s = "\n".join(txt)
    print(s)
    tag = "" if not args else "_" + "_".join(names)
    with open(f"{OUT}/study_kharaj{tag}.txt", "w") as fh:
        fh.write(s + "\n")
    with open(f"{OUT}/study_kharaj{tag}.json", "w") as fh:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ("report", "report_nut")} for k, v in results.items()},
                  fh, indent=1, default=float)
