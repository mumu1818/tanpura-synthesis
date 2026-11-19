"""Render the thread-placement sweep (1-D vs 2-D variants), friction on/off and noise-floor twins.

    python render2d.py [--jobs 3] [--force] [case ...]      # default: all cases in CASES
    python render2d.py --list
Case names are <variant>_<position>, e.g. 2d_xc5, 1d_xc8, 2dnf_xc5p001 (noise twin at x_c = 5.001 mm).
Each case writes runs/<case>.npz (44.1 kHz bridge/nut forces + metric tracks + stats); "main" cases also write
WAVs and spectrogram PNGs.  A case whose npz exists is skipped unless --force.  Text summaries are appended to
render2d_results.txt.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf

from common2d import HERE, GZ_MONO, analyze, report, spectrogram_png, to_audio, mean_metric
import tanpura2d as t2

RUNS = f"{HERE}/runs"
WAVS = f"{HERE}/wav"
PNGS = f"{HERE}/png"
DUR = 10.0

# variant: (overrides, description)
VARIANTS = {
    "1d":    (dict(pol2=False, tm=False), "1-D: vertical polarisation only, no tension modulation (= tanpura1d.py)"),
    "1dtm":  (dict(pol2=False, tm=True), "1-D + Kirchhoff-Carrier tension modulation (horizontal polarisation off)"),
    "2d":    (dict(), "2-D full: y+z, nut coupling (30 deg, 1 cent), TM, friction mu=0.39 atan(2.84 v), pluck 20 deg"),
    "2dnf":  (dict(friction=False), "2-D, friction off"),
    "2dcf":  (dict(mu_c2=2840.0), "2-D, near-Coulomb friction mu=0.39 atan(2840 v)"),
    "2dnc":  (dict(theta_nut=0.0, detune_cents=0.0), "2-D, no nut coupling (polarisations couple only via friction + TM)"),
    "2dp5":  (dict(pluck_deg=5.0), "2-D, pluck 5 deg from vertical"),
    "2dp45": (dict(pluck_deg=45.0), "2-D, pluck 45 deg from vertical"),
    "2dsz":  (dict(kcz=1.2e3, rcz=0.05), "2-D, laterally soft thread (k_cz = 1.2e3 N/m, r_cz = 0.05 kg/s)"),
    "2dd5":  (dict(detune_cents=5.0), "2-D, stronger nut coupling: 5 cent split of the principal axes (30 deg)"),
    "2dd02": (dict(detune_cents=0.2), "2-D, weaker nut coupling: 0.2 cent split of the principal axes (30 deg)"),
}
POSITIONS = {"xc3": 0.003, "xc4": 0.004, "xc5": 0.005, "xc6p5": 0.0065, "xc8": 0.008}
TWINS = {"xc4p999": 0.004999, "xc5p001": 0.005001, "xc5p002": 0.005002}
MAIN_VARIANTS = ["1d", "1dtm", "2d", "2dnf", "2dcf"]
EXTRA_VARIANTS = ["2dnc", "2dp5", "2dp45", "2dsz", "2dd5", "2dd02"]


def all_cases():
    out = []
    for v in MAIN_VARIANTS:
        out += [f"{v}_{k}" for k in POSITIONS] + [f"{v}_{k}" for k in TWINS]
    for v in EXTRA_VARIANTS:
        out += [f"{v}_{k}" for k in POSITIONS] + [f"{v}_xc5p001"]
    return out


def case_params(name):
    v, pos = name.split("_", 1)
    suffix = ""
    if "@" in pos:
        pos, suffix = pos.split("@", 1); suffix = "@" + suffix
    xc = POSITIONS.get(pos, TWINS.get(pos))
    over, desc = VARIANTS[v]
    over = dict(over, xc=xc, dur=DUR)
    if xc < 0.004:
        over["br_x0"] = xc               # bridge sample region must start at the thread
    extra = ""
    if suffix == "@2x":
        over["fs"] = 352800.0; extra = " ; fs 352.8 kHz"
    elif suffix == "@m40":
        over["fmax"] = 40000.0; extra = " ; modes to 40 kHz"
    elif suffix:
        raise ValueError(suffix)
    is_main = pos in POSITIONS and suffix == ""
    return over, f"{desc}; x_c = {xc * 1e3:.3f} mm" + extra, is_main


def run_case(name):
    over, desc, is_main = case_params(name)
    p, s, r = t2.simulate(over)
    fs = p["fs"]
    y = to_audio(r["Fy"], fs); z = to_audio(r["Fz"], fs)
    ny = to_audio(r["Fny"], fs); nz = to_audio(r["Fnz"], fs)
    mono = y + GZ_MONO * z
    my = analyze(y, 44100)
    mm = analyze(mono, 44100) if p["pol2"] else my
    mz = analyze(z, 44100) if p["pol2"] else None
    st = dict(r["stats"]); st.pop("static", None)
    keep = ("t", "centroid_hz", "hf_frac", "buzz_db", "peak_hz", "frame_db")
    np.savez_compressed(f"{RUNS}/{name}.npz", y=y.astype(np.float32), z=z.astype(np.float32),
                        ny=ny.astype(np.float32), nz=nz.astype(np.float32),
                        t_rec=r["t_rec"], Ey=r["Ey"], Ez=r["Ez"], dT=r["dT"], slip=r["slip"],
                        **{f"my_{k}": my[k] for k in keep}, **{f"mm_{k}": mm[k] for k in keep},
                        stats_keys=np.array(list(st)), stats_vals=np.array([float(v) for v in st.values()]),
                        cpu=r["cpu"], dur=p["dur"], xc=p["xc"], pol2=p["pol2"])
    lines = [f"\n##### {name} -- {desc}",
             f"M={s['M']} modes, bridge points {s['Nb']}, fs_int {fs:.0f} Hz, cpu {r['cpu']:.1f} s for {p['dur']:.0f} s "
             f"({r['cpu'] / p['dur']:.2f} s per s); contact {100 * st['contact_frac']:.1f}% of samples, max pen "
             f"{st['max_pen'] * 1e9:.0f} nm, Newton {st['newton_mean']:.2f}/{st['newton_max']} fail {st['newton_fail']}, "
             f"friction Newton {st['fric_newton_mean']:.2f}/{st['fric_newton_max']} fail {st['fric_fail']}, edge pen "
             f"{st['edge_pen']:.0e}, max slip {st['max_slip'] * 1e3:.1f} mm/s, max mu {st['max_mu']:.3f}, "
             f"friction loss {st['E_friction'] * 1e6:.3f} uJ",
             report(my, f"{name} / vertical bridge force F_y")]
    if p["pol2"]:
        lines += [report(mm, f"{name} / mono mix F_y + {GZ_MONO} F_z"),
                  report(mz, f"{name} / horizontal bridge force F_z"),
                  f"rms F_z / rms F_y = {20 * np.log10(np.std(z) / np.std(y)):.1f} dB"]
    lines.append("means (vertical): buzz 0.5-3 s {:.1f} dB, 3-6 s {:.1f} dB, hf 0.5-3 s {:.3f}; (mono) buzz {:.1f} / {:.1f} dB".format(
        mean_metric(my, "buzz_db", 0.5, 3.0), mean_metric(my, "buzz_db", 3.0, 6.0), mean_metric(my, "hf_frac", 0.5, 3.0),
        mean_metric(mm, "buzz_db", 0.5, 3.0), mean_metric(mm, "buzz_db", 3.0, 6.0)))
    if is_main:
        g = 0.9 / np.max(np.abs(y))
        sf.write(f"{WAVS}/{name}_vertical.wav", (g * y).astype(np.float32), 44100, subtype="PCM_24")
        spectrogram_png(my, f"{PNGS}/{name}_vertical.png", title=f"{name}: vertical bridge force F_y -- {desc}")
        if p["pol2"]:
            gm = 0.9 / np.max(np.abs(mono))
            sf.write(f"{WAVS}/{name}_mono.wav", (gm * mono).astype(np.float32), 44100, subtype="PCM_24")
            gs = 0.9 / max(np.max(np.abs(y)), np.max(np.abs(z)))
            sf.write(f"{WAVS}/{name}_stereo_y_z.wav", (gs * np.stack([y, z], 1)).astype(np.float32), 44100,
                     subtype="PCM_24")
            spectrogram_png(mm, f"{PNGS}/{name}_mono.png", title=f"{name}: mono mix F_y+{GZ_MONO}F_z -- {desc}")
    return name, "\n".join(lines), r["cpu"]


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--list" in args:
        print("\n".join(all_cases())); sys.exit(0)
    jobs = 1
    if "--jobs" in args:
        i = args.index("--jobs"); jobs = int(args[i + 1]); del args[i:i + 2]
    force = "--force" in args
    args = [a for a in args if a != "--force"]
    for d in (RUNS, WAVS, PNGS):
        os.makedirs(d, exist_ok=True)
    names = args or all_cases()
    todo = [n for n in names if force or not os.path.exists(f"{RUNS}/{n}.npz")]
    print(f"{len(todo)} cases to render ({len(names) - len(todo)} already done)", flush=True)
    t0 = time.time()
    with open(f"{HERE}/render2d_results.txt", "a") as fh:
        fh.write(f"\n==== render2d run {time.strftime('%Y-%m-%d %H:%M:%S')} ({len(todo)} cases) ====\n")
        if jobs == 1:
            for n in todo:
                _, txt, cpu = run_case(n)
                print(txt, flush=True); fh.write(txt + "\n"); fh.flush()
        else:
            with ProcessPoolExecutor(jobs) as ex:
                futs = {ex.submit(run_case, n): n for n in todo}
                for f in as_completed(futs):
                    _, txt, cpu = f.result()
                    print(txt, flush=True); fh.write(txt + "\n"); fh.flush()
    print(f"done in {time.time() - t0:.0f} s wall", flush=True)
