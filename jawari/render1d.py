"""Render the cases to 44.1 kHz WAV (bridge force, DC-blocked, peak-normalised) + metrics + spectrograms.

    python render.py               # all cases
    python render.py nominal xc8   # selected cases
Results are appended to render_results.txt.
"""
import sys
import time

import numpy as np
import soundfile as sf

from common import HERE, analyze, report, spectrogram_png, to_audio, mean_metric
from tanpura1d import simulate

CASES = {
    # name: (overrides, duration, description)
    "linear":      (dict(bridge=False, thread=False), 10.0, "linear string: no bridge, no thread"),
    "bridge_only": (dict(thread=False), 10.0, "curved bridge, no thread"),
    "nominal":     (dict(), 10.0, "bridge + thread at x_c = 5 mm (nominal, DAFx-16/17 C3 values)"),
    "xc4":         (dict(xc=0.004), 10.0, "bridge + thread at x_c = 4 mm"),
    "xc6p5":       (dict(xc=0.0065), 10.0, "bridge + thread at x_c = 6.5 mm"),
    "xc8":         (dict(xc=0.008), 10.0, "bridge + thread at x_c = 8 mm"),
    "no_stiffness": (dict(EI_scale=1e-6), 10.0, "nominal geometry, string bending stiffness removed (EI x 1e-6)"),
}


def sweep_trace(m, t0=0.25, t1=None, step=0.5):
    t1 = t1 or m["dur"] - 0.1
    out = []
    for a in np.arange(t0, t1, step):
        sel = (m["t"] >= a) & (m["t"] < a + step)
        if np.any(sel):
            out.append((a + step / 2, float(np.median(m["peak_hz"][sel])), float(np.mean(m["buzz_db"][sel])),
                        float(np.mean(m["frame_db"][sel]))))
    return out


def render(name, over, dur, desc, fh):
    p, s, r = simulate(dict(dict(dur=dur), **over))
    cpu_per_s = r["cpu"] / dur
    y = to_audio(r["Fbridge"], p["fs"])
    y = 0.9 * y / np.max(np.abs(y))
    wav = f"{HERE}/{name}.wav"
    sf.write(wav, y, 44100, subtype="PCM_24")
    m = analyze(y, 44100)
    png = f"{HERE}/{name}.png"
    spectrogram_png(m, png, title=f"{name}: {desc} (bridge force)")
    mn = analyze(to_audio(r["Fnut"], p["fs"]), 44100)
    st = r["stats"]
    lines = [f"\n##### {name} -- {desc}",
             f"fs_int={p['fs']:.0f} Hz, M={s['M']} modes (f_M={s['om'][-1] / 2 / np.pi:.0f} Hz), bridge points={s['Nb']}, "
             f"cpu {r['cpu']:.1f} s for {dur:.0f} s audio ({cpu_per_s:.2f} s per s)",
             f"contact in {100 * st['contact_frac']:.1f}% of samples, max penetration {st['max_pen'] * 1e9:.0f} nm, "
             f"Newton mean {st['newton_mean']:.1f} it (max {st['newton_max']}, failures {st['newton_fail']}), "
             f"max |y(L/2)| {np.abs(r['y']).max() * 1e3:.2f} mm",
             report(m, f"{name} / bridge force (WAV)"),
             report(mn, f"{name} / nut force"),
             "means: buzz(0.3-3 s) {:.1f} dB, buzz(3-8 s) {:.1f} dB, hf_frac(0.3-3 s) {:.3f}, hf_frac(3-8 s) {:.3f}".format(
                 mean_metric(m, "buzz_db", 0.3, 3.0), mean_metric(m, "buzz_db", 3.0, 8.0),
                 mean_metric(m, "hf_frac", 0.3, 3.0), mean_metric(m, "hf_frac", 3.0, 8.0)),
             "formant trace (t: median peak Hz / mean buzz dB / level dB): " + "  ".join(
                 f"{t:.2f}s:{pk:.0f}/{b:.0f}/{l:.0f}" for t, pk, b, l in sweep_trace(m))]
    txt = "\n".join(lines)
    print(txt, flush=True)
    fh.write(txt + "\n")
    fh.flush()
    return m


if __name__ == "__main__":
    names = sys.argv[1:] or list(CASES)
    with open(f"{HERE}/render_results.txt", "a") as fh:
        fh.write(f"\n==== render run {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
        for n in names:
            if n.startswith("best:"):
                # best:<case>  -> 20 s render of an existing case
                base = n.split(":", 1)[1]
                over, _, desc = CASES[base]
                render(f"best_{base}_20s", over, 20.0, desc + " [20 s]", fh)
            else:
                over, dur, desc = CASES[n]
                render(n, over, dur, desc, fh)
