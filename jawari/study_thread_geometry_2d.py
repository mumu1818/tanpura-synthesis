"""Fixed-thread sweep in 1-D and 2-D (EVALUATION §4c): does the 1-D vs 2-D difference survive a geometrically consistent thread sweep?
A FIXED thread (75 um lift above the local bridge surface) slid by delta along the parabola gives a crest gap
h0 = -3 delta (10 - delta) um (delta in mm; h0>0 = string pressed onto crest; prototype B solves the static
equilibrium for h0>0).  Compare 1-D (full 0.2 N vertical), 1-D with the same vertical force as 2-D (0.2 cos20),
2-D nominal (1 cent) and 2-D 5 cent, at x_c = 4.9, 5.0, 5.1, 5.2 mm.  5 s runs, vertical bridge force F_y."""
import sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
import common  # noqa: F401  (puts this directory on sys.path)
VAR = {"1d": dict(pol2=False, tm=False), "1d_eqF": dict(pol2=False, tm=False, Fe_amp=0.2 * np.cos(np.deg2rad(20))),
       "2d": dict(), "2dd5": dict(detune_cents=5.0)}

def job(args):
    v, d = args
    import tanpura2d as t2
    from common import to_audio
    from metrics import analyze
    xc = (5 + d) * 1e-3; h0 = -3 * d * (10 - d) * 1e-6
    p, s, r = t2.simulate(dict(VAR[v], xc=xc, h0=h0, dur=5.0))
    m = analyze(to_audio(r["Fy"], p["fs"]), 44100)
    t = m["t"]; sel1 = (t > 0.3) & (t < 3); sel2 = (t > 3) & (t < 4.8)
    st = r["stats"]
    return v, d, h0, float(np.mean(m["buzz_db"][sel1])), float(np.mean(m["buzz_db"][sel2])), st["contact_frac"], st["static"]

if __name__ == "__main__":
    cases = [(v, d) for d in (-0.1, 0.0, 0.1, 0.2) for v in VAR]
    res = {}
    with ProcessPoolExecutor(3) as ex:
        for v, d, h0, b1, b2, cf, stt in ex.map(job, cases):
            res[(v, d)] = (b1, b2)
            print(f"{v:7s} x_c={5+d:.1f} mm h0={h0*1e6:+5.1f} um: mean buzz 0.3-3 s {b1:6.1f} dB, 3-4.8 s {b2:6.1f} dB, "
                  f"contact {100*cf:4.0f}%  static={stt}", flush=True)
    print("\nrange over the 0.3 mm consistent sweep (max-min of mean buzz 0.3-3 s):")
    for v in VAR:
        a = [res[(v, d)][0] for d in (-0.1, 0.0, 0.1, 0.2)]
        print(f"  {v:7s} {max(a)-min(a):6.1f} dB   values " + " ".join(f"{x:6.1f}" for x in a))
