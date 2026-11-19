"""Confound check on the thread-placement sweep.  The prototype's x_c sweep keeps the bridge crest grazing (h0=0)
for every x_c, i.e. it silently re-sizes the thread.  With a FIXED thread (constant 75 um lift above the local
bridge surface), sliding it by delta along the curved bridge raises the string by 3*delta*(10-delta) um near the
bridge, so the crest gap becomes h0 = -3*delta*(10-delta) um (delta in mm, + toward the crest).
NB: h0>0 (string pressed onto the crest) starts from a non-equilibrium state (small settling transient)."""
import numpy as np
from jlib import simulate_sig, to_audio, analyze
for amp in (0.2, 0.05):
    for d in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.4):
        xc = (5 + d) * 1e-3
        h0 = -3 * d * (10 - d) * 1e-6
        p, s, r = simulate_sig(dict(dur=3.0, xc=xc, h0=h0, Fe_amp=amp))
        m = analyze(to_audio(r["Fnut"], p["fs"]), 44100)
        sel = (m["t"] > 0.3) & (m["t"] < 2.8)
        b05 = m["buzz_db"][int(np.argmin(np.abs(m["t"] - 0.5)))]
        b1 = m["buzz_db"][int(np.argmin(np.abs(m["t"] - 1.0)))]
        print(f"pluck {amp:.2f} N  thread {xc*1e3:.1f} mm (shift {d:+.1f} mm) crest gap h0={h0*1e6:+5.1f} um: "
              f"mean buzz(0.3-2.8s) {np.mean(m['buzz_db'][sel]):6.1f} dB, buzz@0.5s {b05:6.1f}, @1s {b1:6.1f}, "
              f"mean hf {np.mean(m['hf_frac'][sel]):.3f}, contact {100*r['st'][2]/len(r['Fnut']):.0f}% of samples", flush=True)
    # prototype-style (h0=0) for reference
    for xc in (0.0048, 0.0052):
        p, s, r = simulate_sig(dict(dur=3.0, xc=xc, Fe_amp=amp))
        m = analyze(to_audio(r["Fnut"], p["fs"]), 44100); sel = (m["t"] > 0.3) & (m["t"] < 2.8)
        print(f"   [prototype-style h0=0] pluck {amp} N thread {xc*1e3:.1f} mm: mean buzz {np.mean(m['buzz_db'][sel]):6.1f} dB")
