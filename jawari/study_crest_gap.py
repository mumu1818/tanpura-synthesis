"""Cross-check of the crest-gap sensitivity with the independent FD model (matched loss), xc = 5 mm."""
import numpy as np
from jlib import simulate_sig, to_audio, analyze
from fdstring import simulate_fd
sig = lambda s: 0.6 + 5e-4 * s["beta"] ** 2
def mb(y, fs):
    m = analyze(to_audio(y, fs), 44100); sel = (m["t"] > 0.3) & (m["t"] < 2.8)
    return np.mean(m["buzz_db"][sel])
for h0 in (0.0, -1e-6, -3e-6):
    for amp in (0.05, 0.2, 0.8):
        p, s, r = simulate_sig(dict(dur=3.0, h0=h0, Fe_amp=amp), sig_fn=sig)
        pf, g, rf = simulate_fd(dict(dur=3.0, h0=h0, Fe_amp=amp, h=1e-3, ovs=20))
        print(f"h0={h0*1e6:+.0f} um pluck {amp:.2f} N: mean buzz modal {mb(r['Fnut'], p['fs']):7.1f} dB | FD(1 mm) {mb(rf['Fnut'], g['fs']):7.1f} dB", flush=True)
