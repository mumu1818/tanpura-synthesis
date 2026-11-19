"""Cross-check helpers: run the tanpura1d kernel with custom per-mode damping,
plus common analysis utilities.  Does not touch the prototype directory."""
import sys
import numpy as np
import scipy.signal as ss

from common import analyze, report, spectrogram_png, to_audio  # noqa  (also puts this dir on sys.path)
import tanpura1d as T1  # noqa


def simulate_sig(over=None, sig_fn=None, energy=False):
    """Like tanpura1d.simulate but optionally replaces the modal decay rates by sig_fn(s) (array, 1/s)."""
    p = dict(T1.DEFAULT)
    if over:
        p.update(over)
    s = T1.build(p)
    if sig_fn is not None:
        sig = np.asarray(sig_fn(s), float)
        dt = s["dt"]; om = s["om"]
        omd = np.sqrt(np.maximum(om ** 2 - sig ** 2, 0.0))
        a = 2 * np.exp(-sig * dt) * np.cos(omd * dt)
        b = np.exp(-2 * sig * dt)
        c = dt ** 2 * (1 + b) / (2 * s["m"])
        s.update(sig=sig, a=a, b=b, c=c, om2s=(2 - 2 * a / (1 + b)) / dt ** 2, sigs=(1 - b) / ((1 + b) * dt))
        # thread elimination constants depend on c
        s["thcc"] = float(np.sum(c * s["gc"] * s["gc"]))
        if p["bridge"]:
            Phi = s["Phi"]; dxb = s["dxb"]
            v = Phi @ (c * s["gc"])
            Q = dxb * (Phi * c[None, :]) @ Phi.T
            Qp = Q - s["kap"] * dxb * np.outer(v, v) / (1 + s["kap"] * s["thcc"])
            s["v"] = v; s["Qp"] = np.ascontiguousarray(0.5 * (Qp + Qp.T))
    N = int(round(p["dur"] * p["fs"]))
    Fe = T1.excitation(p)
    # initial state (zeros unless p['static_init']); computed after the damping override because the static
    # fixed point of the scheme uses its omega*^2 (which depends on sigma)
    q_init, _ = T1.initial_state(p, s)
    out = T1._run(N, s["a"], s["b"], s["c"], s["ge"], Fe, s["gc"], s["kap"], s["lam"], s["thcc"], s["kc"],
                  s["rc"], bool(p["bridge"]), s["Phi"], s["hj"], s["dxb"], s["v"], s["Qp"], float(p["kb"]),
                  float(p["alpha"]), s["w0"], s["wn"], s["phio"], float(p["tol"]), int(p["maxit"]),
                  bool(energy), s["m"], s["om2s"], s["sigs"], s["dt"], bool(p["guess"]), q_init)
    Fbr, Fnut, Fbs, Fcs, F0s, yo, H, Dcum, Wcum, pen, st = out
    return p, s, dict(Fbridge=Fbr, Fnut=Fnut, Fb=Fbs, Fc=Fcs, F0=F0s, y=yo, H=H, D=Dcum, W=Wcum, pen=pen, st=st)


def mean_in(m, key, t0, t1):
    sel = (m["t"] >= t0) & (m["t"] <= t1)
    return float(np.mean(m[key][sel]))


def summary_line(m, label):
    return (f"{label:34s} buzz@0.5/1/2/4/8s " +
            "/".join(f"{m['buzz_db'][int(np.argmin(np.abs(m['t'] - t)))]:6.1f}" for t in (0.5, 1, 2, 4, 8) if t < m['dur'] - 0.05)
            + f" | mean buzz 0.3-3s {mean_in(m, 'buzz_db', 0.3, 3):6.1f} 3-8s "
            + (f"{mean_in(m, 'buzz_db', 3, 8):6.1f}" if m['dur'] > 8 else "   n/a")
            + f" | hf 0.3-3s {mean_in(m, 'hf_frac', 0.3, 3):.3f}"
            + f" | level@4s {m['frame_db'][int(np.argmin(np.abs(m['t'] - 4)))] if m['dur'] > 4.1 else float('nan'):6.1f}")


def partial_envelope(y, sr, f0_est, nmax=60, nperseg=8192, hop=2048, inharm_B=0.0):
    """Per-frame amplitude (dB) of partials n=1..nmax located by peak search near n*f0*sqrt(1+B n^2)."""
    f, t, Z = ss.stft(y, sr, window="hann", nperseg=nperseg, noverlap=nperseg - hop, boundary=None)
    S = np.abs(Z) ** 2 + 1e-30
    df = f[1]
    A = np.zeros((nmax, len(t)))
    for n in range(1, nmax + 1):
        fn = n * f0_est * np.sqrt(1 + inharm_B * n * n)
        k = int(round(fn / df)); w = max(2, int(0.25 * f0_est / df))
        A[n - 1] = S[max(k - w, 0):k + w + 1].max(0)
    return t, 10 * np.log10(A)
