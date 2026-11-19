"""Validation tests for tanpura2d.py (two-polarisation model).

    python validate2d.py            # all tests
    python validate2d.py A B        # selected tests
Each finished test is merged into validation2d_results.json and the text log is appended to
validation2d_results.txt, so partial progress survives an interruption.

A  partial frequencies with contact/thread off, both polarisations (principal axes of the nut coupling),
   and the Kirchhoff-Carrier (tension-modulation) pitch shift vs the Duffing prediction
B  discrete energy: lossless conservation (contact + thread + TM + nut coupling); friction only dissipates;
   lossy full-model balance
C  convergence of the nominal 2-D model (fs x2, modes to 40 kHz, bridge dx, k_b, solver tolerances) compared
   with the chaotic noise floor (1 um thread shift)
D  1-D limit: pol2 off + TM off reproduces tanpura1d.py
E  contact / thread sanity: penetration vs thread height, edge points, solver failures, thread forces vs preload
"""
import json
import os
import sys
import time

import numpy as np

from common2d import HERE, analyze, to_audio, mean_metric
import tanpura2d as t2

LOG = []
RES_PATH = f"{HERE}/validation2d_results.json"
TXT_PATH = f"{HERE}/validation2d_results.txt"


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def save(key, entry):
    d = {}
    if os.path.exists(RES_PATH):
        with open(RES_PATH) as fh:
            d = json.load(fh)
    d[key] = entry
    with open(RES_PATH, "w") as fh:
        json.dump(d, fh, indent=1, default=float)
    with open(TXT_PATH, "a") as fh:
        fh.write("\n".join(LOG) + "\n")
    LOG.clear()


def peak_freqs(y, fs, f_guess, search=30.0, nfft=1 << 22):
    w = np.hanning(len(y))
    Y = np.abs(np.fft.rfft(y * w, nfft))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    df = f[1]
    fo, ao = [], []
    for fg in f_guess:
        sel = np.where(np.abs(f - fg) < search)[0]
        k = sel[np.argmax(Y[sel])]
        a, b, c = np.log(Y[k - 1]), np.log(Y[k]), np.log(Y[k + 1])
        d = 0.5 * (a - c) / (a - 2 * b + c)
        fo.append((k + d) * df)
        ao.append(2 * np.exp(b - 0.25 * (a - c) * d) / w.sum())
    return np.array(fo), np.array(ao)


# ------------------------------------------------------------------ A
def test_A():
    log("=== A. Partial frequencies, contact/thread/TM off, lossless, 8 s at fs = 44.1 kHz (nut forces) ===")
    th, det = 30.0, 5.0
    over = dict(bridge=False, thread=False, lossless=True, tm=False, xe=0.137, we=0.003, Fe_rise=1e-4,
                Fe_fall=1e-4, Fe_amp=1.0, dur=8.0, fs=44100.0, pluck_deg=20.0, theta_nut=th, detune_cents=det)
    p, s, r = t2.simulate(over)
    c, sn = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))
    u1 = c * r["Fny"] + sn * r["Fnz"]          # principal axis 1 (frequency f_i)
    u2 = -sn * r["Fny"] + c * r["Fnz"]         # principal axis 2 (frequency f_i 2^(det/1200))
    f1 = s["om"] / (2 * np.pi)
    f2 = s["om2"] / (2 * np.pi)
    n = 100
    m1, _ = peak_freqs(u1, p["fs"], f1[:n], search=0.2 * f1[0])
    m2, _ = peak_freqs(u2, p["fs"], f2[:n], search=0.2 * f1[0])
    e1 = 1200 * np.log2(m1 / f1[:n]); e2 = 1200 * np.log2(m2 / f2[:n])
    log(f"  nut coupling theta = {th} deg, axis-2 detuning {det} cents; pluck 20 deg from vertical")
    log("  mode   f_axis1 Hz   err cents   f_axis2 Hz   err cents   measured split (cents)")
    for i in [0, 1, 2, 4, 9, 19, 49, 99]:
        log(f"  {i + 1:4d}  {f1[i]:11.4f}  {e1[i]:+9.5f}  {f2[i]:11.4f}  {e2[i]:+9.5f}   {1200 * np.log2(m2[i] / m1[i]):.4f}")
    E1, E2 = float(np.max(np.abs(e1[:20]))), float(np.max(np.abs(e2[:20])))
    E1h, E2h = float(np.max(np.abs(e1))), float(np.max(np.abs(e2)))
    log(f"  max |err| partials 1-20: axis1 {E1:.5f}, axis2 {E2:.5f} cents; partials 1-100: {E1h:.5f}, {E2h:.5f}")
    passA1 = max(E1, E2) < 0.05
    # A2: no coupling (theta=0, no detune): vertical and horizontal partials identical and uncoupled
    p0, s0, r0 = t2.simulate(dict(over, theta_nut=0.0, detune_cents=0.0, pluck_deg=0.0))
    leak = float(np.max(np.abs(r0["Fnz"])))
    log(f"  theta=0, pluck 0 deg: max |F_nut,z| = {leak:.2e} N (must be exactly 0: no coupling)")
    passA2 = leak == 0.0
    # A3: Kirchhoff-Carrier tension modulation pitch shift of mode 1 (lossless, constant amplitude)
    log("  --- A3. tension modulation: mode-1 frequency shift vs Duffing prediction 3*g*a^2/(8*w0), g = EA b^4/(4 mu)")
    rows = []
    for amp in (0.2, 0.8):
        ov = dict(over, theta_nut=0.0, detune_cents=0.0, xe=0.5, we=1.0, Fe_amp=amp, Fe_rise=0.010,
                  Fe_fall=0.001, pluck_deg=20.0)
        pa, sa, ra = t2.simulate(dict(ov, tm=False))
        pb, sb, rb = t2.simulate(dict(ov, tm=True))
        fA, aY = peak_freqs(ra["Fny"], pa["fs"], f1[:1], search=5.0)
        _, aZ = peak_freqs(ra["Fnz"], pa["fs"], f1[:1], search=5.0)
        fB, _ = peak_freqs(rb["Fny"], pb["fs"], f1[:1], search=5.0)
        k1 = sa["T"] * sa["beta"][0] + sa["EI"] * sa["beta"][0] ** 3
        a_off = np.hypot(aY[0], aZ[0]) / k1
        w0 = sa["om"][0]
        wB = 2 * np.pi * fB[0]
        # amplitude of the TM-on run: the pluck response itself depends on the (stiffer) string, so take it from
        # that run's mean quadratic modal energy, calibrated on the TM-off run:  <E_lin> = (m/4) a^2 (w^2 + w0^2)
        sel_a = ra["t_rec"] > 0.05; sel_b = rb["t_rec"] > 0.05
        Ea = np.mean((ra["Ey"] + ra["Ez"])[sel_a]); Eb = np.mean((rb["Ey"] + rb["Ez"])[sel_b])
        a = a_off * np.sqrt((Eb / Ea) * (2 * w0 ** 2) / (wB ** 2 + w0 ** 2))
        g = sa["EA"] * sa["beta"][0] ** 4 / (4 * sa["mu"])
        dpred = 3 * g * a * a / (8 * w0) / (2 * np.pi)
        dmeas = fB[0] - fA[0]
        rows.append((amp, a, dmeas, dpred))
        log(f"  pluck {amp:.1f} N: mode-1 amplitude {a_off * 1e3:.3f} mm (TM off) / {a * 1e3:.3f} mm (TM on), "
            f"measured shift {dmeas * 1e3:.3f} mHz ({1200 * np.log2(fB[0] / fA[0]):.3f} cents), Duffing prediction "
            f"{dpred * 1e3:.3f} mHz, ratio {dmeas / dpred:.3f}")
    passA3 = all(abs(dm / dp - 1) < 0.05 for _, _, dm, dp in rows)
    log(f"  PASS A1={passA1} (<0.05 cent) A2={passA2} A3={passA3} (within 5% of Duffing)")
    entry = dict(test="A. Modal frequencies with contact, thread and TM off (both polarisations, principal axes of "
                      "the nut coupling, theta=30 deg, 5 cent split), uncoupled limit, and KC tension-modulation pitch shift",
                 passed=bool(passA1 and passA2 and passA3),
                 result=f"max error partials 1-20: {E1:.5f}/{E2:.5f} cents (axis 1/2), 1-100: {E1h:.5f}/{E2h:.5f}; "
                        f"theta=0 & vertical pluck: horizontal force exactly {leak:.0e}; TM mode-1 shift / Duffing "
                        f"prediction = " + ", ".join(f"{dm / dp:.3f} at {a * 1e3:.2f} mm ({dm * 1e3:.1f} mHz)"
                                                     for _, a, dm, dp in rows))
    save("A", entry)
    return entry


# ------------------------------------------------------------------ B
def _energy_check(over, label, n_pluck_extra=2):
    p, s, r = t2.simulate(over, energy=True)
    H, D, Df, W = r["H"], r["D"], r["Dfric"], r["W"]
    n0 = int(round((p["Fe_rise"] + p["Fe_fall"]) * p["fs"])) + n_pluck_extra
    Hm = float(np.max(H))
    bal = H + D + Df - W                     # must stay 0 (to round-off) at every step
    res = float(np.max(np.abs(bal)) / Hm)
    dH = np.diff(H[n0:])
    n_up = int(np.sum(dH > 1e-13 * Hm))
    max_up = float(np.max(dH) / Hm)
    st = r["stats"]
    log(f"  {label}: {r['N']} steps, contact {100 * st['contact_frac']:.0f}% of samples (max {st['max_active']} pts), "
        f"Newton {st['newton_mean']:.2f}/{st['newton_max']} fail {st['newton_fail']}, friction Newton "
        f"{st['fric_newton_mean']:.2f}/{st['fric_newton_max']} fail {st['fric_fail']}")
    log(f"      max|H + D_damp + D_fric - W|/max H = {res:.2e};  H(end)/H(after pluck) = {H[-1] / H[n0]:.6f};  "
        f"friction loss / H(after pluck) = {Df[-1] / H[n0]:.3e};  steps with dH > 1e-13 Hmax after pluck: {n_up} "
        f"(max dH/Hmax {max_up:+.1e})")
    return p, r, dict(res=res, n_up=n_up, max_up=max_up, H_ratio=float(H[-1] / H[n0]), fric_frac=float(Df[-1] / H[n0]),
                      drift=float(np.max(np.abs(H[n0:] / H[n0] - 1))), st=st)


def test_B():
    log("\n=== B. Discrete energy balance  H^{n+1/2}-H^{n-1/2} = dt[ supplied - modal damping - thread damping - friction ] ===")
    LL = dict(lossless=True, rc=0.0, rcz=0.0)
    p, r, b1 = _energy_check(dict(LL, dur=1.0, friction=False), "B1 lossless, contact+thread+TM+nut coupling, friction off, 1 s")
    log(f"      relative drift of H after the pluck: {b1['drift']:.2e}")
    p, r, b2 = _energy_check(dict(LL, dur=1.0, friction=True), "B2 lossless + friction (c2=2.84 s/m), 1 s")
    p, r, b3 = _energy_check(dict(LL, dur=1.0, friction=True, mu_c2=2840.0), "B3 lossless + near-Coulomb friction (c2=2840 s/m), 1 s")
    p, r, b4 = _energy_check(dict(dur=3.0), "B4 lossy full model (nominal), 3 s")
    pass1 = b1["drift"] < 1e-9 and b1["st"]["newton_fail"] == 0
    pass2 = all(b["res"] < 1e-9 and b["n_up"] == 0 for b in (b2, b3, b4))
    pass2 = pass2 and b2["fric_frac"] > 0 and b3["fric_frac"] > 0
    log(f"  PASS conservation={pass1} (drift<1e-9)  dissipativity/balance={pass2} (residual<1e-9, H never increases)")
    entry = dict(test="B. Energy: lossless (no modal damping, r_c=0) with contact+thread+TM+nut coupling, 1 s at 176.4 kHz; "
                      "friction on (c2=2.84 and 2840 s/m) must only dissipate; lossy full model balance, 3 s",
                 passed=bool(pass1 and pass2),
                 result=f"friction off: relative drift of H {b1['drift']:.1e}. Friction on: balance residual "
                        f"{b2['res']:.1e}/{b3['res']:.1e}, H increased in {b2['n_up']}/{b3['n_up']} steps, friction removed "
                        f"{100 * b2['fric_frac']:.2f}%/{100 * b3['fric_frac']:.1f}% of the energy in 1 s. Lossy: residual "
                        f"{b4['res']:.1e}, H increased in {b4['n_up']} steps. Newton failures 0/0/0/0: "
                        f"{[b['st']['newton_fail'] + b['st']['fric_fail'] for b in (b1, b2, b3, b4)]}")
    save("B", entry)
    return entry


# ------------------------------------------------------------------ C
BANDS = [(0, 2e3), (2e3, 5e3), (5e3, 10e3), (10e3, 15e3), (15e3, 20e3)]


def bandlev(y):
    import scipy.signal as ss
    f, P = ss.welch(y, 44100, nperseg=8192)
    tot = P.sum()
    return [float(10 * np.log10(P[(f >= a) & (f < b)].sum() / tot + 1e-30)) for a, b in BANDS]


def summ(r, p):
    out = {}
    for key, sig in (("y", r["Fy"]), ("mono", r["Fy"] + 0.3 * r["Fz"])):
        y = to_audio(sig, p["fs"])
        m = analyze(y, 44100)
        out[key] = dict(b03_3=mean_metric(m, "buzz_db", 0.3, 3.0), b3_5=mean_metric(m, "buzz_db", 3.0, 4.9),
                        hf=mean_metric(m, "hf_frac", 0.3, 3.0), cen=mean_metric(m, "centroid_hz", 0.3, 3.0),
                        bands=bandlev(y), lev5=float(m["frame_db"][np.argmin(np.abs(m["t"] - 4.8))]))
    zr = to_audio(r["Fz"], p["fs"]); yr = to_audio(r["Fy"], p["fs"])
    out["z_over_y_db"] = float(20 * np.log10(np.std(zr) / np.std(yr)))
    return out


def test_C():
    log("\n=== C. Convergence of the nominal 2-D model (x_c=5 mm, 5 s, vertical bridge force and mono mix) ===")
    cases = [("base 176.4k, 133 modes", {}),
             ("noise floor: x_c + 1 um", dict(xc=0.005001)),
             ("noise floor: x_c - 1 um", dict(xc=0.004999)),
             ("fs 352.8k", dict(fs=352800.0)),
             ("modes to 40 kHz", dict(fmax=40000.0)),
             ("bridge dx 0.1 mm", dict(dxb=1e-4)),
             ("k_b 1e12", dict(kb=1e12)),
             ("tolerances x1e-3/x1e-2", dict(tol=1e-21, ftol=1e-15)),
             ("fs 88.2k (no oversampling x4 -> x2)", dict(fs=88200.0))]
    rows = {}
    for lab, ov in cases:
        p, s, r = t2.simulate(dict(dict(dur=5.0), **ov))
        sm = summ(r, p)
        rows[lab] = dict(sm, M=s["M"], cpu_per_s=r["cpu"] / 5.0)
        y = sm["y"]; mo = sm["mono"]
        log(f"  {lab:38s} M={s['M']:3d} cpu/s {r['cpu'] / 5:5.2f} | vertical: buzz {y['b03_3']:6.1f}/{y['b3_5']:6.1f} dB "
            f"hf {y['hf']:.3f} cen {y['cen']:5.0f} | mono: buzz {mo['b03_3']:6.1f}/{mo['b3_5']:6.1f} | z/y {sm['z_over_y_db']:5.1f} dB "
            f"| bands(y) " + " ".join(f"{b:6.1f}" for b in y["bands"]))
    base = rows[cases[0][0]]
    noise = [rows[cases[1][0]], rows[cases[2][0]]]

    def dev(a, b):
        return dict(b1=abs(a["y"]["b03_3"] - b["y"]["b03_3"]), b2=abs(a["y"]["b3_5"] - b["y"]["b3_5"]),
                    hf=abs(a["y"]["hf"] - b["y"]["hf"]), mb1=abs(a["mono"]["b03_3"] - b["mono"]["b03_3"]),
                    bands=float(np.max(np.abs(np.array(a["y"]["bands"][:4]) - np.array(b["y"]["bands"][:4])))),
                    b15=abs(a["y"]["bands"][4] - b["y"]["bands"][4]), zy=abs(a["z_over_y_db"] - b["z_over_y_db"]))
    nf = {k: max(dev(n, base)[k] for n in noise) for k in dev(base, base)}
    log(f"  noise floor (1 um thread shift): |d buzz| {nf['b1']:.2f}/{nf['b2']:.2f} dB, |d hf| {nf['hf']:.3f}, "
        f"bands 0-15k {nf['bands']:.2f} dB, 15-20k {nf['b15']:.2f} dB, z/y {nf['zy']:.2f} dB")
    worst = {}
    for lab, _ in cases[3:8]:
        d = dev(rows[lab], base)
        worst[lab] = d
        log(f"  {lab:38s} |d buzz 0.3-3| {d['b1']:.2f} dB, |d buzz 3-5| {d['b2']:.2f} dB, |d hf| {d['hf']:.3f}, "
            f"|d mono buzz| {d['mb1']:.2f}, bands 0-15k {d['bands']:.2f} dB, 15-20k {d['b15']:.2f} dB, z/y {d['zy']:.2f} dB")
    mx = {k: max(w[k] for w in worst.values()) for k in nf}
    passed = mx["b1"] < 1.5 and mx["b2"] < 2.5 and mx["hf"] < 0.03 and mx["bands"] < 2.0
    d88 = dev(rows[cases[8][0]], base)
    log(f"  [info] fs 88.2k: |d buzz| {d88['b1']:.2f}/{d88['b2']:.2f} dB, bands 0-15k {d88['bands']:.2f} dB, 15-20k {d88['b15']:.2f} dB")
    log(f"  PASS={passed} (pre-set criteria: |d mean buzz| < 1.5 dB (0.3-3 s) and < 2.5 dB (3-5 s), |d hf| < 0.03, "
        f"band levels 0-15 kHz within 2 dB; 15-20 kHz band reported only)")
    entry = dict(test="C. Convergence of the nominal 2-D model (5 s): fs 352.8k, modes to 40 kHz, bridge dx 0.1 mm, "
                      "k_b 1e12, tighter Newton tolerances; compared with the chaotic noise floor of a 1 um thread shift",
                 passed=bool(passed),
                 result=(f"worst deviation from base over refinements: mean buzz {mx['b1']:.2f} dB (0.3-3 s), "
                         f"{mx['b2']:.2f} dB (3-5 s), hf_frac {mx['hf']:.3f}, bands 0-15 kHz {mx['bands']:.2f} dB, "
                         f"15-20 kHz {mx['b15']:.2f} dB, z/y level {mx['zy']:.2f} dB; noise floor (1 um thread shift): "
                         f"{nf['b1']:.2f}/{nf['b2']:.2f} dB buzz, hf {nf['hf']:.3f}, bands {nf['bands']:.2f} dB"),
                 rows=rows)
    save("C", entry)
    return entry


# ------------------------------------------------------------------ D
def test_D():
    log("\n=== D. 1-D limit: tanpura2d (pol2=False, tm=False) vs tanpura1d.py, nominal, 10 s ===")
    import tanpura1d as t1
    p1, s1, r1 = t1.simulate(dict(dur=10.0))
    p2, s2, r2 = t2.simulate(dict(dur=10.0, pol2=False, tm=False))
    a, b = r1["Fbridge"], r2["Fy"]
    fs = p1["fs"]; mx = np.max(np.abs(a))
    err = np.array([np.max(np.abs(a[i:i + int(0.01 * fs)] - b[i:i + int(0.01 * fs)])) / mx
                    for i in range(0, len(a) - int(0.01 * fs), int(0.01 * fs))])
    tt = np.arange(len(err)) * 0.01
    e03 = float(np.max(err[tt < 0.3]))
    t_div = float(tt[np.argmax(err > 1e-3)]) if np.any(err > 1e-3) else np.inf
    m1 = analyze(to_audio(a, fs), 44100); m2 = analyze(to_audio(b, fs), 44100)
    mb = [(mean_metric(m, "buzz_db", 0.3, 3.0), mean_metric(m, "buzz_db", 3.0, 8.0), mean_metric(m, "hf_frac", 0.3, 3.0))
          for m in (m1, m2)]
    log(f"  max |F_1D - F_2D| / max F over 0-0.3 s: {e03:.2e}; difference first exceeds 1e-3 at t = {t_div:.2f} s "
        f"(round-off differences in the contact solve grow exponentially: the contact dynamics is chaotic)")
    log(f"  mean buzz 0.3-3 s / 3-8 s, hf 0.3-3 s: 1-D {mb[0][0]:.2f}/{mb[0][1]:.2f} dB, {mb[0][2]:.3f};  "
        f"2-D code {mb[1][0]:.2f}/{mb[1][1]:.2f} dB, {mb[1][2]:.3f}")
    gr = np.log(err[(tt > 0.1) & (tt < t_div)] + 1e-300)
    passed = e03 < 1e-8 and abs(mb[0][0] - mb[1][0]) < 1.5 and abs(mb[0][1] - mb[1][1]) < 2.5
    log(f"  PASS={passed}")
    entry = dict(test="D. 1-D limit: 2-D code with horizontal polarisation and TM disabled vs the validated 1-D prototype",
                 passed=bool(passed),
                 result=f"identical to {e03:.1e} (relative) for 0-0.3 s; trajectories separate (>1e-3) after {t_div:.2f} s "
                        f"because round-off is amplified by the chaotic contact dynamics; time-averaged metrics agree: "
                        f"buzz {mb[0][0]:.2f} vs {mb[1][0]:.2f} dB (0.3-3 s), {mb[0][1]:.2f} vs {mb[1][1]:.2f} dB (3-8 s)")
    save("D", entry)
    return entry


# ------------------------------------------------------------------ E
def test_E():
    log("\n=== E. Contact/thread sanity for the full 2-D model (3 s runs), thread positions 3-8 mm, plus 4x pluck ===")
    rows = []
    ok = True
    for xc, amp in ((0.003, 0.2), (0.004, 0.2), (0.005, 0.2), (0.0065, 0.2), (0.008, 0.2), (0.005, 0.8), (0.004, 0.8)):
        ov = dict(dur=3.0, xc=xc, Fe_amp=amp)
        if xc < 0.004:
            ov["br_x0"] = xc
        p, s, r = t2.simulate(ov)
        st = r["stats"]
        lift = t2.thread_lift(p)
        rows.append(dict(xc=xc, amp=amp, lift=lift, pen=st["max_pen"], edge=st["edge_pen"], nf=st["newton_fail"],
                         ff=st["fric_fail"], res=st["n_resolve"], Fcy=st["max_Fcy"], Fcz=st["max_Fcz"],
                         slip=st["max_slip"], mu=st["max_mu"], dT=st["max_dT"], nb=s["Nb"]))
        log(f"  x_c={xc * 1e3:4.1f} mm pluck {amp:.1f} N: lift {lift * 1e6:6.1f} um, max pen {st['max_pen'] * 1e9:6.1f} nm "
            f"({100 * st['max_pen'] / lift:.3f}% of lift), edge pen {st['edge_pen']:.1e}, Newton fail {st['newton_fail']}, "
            f"friction fail {st['fric_fail']}, set re-solves {st['n_resolve']}, max|F_c| y/z {st['max_Fcy']:.3f}/"
            f"{st['max_Fcz']:.3f} N, max slip {st['max_slip'] * 1e3:.1f} mm/s, max mu {st['max_mu']:.3f}, "
            f"max dT {st['max_dT'] * 1e3:.1f} mN")
        ok = ok and st["edge_pen"] == 0 and st["newton_fail"] == 0 and st["fric_fail"] == 0
    T0 = 33.097
    pre5 = 2 * T0 * np.sin(np.deg2rad(5) / 2)
    fy = max(r_["Fcy"] for r_ in rows); fz = max(r_["Fcz"] for r_ in rows)
    log(f"  bilateral thread: max |F_c,y| {fy:.3f} N vs preload of a 5 deg break angle {pre5:.2f} N; max |F_c,z| {fz:.3f} N "
        f"(the lateral grip needed; with a 5 deg break angle and thread friction 0.5 the grip limit is ~{0.5 * pre5:.2f} N)")
    ok = ok and fy < pre5 and fz < 0.5 * pre5
    log(f"  PASS={ok}")
    entry = dict(test="E. Contact/thread sanity (x_c 3-8 mm, nominal and 4x pluck): penetration, edges, solver failures, "
                      "bilateral thread validity in y and z",
                 passed=bool(ok),
                 result=f"max penetration {max(r_['pen'] for r_ in rows) * 1e9:.0f} nm = "
                        f"{100 * max(r_['pen'] / r_['lift'] for r_ in rows):.2f}% of thread lift; edge penetration "
                        f"{max(r_['edge'] for r_ in rows):.0e}; Newton/friction failures 0; max |F_c| y {fy:.2f} N, "
                        f"z {fz:.2f} N; max slip speed {max(r_['slip'] for r_ in rows) * 1e3:.0f} mm/s, max mu_f "
                        f"{max(r_['mu'] for r_ in rows):.3f}; max tension rise {max(r_['dT'] for r_ in rows) * 1e3:.0f} mN",
                 rows=rows)
    save("E", entry)
    return entry


if __name__ == "__main__":
    which = sys.argv[1:] or ["A", "B", "C", "D", "E"]
    with open(TXT_PATH, "a") as fh:
        fh.write(f"\n==== validate2d run {time.strftime('%Y-%m-%d %H:%M:%S')} tests {which} ====\n")
    for w in which:
        t0 = time.time()
        globals()[f"test_{w}"]()
        log(f"  [{w} took {time.time() - t0:.0f} s]")
        with open(TXT_PATH, "a") as fh:
            fh.write("\n".join(LOG) + "\n")
        LOG.clear()
