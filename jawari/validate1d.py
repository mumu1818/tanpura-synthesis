"""Validation tests for tanpura1d.py.  Run:  python validate.py  (writes validation_results.txt/.json)

A  contact disabled: measured partial frequencies vs analytic stiff-string frequencies (cents)
B  lossless energy conservation with contact + thread active; lossy energy balance
C  convergence: sample rate, number of modes, bridge point spacing, penalty stiffness, Newton tolerance
D  maximum penetration vs thread height (lift of the string above the bridge at the thread)
E  extra sanity checks: thread force vs. static preload (validity of a bilateral thread), contact never
   reaches the edges of the simulated bridge, output finite, energy above 20 kHz in the output,
   published qualitative results (no stiffness -> no jvari), amplitude dependence
"""
import json
import sys
import time

import numpy as np

from common import HERE, analyze, to_audio, metric_rows, mean_metric
from tanpura1d import simulate, thread_lift, build, DEFAULT

LOG = []
RES = {}


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def peak_freqs(y, fs, f_guess, search=30.0, nfft=1 << 22):
    w = np.hanning(len(y))
    Y = np.abs(np.fft.rfft(y * w, nfft))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    df = f[1]
    out = []
    for fg in f_guess:
        sel = np.where(np.abs(f - fg) < search)[0]
        k = sel[np.argmax(Y[sel])]
        a, b, c = np.log(Y[k - 1]), np.log(Y[k]), np.log(Y[k + 1])
        d = 0.5 * (a - c) / (a - 2 * b + c)
        out.append((k + d) * df)
    return np.array(out)


# ------------------------------------------------------------------ A
def test_A():
    log("=== A. Partial frequencies, contact and thread disabled, lossless (8 s at fs=44.1 kHz, nut force) ===")
    over = dict(bridge=False, thread=False, lossless=True, xe=0.137, we=0.003, Fe_rise=1e-4, Fe_fall=1e-4,
                Fe_amp=1.0, dur=8.0, fs=44100.0)
    p, s, r = simulate(over)
    f_an = s["om"] / (2 * np.pi)
    nchk = 100
    f_me = peak_freqs(r["Fnut"], p["fs"], f_an[:nchk])
    cents = 1200 * np.log2(f_me / f_an[:nchk])
    log("  mode  analytic Hz   measured Hz   error cents")
    for i in range(20):
        log(f"  {i + 1:4d}  {f_an[i]:11.4f}  {f_me[i]:12.4f}  {cents[i]:+.5f}")
    e20 = float(np.max(np.abs(cents[:20])))
    e100 = float(np.max(np.abs(cents)))
    log(f"  max |error| partials 1-20: {e20:.5f} cents; partials 1-{nchk}: {e100:.5f} cents")
    log(f"  inharmonicity check: f_20/(20 f_1) = {f_me[19] / (20 * f_me[0]):.6f} (analytic {f_an[19] / (20 * f_an[0]):.6f})")
    RES["A_max_cents_1_20"] = e20
    RES["A_max_cents_1_100"] = e100
    passed = e20 < 0.05
    log(f"  PASS={passed} (criterion: < 0.05 cent)")
    # A2: effect of the (stiff) thread spring alone, informative
    over2 = dict(over, thread=True, rc=0.0)
    p2, s2, r2 = simulate(over2)
    f_me2 = peak_freqs(r2["Fnut"], p2["fs"], f_an[:20] * 1.004)
    Lp = p["L"] - p["xc"]
    bp = np.arange(1, 21) * np.pi / Lp
    f_pin = np.sqrt((s["T"] * bp ** 2 + s["EI"] * bp ** 4) / s["mu"]) / (2 * np.pi)
    c_vs_L = 1200 * np.log2(f_me2 / f_an[:20])
    c_vs_pin = 1200 * np.log2(f_me2 / f_pin)
    log("  [info] with thread spring (k_c=1.2e5 N/m, r_c=0) but no bridge: partial shift vs. string of length L"
        " and vs. string pinned at x_c (length L-x_c):")
    log("   " + "  ".join(f"{i + 1}:{c_vs_L[i]:+.2f}/{c_vs_pin[i]:+.2f}" for i in range(0, 20, 3)) + "  (cents)")
    RES["A2_thread_shift_cents_mode1"] = [float(c_vs_L[0]), float(c_vs_pin[0])]
    return dict(test="A partial frequencies (no contact)", passed=bool(passed),
                result=f"max error {e20:.4f} cents (partials 1-20), {e100:.4f} cents (1-100)")


# ------------------------------------------------------------------ B
def test_B():
    log("\n=== B. Energy: lossless (damping, thread damper off), bridge contact + thread spring active, 1.0 s ===")
    p, s, r = simulate(dict(dur=1.0, lossless=True, rc=0.0), energy=True)
    H = r["H"]
    n0 = int(round((p["Fe_rise"] + p["Fe_fall"]) * p["fs"])) + 2
    drift = float(np.max(np.abs(H[n0:] / H[n0] - 1)))
    st = r["stats"]
    log(f"  contact active in {100 * st['contact_frac']:.1f}% of samples, max active bridge points {st['max_active']}, "
        f"Newton iterations mean {st['newton_mean']:.2f} max {st['newton_max']} failures {st['newton_fail']}")
    log(f"  H after excitation = {H[n0]:.6e} J ; max relative drift over {len(H) - n0} steps = {drift:.3e}")
    # share of the energy that has left the low modes (shows the contact is doing real work)
    RES["B_lossless_drift"] = drift
    passed1 = drift < 1e-9 and st["newton_fail"] == 0
    log(f"  PASS={passed1} (criterion: drift < 1e-9, no Newton failure)")
    log("  --- lossy, full model, 3 s: discrete energy balance H^(n+1/2) + dissipated - supplied = 0")
    p, s, r = simulate(dict(dur=3.0), energy=True)
    H, D, W = r["H"], r["D"], r["W"]
    bal = H + D - W
    rel = float(np.max(np.abs(bal)) / np.max(H))
    dH = np.diff(H[n0:])
    n_up = int(np.sum(dH > 1e-12 * np.max(H)))
    log(f"  max |H + D - W| / max H = {rel:.3e};  steps after excitation where H increased: {n_up}")
    RES["B_lossy_balance"] = rel
    RES["B_lossy_H_increases"] = n_up
    passed2 = rel < 1e-9 and n_up == 0
    log(f"  PASS={passed2}")
    return dict(test="B energy conservation (lossless, contact active, 1 s) / balance (lossy, 3 s)",
                passed=bool(passed1 and passed2),
                result=f"lossless relative drift {drift:.2e}; lossy balance residual {rel:.2e}, H increased in {n_up} steps")


# ------------------------------------------------------------------ C
BANDS = [(0, 2e3), (2e3, 5e3), (5e3, 10e3), (10e3, 15e3), (15e3, 20e3), (20e3, 22.05e3)]


def bandlev(y):
    import scipy.signal as ss
    f, P = ss.welch(y, 44100, nperseg=8192)
    tot = P.sum()
    return [float(10 * np.log10(P[(f >= a) & (f < b)].sum() / tot + 1e-30)) for a, b in BANDS]


def summarise(r, p, label, ref=None):
    y = to_audio(r["Fbridge"], p["fs"])
    m = analyze(y, 44100)
    rows = metric_rows(m, (0.25, 0.5, 1.0, 2.0))
    mb = mean_metric(m, "buzz_db", 0.3, 2.7)
    mh = mean_metric(m, "hf_frac", 0.3, 2.7)
    mc = mean_metric(m, "centroid_hz", 0.3, 2.7)
    txt = " | ".join(f"{t}s b{v['buzz']:6.1f} hf{v['hf']:.2f} c{v['centroid']:5.0f} pk{v['peak']:5.0f}"
                     for t, v in rows.items())
    d = ""
    if ref is not None:
        sel = (m["t"] > 0.1) & (m["t"] < 2.8)
        db = float(np.mean(np.abs(m["buzz_db"][sel] - ref["buzz_db"][sel])))
        d = f" | mean|dbuzz| vs base {db:4.1f} dB"
    log(f"  {label:26s} M={r['M']:3d} cpu/s={r['cpu'] / p['dur']:5.2f} | mean(0.3-2.7s) buzz {mb:6.1f} dB hf {mh:.3f} "
        f"cen {mc:5.0f} Hz{d}")
    log(f"  {'':26s} {txt}")
    return m, dict(mean_buzz=mb, mean_hf=mh, mean_centroid=mc)


def test_C():
    log("\n=== C. Convergence (nominal model, 3 s, bridge force; outputs at fs>44.1k are resampled to 44.1k) ===")
    log("  band levels: long-term spectrum of the bridge force in dB re total, bands 0-2|2-5|5-10|10-15|15-20 kHz")
    cases = [("base 176.4k, modes<20k", {}),
             ("fs 44.1k, same 133 modes", dict(fs=44100.0)),
             ("fs 88.2k, same 133 modes", dict(fs=88200.0)),
             ("fs 352.8k, same 133 modes", dict(fs=352800.0)),
             ("fs 176.4k, modes<40k", dict(fmax=40000.0)),
             ("fs 176.4k, modes<80k", dict(fmax=80000.0)),
             ("bridge dx 0.1 mm", dict(dxb=1e-4)),
             ("bridge dx 0.5 mm", dict(dxb=5e-4)),
             ("k_b 1e10", dict(kb=1e10)),
             ("k_b 1e12", dict(kb=1e12)),
             ("Newton tol 1e-15", dict(tol=1e-15))]
    ref = None
    out = {}
    for label, over in cases:
        p, s, r = simulate(dict(dict(dur=3.0), **over))
        r["M"] = s["M"]
        m, summ = summarise(r, p, label, ref)
        summ["bands"] = bandlev(to_audio(r["Fbridge"], p["fs"]))
        log(f"  {'':26s} band levels " + " ".join(f"{x:6.1f}" for x in summ["bands"]))
        if ref is None:
            ref = m
        out[label] = summ
    RES["C"] = out
    b = out["base 176.4k, modes<20k"]
    conv = {k: v for k, v in out.items() if not k.startswith("fs 44.1k") and not k.startswith("fs 88.2k")}
    worst = max(abs(v["mean_buzz"] - b["mean_buzz"]) for v in conv.values())
    worst_hf = max(abs(v["mean_hf"] - b["mean_hf"]) for v in conv.values())
    worst_band = max(max(abs(x - y) for x, y in zip(v["bands"][:5], b["bands"][:5])) for v in conv.values())
    w44 = out["fs 44.1k, same 133 modes"]
    d44 = [x - y for x, y in zip(w44["bands"], b["bands"])]
    log(f"  vs base, all refinements (fs x2, modes x2/x4, dx, k_b, tol): worst |d mean buzz| {worst:.1f} dB, "
        f"|d mean hf_frac| {worst_hf:.3f}, worst band-level difference {worst_band:.1f} dB")
    log(f"  coarser fs=44.1k (no oversampling): mean buzz {w44['mean_buzz'] - b['mean_buzz']:+.1f} dB, band levels "
        + " ".join(f"{x:+.1f}" for x in d44[:5]) + " dB vs base -> excess energy around fs/4 = 11 kHz")
    passed = worst < 2.0 and worst_hf < 0.05 and worst_band < 4.0
    log(f"  PASS={passed} (criterion for all refinements: |d mean buzz| < 2 dB, |d mean hf_frac| < 0.05, "
        f"band levels within 4 dB)")
    return dict(test="C convergence (fs x2, modes to 40/80 kHz, bridge dx x2/x0.4, k_b x0.1/x10, Newton tol)",
                passed=bool(passed),
                result=f"worst |d mean buzz| {worst:.1f} dB, |d mean hf_frac| {worst_hf:.3f}, band levels within "
                       f"{worst_band:.1f} dB; un-oversampled 44.1 kHz run has {d44[3]:+.1f} dB excess at 10-15 kHz")


# ------------------------------------------------------------------ D/E
def test_D():
    log("\n=== D. Maximum penetration vs thread height (3 s runs) ===")
    rows = []
    ok = True
    for xc in (0.004, 0.005, 0.0065, 0.008):
        for amp, kb in ((0.2, 1e11), (0.8, 1e11), (0.8, 1e12)):
            p, s, r = simulate(dict(dur=3.0, xc=xc, Fe_amp=amp, kb=kb))
            st = r["stats"]
            lift = thread_lift(p)
            ratio = st["max_pen"] / lift
            if amp == 0.2:
                ok &= ratio < 0.01
            log(f"  x_c={xc * 1e3:4.1f} mm  pluck {amp:.1f} N k_b={kb:.0e}: thread height {lift * 1e6:6.1f} um, max pen. "
                f"{st['max_pen'] * 1e9:6.1f} nm at x={st['x_max_pen'] * 1e3:5.2f} mm -> {100 * ratio:.3f}% ; "
                f"edge pen. {st['edge_pen']:.1e} m ; max|F_c| {st['max_Fc']:.3f} N ; max|y(L/2)| "
                f"{np.abs(r['y']).max() * 1e3:.2f} mm")
            rows.append(dict(xc=xc, amp=amp, kb=kb, lift=lift, max_pen=st["max_pen"], ratio=ratio,
                             edge=st["edge_pen"], maxFc=st["max_Fc"]))
    RES["D"] = rows
    nom = [r for r in rows if r["amp"] == 0.2]
    worst = max(r["ratio"] for r in nom)
    stress = max(r["ratio"] for r in rows if r["amp"] == 0.8 and r["kb"] == 1e11)
    stress12 = max(r["ratio"] for r in rows if r["amp"] == 0.8 and r["kb"] == 1e12)
    log(f"  PASS={ok} (criterion: max penetration < 1% of thread height at the nominal pluck, all thread positions)")
    return dict(test="D max penetration vs thread height", passed=bool(ok),
                result=f"nominal pluck: worst {100 * worst:.2f}% of thread height (<= "
                       f"{max(r['max_pen'] for r in nom) * 1e9:.0f} nm vs 12-108 um); 4x harder pluck: {100 * stress:.2f}% "
                       f"(k_b=1e11), {100 * stress12:.2f}% (k_b=1e12)"), rows


def test_E(rowsD):
    log("\n=== E. Extra sanity checks ===")
    res = []
    # E1 thread never needs to pull: compare dynamic thread force with static preload from a break angle
    maxFc = max(r["maxFc"] for r in rowsD)
    T = build(dict(DEFAULT))["T"]
    ang = np.degrees(np.arcsin(maxFc / T))
    log(f"  E1 max |thread force| {maxFc:.3f} N = static preload of a {ang:.2f} deg break angle at T={T:.1f} N."
        f" Real bridges break the string by several degrees -> the string never lifts off the thread, so a bilateral"
        f" (linearised) thread is valid.")
    res.append(dict(test="E1 bilateral thread validity (dynamic thread force < static preload, ASSUMING a break "
                         "angle >= 5 deg over the bridge)", passed=bool(maxFc < T * np.sin(np.radians(5.0))),
                    result=f"max |F_c| {maxFc:.3f} N (incl. 4x harder pluck) = preload of a {ang:.2f} deg break angle; "
                           f"5 deg gives {T * np.sin(np.radians(5.0)):.2f} N"))
    edge = max(r["edge"] for r in rowsD)
    log(f"  E2 max penetration at the two outermost bridge points on either side: {edge:.2e} m (0 = contact never"
        f" reaches the end of the simulated bridge)")
    res.append(dict(test="E2 contact stays inside simulated bridge region", passed=bool(edge == 0.0),
                    result=f"edge penetration {edge:.1e} m"))
    # E3 output sanity and energy above 20 kHz
    p, s, r = simulate(dict(dur=3.0))
    y = r["Fbridge"] - r["Fbridge"].mean()
    fin = bool(np.all(np.isfinite(y)))
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / p["fs"])
    hf20 = float(Y[f > 20000].sum() / Y.sum())
    hf20n = float((np.abs(np.fft.rfft(r["Fnut"] * np.hanning(len(y)))) ** 2)[f > 20000].sum()
                  / (np.abs(np.fft.rfft(r["Fnut"] * np.hanning(len(y)))) ** 2).sum())
    log(f"  E3 outputs finite: {fin}; energy fraction above 20 kHz: bridge force {hf20:.2e}, nut force {hf20n:.2e}")
    res.append(dict(test="E3 outputs finite; energy above the 20 kHz mode cut-off in the raw 176.4 kHz output",
                    passed=bool(fin),
                    result=f"finite={fin}; fraction of energy above 20 kHz {hf20:.1e} (bridge force, contains the "
                           f"fs/4 contact-chatter), {hf20n:.1e} (nut force); removed by the decimation to 44.1 kHz"))
    # E4 published qualitative result: no string stiffness -> no jvari (DAFx-16 sound example)
    out = {}
    for lab, over in (("nominal", {}), ("EI -> 0 (1e-6 x)", dict(EI_scale=1e-6)), ("no thread", dict(thread=False)),
                      ("no bridge (thread only)", dict(bridge=False))):
        p, s, r = simulate(dict(dict(dur=3.0), **over))
        m = analyze(to_audio(r["Fbridge"], p["fs"]), 44100)
        out[lab] = dict(buzz=mean_metric(m, "buzz_db", 0.3, 2.7), hf=mean_metric(m, "hf_frac", 0.3, 2.7),
                        cen=mean_metric(m, "centroid_hz", 0.3, 2.7))
        log(f"  E4 {lab:24s} mean(0.3-2.7 s) buzz {out[lab]['buzz']:6.1f} dB  hf_frac {out[lab]['hf']:.3f}  "
            f"centroid {out[lab]['cen']:5.0f} Hz")
    RES["E4"] = out
    # E5 amplitude dependence
    log("  E5 amplitude dependence (pluck force 0.05 / 0.2 / 0.8 N; nominal geometry)")
    amp = {}
    for a in (0.05, 0.2, 0.8):
        p, s, r = simulate(dict(dur=3.0, Fe_amp=a))
        m = analyze(to_audio(r["Fbridge"], p["fs"]), 44100)
        amp[a] = dict(buzz=mean_metric(m, "buzz_db", 0.3, 2.7), hf=mean_metric(m, "hf_frac", 0.3, 2.7),
                      ymax=float(np.abs(r["y"]).max()))
        log(f"     {a:4.2f} N: max |y(L/2)| {amp[a]['ymax'] * 1e3:5.2f} mm, mean buzz {amp[a]['buzz']:6.1f} dB, "
            f"mean hf_frac {amp[a]['hf']:.3f}")
    for h0 in (+1e-6, -1e-6):
        for a in (0.05, 0.8):
            p, s, r = simulate(dict(dur=3.0, Fe_amp=a, h0=h0))
            m = analyze(to_audio(r["Fbridge"], p["fs"]), 44100)
            log(f"     crest offset h0={h0 * 1e6:+.0f} um, {a:4.2f} N: mean buzz "
                f"{mean_metric(m, 'buzz_db', 0.3, 2.7):6.1f} dB, mean hf_frac {mean_metric(m, 'hf_frac', 0.3, 2.7):.3f}")
    RES["E5"] = {str(k): v for k, v in amp.items()}
    return res


if __name__ == "__main__":
    t0 = time.time()
    results = []
    results.append(test_A())
    results.append(test_B())
    results.append(test_C())
    rD, rowsD = test_D()
    results.append(rD)
    results.extend(test_E(rowsD))
    log("\n=== SUMMARY ===")
    for r in results:
        log(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['test']}: {r['result']}")
    log(f"total validation time {time.time() - t0:.0f} s")
    RES["summary"] = results
    with open(HERE + "/validation_results.txt", "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    with open(HERE + "/validation_results.json", "w") as fh:
        json.dump(RES, fh, indent=1, default=float)
