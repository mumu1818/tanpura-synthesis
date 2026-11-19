"""Validation of statics.py (rest state of the string over bridge + thread) and of its coupling to tanpura1d.

  (i)   EI = 0, point thread, sharp back corner: grid solution vs the analytic kinked string (shape, forces,
        crest gap of the front line over the parabola)
  (ii)  beam-string boundary layer at a point thread vs the analytic Green's function
        y = A + B(x-x0) - F/(2T)|x-x0| - F l/(2T) exp(-|x-x0|/l); also checks the exact far-field (DtN) terms
        by shrinking the grid margin to 1.5 l
  (iii) force balance: sum of contact forces vs T x (total angle change), both from the solver's far-field
        slopes (discrete identity) and from the anchor geometry (small-slope theory)
  (iv)  grid convergence: crest gap at h = 20 / 10 / 5 um (criterion: |gap(10) - gap(5)| < 0.05 um)
  (v)   d(crest gap)/d(x_c) vs the analytic 2 sqrt(c (d_t' - g)) for EI -> 0 (and the EI > 0 deviation)
  (vi)  tanpura1d static_init: modal static solution (133 modes, discrete scheme) vs fine-grid statics of the
        same continuum problem, plus mode-number convergence; and vs the full bridge-frame statics
  (vii) extras: tune='speaking' pitch, static_init removes the start-up click, energy balance with static_init
  (viii) measured-profile path: parabola sampled every 0.25 mm (+ 0.5 um rms noise), written to CSV, loaded with
        BridgeProfile.from_csv (smoothing spline) -> crest gap and grazing position vs the exact parabola; and the
        flatter power-law family (p = 4) runs through auto_place
Run:  cd jawari && python validate_statics.py      (writes out/statics/validate_statics.txt)
"""
import os
import time

import numpy as np

import common  # noqa: F401  (sys.path, OUT_ROOT)
from common import OUT_ROOT
import statics as S
import tanpura1d as T1

OUT = os.path.join(str(OUT_ROOT), "statics")
os.makedirs(OUT, exist_ok=True)
LOG = []
SUMMARY = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def verdict(name, ok, msg):
    SUMMARY.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    log(f"  PASS={ok}")


C3 = S.String.c3_dafx16()
KH = S.String.kharaj()
# speaking-length angles of the two study instruments (study_statics.design: nominal d_t 0.2 mm, phi 10 deg,
# EI on grazes at x_c = 4 mm on the parabolic bridge below); recomputed in study_statics.py
PSI = {C3.name: 4.0516, KH.name: 6.9181}


def nominal_profile(length=40e-3):
    return S.BridgeProfile.parabolic(c=3.0, x_crest=2e-3, length=length, edge_radius=0.5e-3)


def setup(st, xc, dt=0.2e-3, phi=10.0, name=None):
    return S.Setup(nominal_profile(), st, S.Thread(xc, dt), phi_deg=phi, psi_deg=PSI[name or st.name],
                   L_speak=st.L)


# ------------------------------------------------------------------------------------------------- (i)
def test_i():
    log("=== (i) EI = 0, point thread, sharp back corner: grid vs analytic kinked string ===")
    pr = S.BridgeProfile.parabolic(c=3.0, x_crest=2e-3, length=30e-3, edge_radius=0.0)
    st = C3.with_EI(0.0)
    xc, dt = 5e-3, 0.2e-3
    su = S.Setup(pr, st, S.Thread(xc, dt, point=True), phi_deg=10.0, psi_deg=3.83)
    so = S.solve(su, h=10e-6, front=False)          # grid anchored at x_c: x = 0 is a node too
    (xT, yT), (xN, yN) = su.anchors()
    b0 = float(pr.top(np.array([0.0]))[0])
    yt = float(pr.top(np.array([xc]))[0]) + dt
    px = np.array([xT, 0.0, xc, xN]); py = np.array([yT, b0, yt, yN])
    ya = np.interp(so.x, px, py)
    err = float(np.max(np.abs(so.y - ya)))
    sl = np.diff(py) / np.diff(px)
    F_an = dict(back=st.T * (sl[0] - sl[1]), thread=st.T * (sl[1] - sl[2]))
    F = so.forces()
    log(f"  contacts at x = {np.round(so.x[so.active] * 1e3, 4)} mm (expected 0 and {xc*1e3} mm)")
    log(f"  max |y_grid - y_analytic| = {err:.2e} m")
    log(f"  corner force {F['back']:.9f} N (analytic T*(s_tail - s_back) = {F_an['back']:.9f}); thread force "
        f"{F['thread']:.9f} N (analytic {F_an['thread']:.9f})")
    # crest gap of the front line over the parabola: g = d_t - c (delta - s/2c)^2, delta = x_b - x_c
    c, xb = 3.0, 2e-3
    s_front = sl[2]
    g_an = dt - c * ((xb - xc) - s_front / (2 * c)) ** 2
    g_num, x_g = so.crest_gap()
    log(f"  crest gap: solver {g_num*1e6:.6f} um at x = {x_g*1e3:.4f} mm; analytic d_t - c(delta - s/2c)^2 = "
        f"{g_an*1e6:.6f} um at x = {(xb + s_front/(2*c))*1e3:.4f} mm")
    ok = err < 1e-12 and abs(F["back"] - F_an["back"]) < 1e-8 and abs(F["thread"] - F_an["thread"]) < 1e-8 \
        and abs(g_num - g_an) < 1e-10
    verdict("(i) EI=0 point thread vs analytic", ok,
            f"shape error {err:.1e} m, force errors {abs(F['back']-F_an['back']):.1e} / "
            f"{abs(F['thread']-F_an['thread']):.1e} N, crest gap error {abs(g_num-g_an)*1e6:.1e} um")


# ------------------------------------------------------------------------------------------------- (ii)
def test_ii():
    log("\n=== (ii) beam-string boundary layer at a point thread vs analytic Green's function ===")
    # bridge 20 mm below everything: the only contact is the point thread 0.2 mm above the datum
    pr = S.BridgeProfile.parabolic(c=0.0, x_crest=10e-3, length=20e-3, edge_radius=0.0, h_crest=-20e-3)
    res = []
    for st in (C3, KH):
        l = st.l
        su = S.Setup(pr, st, S.Thread(10e-3, 20.2e-3, point=True), phi_deg=5.0, psi_deg=3.0)
        (xT, yT), (xN, yN) = su.anchors()
        x0, ytop = 10e-3, 0.2e-3
        # analytic: lines y = A + (B +- F/2T)(x - x0) through the anchors, A = ytop + F l/(2T)
        # unknowns B, F:  yT = ytop + F l/2T + (B + F/2T)(xT - x0);  yN = ytop + F l/2T + (B - F/2T)(xN - x0)
        T = st.T
        M = np.array([[xT - x0, (l + (xT - x0)) / (2 * T)], [xN - x0, (l - (xN - x0)) / (2 * T)]])
        B, F = np.linalg.solve(M, [yT - ytop, yN - ytop])
        for mfac in (4.0, 1.5):
            so = S.solve(su, h=10e-6, front=False, margin=mfac * l)
            xx = so.x
            ya = ytop + F * l / (2 * T) + B * (xx - x0) - F / (2 * T) * np.abs(xx - x0) \
                - F * l / (2 * T) * np.exp(-np.abs(xx - x0) / l)
            err = float(np.max(np.abs(so.y - ya)))
            Fn = so.forces()["thread"]
            res.append((st.name, mfac, err, Fn, F, F * l / (2 * T)))
            log(f"  {st.name:22s} l = {l*1e3:.3f} mm, grid margin {mfac:.1f} l: max |y - y_an| = {err*1e9:8.3f} nm "
                f"(BL lift F l/2T = {F*l/(2*T)*1e6:.2f} um); thread force {Fn:.8f} N vs analytic {F:.8f} N")
    worst = max(r[2] for r in res)
    worst_rel = max(r[2] / r[5] for r in res)
    worst_def = max(r[2] for r in res if r[1] == 4.0)
    ok = worst_rel < 1e-4 and worst_def < 2e-9 and max(abs(r[3] - r[4]) / r[4] for r in res) < 1e-6
    verdict("(ii) beam-string boundary layer vs analytic", ok,
            f"max shape error {worst_def*1e9:.2f} nm with the default grid margin 4 l, {worst*1e9:.1f} nm "
            f"({worst_rel:.1e} of the BL lift) with a margin of only 1.5 l (far-field terms exact up to the O(h) "
            f"boundary-slope estimate); thread force to {max(abs(r[3] - r[4]) / r[4] for r in res):.0e}")


# ------------------------------------------------------------------------------------------------- (iii)
def test_iii():
    log("\n=== (iii) force balance: sum of contact forces vs T x total angle change ===")
    pr = nominal_profile()
    worst = 0.0
    for st in (C3, KH):
        for xc in (3.5e-3, 4.6e-3):
            su = setup(st, xc)
            so = S.solve(su, front=True)
            F = so.forces()
            tot = sum(F.values())
            BL, BR = so.slopes_far()
            (xT, yT), (xN, yN) = su.anchors()
            geo = st.T * (np.tan(np.deg2rad(su.phi_deg)) + np.tan(np.deg2rad(su.psi_deg)))
            e1 = abs(tot - st.T * (BL - BR))
            worst = max(worst, e1)
            log(f"  {st.name:22s} x_c={xc*1e3:.1f} mm: back {F['back']:.6f} + thread {F['thread']:.6f} + front "
                f"{F['front']:.6f} = {tot:.9f} N; T(s_tail - s_nut) = {st.T*(BL-BR):.9f} N (diff {e1:.1e}); "
                f"T(tan phi + tan psi) = {geo:.6f} N ({100*(tot/geo-1):+.3f}%: the anchors are not exactly at the "
                f"datum angles once the string sits on the bridge)")
    verdict("(iii) force balance", worst < 1e-8, f"sum of contact forces = T x total slope change to {worst:.1e} N")


# ------------------------------------------------------------------------------------------------- (iv)
def test_iv():
    log("\n=== (iv) grid convergence of the crest gap (h = 20 / 10 / 5 um) ===")
    pr = nominal_profile()
    worst = 0.0
    rows = []
    cases = [(C3, 4.0e-3, 10, None), (C3, 5.0e-3, 10, None), (C3.with_EI(0.0), 4.0e-3, 10, C3.name),
             (C3, 4.0e-3, 5, None), (KH, 4.0e-3, 10, None), (KH, 6.0e-3, 10, None),
             (KH.with_EI(0.0), 8.0e-3, 10, KH.name)]
    for st, xc, phi, nm in cases:
        su = setup(st, xc, phi=phi, name=nm)
        g = []
        for h in (20e-6, 10e-6, 5e-6):
            so = S.solve(su, h=h, front=False)
            g.append(so.crest_gap()[0])
        d = abs(g[1] - g[2])
        if np.isfinite(d):
            worst = max(worst, d)
        rows.append(d)
        log(f"  {st.name:22s} EI={st.EI:.2e} x_c={xc*1e3:.1f} mm phi={phi:2d}: crest gap {g[0]*1e6:10.4f} / "
            f"{g[1]*1e6:10.4f} / {g[2]*1e6:10.4f} um;  |g(10um) - g(5um)| = {d*1e6:.4f} um")
    verdict("(iv) grid convergence", worst < 0.05e-6 and all(np.isfinite(rows)),
            f"max |gap(h=10um) - gap(h=5um)| = {worst*1e6:.4f} um (criterion 0.05 um)")


# ------------------------------------------------------------------------------------------------- (v)
def test_v():
    log("\n=== (v) d(crest gap)/d(x_c) vs analytic 2 sqrt(c d_t) for EI -> 0 ===")
    pr = nominal_profile()
    c = 3.0
    out = []
    for st, lab, nm in ((C3.with_EI(0.0), "EI = 0", C3.name), (C3.with_EI(C3.EI * 1e-6), "EI x 1e-6", C3.name),
                        (C3, "C3 EI", None), (KH, "kharaj EI", None)):
        for dt in (0.1e-3, 0.2e-3, 0.3e-3):
            su = setup(st, 4e-3, dt=dt, name=nm)
            try:
                xg, rs = S.auto_place(su, 0.0, x_lo=0.4e-3, x_hi=12e-3, n_scan=25)
            except ValueError as e:
                log(f"  {lab:10s} d_t={dt*1e3:.1f} mm: no grazing position on this bridge ({str(e)[:60]}...)")
                continue
            dx = 2e-6
            gp = S.crest_gap(su.replace(x_c=xg + dx)); gm = S.crest_gap(su.replace(x_c=xg - dx))
            slope = (gp - gm) / (2 * dx)
            an = 2 * np.sqrt(c * dt)
            # effective thread height for the round thread: front line tangent to the circle
            cx, cy, r = S.thread_centre(pr, S.Thread(xg, dt))
            sF = rs.to.slopes_far()[1]
            d_eff = cy + r * np.sqrt(1 + sF * sF) - float(pr.top(np.array([xg]))[0])
            # finite nut distance D: the front line also tilts when the thread moves along the sloped bridge,
            # d g/d x_c = 2 sqrt(c d) (1 - u/D), u = sqrt(d/c) (thread -> graze distance)
            D = su.anchors()[1][0] - xg
            an_eff = 2 * np.sqrt(c * d_eff) * (1 - np.sqrt(d_eff / c) / D)
            out.append((lab, dt, slope, an, an_eff))
            log(f"  {lab:10s} d_t={dt*1e3:.1f} mm: grazing at x_c = {xg*1e3:.4f} mm; dg/dx_c = {slope*1e3:.4f} um/mm "
                f"vs 2 sqrt(c d_t) = {an*1e3:.4f} ({100*(slope/an-1):+.3f}%), vs 2 sqrt(c d_eff)(1 - u/D) = "
                f"{an_eff*1e3:.4f} um/mm (d_eff = {d_eff*1e6:.2f} um; {100*(slope/an_eff-1):+.3f}%)")
    e0 = [abs(o[2] / o[4] - 1) for o in out if o[0] in ("EI = 0", "EI x 1e-6")]
    eC = [100 * (o[2] / o[4] - 1) for o in out if o[0] == "C3 EI"]
    eK = [100 * (o[2] / o[4] - 1) for o in out if o[0] == "kharaj EI"]
    verdict("(v) dg/dx_c vs 2 sqrt(c d_t) (EI -> 0)", max(e0) < 1e-3,
            f"EI->0: within {100*max(e0):.3f}% of 2 sqrt(c d_eff)(1 - u/D) (d_eff = thread-top height of the front "
            f"tangent line = d_t + 0.2-0.4 um; u/D = finite-nut-distance correction, -0.6..-1.0%); with bending "
            f"stiffness the sensitivity departs from the EI=0 rule by {min(eC):+.0f}..{max(eC):+.0f}% (C3) and "
            f"{min(eK):+.0f}..{max(eK):+.0f}% (kharaj)")


# ------------------------------------------------------------------------------------------------- (vi)
def test_vi():
    log("\n=== (vi) tanpura1d static_init (modal, discrete scheme) vs fine-grid statics ===")
    pr = nominal_profile()
    su = setup(C3, 4e-3)
    xg, rs = S.auto_place(su, -2e-6, x_lo=3e-3, x_hi=5e-3)
    dyn = S.profile_for_dynamics(rs, h_floor=-0.3e-3)
    F_full = rs.full.forces()
    log(f"  configuration: C3, d_t 0.2 mm, phi 10 deg, thread auto-placed for crest gap -2 um at x_c = {xg*1e3:.4f} mm;"
        f" x_term = {dyn['x_term']*1e3:.4f} mm, xc_rel = {dyn['xc_rel']*1e3:.4f} mm, L_rel = {dyn['L_rel']:.6f} m")
    log(f"  (a) bridge-frame statics (rigid thread, real termination): front force {F_full['front']*1e3:.4f} mN, "
        f"thread force {dyn['F_thread']:.6f} -> {dyn['F_thread_full']:.6f} N "
        f"(change {(dyn['F_thread_full']-dyn['F_thread'])*1e3:+.4f} mN); front contact "
        f"{rs.summary()['front_contact']}")
    p = dict(T1.DEFAULT)
    p.update(dyn["t1"])
    fine = S.dynamic_frame_statics(C3.T, C3.EI, dyn["L_rel"], dyn["xc_rel"], p["wc"], p["kc"],
                                   dyn["x_rel"], dyn["h_rel"], h=10e-6)
    xa = fine["x_active"]
    log(f"  (b) fine-grid statics of the tanpura1d continuum problem (pinned at x_term, thread spring k_c = "
        f"{p['kc']:.1e} N/m on the 1 mm window, rigid bridge h_rel): bridge force {fine['F_bridge']*1e3:.4f} mN, "
        f"thread force change {fine['F_thread_change']*1e3:+.4f} mN, contact x_rel {xa.min()*1e3:.3f}-"
        f"{xa.max()*1e3:.3f} mm ({xa.size} nodes), iterations {fine['iters']}")
    rows = []
    for fmax, fsx in ((20000.0, 176400.0), (40000.0, 176400.0), (80000.0, 352800.0), (160000.0, 705600.0)):
        for dxb in (2e-4, 5e-5):
            pp = dict(p, fmax=fmax, fs=fsx, dxb=dxb, static_init=True)
            s = T1.build(pp)
            q, inf = T1.static_state(pp, s)
            # continuous-time modal statics (omega^2 instead of the scheme's omega*^2) for comparison
            s2 = dict(s, om2s=s["om"] ** 2)
            q2, inf2 = T1.static_state(pp, s2)
            rows.append((fmax, s["M"], dxb, inf["static_force"], inf["thread_force"], inf["max_pen"], inf["n_active"],
                         inf2["static_force"]))
            log(f"  (c) modal static, modes < {fmax/1e3:5.0f} kHz (M = {s['M']:4d}), bridge dx {dxb*1e3:.2f} mm: "
                f"bridge force {inf['static_force']*1e3:8.4f} mN ({100*(inf['static_force']/fine['F_bridge']-1):+6.1f}% vs "
                f"fine), thread force {inf['thread_force']*1e3:+8.4f} mN, max penetration {inf['max_pen']*1e9:.2f} nm, "
                f"{inf['n_active']} active point(s) at x_rel {np.round(inf['x_active']*1e3, 2)} mm; continuous-omega "
                f"modal: {inf2['static_force']*1e3:.4f} mN; Newton its {inf['iters']}, residual {inf['residual']:.1e}")
    e133 = rows[0][3] / fine["F_bridge"] - 1
    log(f"  modal-truncation error of the static bridge force with the default 133 modes: {100*e133:+.1f}%")
    log(f"  mapping error (a) vs (b): {100*(fine['F_bridge']/F_full['front']-1):+.1f}% (spring vs rigid thread, "
        f"pinned vs real termination)")
    verdict("(vi) modal static vs fine statics", True,
            f"default 133 modes: static bridge force {100*e133:+.0f}% vs fine-grid statics of the same continuum "
            f"problem (reported, not a pass/fail criterion); fine (b) vs bridge-frame (a): "
            f"{100*(fine['F_bridge']/F_full['front']-1):+.1f}%")


# ------------------------------------------------------------------------------------------------- (vii)
def test_vii():
    log("\n=== (vii) extras: tune='speaking', static_init click, energy with static_init ===")
    from validate1d import peak_freqs
    out = []
    for tune in (None, "speaking"):
        p, s, r = T1.simulate(dict(bridge=False, thread=True, lossless=True, rc=0.0, xe=0.137, we=0.003, Fe_rise=1e-4,
                                   Fe_fall=1e-4, Fe_amp=1.0, dur=4.0, tune=tune))
        f = peak_freqs(r["Fnut"], p["fs"], [p["f1"] * 1.005])[0]
        out.append(1200 * np.log2(f / p["f1"]))
        log(f"  tune={str(tune):9s}: T = {s['T']:.5f} N, lowest partial with thread spring {f:.5f} Hz "
            f"({out[-1]:+.4f} cents re f1 = {p['f1']} Hz)")
    clicks = []
    for si in (False, True):
        p, s, r = T1.simulate(dict(dur=0.2, h0=3e-6, Fe_amp=0.0, static_init=si))
        clicks.append(np.ptp(r["Fbridge"]))
        log(f"  crest pressed by 3 um, no pluck, static_init={si}: bridge-force peak-to-peak {clicks[-1]:.3e} N")
    p, s, r = T1.simulate(dict(dur=0.5, h0=3e-6, static_init=True, lossless=True, rc=0.0), energy=True)
    n0 = int(0.012 * p["fs"])
    drift = float(np.max(np.abs(r["H"][n0:] - r["H"][n0])) / np.max(r["H"][n0:]))
    p, s, r = T1.simulate(dict(dur=0.5, h0=3e-6, static_init=True), energy=True)
    bal = r["H"] + r["D"] - r["W"]
    resid = float(np.max(np.abs(bal - bal[0])) / np.max(r["H"]))
    log(f"  static_init + pluck: lossless energy drift {drift:.1e}, lossy balance residual {resid:.1e}")
    ok = abs(out[1]) < 0.01 and clicks[1] < 1e-9 and drift < 1e-9 and resid < 1e-9
    verdict("(vii) tune / static_init", ok, f"tuned pitch {out[1]:+.4f} cents (untuned {out[0]:+.2f}); start-up click "
            f"{clicks[0]:.2f} N -> {clicks[1]:.1e} N; energy drift {drift:.0e}, balance {resid:.0e}")


# ------------------------------------------------------------------------------------------------- (viii)
def test_viii():
    log("\n=== (viii) measured profile (CSV + smoothing spline) and power-law profile ===")
    ex = nominal_profile()
    xs_mm = np.arange(0.0, 40.0001, 0.25)
    h_um = ex.top(xs_mm * 1e-3) * 1e6
    rng = np.random.default_rng(1)
    su0 = setup(C3, 4e-3)
    x0, rs0 = S.auto_place(su0, 0.0, x_lo=3e-3, x_hi=5e-3)
    rows = []
    for noise, smooth in ((0.0, 0.0), (0.5, 0.5), (0.5, 0.0)):
        path = os.path.join(OUT, f"profile_test_noise{noise}.csv")
        hn = h_um + noise * rng.standard_normal(h_um.size)
        with open(path, "w") as fh:
            fh.write("x_mm,height_um\n")
            for a, b in zip(xs_mm, hn):
                fh.write(f"{a:.4f},{b:.5f}\n")
        pr = S.BridgeProfile.from_csv(path, smoothing=smooth, edge_radius=0.5e-3)
        su = S.Setup(pr, C3, S.Thread(x0, 0.2e-3), phi_deg=10.0, psi_deg=PSI[C3.name], L_speak=C3.L,
                     x_ref=ex.x_crest)
        g = S.crest_gap(su)
        try:
            x1, _ = S.auto_place(su, 0.0, x_lo=x0 - 0.5e-3, x_hi=x0 + 0.5e-3, n_scan=9)
        except ValueError:
            x1 = np.nan
        rows.append((noise, smooth, g, x1))
        log(f"  CSV profile, noise {noise} um rms, smoothing {smooth} um: crest gap at the exact grazing position "
            f"{g*1e6:+.3f} um (exact 0); re-auto-placed grazing x_c {x1*1e3:.4f} mm (exact {x0*1e3:.4f} mm, "
            f"shift {(x1-x0)*1e6:+.1f} um)")
    pw = S.BridgeProfile.power(c=3.0, p=4.0, x_crest=2e-3, x_ref=5e-3, length=40e-3, edge_radius=0.5e-3)
    su = S.Setup(pw, C3, S.Thread(4e-3, 0.2e-3), phi_deg=10.0, psi_deg=PSI[C3.name], L_speak=C3.L)
    try:
        xp, rsp = S.auto_place(su, 0.0, x_lo=0.4e-3, x_hi=12e-3, n_scan=25)
        dx = 2e-6
        sl = (S.crest_gap(su.replace(x_c=xp + dx)) - S.crest_gap(su.replace(x_c=xp - dx))) / (2 * dx)
        log(f"  power-law profile p = 4 (same drop as the parabola 5 mm from the crest), same psi: grazing at x_c = "
            f"{xp*1e3:.3f} mm (parabola: {x0*1e3:.3f} mm), graze point x = {rsp.x_graze*1e3:.2f} mm, "
            f"dg/dx_c = {sl*1e3:.1f} um/mm")
        okp = True
    except ValueError as e:
        log(f"  power-law profile: no grazing position ({e})")
        okp = False
    ok = abs(rows[0][2]) < 0.05e-6 and abs(rows[1][2]) < 2e-6 and okp
    verdict("(viii) measured-profile path", ok,
            f"noise-free CSV/spline reproduces the crest gap to {abs(rows[0][2])*1e6:.3f} um; 0.5 um rms profile noise "
            f"-> crest-gap error {rows[1][2]*1e6:+.2f} um with smoothing, {rows[2][2]*1e6:+.2f} um without; "
            f"power-law family runs")


if __name__ == "__main__":
    t0 = time.time()
    for t in (test_i, test_ii, test_iii, test_iv, test_v, test_vi, test_vii, test_viii):
        t1 = time.time()
        t()
        log(f"  [{time.time() - t1:.1f} s]")
    log("\n=== SUMMARY ===")
    for s_ in SUMMARY:
        log(s_)
    log(f"total {time.time() - t0:.0f} s")
    with open(os.path.join(OUT, "validate_statics.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")
