"""Study: rest-state statics of the tanpura string over bridge + thread, and what it does to the jawari.

Part 1 (statics, both strings: C3 / DAFx-16 and kharaj 0.7 mm steel):
  * one 'instrument' per string: parabolic bridge c = 3/m, crest 2 mm behind... (x_crest = 2 mm from the back
    face, surface falling towards the nut), rounded back edge r_e = 0.5 mm, tailpiece 0.15 m behind the bridge;
    the speaking-length angle psi (equivalently the bridge tilt) is DESIGNED so that the nominal set-up
    (d_t = 0.2 mm, phi = 10 deg, real bending stiffness) grazes with the thread at x_c = 4 mm;
  * on that fixed instrument: crest gap vs thread position for d_t in {0.1, 0.2, 0.3} mm, tail break angle
    phi in {2, 5, 10} deg, bending stiffness on / off (EI = 0);
  * auto-place: thread position for a target crest gap (0 = grazing), i.e. how a player sets the jawari.
Part 2 (dynamics, C3): thread swept in 0.05 mm steps around the auto-placed grazing position; each position
  gets its own rest state -> profile_for_dynamics -> tanpura1d with static_init, 3 s, plucks 0.2 N and 0.05 N.
  Compared with the earlier parabola-based sweep (study_thread_geometry.py: string pinned at x = 0, fixed
  75 um thread slid along the parabola, crest gap h0 = -3 d (10 - d) um, no static init).
Outputs: out/statics/*.txt, *.png, *.npz.   Run: cd jawari && python study_statics.py [statics|dynamics]
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

import common  # noqa: E402,F401
from common import OUT_ROOT, to_audio  # noqa: E402
from metrics import analyze  # noqa: E402
import statics as S  # noqa: E402

OUT = os.path.join(str(OUT_ROOT), "statics")
os.makedirs(OUT, exist_ok=True)
NPROC = 2
PROFILE = dict(c=3.0, x_crest=2e-3, length=40e-3, edge_radius=0.5e-3)
XC0 = 4e-3                      # nominal thread position used to design psi
DT_LIST = (0.1e-3, 0.2e-3, 0.3e-3)
PHI_LIST = (2.0, 5.0, 10.0)
XC_GRID = np.round(np.arange(0.25e-3, 12.0001e-3, 0.125e-3), 7)
STRINGS = {"C3": S.String.c3_dafx16(), "kharaj": S.String.kharaj()}
LOGL = []


def log(s=""):
    print(s, flush=True)
    LOGL.append(s)


def profile():
    return S.BridgeProfile.parabolic(**PROFILE)


def instrument(key, psi=None):
    st = STRINGS[key]
    su = S.Setup(profile(), st, S.Thread(XC0, 0.2e-3), phi_deg=10.0, psi_deg=0.0, L_speak=st.L)
    if psi is None:
        psi = S.design_psi(su, XC0)
    return su.replace(psi_deg=psi), psi


def _sweep_job(args):
    key, psi, dt, phi, ei_on = args
    su, _ = instrument(key, psi)
    st = su.string if ei_on else su.string.with_EI(0.0)
    su = su.replace(string=st, phi_deg=phi, d_t=dt)
    rows = []
    for xc in XC_GRID:
        so = S.solve(su.replace(x_c=xc), front=False)
        g, xg = so.crest_gap()
        gm, _ = so.min_front_clearance()
        F = so.forces()
        rows.append((xc, g, xg, gm, so.x_term(), F["thread"], F["back"], so.converged))
    return (key, dt, phi, ei_on), np.array(rows, float)


def crossings(xc, g, target=0.0):
    """x_c where the crest gap crosses target (linear interpolation between sweep points)."""
    out = []
    v = g - target
    for k in range(len(xc) - 1):
        if np.isfinite(v[k]) and np.isfinite(v[k + 1]) and np.sign(v[k]) != np.sign(v[k + 1]):
            out.append(xc[k] - v[k] * (xc[k + 1] - xc[k]) / (v[k + 1] - v[k]))
    return out


# ------------------------------------------------------------------------------------------------ part 1
def part_statics():
    t0 = time.time()
    log("=== Part 1: crest gap vs thread position (statics) ===")
    log(f"bridge: parabolic c = {PROFILE['c']}/m, crest at {PROFILE['x_crest']*1e3} mm from the back face, length "
        f"{PROFILE['length']*1e3} mm, back edge radius {PROFILE['edge_radius']*1e3} mm; tailpiece 0.15 m behind the "
        f"back face; nut at L from the crest")
    psis = {}
    for key, st in STRINGS.items():
        su, psi = instrument(key)
        psis[key] = psi
        rs = S.rest_state(su)
        sm = rs.summary()
        log(f"{key:7s} T = {st.T:.3f} N, EI = {st.EI:.3e} N m^2, l = sqrt(EI/T) = {st.l*1e3:.3f} mm, mu = {st.mu:.4e} kg/m")
        log(f"        designed speaking-length angle psi = {psi:.4f} deg (nominal d_t 0.2 mm, phi 10 deg, EI on, "
            f"grazing at x_c = {XC0*1e3} mm): x_term = {sm['x_term']*1e3:.4f} mm, graze at x = "
            f"{sm['x_graze']*1e3:.3f} mm, thread preload {sm['F_thread_TO']:.3f} N, back {sm['F_back_TO']:.3f} N")
    jobs = [(key, psis[key], dt, phi, ei) for key in STRINGS for dt in DT_LIST for phi in PHI_LIST
            for ei in (True, False)]
    with Pool(NPROC) as pool:
        res = dict(pool.map(_sweep_job, jobs))
    log(f"sweep: {len(jobs)} configurations x {len(XC_GRID)} thread positions in {time.time()-t0:.0f} s")
    np.savez(os.path.join(OUT, "crest_gap_sweep.npz"), xc=XC_GRID, psi=np.array([psis[k] for k in STRINGS]),
             keys=np.array([f"{k}|{dt}|{phi}|{ei}" for (k, dt, phi, ei) in res]),
             data=np.array([res[k] for k in res]),
             columns=np.array(["xc", "crest_gap", "x_graze", "min_front_clearance", "x_term", "F_thread", "F_back",
                               "converged"]))
    # --- table 1: grazing positions (auto-place on the sweep, refined with Brent)
    log("\n--- Table 1: grazing thread positions x_c* (crest gap = 0), auto-placed (Brent from the sweep brackets) ---")
    log("all crossings in 0.25-12 mm are listed: x_c* [mm] (back = string rests on the back of the bridge, LIFT = it "
        "hangs from the thread; dg/dx_c [um/mm]; thread preload [N])")
    T1rows = {}
    for key in STRINGS:
        for dt in DT_LIST:
            for phi in PHI_LIST:
                cells = []
                first = []
                for ei in (True, False):
                    d = res[(key, dt, phi, ei)]
                    cr = crossings(d[:, 0], d[:, 1])
                    if not cr:
                        cells.append("never (gap > 0 or no graze up to 12 mm)" if np.nanmin(d[:, 1]) > 0 else
                                     "never (pressed onto the bridge up to 12 mm)")
                        first.append(np.nan)
                        continue
                    su, _ = instrument(key, psis[key])
                    st = su.string if ei else su.string.with_EI(0.0)
                    su = su.replace(string=st, phi_deg=phi, d_t=dt)
                    items = []
                    for x0 in cr:
                        try:
                            xg, rs = S.auto_place(su, 0.0, x_lo=max(x0 - 0.13e-3, XC_GRID[0]), x_hi=x0 + 0.13e-3,
                                                  n_scan=5)
                        except ValueError:
                            xg, rs = x0, S.rest_state(su.replace(x_c=x0))
                        sm = rs.summary()
                        dx = 2e-6
                        sl = (S.crest_gap(su.replace(x_c=xg + dx)) - S.crest_gap(su.replace(x_c=xg - dx))) / (2 * dx)
                        items.append(f"{xg*1e3:6.3f} ({'back' if sm['back_contact'] else 'LIFT'}, {sl*1e3:5.1f}, "
                                     f"{sm['F_thread_TO']:5.2f})")
                        T1rows.setdefault((key, dt, phi, ei), []).append((xg, sm["back_contact"], sl, sm["F_thread_TO"]))
                    cells.append("; ".join(items))
                    first.append(T1rows[(key, dt, phi, ei)][0][0])
                log(f"{key:7s} d_t {dt*1e3:.1f} phi {phi:4.1f} | EI on: {cells[0]}")
                log(f"{'':22s}| EI=0 : {cells[1]}" + (f"   [EI moves the first graze by {(first[0]-first[1])*1e3:+.2f} mm]"
                                                     if np.isfinite(first[0] - first[1]) else ""))
    # --- table 2: crest gap at fixed thread positions
    log("\n--- Table 2: crest gap [um] at fixed thread positions: EI on vs EI = 0, and the break-angle effect ---")
    log("('nan' = no interior clearance minimum: the string rises away from the bridge in front of the thread; "
        "'*' = string lifted off the back of the bridge)")
    for key in STRINGS:
        for xcf in (3e-3, 4e-3, 6e-3):
            k = int(np.argmin(np.abs(XC_GRID - xcf)))
            log(f"  (grid point x_c = {XC_GRID[k]*1e3:.3f} mm)")
            for dt in DT_LIST:
                parts = []
                for phi in PHI_LIST:
                    on = res[(key, dt, phi, True)][k]
                    off = res[(key, dt, phi, False)][k]
                    parts.append(f"phi {phi:2.0f}: {on[1]*1e6:7.1f}{'*' if not np.isfinite(on[4]) else ' '} "
                                 f"(EI=0 {off[1]*1e6:7.1f}{'*' if not np.isfinite(off[4]) else ' '})")
                on2, on10 = res[(key, dt, 2.0, True)][k][1], res[(key, dt, 10.0, True)][k][1]
                on5 = res[(key, dt, 5.0, True)][k][1]
                log(f"{key:7s} x_c={xcf*1e3:.0f} mm d_t={dt*1e3:.1f}: " + " | ".join(parts)
                    + f" || EI shift (phi 10) {(on10 - res[(key, dt, 10.0, False)][k][1])*1e6:+7.1f} um; "
                      f"phi 5->10: {(on10-on5)*1e6:+6.1f} um, 2->10: {(on10-on2)*1e6:+6.1f} um")
    # --- analytic interpretation of the EI shift: far-field lift l * dtheta / 2 at the thread kink
    log("\n--- EI shift vs the single-kink estimate l*dtheta_thread/2 (dtheta from the EI = 0 solution) ---")
    for key, st in STRINGS.items():
        su, _ = instrument(key, psis[key])
        for xcf in (3e-3, 4e-3, 6e-3):
            so0 = S.solve(su.replace(string=st.with_EI(0.0), x_c=xcf), front=False)
            so1 = S.solve(su.replace(x_c=xcf), front=False)
            dth = so0.forces()["thread"] / st.T
            g0, g1 = so0.crest_gap()[0], so1.crest_gap()[0]
            log(f"{key:7s} x_c={xcf*1e3:.0f} mm: thread kink {np.rad2deg(dth):.2f} deg -> l*dtheta/2 = "
                f"{st.l*dth/2*1e6:7.1f} um; computed EI shift of the crest gap {(g1-g0)*1e6:7.1f} um; "
                f"termination-thread distance {(xcf - so1.x_term())*1e3:.2f} mm = {(xcf - so1.x_term())/st.l:.2f} l")
    plot_sweep(res, psis)
    plot_shapes(psis)
    with open(os.path.join(OUT, "study_statics_part1.txt"), "w") as fh:
        fh.write("\n".join(LOGL) + "\n")
    return psis


def plot_sweep(res, psis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {2.0: "#1f77b4", 5.0: "#ff7f0e", 10.0: "#2ca02c"}
    for key in STRINGS:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
        for ax, dt in zip(axs, DT_LIST):
            for phi in PHI_LIST:
                for ei, ls in ((True, "-"), (False, "--")):
                    d = res[(key, dt, phi, ei)]
                    x, g, xt = d[:, 0] * 1e3, d[:, 1] * 1e6, d[:, 4]
                    ax.plot(x, g, ls, color=cols[phi], lw=1.6 if ei else 1.0,
                            label=f"phi {phi:.0f} deg, {'EI on' if ei else 'EI = 0'}")
                    lift = ~np.isfinite(xt)
                    ax.plot(x[lift], g[lift], "o", mfc="none", color=cols[phi], ms=3)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_ylim(*((-300, 250) if key == "C3" else (-1100, 300)))
            ax.set_xlabel("thread position x_c [mm from back face]")
            ax.set_title(f"{key}: d_t = {dt*1e3:.1f} mm")
            ax.grid(alpha=0.3)
        axs[0].set_ylabel("crest gap [um]  (<0: string pressed onto bridge)")
        axs[0].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"{key}: crest gap of the thread-only rest shape (bridge c=3/m, psi = {psis[key]:.2f} deg); "
                     f"open circles: string lifted off the back of the bridge; gaps: no interior graze")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"crest_gap_vs_xc_{key}.png"), dpi=110)
        plt.close(fig)


def plot_shapes(psis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    for row, key in enumerate(STRINGS):
        su, _ = instrument(key, psis[key])
        st = su.string
        pr = su.profile
        for phi, col in ((2.0, "#1f77b4"), (5.0, "#ff7f0e"), (10.0, "#2ca02c")):
            for ei, ls in ((True, "-"), (False, "--")):
                so = S.solve(su.replace(string=st if ei else st.with_EI(0.0), phi_deg=phi), front=False)
                m = (so.x > -3e-3) & (so.x < 25e-3)
                axs[row, 0].plot(so.x[m] * 1e3, so.y[m] * 1e6, ls, color=col, lw=1.4 if ei else 0.9,
                                 label=f"phi {phi:.0f}, {'EI on' if ei else 'EI=0'}")
                xg, g = so.front_clearance()
                axs[row, 1].plot(xg * 1e3, g * 1e6, ls, color=col, lw=1.4 if ei else 0.9)
        xs = np.linspace(-3e-3, 25e-3, 3000)
        bb = pr.surface(xs)
        axs[row, 0].plot(xs * 1e3, np.where(np.isfinite(bb), bb, np.nan) * 1e6, "k", lw=2, label="bridge")
        cx, cy, r = S.thread_centre(pr, su.thread)
        th = np.linspace(0, 2 * np.pi, 200)
        axs[row, 0].plot((cx + r * np.cos(th)) * 1e3, (cy + r * np.sin(th)) * 1e6, "k", lw=1)
        axs[row, 0].set_ylim(-600, 400)
        axs[row, 0].set_title(f"{key}: thread-only rest shape, thread d_t=0.2 mm at x_c={XC0*1e3} mm "
                              f"(l = {st.l*1e3:.2f} mm)")
        axs[row, 0].legend(fontsize=7, ncol=2)
        axs[row, 1].set_ylim(-150, 400)
        axs[row, 1].axhline(0, color="k", lw=0.8)
        axs[row, 1].set_title(f"{key}: front clearance g = y - b")
        for ax in axs[row]:
            ax.grid(alpha=0.3)
            ax.set_xlabel("x [mm from back face]")
        axs[row, 0].set_ylabel("height [um] (vertical scale exaggerated)")
        axs[row, 1].set_ylabel("clearance [um]")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rest_shapes.png"), dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ part 2
D_LIST = np.round(np.arange(-0.30, 0.3001, 0.05), 2)
AMPS = (0.2, 0.05)


def _mean_buzz(y, fs):
    m = analyze(to_audio(y, fs), 44100)
    sel = (m["t"] > 0.3) & (m["t"] < 2.8)
    return float(np.mean(m["buzz_db"][sel])), float(np.mean(m["hf_frac"][sel]))


def _dyn_job(args):
    kind, d, amp, psi, xg = args
    import tanpura1d as T1
    if kind == "new":
        su, _ = instrument("C3", psi)
        rs = S.rest_state(su.replace(x_c=xg + d * 1e-3))
        dyn = S.profile_for_dynamics(rs, h_floor=-0.3e-3)
        over = dict(dyn["t1"], dur=3.0, static_init=True, Fe_amp=amp)
        p, s, r = T1.simulate(over)
        st = r["stats"].get("static") or {}
        info = dict(gap=dyn["crest_gap"], F_thread=dyn["F_thread"], F_static=st.get("static_force", 0.0),
                    Nb=int(s["Nb"]), xc_rel=dyn["xc_rel"], L_rel=dyn["L_rel"])
    else:
        xc = (5 + d) * 1e-3
        h0 = -3 * d * (10 - d) * 1e-6
        p, s, r = T1.simulate(dict(dur=3.0, xc=xc, h0=h0, Fe_amp=amp, static_init=(kind == "old_si")))
        info = dict(gap=-h0, F_thread=np.nan, F_static=np.nan, Nb=int(s["Nb"]), xc_rel=xc, L_rel=1.0)
    bn, hn = _mean_buzz(r["Fnut"], p["fs"])
    bb, hb = _mean_buzz(r["Fbridge"], p["fs"])
    stt = r["stats"]
    return (kind, d, amp), dict(info, buzz_nut=bn, hf_nut=hn, buzz_bridge=bb, hf_bridge=hb,
                                contact=stt["contact_frac"], edge_pen=stt["edge_pen"], fail=stt["newton_fail"],
                                max_Fc=stt["max_Fc"], cpu=r["cpu"])


def part_dynamics(psi=None):
    t0 = time.time()
    LOGL.clear()
    log("\n=== Part 2: dynamic thread sweep (C3, 3 s, nut force; mean buzz 0.3-2.8 s) ===")
    su, psi = instrument("C3", psi)
    xg, rs = S.auto_place(su, 0.0, x_lo=3e-3, x_hi=5e-3)
    log(f"C3 instrument psi = {psi:.4f} deg; auto-placed grazing thread position x_c* = {xg*1e3:.4f} mm "
        f"(bridge frame); x_term = {rs.x_term*1e3:.4f} mm")
    jobs = [(kind, d, amp, psi, xg) for kind in ("new", "old", "old_si") for amp in AMPS for d in D_LIST]
    with Pool(NPROC) as pool:
        res = dict(pool.map(_dyn_job, jobs))
    log(f"{len(jobs)} runs in {time.time()-t0:.0f} s")
    log("new = statics-derived profile (finite thread d_t 0.2 mm, phi 10 deg, EI, real termination) + static_init; "
        "old = study_thread_geometry.py (parabola about a straight line, fixed 75 um thread, NO static init: a pressed "
        "crest starts out of equilibrium with a click); old+si = old geometry started from its static equilibrium")
    log("   d [mm] | new: crest gap um, thread preload N, static bridge force mN | buzz 0.2 N / 0.05 N (nut) | "
        "buzz (bridge force) 0.2 N || old: crest gap um | buzz 0.2 / 0.05 N || old+si: buzz 0.2 / 0.05 N")
    for d in D_LIST:
        n2, n5 = res[("new", d, 0.2)], res[("new", d, 0.05)]
        o2, o5 = res[("old", d, 0.2)], res[("old", d, 0.05)]
        q2, q5 = res[("old_si", d, 0.2)], res[("old_si", d, 0.05)]
        log(f"  {d:+.2f}  | {n2['gap']*1e6:+7.2f} {n2['F_thread']:6.3f} {n2['F_static']*1e3:7.3f} | "
            f"{n2['buzz_nut']:7.1f} / {n5['buzz_nut']:7.1f} | {n2['buzz_bridge']:7.1f} || "
            f"{o2['gap']*1e6:+7.2f} | {o2['buzz_nut']:7.1f} / {o5['buzz_nut']:7.1f} || "
            f"{q2['buzz_nut']:7.1f} / {q5['buzz_nut']:7.1f}")
    for kind in ("new", "old", "old_si"):
        for amp in AMPS:
            b = [res[(kind, d, amp)]["buzz_nut"] for d in D_LIST]
            sel = np.abs(D_LIST) <= 0.1001
            log(f"  {kind} {amp:.2f} N: buzz range over +-0.3 mm {max(b)-min(b):5.1f} dB, over +-0.1 mm "
                f"{np.max(np.array(b)[sel]) - np.min(np.array(b)[sel]):5.1f} dB")
    ep = max(v["edge_pen"] for k, v in res.items() if k[0] == "new")
    epo = max(v["edge_pen"] for k, v in res.items() if k[0] == "old")
    eps = max(v["edge_pen"] for k, v in res.items() if k[0] == "old_si")
    nf = sum(v["fail"] for v in res.values())
    mfc = max(v["max_Fc"] for k, v in res.items() if k[0] == "new")
    pre = min(v["F_thread"] for k, v in res.items() if k[0] == "new")
    log(f"  checks: max edge penetration new {ep:.1e} m / old {epo:.1e} m (start-up transient of the pressed, "
        f"non-equilibrium start) / old+si {eps:.1e} m; Newton failures {nf}; max dynamic thread force {mfc:.3f} N "
        f"vs min static thread preload {pre:.2f} N (bilateral thread valid: {mfc < pre}); bridge points per run "
        f"{sorted(set(v['Nb'] for k, v in res.items() if k[0] == 'new'))}; cpu/run "
        f"{np.mean([v['cpu'] for v in res.values()]):.1f} s")
    np.savez(os.path.join(OUT, "dynamic_sweep.npz"), d=D_LIST, amps=np.array(AMPS),
             new_gap=np.array([res[("new", d, 0.2)]["gap"] for d in D_LIST]),
             old_gap=np.array([res[("old", d, 0.2)]["gap"] for d in D_LIST]),
             **{f"{k}_{a}_{q}": np.array([res[(k, d, a)][q] for d in D_LIST]) for k in ("new", "old", "old_si")
                for a in AMPS for q in ("buzz_nut", "buzz_bridge", "hf_nut")})
    plot_dyn(res)
    with open(os.path.join(OUT, "study_statics_part2.txt"), "w") as fh:
        fh.write("\n".join(LOGL) + "\n")


def plot_dyn(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    names = {"new": "statics-derived + static_init (new)", "old": "old parabola sweep, no static init",
             "old_si": "old parabola sweep + static_init"}
    for kind, mk in (("new", "o-"), ("old", "s:"), ("old_si", "^--")):
        for amp, col in ((0.2, "#1f77b4"), (0.05, "#d62728")):
            b = [res[(kind, d, amp)]["buzz_nut"] for d in D_LIST]
            g = [res[(kind, d, amp)]["gap"] * 1e6 for d in D_LIST]
            lab = f"{names[kind]}, {amp} N"
            axs[0].plot(D_LIST, b, mk, color=col, label=lab, ms=4)
            axs[1].plot(g, b, mk, color=col, label=lab, ms=4)
    axs[0].set_xlabel("thread shift from grazing position [mm]")
    axs[1].set_xlabel("crest gap [um]  (<0: pressed onto the bridge)")
    for ax in axs:
        ax.set_ylabel("mean buzz 0.3-2.8 s [dB] (nut force)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("C3 thread sweep: buzz vs thread position and vs crest gap (old sweep: gap = -h0)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dynamic_sweep.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["statics", "dynamics"]
    psis = None
    if "statics" in which:
        psis = part_statics()
    if "dynamics" in which:
        part_dynamics(psis["C3"] if psis else None)
