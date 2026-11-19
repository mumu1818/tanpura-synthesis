"""INDEPENDENT cross-check of statics.py (judge's script; not used by the models).

Thread-only rest state of a small-slope beam-string over the bridge, solved ANALYTICALLY: exact piecewise solutions
of EI y'''' - T y'' = 0 on the three free spans (tailpiece -> back-edge tangent point -> thread tangent point -> nut,
pinned ends, all exponentials kept), with value and slope matched to the obstacle at the two tangent points; the two
unknown tangent positions are found by Newton (fsolve) on curvature continuity (no point moments).  EI = 0: an upper
convex hull coded from scratch.  Geometry (parabola, edge circle tangent to the vertical back face, lowest
non-penetrating thread circle, tail/nut anchors) is recomputed from the documented definitions; statics.py is only
called for the comparison.

    cd jawari && python crosscheck_statics.py      (~30 s; writes out/statics/crosscheck_statics.txt)
"""
import os
import time

import numpy as np
from scipy.optimize import brentq, fsolve, minimize_scalar

import common  # noqa: F401  (sys.path, OUT_ROOT)
from common import OUT_ROOT
import presets as P
import statics as S


class Geo:
    def __init__(s, c, xcr, blen, re, xc, dt, phi, psi, Ltail, Lspeak):
        s.c, s.xcr, s.blen, s.re = c, xcr, blen, re
        s.b = lambda x: -c * (np.asarray(x, float) - xcr) ** 2
        s.db = lambda x: -2 * c * (np.asarray(x, float) - xcr)

        def F(xa):      # edge circle centre (re, ye): tangent to x = 0 and to the parabola at x_a
            bp = s.db(xa)
            return xa + re * bp / np.sqrt(1 + bp * bp) - re
        s.xa = brentq(F, 0.0, 2 * re, xtol=1e-16)
        bp = s.db(s.xa)
        s.ye = s.b(s.xa) - re / np.sqrt(1 + bp * bp)
        s.xe0 = re
        r = dt / 2                          # thread circle: lowest centre over x_c not penetrating the parabola
        s.r, s.cx = r, xc
        u = np.linspace(-r, r, 400001)
        v = s.b(xc + u) + np.sqrt(np.maximum(r * r - u * u, 0))
        k = int(np.argmax(v))
        res = minimize_scalar(lambda uu: -(s.b(xc + uu) + np.sqrt(max(r * r - uu * uu, 0))),
                              bounds=(u[max(k - 2, 0)], u[min(k + 2, u.size - 1)]), method="bounded",
                              options=dict(xatol=1e-14))
        s.cy = max(-res.fun, v[k])
        s.xT, s.yT = -Ltail, float(s.b(0.0)) - Ltail * np.tan(np.deg2rad(phi))
        s.xN, s.yN = xcr + Lspeak, float(s.b(xcr)) - Lspeak * np.tan(np.deg2rad(psi))

    def surf(s, x):
        x = np.asarray(x, float)
        out = np.where(x < s.xa, s.ye + np.sqrt(np.maximum(s.re ** 2 - (x - s.xe0) ** 2, 0)), s.b(x))
        return np.where((x >= 0) & (x <= s.blen), out, -np.inf)

    def obsB(s, x):
        if x < s.xa:
            w = np.sqrt(s.re ** 2 - (x - s.xe0) ** 2)
            return s.ye + w, -(x - s.xe0) / w
        return float(s.b(x)), float(s.db(x))

    def obsT(s, x):
        w = np.sqrt(s.r ** 2 - (x - s.cx) ** 2)
        return s.cy + w, -(x - s.cx) / w


def basis(x, a, bnd, l):
    """value/slope/curvature/third-derivative rows of A + B(x-a) + C e^{-(x-a)/l} + E e^{-(b-x)/l}."""
    e1 = np.exp(-(x - a) / l)
    e2 = np.exp(-(bnd - x) / l)
    return (np.array([1, x - a, e1, e2]), np.array([0, 1, -e1 / l, e2 / l]),
            np.array([0, 0, e1 / l ** 2, e2 / l ** 2]), np.array([0, 0, -e1 / l ** 3, e2 / l ** 3]))


def solve_beam(g, T, EI, x1, x2, full=False):
    l = np.sqrt(EI / T)
    spans = [(g.xT, x1), (x1, x2), (x2, g.xN)]
    Mtx = np.zeros((12, 12))
    rhs = np.zeros(12)
    rows = []

    def put(span, row, val):
        k = len(rows)
        Mtx[k, 4 * span:4 * span + 4] = row
        rhs[k] = val
        rows.append(k)
    B = [lambda x, k=k: basis(x, spans[k][0], spans[k][1], l) for k in range(3)]
    put(0, B[0](g.xT)[0], g.yT)
    put(0, B[0](g.xT)[2], 0.0)
    o1, s1 = g.obsB(x1)
    o2, s2 = g.obsT(x2)
    for (xk, ok, sk, kl) in ((x1, o1, s1, 0), (x2, o2, s2, 1)):
        put(kl, B[kl](xk)[0], ok)
        put(kl + 1, B[kl + 1](xk)[0], ok)
        put(kl, B[kl](xk)[1], sk)
        put(kl + 1, B[kl + 1](xk)[1], sk)
    put(2, B[2](g.xN)[0], g.yN)
    put(2, B[2](g.xN)[2], 0.0)
    coef = np.linalg.solve(Mtx, rhs)
    cf = [coef[4 * k:4 * k + 4] for k in range(3)]
    res = np.array([B[0](x1)[2] @ cf[0] - B[1](x1)[2] @ cf[1], B[1](x2)[2] @ cf[1] - B[2](x2)[2] @ cf[2]])
    if not full:
        return res * l
    F1 = EI * (B[1](x1)[3] @ cf[1] - B[0](x1)[3] @ cf[0])     # point force = EI x jump of y'''
    F2 = EI * (B[2](x2)[3] @ cf[2] - B[1](x2)[3] @ cf[1])

    def y(x):
        x = np.asarray(x, float)
        out = np.empty_like(x)
        for k, (a, bb) in enumerate(spans):
            m = ((x >= a) if k == 0 else (x > a)) & (x <= bb)
            out[m] = cf[k][0] + cf[k][1] * (x[m] - a) + cf[k][2] * np.exp(-(x[m] - a) / l) + \
                cf[k][3] * np.exp(-(bb - x[m]) / l)
        return out
    return y, F1, F2, res


def thread_only(g, T, EI):
    sol, info, ier, msg = fsolve(lambda z: solve_beam(g, T, EI, z[0] * 1e-3, z[1] * 1e-3),
                                 [0.5 * g.xa * 1e3, (g.cx - 0.3 * g.r) * 1e3], xtol=1e-14, full_output=True)
    x1, x2 = sol * 1e-3
    y, F1, F2, res = solve_beam(g, T, EI, x1, x2, full=True)
    return dict(x1=x1, x2=x2, y=y, F_back=F1, F_thread=F2, res=res, ier=ier)


def crest_gap_curve(g, y):
    x = np.linspace(g.cx + g.r, g.blen, 400001)
    gg = y(x) - g.b(x)
    k = np.where((gg[1:-1] <= gg[:-2]) & (gg[1:-1] < gg[2:]))[0] + 1
    if k.size == 0:
        return np.nan, np.nan
    k = k[np.argmin(gg[k])]
    return float(gg[k]), float(x[k])


def hull_EI0(g):
    """EI = 0: upper convex hull of anchors + obstacle (edge arc + parabola up to the thread + thread arc)."""
    xs = np.concatenate([np.linspace(0, g.xa, 20001), np.linspace(g.xa, g.cx - g.r, 20001)])
    ys = g.surf(xs)
    th = np.linspace(g.cx - g.r, g.cx + g.r, 200001)
    yt = np.maximum(g.cy + np.sqrt(np.maximum(g.r ** 2 - (th - g.cx) ** 2, 0)), g.b(th))
    Pts = np.concatenate([[[g.xT, g.yT]], np.stack([xs, ys], 1), np.stack([th, yt], 1), [[g.xN, g.yN]]])
    Pts = Pts[np.argsort(Pts[:, 0], kind="stable")]
    H = []
    for q in Pts:
        while len(H) >= 2:
            (x0, y0), (x1, y1) = H[-2], H[-1]
            if (x1 - x0) * (q[1] - y0) - (y1 - y0) * (q[0] - x0) >= 0:
                H.pop()
            else:
                break
        H.append(tuple(q))
    H = np.array(H)
    return lambda x: np.interp(x, H[:, 0], H[:, 1])


LINES = []


def say(s):
    print(s, flush=True)
    LINES.append(s)


def compare(label, c, xcr, blen, re, xc, dt, phi, psi, Ltail, Lspeak, T, EI, h=10e-6):
    g = Geo(c, xcr, blen, re, xc, dt, phi, psi, Ltail, Lspeak)
    pr = S.BridgeProfile.parabolic(c=c, x_crest=xcr, length=blen, edge_radius=re)
    su = S.Setup(pr, S.String(T=T, EI=EI, mu=1.0), S.Thread(xc, dt), phi_deg=phi, psi_deg=psi, L_tail=Ltail,
                 L_speak=Lspeak)
    t0 = time.time()
    sol = S.solve(su, h=h, front=False)
    ts = time.time() - t0
    gS, xgS = sol.crest_gap()
    fS = sol.forces()
    cx, cy, r = sol.thread_c
    say(f"== {label}: T={T:.4f} N EI={EI:.4e} l={np.sqrt(EI / T) * 1e3 if EI > 0 else 0:.3f} mm xc={xc * 1e3:.4f} mm "
        f"psi={psi:.4f} phi={phi:g}")
    say(f"   geometry: thread centre y {cy * 1e6:.6f} (statics) vs {g.cy * 1e6:.6f} um; edge centre y "
        f"{pr.edge_c[1] * 1e6:.6f} vs {g.ye * 1e6:.6f} um")
    m = (sol.x > 0) & (sol.x < blen)
    if EI > 0:
        a = thread_only(g, T, EI)
        gA, xgA = crest_gap_curve(g, a["y"])
        dy = sol.y[m] - a["y"](sol.x[m])
        xx = np.linspace(1e-7, g.cx - g.r, 20001)
        say(f"   tangent points: edge {a['x1'] * 1e6:.3f} um (statics x_term {sol.x_term() * 1e6:.3f}), thread "
            f"{a['x2'] * 1e6:.3f} um (statics {sol.info['x_tangent_thread'] * 1e6:.3f}); Newton ier={a['ier']} "
            f"residual {np.abs(a['res']).max():.1e}")
        say(f"   forces: back {a['F_back']:.6f} N (statics {fS['back']:.6f}), thread {a['F_thread']:.6f} N "
            f"(statics {fS['thread']:.6f}){'  LIFT-OFF: analytic assumption invalid' if a['F_back'] < 0 else ''}")
        say(f"   feasibility of the analytic shape: min(y - bridge) on (0, thread) = "
            f"{np.min(a['y'](xx) - g.surf(xx)) * 1e6:.3f} um (>= 0 required)")
        say(f"   crest gap: statics {gS * 1e6:+.4f} um at {xgS * 1e3:.3f} mm | analytic {gA * 1e6:+.4f} um at "
            f"{xgA * 1e3:.3f} mm | diff {(gS - gA) * 1e6:+.4f} um; shape max |dy| on the bridge "
            f"{np.abs(dy).max() * 1e6:.4f} um ({ts:.2f} s)")
        return gS, gA
    yh = hull_EI0(g)
    xx = np.linspace(g.cx + g.r, blen, 400001)
    gg = yh(xx) - g.b(xx)
    k = np.where((gg[1:-1] <= gg[:-2]) & (gg[1:-1] < gg[2:]))[0] + 1
    gA = gg[k[np.argmin(gg[k])]] if k.size else np.nan
    say(f"   EI=0 hull: crest gap statics {gS * 1e6:+.4f} um vs independent {gA * 1e6:+.4f} um; shape max |dy| "
        f"{np.abs(sol.y[m] - yh(sol.x[m])).max() * 1e6:.4f} um")
    return gS, gA


def main():
    G = P.KHARAJ_GEOM
    t1, info = P.kharaj_geometry("bronze")          # the bronze kharaj preset (used by render_demo.py)
    d = G["d"]
    EI = 1.1e11 * np.pi * d ** 4 / 64
    T = info["T_statics"]
    base = [G["c"], G["x_crest"], G["bridge_len"], G["edge_radius"], info["x_c_graze"], G["d_t"], G["phi_deg"],
            info["psi_deg"], G["L_tail"], G["L_speak"]]
    a1 = compare("bronze kharaj preset, grazing x_c, EI on", *base, T, EI)
    a0 = compare("bronze kharaj preset, grazing x_c, EI = 0", *base, T, 0.0)
    say(f"   ==> EI-induced crest-gap shift (bronze preset): statics {(a1[0] - a0[0]) * 1e6:+.2f} um, independent "
        f"{(a1[1] - a0[1]) * 1e6:+.2f} um")
    for sh in (-0.3e-3, -0.1e-3, 0.1e-3, 0.3e-3):
        b2 = list(base)
        b2[4] = info["x_c_graze"] + sh
        compare(f"bronze kharaj, thread {sh * 1e3:+.1f} mm, EI on", *b2, T, EI)
    st = S.String.kharaj()
    geo = (3.0, 2e-3, 40e-3, 0.5e-3, 4e-3, 0.2e-3, 10.0, 6.918, 0.15, 1.05)
    a1 = compare("steel kharaj, STATICS.md design point, EI on", *geo, st.T, st.EI)
    a0 = compare("steel kharaj, STATICS.md design point, EI = 0", *geo, st.T, 0.0)
    say(f"   ==> EI-induced crest-gap shift (steel, x_c 4 mm): statics {(a1[0] - a0[0]) * 1e6:+.2f} um, independent "
        f"{(a1[1] - a0[1]) * 1e6:+.2f} um")
    c3 = S.String.c3_dafx16()
    geo = [3.0, 2e-3, 40e-3, 0.5e-3, 4e-3, 0.2e-3, 10.0, 4.052, 0.15, 1.0]
    a1 = compare("C3 design point, EI on", *geo, c3.T, c3.EI)
    a0 = compare("C3 design point, EI = 0", *geo, c3.T, 0.0)
    say(f"   ==> EI-induced crest-gap shift (C3, x_c 4 mm): statics {(a1[0] - a0[0]) * 1e6:+.2f} um, independent "
        f"{(a1[1] - a0[1]) * 1e6:+.2f} um")
    geo[6] = 5.0
    a5 = compare("C3 design point, phi = 5 deg, EI on", *geo, c3.T, c3.EI)
    say(f"   ==> C3 break-angle leak phi 5 -> 10 deg: statics {(a1[0] - a5[0]) * 1e6:+.2f} um, independent "
        f"{(a1[1] - a5[1]) * 1e6:+.2f} um")
    os.makedirs(os.path.join(str(OUT_ROOT), "statics"), exist_ok=True)
    with open(os.path.join(str(OUT_ROOT), "statics", "crosscheck_statics.txt"), "w") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
