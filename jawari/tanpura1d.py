"""1-D (single, bridge-normal polarisation) tanpura string model with jawari.

Model (after van Walstijn et al., DAFx-16 / DAFx-17; see README.md):
  * stiff, damped string, simply supported at x=0 (bridge-side termination) and x=L (nut),
    modal expansion y(x,t) = sum_i q_i(t) sin(beta_i x), beta_i = i pi / L, all modes < fmax;
  * exact modal frequencies and decay rates (exact-pole two-step recursion, explicit-stiffness form);
  * thread (jiva) at x_c: bilateral linear spring-damper (k_c, r_c) on a sin^2 window of width w_c;
  * bridge: rigid curved barrier h_b(x) = h0 - curv (x - x_b)^2, crest at x_b grazing the string at rest,
    DISTRIBUTED one-sided power-law penalty contact  f(x) = k_b * max(h_b(x) - y(x), 0)^alpha  [N/m],
    sampled at N_b points with spacing dx_b;
  * discrete-gradient (energy-conserving) contact force, solved every sample by Newton-Raphson on the
    active contact points (thread force eliminated analytically);
  * output: total vertical force the string exerts on the bridge assembly (termination + thread + barrier),
    plus nut force, contact diagnostics and (optionally) the discrete energy balance.

All of the time loop runs inside one numba @njit function (_run).
"""
from __future__ import annotations

import time
import numpy as np
from numba import njit

DEFAULT = dict(
    # string (DAFx-16 Table 1, C3 steel string)
    L=1.0, rho=7850.0, A=6.16e-8, E=2.0e11, I=3.02e-16, f1=130.81, EI_scale=1.0,
    # discretisation: modes below 20 kHz (133 modes), integrated at 4 x 44.1 kHz.  4x oversampling moves the
    # unresolved stiff-contact oscillation of the energy-conserving scheme (which sits at fs/4) out of the
    # audio band; output is decimated to 44.1 kHz (see validate.py, test C).
    fs=176400.0, fmax=20000.0, M=None,
    # losses (Woodhouse form; values are engineering defaults, not from the papers)
    etaF=5e-5, etaA=1.2, etaB=1e-2, lossless=False,
    # thread (jiva): bilateral spring-damper
    thread=True, xc=0.005, wc=0.001, kc=1.2e5, rc=0.5,
    # bridge: curved rigid barrier, distributed power-law penalty
    bridge=True, xb=0.010, h0=0.0, curv=3.0, kb=1e11, alpha=1.0,
    br_x0=None, br_x1=None, dxb=2e-4,
    # excitation: force density psi_e(x) F_e(t)
    xe=0.5, we=1.0, Fe_amp=0.2, Fe_rise=0.010, Fe_fall=0.001,
    # displacement output point
    x_out=0.5,
    # Newton
    tol=1e-18, maxit=60, guess=True,
    dur=3.0,
)


def sin2win(x, xc, w):
    """sin^2 window of total width w centred at xc, unit integral."""
    out = np.zeros_like(x)
    m = np.abs(x - xc) < w / 2
    out[m] = (2.0 / w) * np.sin(np.pi * (x[m] - xc + w / 2) / w) ** 2
    return out


def project(L, beta, xc, w, npts=8001):
    """g_i = int psi(x) sin(beta_i x) dx for the sin^2 window psi (trapezoidal quadrature)."""
    lo, hi = max(xc - w / 2, 0.0), min(xc + w / 2, L)
    x = np.linspace(lo, hi, npts)
    psi = sin2win(x, xc, w)
    return np.trapezoid(psi[None, :] * np.sin(beta[:, None] * x[None, :]), x, axis=1)


def build(p):
    L = p["L"]
    mu = p["rho"] * p["A"]
    EI = p["E"] * p["I"] * p["EI_scale"]
    f1 = p["f1"]
    # tension such that mode 1 is exactly f1 including stiffness
    T = mu * (2 * L * f1) ** 2 - EI * (np.pi / L) ** 2
    fs = p["fs"]; dt = 1.0 / fs
    ii = np.arange(1, 5000)
    beta = ii * np.pi / L
    om = np.sqrt((T * beta ** 2 + EI * beta ** 4) / mu)
    if p.get("M"):
        M = int(p["M"])
    else:
        M = int(np.sum(om / (2 * np.pi) < p["fmax"]))
    if om[M - 1] / (2 * np.pi) >= 0.5 * fs:
        raise ValueError("highest mode above Nyquist")
    ii, beta, om = ii[:M], beta[:M], om[:M]
    if p["lossless"]:
        sig = np.zeros(M)
    else:
        eta = (T * (p["etaF"] + p["etaA"] / om) + EI * p["etaB"] * beta ** 2) / (T + EI * beta ** 2)
        sig = om * eta / 2
    omd = np.sqrt(np.maximum(om ** 2 - sig ** 2, 0.0))
    a = 2 * np.exp(-sig * dt) * np.cos(omd * dt)
    b = np.exp(-2 * sig * dt)
    m = mu * L / 2
    c = dt ** 2 * (1 + b) / (2 * m)
    om2s = (2 - 2 * a / (1 + b)) / dt ** 2          # omega*^2 of the equivalent energy-consistent scheme
    sigs = (1 - b) / ((1 + b) * dt)                   # sigma* of the equivalent scheme
    ge = project(L, beta, p["xe"], p["we"])
    if p["thread"]:
        gc = project(L, beta, p["xc"], p["wc"])
        kc, rc = p["kc"], p["rc"]
    else:
        gc = np.zeros(M); kc = 0.0; rc = 0.0
    kap = 0.5 * kc + rc / (2 * dt)
    lam = 0.5 * kc - rc / (2 * dt)
    thcc = float(np.sum(c * gc * gc))
    # bridge points
    xb = p["xb"]
    if p["bridge"]:
        x0 = p["br_x0"] if p["br_x0"] is not None else (max(p["xc"], xb - 0.006) if p["thread"] else xb - 0.006)
        x1 = p["br_x1"] if p["br_x1"] is not None else xb + 0.006
        Nb = int(round((x1 - x0) / p["dxb"])) + 1
        xj = np.linspace(x0, x1, Nb)
        dxb = xj[1] - xj[0]
        # make sure the crest itself is one of the points
        k0 = int(np.argmin(np.abs(xj - xb))); xj = xj + (xb - xj[k0])
        hj = p["h0"] - p["curv"] * (xj - xb) ** 2
        Phi = np.ascontiguousarray(np.sin(xj[:, None] * beta[None, :]))   # (Nb, M)
        v = Phi @ (c * gc)                                               # (Nb,)
        Q = dxb * (Phi * c[None, :]) @ Phi.T
        Qp = Q - kap * dxb * np.outer(v, v) / (1 + kap * thcc)
        Qp = np.ascontiguousarray(0.5 * (Qp + Qp.T))
    else:
        xj = np.array([xb]); dxb = 1.0; hj = np.array([-1.0]); Nb = 1
        Phi = np.zeros((1, M)); v = np.zeros(1); Qp = np.zeros((1, 1))
    w0 = T * beta + EI * beta ** 3                       # F0 = sum q_i w0_i (force on x=0 termination)
    wn = -np.cos(ii * np.pi) * (T * beta + EI * beta ** 3)  # F_nut = sum q_i wn_i
    phio = np.sin(beta * p["x_out"])
    return dict(L=L, mu=mu, EI=EI, T=T, fs=fs, dt=dt, M=M, beta=beta, om=om, sig=sig, a=a, b=b, c=c,
                m=m, om2s=om2s, sigs=sigs, ge=ge, gc=gc, kc=kc, rc=rc, kap=kap, lam=lam, thcc=thcc,
                xj=xj, dxb=dxb, hj=hj, Nb=Nb, Phi=Phi, v=v, Qp=Qp, w0=w0, wn=wn, phio=phio)


def excitation(p):
    fs = p["fs"]
    tr = np.arange(int(round(p["Fe_rise"] * fs))) / fs
    tf = np.arange(int(round(p["Fe_fall"] * fs))) / fs
    return np.concatenate([p["Fe_amp"] * np.sin(0.5 * np.pi * tr / p["Fe_rise"]) ** 2,
                           p["Fe_amp"] * np.cos(0.5 * np.pi * tf / p["Fe_fall"]) ** 2, [0.0]])


@njit(cache=True, inline="always")
def _dgrad(a, s, k, alpha):
    """Discrete gradient f = (V(a+s)-V(a))/s of V(eta)=k/(alpha+1) max(eta,0)^(alpha+1), and df/ds."""
    e = a + s
    if alpha == 1.0:
        if a > 0.0:
            if e > 0.0:
                return k * (a + 0.5 * s), 0.5 * k
            return -0.5 * k * a * a / s, 0.5 * k * a * a / (s * s)
        if e > 0.0:
            f = 0.5 * k * e * e / s
            return f, (k * e - f) / s
        return 0.0, 0.0
    # general alpha
    if a <= 0.0 and e <= 0.0:
        return 0.0, 0.0
    ap = a if a > 0.0 else 0.0
    ep = e if e > 0.0 else 0.0
    scale = ap if ap > ep else ep
    if abs(s) < 1e-7 * scale:
        mid = a + 0.5 * s
        if mid <= 0.0:
            return 0.0, 0.0
        return k * mid ** alpha, 0.5 * alpha * k * mid ** (alpha - 1.0)
    Va = k / (alpha + 1.0) * ap ** (alpha + 1.0)
    Ve = k / (alpha + 1.0) * ep ** (alpha + 1.0)
    f = (Ve - Va) / s
    return f, (k * ep ** alpha - f) / s


@njit(cache=True, inline="always")
def _scalar_root(a, eta_star, phi):
    """eta^{n+1} = z solving z = eta* - phi (max(z,0)^2 - max(a,0)^2)/(z - a)  (alpha = 1, one point)."""
    if a > 0.0:
        if eta_star > phi * a:
            return (eta_star - phi * a) / (1.0 + phi)
        return 0.5 * ((a + eta_star) - np.sqrt((a - eta_star) ** 2 + 4.0 * phi * a * a))
    if eta_star <= 0.0:
        return eta_star
    sm = a + eta_star
    return (sm + np.sqrt(sm * sm - 4.0 * (1.0 + phi) * a * eta_star)) / (2.0 * (1.0 + phi))


@njit(cache=True)
def _run(N, a, b, c, ge, Fe, gc, kap, lam, thcc, kc, rc, use_bridge, Phi, hj, dxb, v, Qp, kb, alpha,
         w0, wn, phio, tol, maxit, rec_energy, m, om2s, sigs, dt, guess):
    M = a.shape[0]
    Nb = hj.shape[0]
    q0 = np.zeros(M); q1 = np.zeros(M); q2 = np.zeros(M); qt = np.zeros(M)
    Fbr = np.zeros(N); Fnut = np.zeros(N); Fbs = np.zeros(N); Fcs = np.zeros(N); F0s = np.zeros(N)
    yo = np.zeros(N); H = np.zeros(N); Dcum = np.zeros(N); Wcum = np.zeros(N); pen = np.zeros(N)
    eta_p = hj.copy()   # eta^{n-1}
    eta_c = hj.copy()   # eta^{n}
    eta_n = hj.copy()   # eta^{n+1}
    yt = np.zeros(Nb); bv = np.zeros(Nb); s = np.zeros(Nb); f = np.zeros(Nb); fp = np.zeros(Nb)
    r = np.zeros(Nb); dl = np.zeros(Nb)
    act = np.zeros(Nb, dtype=np.int64)
    J = np.zeros((Nb, Nb)); rhs = np.zeros(Nb)
    uc_p = 0.0; uc_c = 0.0; ue_p = 0.0; ue_c = 0.0
    # stats
    max_pen = 0.0; j_max_pen = -1; n_contact = 0; tot_it = 0; max_it = 0; n_fail = 0; max_res = 0.0
    max_Fc = 0.0; edge_pen = 0.0; max_nact = 0
    Dacc = 0.0; Wacc = 0.0
    nFe = Fe.shape[0]
    for n in range(N):
        fe = Fe[n] if n < nFe else 0.0
        uct = 0.0; uet = 0.0
        for i in range(M):
            qt[i] = a[i] * q1[i] - b[i] * q0[i] + c[i] * ge[i] * fe
            uct += gc[i] * qt[i]
        g0 = (-kap * uct - lam * uc_p) / (1.0 + kap * thcc)
        g1 = -kap / (1.0 + kap * thcc)   # F_c = g0 + g1 * dxb * (v . f)
        nact = 0
        Fb_tot = 0.0
        vf = 0.0
        if use_bridge:
            # predicted (force-free + thread-only) penetration change
            anyc = False
            for j in range(Nb):
                acc = 0.0
                for i in range(M):
                    acc += Phi[j, i] * qt[i]
                yt[j] = acc
                bv[j] = hj[j] - acc - v[j] * g0 - eta_p[j]
                s[j] = bv[j]
                if eta_p[j] > 0.0 or eta_p[j] + s[j] > 0.0:
                    anyc = True
                    if alpha == 1.0 and guess:
                        # initial guess: exact solution of the decoupled scalar problem (lumped, 4 cases)
                        s[j] = _scalar_root(eta_p[j], eta_p[j] + bv[j], 0.5 * kb * Qp[j, j]) - eta_p[j]
            if anyc:
                it = 0
                while True:
                    nact = 0
                    for j in range(Nb):
                        fj, fpj = _dgrad(eta_p[j], s[j], kb, alpha)
                        f[j] = fj; fp[j] = fpj
                        if fj != 0.0 or fpj != 0.0:
                            act[nact] = j; nact += 1
                    res = 0.0
                    for j in range(Nb):
                        acc = s[j] - bv[j]
                        for kk in range(nact):
                            k = act[kk]
                            acc += Qp[j, k] * f[k]
                        r[j] = acc
                        if abs(acc) > res:
                            res = abs(acc)
                    if res <= tol:
                        break
                    if it >= maxit:
                        n_fail += 1
                        break
                    # reduced Newton system (I + Qp_AA diag(fp_A)) d_A = -r_A
                    for ii in range(nact):
                        jr = act[ii]
                        for kk in range(nact):
                            k = act[kk]
                            J[ii, kk] = Qp[jr, k] * fp[k]
                        J[ii, ii] += 1.0
                        rhs[ii] = -r[jr]
                    # Gaussian elimination with partial pivoting
                    for col in range(nact):
                        piv = col; pmax = abs(J[col, col])
                        for rr in range(col + 1, nact):
                            if abs(J[rr, col]) > pmax:
                                pmax = abs(J[rr, col]); piv = rr
                        if piv != col:
                            for cc in range(nact):
                                tmp = J[col, cc]; J[col, cc] = J[piv, cc]; J[piv, cc] = tmp
                            tmp = rhs[col]; rhs[col] = rhs[piv]; rhs[piv] = tmp
                        inv = 1.0 / J[col, col]
                        for rr in range(col + 1, nact):
                            fac = J[rr, col] * inv
                            if fac != 0.0:
                                for cc in range(col, nact):
                                    J[rr, cc] -= fac * J[col, cc]
                                rhs[rr] -= fac * rhs[col]
                    for col in range(nact - 1, -1, -1):
                        acc = rhs[col]
                        for cc in range(col + 1, nact):
                            acc -= J[col, cc] * dl[cc]
                        dl[col] = acc / J[col, col]
                    # full update: delta_j = -r_j - sum_A Qp[j,k] fp_k d_k  (exact for active rows too)
                    for j in range(Nb):
                        acc = -r[j]
                        for kk in range(nact):
                            k = act[kk]
                            acc -= Qp[j, k] * fp[k] * dl[kk]
                        s[j] += acc
                    it += 1
                tot_it += it
                if it > max_it:
                    max_it = it
                if res > max_res:
                    max_res = res
                if nact > max_nact:
                    max_nact = nact
                for kk in range(nact):
                    k = act[kk]
                    vf += v[k] * f[k]
                    Fb_tot += f[k]
                vf *= dxb
                Fb_tot *= dxb
            else:
                for j in range(Nb):
                    f[j] = 0.0
            if nact > 0:
                n_contact += 1
        Fc = g0 + g1 * vf
        # modal update
        for i in range(M):
            acc = gc[i] * Fc
            for kk in range(nact):
                k = act[kk]
                acc += dxb * Phi[k, i] * f[k]
            q2[i] = qt[i] + c[i] * acc
        uc_n = 0.0; ue_n = 0.0; F0 = 0.0; Fn = 0.0; yv = 0.0
        for i in range(M):
            uc_n += gc[i] * q2[i]
            ue_n += ge[i] * q2[i]
            F0 += w0[i] * q2[i]
            Fn += wn[i] * q2[i]
            yv += phio[i] * q2[i]
        if use_bridge:
            pmax = -1e9
            for j in range(Nb):
                eta_n[j] = eta_p[j] + s[j]
                if eta_n[j] > pmax:
                    pmax = eta_n[j]
                if eta_n[j] > max_pen:
                    max_pen = eta_n[j]; j_max_pen = j
            pen[n] = pmax
            for j in (0, 1, Nb - 2, Nb - 1):
                if eta_n[j] > edge_pen:
                    edge_pen = eta_n[j]
        if abs(Fc) > max_Fc:
            max_Fc = abs(Fc)
        # outputs (force ON the bridge assembly: termination + thread + barrier)
        F0s[n] = F0; Fcs[n] = Fc; Fbs[n] = Fb_tot
        Fbr[n] = F0 - Fc - Fb_tot
        Fnut[n] = Fn
        yo[n] = yv
        if rec_energy:
            e = 0.0
            for i in range(M):
                vel = (q2[i] - q1[i]) / dt
                e += 0.5 * m * (vel * vel + om2s[i] * q2[i] * q1[i])
            if use_bridge:
                for j in range(Nb):
                    en = eta_n[j] if eta_n[j] > 0.0 else 0.0
                    ec = eta_c[j] if eta_c[j] > 0.0 else 0.0
                    e += 0.5 * dxb * kb / (alpha + 1.0) * (en ** (alpha + 1.0) + ec ** (alpha + 1.0))
            e += 0.25 * kc * (uc_n * uc_n + uc_c * uc_c)
            H[n] = e
            # dissipated power and supplied power at step n
            dloss = 0.0
            for i in range(M):
                vd = (q2[i] - q0[i]) / (2 * dt)
                dloss += 2 * m * sigs[i] * vd * vd
            vu = (uc_n - uc_p) / (2 * dt)
            dloss += rc * vu * vu
            Dacc += dt * dloss
            Wacc += dt * fe * (ue_n - ue_p) / (2 * dt)
            Dcum[n] = Dacc; Wcum[n] = Wacc
        # shift
        for i in range(M):
            q0[i] = q1[i]; q1[i] = q2[i]
        uc_p = uc_c; uc_c = uc_n; ue_p = ue_c; ue_c = ue_n
        if use_bridge:
            for j in range(Nb):
                eta_p[j] = eta_c[j]; eta_c[j] = eta_n[j]
    stats = np.array([max_pen, j_max_pen, n_contact, tot_it, max_it, n_fail, max_res, max_Fc, edge_pen,
                      max_nact])
    return Fbr, Fnut, Fbs, Fcs, F0s, yo, H, Dcum, Wcum, pen, stats


def simulate(over=None, energy=False):
    p = dict(DEFAULT)
    if over:
        p.update(over)
    s = build(p)
    N = int(round(p["dur"] * p["fs"]))
    Fe = excitation(p)
    t0 = time.time()
    out = _run(N, s["a"], s["b"], s["c"], s["ge"], Fe, s["gc"], s["kap"], s["lam"], s["thcc"], s["kc"],
               s["rc"], bool(p["bridge"]), s["Phi"], s["hj"], s["dxb"], s["v"], s["Qp"], float(p["kb"]),
               float(p["alpha"]), s["w0"], s["wn"], s["phio"], float(p["tol"]), int(p["maxit"]),
               bool(energy), s["m"], s["om2s"], s["sigs"], s["dt"], bool(p["guess"]))
    cpu = time.time() - t0
    Fbr, Fnut, Fbs, Fcs, F0s, yo, H, Dcum, Wcum, pen, st = out
    stats = dict(max_pen=st[0], x_max_pen=(s["xj"][int(st[1])] if st[1] >= 0 else np.nan),
                 contact_frac=st[2] / N, newton_mean=(st[3] / st[2] if st[2] > 0 else 0.0),
                 newton_max=int(st[4]), newton_fail=int(st[5]), max_res=st[6], max_Fc=st[7], edge_pen=st[8],
                 max_active=int(st[9]))
    return p, s, dict(Fbridge=Fbr, Fnut=Fnut, Fb=Fbs, Fc=Fcs, F0=F0s, y=yo, H=H, D=Dcum, W=Wcum, pen=pen,
                      stats=stats, cpu=cpu, N=N)


def thread_lift(p):
    """height of the string above the bridge surface at the thread position (the 'thread height')."""
    return -(p["h0"] - p["curv"] * (p["xc"] - p["xb"]) ** 2)
