"""Two-polarisation (y = bridge-normal / vertical, z = horizontal, parallel to the jawari face) modal tanpura
string model.  Extension of tanpura1d.py (van Walstijn et al. DAFx-16/17 style) towards the
Bridges & van Walstijn (Vienna Talk 2015) 2-D model.

Physics
  * stiff damped string, simply supported at x=0 (bridge-side termination) and x=L (nut), same sine basis
    for both polarisations: y = sum q_i sin(beta_i x), z = sum p_i sin(beta_i x), modal mass m = mu L/2.
  * termination coupling ("coupling of controllable level ... at the termination opposite the barrier"):
    the nut support is anisotropic with principal axes rotated by theta from the bridge normal; the mode
    pair (q_i, p_i) has frequencies omega_i (axis 1) and omega_i * 2^(detune/1200) (axis 2) along those
    axes, common damping sigma_i.  theta = 0 or detune = 0 -> no linear coupling.
  * thread: bilateral linear spring-damper in y (k_c, r_c) and in z (k_cz, r_cz), same sin^2 window.
  * bridge: rigid curved barrier h_b(x) = h0 - curv (x - x_b)^2, face ruled across the string, so the
    contact normal is +y.  Distributed one-sided power-law penalty  f_y = k_b max(h_b - y, 0)^alpha.
  * friction along the face (z):  f_z = -mu_f(v_z) f_y,  mu_f(v) = c1 arctan(c2 v)  (VT-2015 kinetic law,
    c1 = 0.39, c2 = 2.84 s/m).  v z mu_f(v) >= 0 -> purely dissipative.
  * Kirchhoff-Carrier tension modulation coupling both polarisations:
    T(t) = T0 + (EA/2L) P,  P = int (y_x^2 + z_x^2) dx = (L/2) sum beta_i^2 (q_i^2 + p_i^2),
    V_TM = (EA/8L) P^2.

Discretisation (all exact per-sample linear algebra, no lagged forces apart from the TM tension):
  * linear part: exact-pole two-step recursion in principal coordinates, written in (y,z) coordinates as
    x^{n+1} = A_i x^n - b_i x^{n-1} + c_i F_i^n  (A_i symmetric 2x2), equivalent to the energy-consistent
    form  m[delta_tt x + 2 sigma* delta_t. x + K*_i x^n] = F^n.
  * TM: linearly-implicit energy-conserving form  F_TM,i = -(EA/8) P^n beta_i^2 (x_i^{n+1} + x_i^{n-1}),
    discrete energy (EA/8L) P^{n+1} P^n >= 0  -> diagonal modification of every mode's update.
  * bridge normal force: discrete gradient [V(eta^{n+1}) - V(eta^{n-1})]/(eta^{n+1} - eta^{n-1}), Newton on
    the candidate contact points (thread forces eliminated analytically, as in the 1-D prototype).
  * friction: f_z = -mu_f(w_j) f_y,j with the centred slip velocity w_j = (z_j^{n+1} - z_j^{n-1})/(2 dt),
    solved by a damped Newton for w on the force-active points.  Within a time step the y-problem does not
    depend on z (TM tension is lagged, friction acts in z only), so the solve is block-triangular and exact.
  * discrete energy balance (see README):
      H^{n+1/2} - H^{n-1/2} = dt * [ supplied - modal damping - thread damping - friction ],
      friction power  dx sum_j mu_f(w_j) w_j f_y,j >= 0.
Outputs: force exerted by the string on the bridge assembly, vertical and horizontal components
  (termination + thread + barrier normal force / friction), nut force (y, z), energy terms, diagnostics.
"""
from __future__ import annotations

import time
import numpy as np
from numba import njit

DEFAULT = dict(
    # string (DAFx-16 Table 1, C3 steel string)
    L=1.0, rho=7850.0, A=6.16e-8, E=2.0e11, I=3.02e-16, f1=130.81, EI_scale=1.0,
    fs=176400.0, fmax=20000.0, M=None,
    etaF=5e-5, etaA=1.2, etaB=1e-2, lossless=False,
    # thread (both polarisations; kcz/rcz None -> same as vertical)
    thread=True, xc=0.005, wc=0.001, kc=1.2e5, rc=0.5, kcz=None, rcz=None,
    # bridge
    bridge=True, xb=0.010, h0=0.0, curv=3.0, kb=1e11, alpha=1.0, br_x0=None, br_x1=None, dxb=2e-4,
    init_static=None,          # None -> static equilibrium initial state if h0 > 0
    # excitation (force density psi_e(x) F_e(t), direction pluck_deg from the bridge normal towards +z)
    xe=0.5, we=1.0, Fe_amp=0.2, Fe_rise=0.010, Fe_fall=0.001, pluck_deg=20.0,
    # second polarisation and its couplings
    pol2=True, theta_nut=30.0, detune_cents=1.0,
    friction=True, mu_c1=0.39, mu_c2=2.84,
    tm=True, EA_scale=1.0,
    # solvers
    tol=1e-18, maxit=60, ftol=1e-13, fmaxit=80, guess=True,
    rec_every=64,
    dur=3.0,
)


def sin2win(x, xc, w):
    out = np.zeros_like(x)
    m = np.abs(x - xc) < w / 2
    out[m] = (2.0 / w) * np.sin(np.pi * (x[m] - xc + w / 2) / w) ** 2
    return out


def project(L, beta, xc, w, npts=8001):
    lo, hi = max(xc - w / 2, 0.0), min(xc + w / 2, L)
    x = np.linspace(lo, hi, npts)
    psi = sin2win(x, xc, w)
    return np.trapezoid(psi[None, :] * np.sin(beta[:, None] * x[None, :]), x, axis=1)


def build(p):
    L = p["L"]
    mu = p["rho"] * p["A"]
    EI = p["E"] * p["I"] * p["EI_scale"]
    EA = p["E"] * p["A"] * p["EA_scale"]
    f1 = p["f1"]
    T = mu * (2 * L * f1) ** 2 - EI * (np.pi / L) ** 2
    fs = p["fs"]; dt = 1.0 / fs
    ii = np.arange(1, 5000)
    beta = ii * np.pi / L
    om = np.sqrt((T * beta ** 2 + EI * beta ** 4) / mu)
    M = int(p["M"]) if p.get("M") else int(np.sum(om / (2 * np.pi) < p["fmax"]))
    ii, beta, om = ii[:M], beta[:M], om[:M]
    use_z = bool(p["pol2"])
    det = 2.0 ** (p["detune_cents"] / 1200.0) if use_z else 1.0
    om_ax = [om, om * det]
    if max(om_ax[1][-1], om[-1]) / (2 * np.pi) >= 0.5 * fs:
        raise ValueError("highest mode above Nyquist")
    if p["lossless"]:
        sig = np.zeros(M)
    else:
        eta = (T * (p["etaF"] + p["etaA"] / om) + EI * p["etaB"] * beta ** 2) / (T + EI * beta ** 2)
        sig = om * eta / 2
    b = np.exp(-2 * sig * dt)
    m = mu * L / 2
    c = dt ** 2 * (1 + b) / (2 * m)
    sigs = (1 - b) / ((1 + b) * dt)
    ax_a, ax_k = [], []
    for o in om_ax:
        omd = np.sqrt(np.maximum(o ** 2 - sig ** 2, 0.0))
        a = 2 * np.exp(-sig * dt) * np.cos(omd * dt)
        ax_a.append(a)
        ax_k.append((2 - 2 * a / (1 + b)) / dt ** 2)      # omega*^2 of the energy-consistent scheme
    th = np.deg2rad(p["theta_nut"]) if use_z else 0.0
    cs, sn = np.cos(th), np.sin(th)
    # R = [[c, -s], [s, c]] maps principal (axis1, axis2) -> (y, z);  A = R diag(a1, a2) R^T
    Ayy = cs * cs * ax_a[0] + sn * sn * ax_a[1]
    Azz = sn * sn * ax_a[0] + cs * cs * ax_a[1]
    Ayz = cs * sn * (ax_a[0] - ax_a[1])
    Kyy = cs * cs * ax_k[0] + sn * sn * ax_k[1]
    Kzz = sn * sn * ax_k[0] + cs * cs * ax_k[1]
    Kyz = cs * sn * (ax_k[0] - ax_k[1])
    if not use_z:
        Ayz = np.zeros(M); Azz = np.zeros(M); Kyz = np.zeros(M); Kzz = np.zeros(M)
    ge = project(L, beta, p["xe"], p["we"])
    if p["thread"]:
        gc = project(L, beta, p["xc"], p["wc"])
        kc, rc = p["kc"], p["rc"]
        kcz = p["kc"] if p["kcz"] is None else p["kcz"]
        rcz = p["rc"] if p["rcz"] is None else p["rcz"]
    else:
        gc = np.zeros(M); kc = rc = kcz = rcz = 0.0
    if not use_z:
        kcz = rcz = 0.0
    kapy, lamy = 0.5 * kc + rc / (2 * dt), 0.5 * kc - rc / (2 * dt)
    kapz, lamz = 0.5 * kcz + rcz / (2 * dt), 0.5 * kcz - rcz / (2 * dt)
    xb = p["xb"]
    if p["bridge"]:
        x0 = p["br_x0"] if p["br_x0"] is not None else (max(p["xc"], xb - 0.006) if p["thread"] else xb - 0.006)
        x1 = p["br_x1"] if p["br_x1"] is not None else xb + 0.006
        Nb = int(round((x1 - x0) / p["dxb"])) + 1
        xj = np.linspace(x0, x1, Nb)
        dxb = xj[1] - xj[0]
        k0 = int(np.argmin(np.abs(xj - xb))); xj = xj + (xb - xj[k0])
        hj = p["h0"] - p["curv"] * (xj - xb) ** 2
        Phi = np.ascontiguousarray(np.sin(xj[:, None] * beta[None, :]))
        Q0 = np.ascontiguousarray(dxb * (Phi * c[None, :]) @ Phi.T)
        Q0 = 0.5 * (Q0 + Q0.T)
        v0 = Phi @ (c * gc)
    else:
        xj = np.array([xb]); dxb = 1.0; hj = np.array([-1.0]); Nb = 1
        Phi = np.zeros((1, M)); Q0 = np.zeros((1, 1)); v0 = np.zeros(1)
    w0 = T * beta + EI * beta ** 3
    sgn = -np.cos(ii * np.pi)
    wn = sgn * (T * beta + EI * beta ** 3)
    ph = np.deg2rad(p["pluck_deg"]) if use_z else 0.0
    return dict(L=L, mu=mu, EI=EI, EA=EA, T=T, fs=fs, dt=dt, M=M, beta=beta, om=om, om2=om_ax[1], sig=sig,
                b=b, c=c, m=m, sigs=sigs, Ayy=Ayy, Ayz=Ayz, Azz=Azz, Kyy=Kyy, Kyz=Kyz, Kzz=Kzz, theta=th,
                ge=ge, gc=gc, kc=kc, rc=rc, kcz=kcz, rcz=rcz, kapy=kapy, lamy=lamy, kapz=kapz, lamz=lamz,
                xj=xj, dxb=dxb, hj=hj, Nb=Nb, Phi=Phi, Q0=Q0, v0=v0, w0=w0, wn=wn, sgn=sgn,
                cphi=np.cos(ph), sphi=np.sin(ph), use_z=use_z, k_ax1=ax_k[0], k_ax2=ax_k[1])


def excitation(p):
    fs = p["fs"]
    tr = np.arange(int(round(p["Fe_rise"] * fs))) / fs
    tf = np.arange(int(round(p["Fe_fall"] * fs))) / fs
    return np.concatenate([p["Fe_amp"] * np.sin(0.5 * np.pi * tr / p["Fe_rise"]) ** 2,
                           p["Fe_amp"] * np.cos(0.5 * np.pi * tf / p["Fe_fall"]) ** 2, [0.0]])


def static_state(p, s):
    """Exact discrete static equilibrium (string + thread + bridge preload), alpha = 1 only.
    Uses the scheme's own stiffness K*_i (so q^{n+1} = q^n = q^{n-1} is an exact fixed point)."""
    assert p["alpha"] == 1.0
    m = s["m"]
    # static compliance of mode pair to a vertical force: K*^{-1} = R diag(1/k1, 1/k2) R^T / m
    k1, k2 = s["k_ax1"], s["k_ax2"]
    if s["use_z"]:
        th = s["theta"]; cs, sn = np.cos(th), np.sin(th)
        Syy = (cs * cs / k1 + sn * sn / k2) / m
        Szy = cs * sn * (1 / k1 - 1 / k2) / m
    else:
        Syy = 1.0 / (k1 * m); Szy = np.zeros_like(Syy)
    Phi, dx, gc, kc, h = s["Phi"], s["dxb"], s["gc"], s["kc"], s["hj"]
    ths = np.sum(gc * gc * Syy)
    vs = Phi @ (Syy * gc)
    G = dx * (Phi * Syy[None, :]) @ Phi.T - kc * dx * np.outer(vs, vs) / (1 + kc * ths)
    kb = p["kb"]
    act = h > 0
    f = np.zeros_like(h)
    for _ in range(200):
        f[:] = 0.0
        if act.any():
            A = np.eye(act.sum()) + kb * G[np.ix_(act, act)]
            f[act] = np.linalg.solve(A, kb * h[act])
        y = G @ f
        gap = h - y
        new = (gap > 0)
        if np.array_equal(new, act):
            break
        act = new
    Fmod = dx * Phi.T @ f
    uc_f = np.sum(gc * Syy * Fmod)
    Fc = -kc * uc_f / (1 + kc * ths)
    q = Syy * (Fmod + gc * Fc)
    pz = Szy * (Fmod + gc * Fc)
    return q, pz, dict(static_force=float(np.sum(f) * dx), thread_force=float(Fc), n_active=int(act.sum()))


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
    if a > 0.0:
        if eta_star > phi * a:
            return (eta_star - phi * a) / (1.0 + phi)
        return 0.5 * ((a + eta_star) - np.sqrt((a - eta_star) ** 2 + 4.0 * phi * a * a))
    if eta_star <= 0.0:
        return eta_star
    sm = a + eta_star
    return (sm + np.sqrt(sm * sm - 4.0 * (1.0 + phi) * a * eta_star)) / (2.0 * (1.0 + phi))


@njit(cache=True)
def _gauss(J, rhs, n, out):
    """solve J[:n,:n] out = rhs[:n] in place (partial pivoting); J and rhs are overwritten."""
    for col in range(n):
        piv = col; pmax = abs(J[col, col])
        for rr in range(col + 1, n):
            if abs(J[rr, col]) > pmax:
                pmax = abs(J[rr, col]); piv = rr
        if piv != col:
            for cc in range(n):
                tmp = J[col, cc]; J[col, cc] = J[piv, cc]; J[piv, cc] = tmp
            tmp = rhs[col]; rhs[col] = rhs[piv]; rhs[piv] = tmp
        inv = 1.0 / J[col, col]
        for rr in range(col + 1, n):
            fac = J[rr, col] * inv
            if fac != 0.0:
                for cc in range(col, n):
                    J[rr, cc] -= fac * J[col, cc]
                rhs[rr] -= fac * rhs[col]
    for col in range(n - 1, -1, -1):
        acc = rhs[col]
        for cc in range(col + 1, n):
            acc -= J[col, cc] * out[cc]
        out[col] = acc / J[col, col]


@njit(cache=True)
def _run(N, Ayy, Ayz, Azz, b, c, beta, beta2, ge, Fe, cphi, sphi,
         gc, kapy, lamy, kcy, rcy, kapz, lamz, kcz, rcz,
         use_z, use_bridge, use_fric, use_tm,
         Phi, hj, dxb, Q0, v0, kb, alpha, mu1, mu2,
         ktm, L, w0, wn, sgn,
         tol, maxit, ftol, fmaxit, guess,
         rec_energy, m, Kyy, Kyz, Kzz, sigs, dt, q_init, p_init, rec_every):
    M = b.shape[0]
    Nb = hj.shape[0]
    q0 = q_init.copy(); q1 = q_init.copy(); q2 = np.zeros(M)
    p0 = p_init.copy(); p1 = p_init.copy(); p2 = np.zeros(M)
    qh = np.zeros(M); ph = np.zeros(M); cp = np.zeros(M); qs = np.zeros(M); ps = np.zeros(M)
    Fby = np.zeros(N); Fbz = np.zeros(N); Fny = np.zeros(N); Fnz = np.zeros(N)
    Fcont = np.zeros(N); Ffric = np.zeros(N)
    H = np.zeros(N); Dcum = np.zeros(N); Dfric = np.zeros(N); Wcum = np.zeros(N); pen = np.zeros(N)
    NR = N // rec_every + 1
    rEy = np.zeros(NR); rEz = np.zeros(NR); rdT = np.zeros(NR); rslip = np.zeros(NR)
    eta_p = np.zeros(Nb); eta_c = np.zeros(Nb); eta_n = np.zeros(Nb)
    for j in range(Nb):
        acc = 0.0
        for i in range(M):
            acc += Phi[j, i] * q_init[i]
        eta_p[j] = hj[j] - acc
        eta_c[j] = eta_p[j]
    yt = np.zeros(Nb); bv = np.zeros(Nb)
    inS = np.zeros(Nb, dtype=np.bool_)
    sidx = np.zeros(Nb, dtype=np.int64)
    s = np.zeros(Nb); f = np.zeros(Nb); fp = np.zeros(Nb); r = np.zeros(Nb); dl = np.zeros(Nb)
    act = np.zeros(Nb, dtype=np.int64)
    vS = np.zeros(Nb)
    QB = np.zeros((Nb, Nb)); QY = np.zeros((Nb, Nb)); QZ = np.zeros((Nb, Nb))
    J = np.zeros((Nb, Nb)); rhs = np.zeros(Nb)
    fA = np.zeros(Nb); hA = np.zeros(Nb); wA = np.zeros(Nb); whA = np.zeros(Nb); wt = np.zeros(Nb)
    muA = np.zeros(Nb); mupA = np.zeros(Nb); rz = np.zeros(Nb); dz = np.zeros(Nb); rzt = np.zeros(Nb)
    jA = np.zeros(Nb, dtype=np.int64)
    ucy_p = 0.0; ucy_c = 0.0; ucz_p = 0.0; ucz_c = 0.0
    for i in range(M):
        ucy_p += gc[i] * q_init[i]; ucz_p += gc[i] * p_init[i]
    ucy_c = ucy_p; ucz_c = ucz_p
    uey_p = 0.0; uez_p = 0.0
    for i in range(M):
        uey_p += ge[i] * q_init[i]; uez_p += ge[i] * p_init[i]
    uey_c = uey_p; uez_c = uez_p
    P_c = 0.0
    for i in range(M):
        P_c += 0.5 * L * beta2[i] * (q_init[i] ** 2 + p_init[i] ** 2)
    thcc0 = 0.0
    for i in range(M):
        thcc0 += c[i] * gc[i] * gc[i]
    inv2dt = 1.0 / (2.0 * dt)
    # stats
    max_pen = 0.0; n_contact = 0; tot_it = 0; max_it = 0; n_fail = 0; max_res = 0.0
    max_Fcy = 0.0; max_Fcz = 0.0; edge_pen = 0.0; max_nact = 0; n_viol = 0
    n_fric = 0; f_tot_it = 0; f_max_it = 0; f_fail = 0; f_max_res = 0.0; max_w = 0.0; max_dT = 0.0
    max_mu = 0.0
    Dacc = 0.0; Facc = 0.0; Wacc = 0.0
    Eyacc = 0.0; Ezacc = 0.0; slipacc = 0.0
    nFe = Fe.shape[0]
    for n in range(N):
        fe = Fe[n] if n < nFe else 0.0
        fy = fe * cphi; fz = fe * sphi
        # ---- free (linear) prediction + TM-modified compliance
        thcc = 0.0; uchy = 0.0; uchz = 0.0
        for i in range(M):
            qt = Ayy[i] * q1[i] - b[i] * q0[i] + c[i] * ge[i] * fy
            pt = 0.0
            if use_z:
                qt += Ayz[i] * p1[i]
                pt = Ayz[i] * q1[i] + Azz[i] * p1[i] - b[i] * p0[i] + c[i] * ge[i] * fz
            if use_tm:
                eps = c[i] * ktm * P_c * beta2[i]
                d = 1.0 / (1.0 + eps)
                qh[i] = d * (qt - eps * q0[i])
                ph[i] = d * (pt - eps * p0[i])
                cp[i] = c[i] * d
            else:
                qh[i] = qt; ph[i] = pt; cp[i] = c[i]
            uchy += gc[i] * qh[i]
            uchz += gc[i] * ph[i]
        if use_tm:
            for i in range(M):
                thcc += cp[i] * gc[i] * gc[i]
        else:
            thcc = thcc0
        den = 1.0 + kapy * thcc
        g0y = -(kapy * uchy + lamy * ucy_p) / den
        g1y = -kapy / den
        den = 1.0 + kapz * thcc
        g0z = -(kapz * uchz + lamz * ucz_p) / den
        g1z = -kapz / den
        for i in range(M):
            qs[i] = qh[i] + cp[i] * gc[i] * g0y
            ps[i] = ph[i] + cp[i] * gc[i] * g0z
        nA = 0
        Fb_tot = 0.0; Fh_tot = 0.0; vf = 0.0; vh = 0.0; pfric = 0.0
        Fcy = g0y; Fcz = g0z
        it = 0; res = 0.0; nact = 0
        contact = False
        if use_bridge:
            anyS = False
            for j in range(Nb):
                acc = 0.0
                for i in range(M):
                    acc += Phi[j, i] * qs[i]
                yt[j] = acc
                bv[j] = hj[j] - acc - eta_p[j]
                inS[j] = (eta_p[j] > 0.0) or (eta_p[j] + bv[j] > 0.0)
                if inS[j]:
                    anyS = True
            if anyS:
                contact = True
                for attempt in range(6):
                    nS = 0
                    for j in range(Nb):
                        if inS[j]:
                            sidx[nS] = j; nS += 1
                    # compliance on the candidate set
                    for a_ in range(nS):
                        ja = sidx[a_]
                        if use_tm:
                            acc = 0.0
                            for i in range(M):
                                acc += Phi[ja, i] * cp[i] * gc[i]
                            vS[a_] = acc
                        else:
                            vS[a_] = v0[ja]
                    for a_ in range(nS):
                        ja = sidx[a_]
                        for b_ in range(a_, nS):
                            jb = sidx[b_]
                            if use_tm:
                                acc = 0.0
                                for i in range(M):
                                    acc += Phi[ja, i] * cp[i] * Phi[jb, i]
                                acc *= dxb
                            else:
                                acc = Q0[ja, jb]
                            QB[a_, b_] = acc; QB[b_, a_] = acc
                    for a_ in range(nS):
                        for b_ in range(nS):
                            QY[a_, b_] = QB[a_, b_] + g1y * dxb * vS[a_] * vS[b_]
                            QZ[a_, b_] = QB[a_, b_] + g1z * dxb * vS[a_] * vS[b_]
                    # ---- Newton, vertical contact (identical structure to the 1-D prototype)
                    for a_ in range(nS):
                        ja = sidx[a_]
                        s[a_] = bv[ja]
                        if alpha == 1.0 and guess:
                            s[a_] = _scalar_root(eta_p[ja], eta_p[ja] + bv[ja], 0.5 * kb * QY[a_, a_]) - eta_p[ja]
                    it = 0
                    res = 0.0
                    nact = 0
                    while True:
                        nact = 0
                        for a_ in range(nS):
                            fj, fpj = _dgrad(eta_p[sidx[a_]], s[a_], kb, alpha)
                            f[a_] = fj; fp[a_] = fpj
                            if fj != 0.0 or fpj != 0.0:
                                act[nact] = a_; nact += 1
                        res = 0.0
                        for a_ in range(nS):
                            acc = s[a_] - bv[sidx[a_]]
                            for kk in range(nact):
                                k = act[kk]
                                acc += QY[a_, k] * f[k]
                            r[a_] = acc
                            if abs(acc) > res:
                                res = abs(acc)
                        if res <= tol:
                            break
                        if it >= maxit:
                            n_fail += 1
                            break
                        for ii in range(nact):
                            jr = act[ii]
                            for kk in range(nact):
                                k = act[kk]
                                J[ii, kk] = QY[jr, k] * fp[k]
                            J[ii, ii] += 1.0
                            rhs[ii] = -r[jr]
                        _gauss(J, rhs, nact, dl)
                        for a_ in range(nS):
                            acc = -r[a_]
                            for kk in range(nact):
                                k = act[kk]
                                acc -= QY[a_, k] * fp[k] * dl[kk]
                            s[a_] += acc
                        it += 1
                    # force-active points
                    nA = 0
                    for a_ in range(nS):
                        if f[a_] != 0.0:
                            jA[nA] = a_; fA[nA] = f[a_]; nA += 1
                    # ---- friction along the bridge face (z)
                    for k in range(nA):
                        hA[k] = 0.0; wA[k] = 0.0; muA[k] = 0.0
                    fit = 0
                    if use_fric and nA > 0:
                        for k in range(nA):
                            ja = sidx[jA[k]]
                            zh = 0.0; zp = 0.0
                            for i in range(M):
                                zh += Phi[ja, i] * ps[i]
                                zp += Phi[ja, i] * p0[i]
                            whA[k] = (zh - zp) * inv2dt
                            wA[k] = whA[k]
                        fres = 0.0
                        while True:
                            for k in range(nA):
                                muA[k] = mu1 * np.arctan(mu2 * wA[k])
                                mupA[k] = mu1 * mu2 / (1.0 + (mu2 * wA[k]) ** 2)
                            base = 0.0; fres = 0.0
                            for k in range(nA):
                                acc = wA[k] - whA[k]
                                for l in range(nA):
                                    acc += inv2dt * QZ[jA[k], jA[l]] * muA[l] * fA[l]
                                rz[k] = acc
                                base += acc * acc
                                if abs(acc) > fres:
                                    fres = abs(acc)
                            if fres <= ftol:
                                break
                            if fit >= fmaxit:
                                f_fail += 1
                                break
                            for k in range(nA):
                                for l in range(nA):
                                    J[k, l] = inv2dt * QZ[jA[k], jA[l]] * mupA[l] * fA[l]
                                J[k, k] += 1.0
                                rhs[k] = -rz[k]
                            _gauss(J, rhs, nA, dz)
                            lam_ = 1.0
                            for ls in range(50):
                                trial = 0.0
                                for k in range(nA):
                                    wt[k] = wA[k] + lam_ * dz[k]
                                for k in range(nA):
                                    acc = wt[k] - whA[k]
                                    for l in range(nA):
                                        acc += inv2dt * QZ[jA[k], jA[l]] * mu1 * np.arctan(mu2 * wt[l]) * fA[l]
                                    trial += acc * acc
                                if trial < base or ls == 49:
                                    break
                                lam_ *= 0.5
                            for k in range(nA):
                                wA[k] = wt[k]
                            fit += 1
                        for k in range(nA):
                            muA[k] = mu1 * np.arctan(mu2 * wA[k])
                            hA[k] = -muA[k] * fA[k]
                        n_fric += 1
                        f_tot_it += fit
                        if fit > f_max_it:
                            f_max_it = fit
                        if fres > f_max_res:
                            f_max_res = fres
                    # ---- thread forces and modal update
                    vf = 0.0; vh = 0.0; Fb_tot = 0.0; Fh_tot = 0.0; pfric = 0.0
                    for k in range(nA):
                        vf += vS[jA[k]] * fA[k]
                        vh += vS[jA[k]] * hA[k]
                        Fb_tot += fA[k]; Fh_tot += hA[k]
                        pfric += muA[k] * wA[k] * fA[k]
                    vf *= dxb; vh *= dxb; Fb_tot *= dxb; Fh_tot *= dxb; pfric *= dxb
                    Fcy = g0y + g1y * vf
                    Fcz = g0z + g1z * vh
                    for i in range(M):
                        accy = gc[i] * Fcy
                        accz = gc[i] * Fcz
                        for k in range(nA):
                            ph_ = Phi[sidx[jA[k]], i]
                            accy += dxb * ph_ * fA[k]
                            accz += dxb * ph_ * hA[k]
                        q2[i] = qh[i] + cp[i] * accy
                        p2[i] = ph[i] + cp[i] * accz
                    # penetrations at all points from the modes; points outside the candidate set must not
                    # penetrate (else enlarge the set and solve again)
                    viol = False
                    for j in range(Nb):
                        acc = 0.0
                        for i in range(M):
                            acc += Phi[j, i] * q2[i]
                        eta_n[j] = hj[j] - acc
                        if (not inS[j]) and eta_n[j] > 0.0:
                            inS[j] = True; viol = True
                    if not viol:
                        break
                    n_viol += 1
                tot_it += it
                if it > max_it:
                    max_it = it
                if res > max_res:
                    max_res = res
                if nact > max_nact:
                    max_nact = nact
                if nA > 0:
                    n_contact += 1
                for k in range(nA):
                    if abs(wA[k]) > max_w:
                        max_w = abs(wA[k])
                    if abs(muA[k]) > max_mu:
                        max_mu = abs(muA[k])
            else:
                for j in range(Nb):
                    eta_n[j] = hj[j] - yt[j]
        if not contact:
            Fcy = g0y; Fcz = g0z
            for i in range(M):
                q2[i] = qs[i]; p2[i] = ps[i]
        # ---- outputs
        ucy_n = 0.0; ucz_n = 0.0; uey_n = 0.0; uez_n = 0.0; F0y = 0.0; F0z = 0.0; Fny_ = 0.0; Fnz_ = 0.0
        P_n = 0.0; by = 0.0; bz = 0.0; byn = 0.0; bzn = 0.0
        for i in range(M):
            ucy_n += gc[i] * q2[i]; ucz_n += gc[i] * p2[i]
            uey_n += ge[i] * q2[i]; uez_n += ge[i] * p2[i]
            F0y += w0[i] * q2[i]; F0z += w0[i] * p2[i]
            Fny_ += wn[i] * q2[i]; Fnz_ += wn[i] * p2[i]
            P_n += 0.5 * L * beta2[i] * (q2[i] * q2[i] + p2[i] * p2[i])
            by += beta[i] * q2[i]; bz += beta[i] * p2[i]
            byn += sgn[i] * beta[i] * q2[i]; bzn += sgn[i] * beta[i] * p2[i]
        dT = 0.0
        if use_tm:
            dT = 4.0 * ktm / L * P_n       # EA/(2L) P
            F0y += dT * by; F0z += dT * bz; Fny_ += dT * byn; Fnz_ += dT * bzn
            if dT > max_dT:
                max_dT = dT
        if use_bridge:
            pmax = -1e9
            for j in range(Nb):
                if eta_n[j] > pmax:
                    pmax = eta_n[j]
            pen[n] = pmax
            if pmax > max_pen:
                max_pen = pmax
            for j in (0, 1, Nb - 2, Nb - 1):
                if eta_n[j] > edge_pen:
                    edge_pen = eta_n[j]
        if abs(Fcy) > max_Fcy:
            max_Fcy = abs(Fcy)
        if abs(Fcz) > max_Fcz:
            max_Fcz = abs(Fcz)
        Fby[n] = F0y - Fcy - Fb_tot
        Fbz[n] = F0z - Fcz - Fh_tot
        Fny[n] = Fny_; Fnz[n] = Fnz_
        Fcont[n] = Fb_tot; Ffric[n] = Fh_tot
        Facc += dt * pfric
        if rec_energy:
            e = 0.0
            for i in range(M):
                vy = (q2[i] - q1[i]) / dt
                vz = (p2[i] - p1[i]) / dt
                e += 0.5 * m * (vy * vy + vz * vz + Kyy[i] * q2[i] * q1[i]
                                + Kyz[i] * (q2[i] * p1[i] + p2[i] * q1[i]) + Kzz[i] * p2[i] * p1[i])
            if use_bridge:
                for j in range(Nb):
                    en = eta_n[j] if eta_n[j] > 0.0 else 0.0
                    ec = eta_c[j] if eta_c[j] > 0.0 else 0.0
                    e += 0.5 * dxb * kb / (alpha + 1.0) * (en ** (alpha + 1.0) + ec ** (alpha + 1.0))
            e += 0.25 * kcy * (ucy_n * ucy_n + ucy_c * ucy_c) + 0.25 * kcz * (ucz_n * ucz_n + ucz_c * ucz_c)
            if use_tm:
                e += ktm / L * P_n * P_c
            H[n] = e
            dloss = 0.0
            for i in range(M):
                vdy = (q2[i] - q0[i]) * inv2dt
                vdz = (p2[i] - p0[i]) * inv2dt
                dloss += 2.0 * m * sigs[i] * (vdy * vdy + vdz * vdz)
            vu = (ucy_n - ucy_p) * inv2dt
            vz_ = (ucz_n - ucz_p) * inv2dt
            dloss += rcy * vu * vu + rcz * vz_ * vz_
            Dacc += dt * dloss
            Wacc += dt * (fy * (uey_n - uey_p) + fz * (uez_n - uez_p)) * inv2dt
            Dcum[n] = Dacc; Wcum[n] = Wacc
        Dfric[n] = Facc
        # polarisation energies (diagnostic, cross term of the nut coupling omitted) and tension
        ey = 0.0; ez = 0.0
        for i in range(M):
            vy = (q2[i] - q1[i]) / dt
            vz = (p2[i] - p1[i]) / dt
            ey += 0.5 * m * (vy * vy + Kyy[i] * q2[i] * q1[i])
            ez += 0.5 * m * (vz * vz + Kzz[i] * p2[i] * p1[i])
        Eyacc += ey; Ezacc += ez
        sl = 0.0
        for k in range(nA):
            if abs(wA[k]) > sl:
                sl = abs(wA[k])
        slipacc += sl
        if (n + 1) % rec_every == 0:
            kr = n // rec_every
            rEy[kr] = Eyacc / rec_every; rEz[kr] = Ezacc / rec_every; rdT[kr] = dT
            rslip[kr] = slipacc / rec_every
            Eyacc = 0.0; Ezacc = 0.0; slipacc = 0.0
        # ---- shift
        for i in range(M):
            q0[i] = q1[i]; q1[i] = q2[i]; p0[i] = p1[i]; p1[i] = p2[i]
        ucy_p = ucy_c; ucy_c = ucy_n; ucz_p = ucz_c; ucz_c = ucz_n
        uey_p = uey_c; uey_c = uey_n; uez_p = uez_c; uez_c = uez_n
        P_c = P_n
        if use_bridge:
            for j in range(Nb):
                eta_p[j] = eta_c[j]; eta_c[j] = eta_n[j]
    stats = np.array([max_pen, n_contact, tot_it, max_it, n_fail, max_res, max_Fcy, max_Fcz, edge_pen,
                      max_nact, n_viol, n_fric, f_tot_it, f_max_it, f_fail, f_max_res, max_w, max_dT, max_mu,
                      Facc])
    return Fby, Fbz, Fny, Fnz, Fcont, Ffric, H, Dcum, Dfric, Wcum, pen, rEy, rEz, rdT, rslip, stats


def simulate(over=None, energy=False):
    p = dict(DEFAULT)
    if over:
        p.update(over)
    s = build(p)
    N = int(round(p["dur"] * p["fs"]))
    Fe = excitation(p)
    q_init = np.zeros(s["M"]); p_init = np.zeros(s["M"]); stat_info = None
    init_static = p["init_static"] if p["init_static"] is not None else (p["bridge"] and p["h0"] > 0)
    if init_static and p["bridge"]:
        q_init, p_init, stat_info = static_state(p, s)
    use_fric = bool(p["friction"]) and s["use_z"] and bool(p["bridge"])
    ktm = s["EA"] / 8.0 if p["tm"] else 0.0
    t0 = time.time()
    out = _run(N, s["Ayy"], s["Ayz"], s["Azz"], s["b"], s["c"], s["beta"], s["beta"] ** 2, s["ge"], Fe,
               float(s["cphi"]), float(s["sphi"]),
               s["gc"], s["kapy"], s["lamy"], float(s["kc"]), float(s["rc"]), s["kapz"], s["lamz"],
               float(s["kcz"]), float(s["rcz"]),
               bool(s["use_z"]), bool(p["bridge"]), use_fric, bool(p["tm"]),
               s["Phi"], s["hj"], float(s["dxb"]), s["Q0"], s["v0"], float(p["kb"]), float(p["alpha"]),
               float(p["mu_c1"]), float(p["mu_c2"]),
               float(ktm), float(s["L"]), s["w0"], s["wn"], s["sgn"],
               float(p["tol"]), int(p["maxit"]), float(p["ftol"]), int(p["fmaxit"]), bool(p["guess"]),
               bool(energy), float(s["m"]), s["Kyy"], s["Kyz"], s["Kzz"], s["sigs"], float(s["dt"]),
               q_init, p_init, int(p["rec_every"]))
    cpu = time.time() - t0
    (Fby, Fbz, Fny, Fnz, Fcont, Ffric, H, D, Df, W, pen, rEy, rEz, rdT, rslip, st) = out
    nc = st[1]
    stats = dict(max_pen=st[0], contact_frac=nc / N, newton_mean=(st[2] / nc if nc > 0 else 0.0),
                 newton_max=int(st[3]), newton_fail=int(st[4]), max_res=st[5], max_Fcy=st[6], max_Fcz=st[7],
                 edge_pen=st[8], max_active=int(st[9]), n_resolve=int(st[10]), fric_steps=int(st[11]),
                 fric_newton_mean=(st[12] / st[11] if st[11] > 0 else 0.0), fric_newton_max=int(st[13]),
                 fric_fail=int(st[14]), fric_max_res=st[15], max_slip=st[16], max_dT=st[17], max_mu=st[18],
                 E_friction=st[19], static=stat_info)
    tr = np.arange(len(rEy)) * p["rec_every"] / p["fs"]
    return p, s, dict(Fy=Fby, Fz=Fbz, Fny=Fny, Fnz=Fnz, Fcont=Fcont, Ffric=Ffric, H=H, D=D, Dfric=Df, W=W,
                      pen=pen, Ey=rEy, Ez=rEz, dT=rdT, slip=rslip, t_rec=tr, stats=stats, cpu=cpu, N=N)


def thread_lift(p):
    return -(p["h0"] - p["curv"] * (p["xc"] - p["xb"]) ** 2)
