"""Independent judge model: explicit finite-difference stiff string (Bilbao-type scheme, loss sigma0 + sigma1*beta^2)
with a point thread (trapezoidal spring-damper, implicit) and a DISTRIBUTED rigid-ish bridge contact evaluated
node-by-node with a discrete-gradient (energy-conserving) penalty f = kb*[h_b - y]_+ (alpha = 1), solved in
closed form per node (the contact only couples to its own node in FD, so no Newton system).
Spatial resolution 0.5-1 mm (vs ~7.5 mm for 133 modes), so the string can wrap the curved bridge.
Written from scratch by the judge; shares no code with prototype A."""
import numpy as np
from numba import njit


@njit(cache=True)
def _contact_root(a, b, K):
    """eta^{n+1}=z solving z = b - K*(V(z)-V(a))/(k (z-a)), V = k/2 [.]_+^2, K = phi*k (dimensionless)."""
    g0 = -b + (0.5 * K * a if a > 0.0 else 0.0)
    if g0 < 0.0:  # z > 0
        if a > 0.0:
            return (b - 0.5 * K * a) / (1.0 + 0.5 * K)
        c2 = 1.0 + 0.5 * K
        sm = a + b
        return (sm + np.sqrt(sm * sm - 4.0 * c2 * a * b)) / (2.0 * c2)
    if a > 0.0:
        return 0.5 * ((a + b) - np.sqrt((a - b) ** 2 + 2.0 * K * a * a))
    return b


@njit(cache=True)
def _run_fd(N, nsteps, h, k, mu, T, EI, s0, s1, ic, kc, rc, j0, j1, hb, kb, psi_e, Fe, rec_energy):
    y0 = np.zeros(N + 1); y1 = np.zeros(N + 1); y2 = np.zeros(N + 1)
    eta_p = np.zeros(j1 - j0); eta_c = np.zeros(j1 - j0)
    for jj in range(j1 - j0):
        eta_p[jj] = hb[jj]; eta_c[jj] = hb[jj]
    Fnut = np.zeros(nsteps); Fbr = np.zeros(nsteps); H = np.zeros(nsteps); pen = np.zeros(nsteps)
    xpen = np.zeros(nsteps)
    lam2 = k * k / mu
    den = 1.0 + s0 * k
    phi = lam2 / den
    kap = 0.5 * kc + rc / (2 * k)
    lamc = 0.5 * kc - rc / (2 * k)
    nFe = Fe.shape[0]
    max_act = 0
    for n in range(nsteps):
        fe = Fe[n] if n < nFe else 0.0
        # free (linear + explicit excitation) update for interior nodes 1..N-1
        for l in range(1, N):
            ylm1 = y1[l - 1]; ylp1 = y1[l + 1]
            ylm2 = y1[l - 2] if l >= 2 else -y1[1]          # simply supported ghost: y_{-1} = -y_1
            ylp2 = y1[l + 2] if l <= N - 2 else -y1[N - 1]  # y_{N+1} = -y_{N-1}
            dxx = (ylp1 - 2 * y1[l] + ylm1) / (h * h)
            # delta_xxxx with ghost points
            d4 = (ylp2 - 4 * ylp1 + 6 * y1[l] - 4 * ylm1 + ylm2) / (h ** 4)
            dxx_old = (y0[l + 1] - 2 * y0[l] + y0[l - 1]) / (h * h)
            y2[l] = (2 * y1[l] - (1 - s0 * k) * y0[l] + lam2 * (T * dxx - EI * d4 + fe * psi_e[l])
                     + 2 * s1 * k * (dxx - dxx_old)) / den
        y2[0] = 0.0; y2[N] = 0.0
        # thread at node ic (point force Fc/h), trapezoidal spring-damper, solved exactly
        ys = y2[ic]
        yn = (ys - (phi / h) * lamc * y0[ic]) / (1.0 + (phi / h) * kap)
        Fc = -kap * yn - lamc * y0[ic]
        y2[ic] = yn
        # distributed contact, node by node
        Fb = 0.0; pmax = -1.0; xm = -1.0; nact = 0
        for jj in range(j1 - j0):
            l = j0 + jj
            bfree = hb[jj] - y2[l]
            a = eta_p[jj]
            if a > 0.0 or bfree > 0.0:
                z = _contact_root(a, bfree, phi * kb)
                f = (bfree - z) / phi
                y2[l] = hb[jj] - z
                Fb += h * f
                if f != 0.0:
                    nact += 1
            else:
                z = bfree
            eta_p[jj] = eta_c[jj]  # shift after use: eta_p <- eta^n
            eta_c[jj] = z          # eta^{n+1}
            if z > pmax:
                pmax = z; xm = l * h
        if nact > max_act:
            max_act = nact
        pen[n] = pmax; xpen[n] = xm
        Fnut[n] = T * y2[N - 1] / h + EI * (2 * y2[N - 1] - y2[N - 2]) / h ** 3
        F0 = T * y2[1] / h + EI * (2 * y2[1] - y2[2]) / h ** 3
        Fbr[n] = F0 - Fc - Fb
        if rec_energy:
            e = 0.0
            for l in range(N):
                v = (y2[l] - y1[l]) / k
                e += h * 0.5 * mu * v * v
                e += h * 0.5 * T * ((y2[l + 1] - y2[l]) / h) * ((y1[l + 1] - y1[l]) / h)
            for l in range(1, N):
                e += h * 0.5 * EI * ((y2[l + 1] - 2 * y2[l] + y2[l - 1]) / (h * h)) * ((y1[l + 1] - 2 * y1[l] + y1[l - 1]) / (h * h))
            # boundary terms of delta_xx are zero (y=0 and ghost antisymmetry)
            for jj in range(j1 - j0):
                zc = eta_c[jj] if eta_c[jj] > 0.0 else 0.0   # eta^{n+1}
                zp = eta_p[jj] if eta_p[jj] > 0.0 else 0.0   # eta^{n}
                e += h * 0.5 * (0.5 * kb * zc * zc + 0.5 * kb * zp * zp)
            e += 0.25 * kc * (y2[ic] ** 2 + y1[ic] ** 2)
            H[n] = e
        for l in range(N + 1):
            y0[l] = y1[l]; y1[l] = y2[l]
    return Fnut, Fbr, H, pen, xpen, max_act


DEF = dict(L=1.0, rho=7850.0, A=6.16e-8, E=2.0e11, I=3.02e-16, f1=130.81, h=1e-3, ovs=20,
           sigma0=0.6, sigma1=5e-4, lossless=False,
           xc=0.005, kc=1.2e5, rc=0.5, thread=True,
           bridge=True, xb=0.010, h0=0.0, curv=3.0, kb=1e11, br_x1=0.006,
           Fe_amp=0.2, Fe_rise=0.010, Fe_fall=0.001, dur=1.0)


def simulate_fd(over=None, energy=False):
    p = dict(DEF)
    if over:
        p.update(over)
    L = p["L"]; mu = p["rho"] * p["A"]; EI = p["E"] * p["I"]
    T = mu * (2 * L * p["f1"]) ** 2 - EI * (np.pi / L) ** 2
    h = p["h"]; N = int(round(L / h)); h = L / N
    fs = 44100.0 * p["ovs"]; k = 1 / fs
    s0, s1 = (0.0, 0.0) if p["lossless"] else (p["sigma0"], p["sigma1"])
    g2, kk2 = T / mu, EI / mu
    q = g2 * k * k + 4 * s1 * k
    hmin = np.sqrt(0.5 * (q + np.sqrt(q * q + 16 * kk2 * k * k)))
    if h < hmin:
        raise ValueError(f"unstable: h={h} < hmin={hmin}")
    x = np.arange(N + 1) * h
    ic = int(round(p["xc"] / h))
    kc, rc = (p["kc"], p["rc"]) if p["thread"] else (0.0, 0.0)
    if p["bridge"]:
        j0 = ic + 1 if p["thread"] else int(round((p["xb"] - 0.006) / h))
        j1 = int(round((p["xb"] + p["br_x1"]) / h)) + 1
        hb = p["h0"] - p["curv"] * (x[j0:j1] - p["xb"]) ** 2
    else:
        j0, j1 = 1, 1; hb = np.zeros(0)
    psi_e = 2 * np.sin(np.pi * x / L) ** 2 / L
    tr = np.arange(int(round(p["Fe_rise"] * fs))) / fs
    tf = np.arange(int(round(p["Fe_fall"] * fs))) / fs
    Fe = np.concatenate([p["Fe_amp"] * np.sin(0.5 * np.pi * tr / p["Fe_rise"]) ** 2,
                         p["Fe_amp"] * np.cos(0.5 * np.pi * tf / p["Fe_fall"]) ** 2, [0.0]])
    nsteps = int(round(p["dur"] * fs))
    out = _run_fd(N, nsteps, h, k, mu, T, EI, s0, s1, ic, kc, rc, j0, j1, hb, float(p["kb"]), psi_e, Fe,
                  bool(energy))
    Fnut, Fbr, H, pen, xpen, max_act = out
    return p, dict(N=N, h=h, fs=fs, T=T, hmin=hmin, j0=j0, j1=j1), dict(Fnut=Fnut, Fbridge=Fbr, H=H, pen=pen,
                                                                           xpen=xpen, max_act=max_act)
