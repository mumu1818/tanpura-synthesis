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

Optional extensions, feature parity with tanpura1d.py (all OFF by default; the default behaviour is bit-identical
to the original model; they change only the set-up -- the time loop _run is unchanged).  See MODEL_2D_PLUCK.md.
  * profile=(x_rel, h_rel): bridge surface relative to the string's rest line (statics.profile_for_dynamics), as
    tanpura1d (same bridge points: tanpura1d.profile_points).
  * static_init=True: start from the EXACT static equilibrium of the discrete 2-D scheme (static_equilibrium():
    bridge contact, thread in y and z, nut anisotropy, tension modulation; Newton on the convex potential).
    The older init_static / static_state() (h0 > 0 only, alpha = 1, z-thread reaction neglected) is kept unchanged.
  * T=<N>: tension override;  tune='speaking': T such that the lowest linear eigenfrequency of the discrete scheme
    with the thread spring (contact off) equals f1, computed by tanpura1d.tune_tension.  It tunes the AXIS-1 series
    of the nut coupling (= y when theta_nut = 0 or pol2 off); axis 2 is detune_cents sharper.  The two series are
    exactly decoupled (so the tuning is exact) when the thread is isotropic (kcz = kc, rcz = rc) or theta_nut = 0.
    tune_ref='mean' instead tunes axis 1 to f1 2^(-detune/2400), i.e. the two series straddle f1.
  * pluck='finger' (xp, wp, pluck_pull, pluck_release, pluck_disp): the tanpura1d finger-pad pluck, applied along
    the direction pluck_deg from the bridge normal y towards +z (90 = parallel to the soundboard).  pluck_disp is
    the static pad displacement ALONG the pluck direction (discrete scheme, thread springs on, contact and TM off).
  * loss='valette' (eta_air, rho_air, delta_ve, Qte_inv, Yb, mb, fb): tanpura1d.loss_sigma, the same sigma_i for
    both polarisations (evaluated at the axis-1 frequencies; common damping keeps the rotated update symmetric).

Round-3 options (MODEL_2D_ROUND3.md; all OFF by default, then the original kernel _run runs, bit-identical).  Any of them
switches simulate() to the kernel _run3, which without new physics reproduces _run bit for bit.
  * groove_kappa (kappa_g, 1/m), groove_z0 (m), groove_slope: transverse bridge profile.  The gap becomes
    eta_j = h_b(x_j) + kappa_g z_j^2 + s z_j - y_j,  s = groove_slope - 2 kappa_g groove_z0
    (kappa_g > 0: the string lies in a groove whose walls rise as it swings sideways; < 0: crowned face; z0: the groove
    bottom is z0 to the side, i.e. the string rests on the groove wall, slope -2 kappa_g z0).  Energy-consistent:
    f = [V(eta^{n+1}) - V(eta^{n-1})] / (eta^{n+1} - eta^{n-1}), force density +f on y and -f (kappa_g (z^{n+1} + z^{n-1}) + s)
    on z, so the contact work equals -[V(eta^{n+1}) - V(eta^{n-1})] exactly (discrete chain rule).  The contact couples y
    and z at every active point, so the step is solved by ONE Newton on (s_a = eta_a^{n+1} - eta_a^{n-1}, Z_a = z_a^{n+1})
    of the candidate points, with friction -mu(w) f along z (w = (Z_a - z_a^{n-1}) / 2 dt, strictly dissipative).
    static_init includes the groove (_static_equilibrium_groove).
  * Yb_y, Yb_z, drain_body, drain_weights: polarisation-dependent body drain (needs loss='valette').
    sigma_y = sigma_str + (T/L) [Re Y_smooth(Yb) + Yb_y shape(f)]  (shape: body.py explicit modes, body_mobility_shape),
    sigma_z = sigma_str + (T/L) Re Y_smooth(Yb_z).  Non-proportional across the rotated nut pair; per pair
    m [delta_tt x + 2 diag(sigma*_y, sigma*_z) delta_t. x + K* x^n] = F^n (energy-consistent, dissipative, stable).
  * solver='joint' forces the joint Newton (kappa_g = 0 test); rec_modes=N records per-mode energies every N steps.
"""
from __future__ import annotations

import time
import numpy as np
from numba import njit

import tanpura1d as T1

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
    # optional extensions (None/False = original behaviour, bit-identical); same meaning as in tanpura1d
    profile=None,        # (x_rel, h_rel) arrays: bridge surface relative to the rest line (+ = pressed)
    static_init=False,   # start from the exact static equilibrium of the discrete scheme (static_equilibrium)
    T=None,              # tension override [N]
    tune=None,           # None | 'speaking'
    tune_ref="axis1",    # 'axis1': nut axis-1 series speaks at f1 | 'mean': the two series straddle f1
    pluck=None,          # None | 'finger'  (direction: pluck_deg)
    xp=None,             # pluck position from the bridge-side termination [m]; None -> L/3
    wp=0.012,            # pad width [m]
    pluck_pull=0.040,    # sin^2 pull [s]
    pluck_release=0.002, # cos^2 release [s]
    pluck_disp=None,     # target static pad displacement along the pluck direction [m] (overrides Fe_amp)
    loss=None,           # None | 'valette'
    eta_air=1.8e-5, rho_air=1.2, delta_ve=4.5e-3, Qte_inv=2.03e-4, Yb=0.0, mb=0.0, fb=0.0,
    # round-3 extensions (MODEL_2D_ROUND3.md); all OFF by default -> the original kernel _run, bit-identical
    groove_kappa=None,   # transverse bridge profile: gap eta = h_b(x) + kappa z^2 + s z - y   [1/m]; None = off
    groove_z0=0.0,       # groove bottom offset [m]: s = groove_slope - 2 kappa z0 (string rests on the groove wall)
    groove_slope=0.0,    # face tilt across the string (dimensionless), adds s z
    Yb_y=0.0,            # bridge-normal MODAL body mobility: peak Re Y of body mode T(0,1) [s/kg] (needs loss='valette')
    Yb_z=None,           # in-plane smooth mobility [s/kg] replacing Yb for z (None: = Yb, isotropic)
    drain_body="tanpura_male",   # body.py preset whose explicit modes shape Re Y_y(f)
    drain_weights="gain",        # 'gain': all explicit modes, body.py gains | 'axisym': A0 + m = 0 tabli modes only
    solver=None,         # None | 'joint': force the round-3 kernel's joint (y, z) contact Newton
    rec_modes=0,         # > 0: record per-mode energies every rec_modes steps (round-3 kernel)
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
    if p.get("T") is not None:
        T = float(p["T"])
    fs = p["fs"]; dt = 1.0 / fs
    ii = np.arange(1, 5000)
    beta = ii * np.pi / L
    if p.get("tune"):
        T = tune_tension(p, T, mu, EI, L, beta)
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
    elif p.get("loss") is not None:
        sig = T1.loss_sigma(p, T, mu, EI, L, beta, om)
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
    if p.get("pluck") is not None:
        xp, wp = T1.pluck_window(p)
        ge = project(L, beta, xp, wp)
    else:
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
    if p["bridge"] and p.get("profile") is not None:
        xj, dxb, hj = T1.profile_points(p)
        Nb = xj.size
        Phi = np.ascontiguousarray(np.sin(xj[:, None] * beta[None, :]))
        Q0 = np.ascontiguousarray(dxb * (Phi * c[None, :]) @ Phi.T)
        Q0 = 0.5 * (Q0 + Q0.T)
        v0 = Phi @ (c * gc)
    elif p["bridge"]:
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
    out = dict(L=L, mu=mu, EI=EI, EA=EA, T=T, fs=fs, dt=dt, M=M, beta=beta, om=om, om2=om_ax[1], sig=sig,
               b=b, c=c, m=m, sigs=sigs, Ayy=Ayy, Ayz=Ayz, Azz=Azz, Kyy=Kyy, Kyz=Kyz, Kzz=Kzz, theta=th,
               ge=ge, gc=gc, kc=kc, rc=rc, kcz=kcz, rcz=rcz, kapy=kapy, lamy=lamy, kapz=kapz, lamz=lamz,
               xj=xj, dxb=dxb, hj=hj, Nb=Nb, Phi=Phi, Q0=Q0, v0=v0, w0=w0, wn=wn, sgn=sgn,
               cphi=np.cos(ph), sphi=np.sin(ph), use_z=use_z, k_ax1=ax_k[0], k_ax2=ax_k[1])
    _round3_setup(p, out)
    return out


def re_y_smooth(f, Yb, mb=0.0, fb=0.0):
    """Smooth bridge mobility of tanpura1d.loss_sigma: Re Y = Yb r(f) / (1 + (2 pi f mb Yb)^2), r = (f/fb)^2/(1+(f/fb)^2)."""
    ReY = np.full_like(np.asarray(f, float), float(Yb))
    if fb:
        r2 = (f / fb) ** 2
        ReY *= r2 / (1.0 + r2)
    if mb:
        ReY /= 1.0 + (2 * np.pi * f * mb * Yb) ** 2
    return ReY


def body_mobility_shape(f, preset="tanpura_male", weights="gain", **overrides):
    """Shape of the bridge-normal MODAL mobility from body.py's explicit low modes (A0, T(m,n), A1):
        Re Y(f) / Yb_y = sum_k w_k (f f_k/Q_k)^2 / ((f_k^2 - f^2)^2 + (f f_k/Q_k)^2),
    i.e. the real part of a sum of single-DOF modal mobilities Y_k = j w Y_pk (w_k/Q_k) / (w_k^2 - w^2 + j w w_k/Q_k),
    each of peak value w_k at f_k.  ASSUMPTION: the relative peak mobilities w_k = 10^((g_k - g_T(0,1))/20) are
    body.py's relative peak radiation gains g_k (pressure ~ mobility x radiation factor; the monopole-like A0 and
    axisymmetric tabli modes dominate both).  weights='axisym': only A0 and the axisymmetric T(0,n) modes, the only
    ones a bridge-normal force at the centre of a circular plate drives (the m >= 1 rocking modes and the cavity
    mode A1 are left out).  Returns an array (1 at the T(0,1) peak if it is isolated)."""
    import body
    if weights not in ("gain", "axisym"):
        raise ValueError(f"unknown drain_weights {weights!r}")
    bp = body.get_params(preset, **overrides)
    modes = bp["modes"] if bp["modes"] is not None else body.explicit_modes(bp)
    f = np.asarray(f, float)
    tot = np.zeros_like(f)
    for (_name, fam, f0, Q, gdb, _m2) in modes:
        if weights == "axisym" and not (_name.startswith("A0") or _name.startswith("T(0,")):
            continue
        fk = f0 * (bp["air_scale"] if fam == "air" else bp["plate_scale"])
        w = 10.0 ** ((gdb - bp["t1_gain_db"]) / 20.0)
        x = (f * fk / Q) ** 2
        tot += w * x / ((fk * fk - f * f) ** 2 + x)
    return tot


def _round3_setup(p, s):
    """Round-3 options (MODEL_2D_ROUND3.md), added to the set-up dict s in place; nothing else in s changes unless
    the per-polarisation damping is on (then K* and the axis stiffnesses are the per-polarisation scheme's).
      groove:  gap eta = h_b(x) + kg z^2 + sg z - y,  kg = groove_kappa, sg = groove_slope - 2 kg groove_z0.
      perpol:  sigma_y = sigma_str + (T/L)[Re Y_smooth(Yb) + Yb_y shape(f)],  sigma_z = sigma_str + (T/L) Re Y_smooth(Yb_z),
               sigma_str = Valette string losses (loss_sigma with Yb = 0), all at the axis-1 frequencies.
               Update per mode pair (energy-consistent, non-proportional damping):
                 m [delta_tt x + 2 S* delta_t. x + K* x^n] = F^n,  S* = diag(sigma*_y, sigma*_z), sigma* = tanh(sigma dt)/dt,
                 K* = R diag(k*_1, k*_2) R^T, k*_k = exact-pole omega*^2 of axis k with sigma_k = R-projected damping,
               i.e. x^{n+1} = (I + S* dt)^-1 [(2 I - dt^2 K*) x^n - (I - S* dt) x^{n-1} + dt^2 F^n / m]."""
    M = s["M"]
    gk = p.get("groove_kappa")
    s["groove"] = gk is not None
    s["kg"] = float(gk) if gk is not None else 0.0
    s["sg"] = float(p.get("groove_slope", 0.0) or 0.0) - 2.0 * s["kg"] * float(p.get("groove_z0", 0.0) or 0.0)
    if s["groove"] and not s["use_z"]:
        raise ValueError("groove_kappa needs pol2=True")
    perpol = bool(p.get("Yb_y", 0.0)) or p.get("Yb_z") is not None
    s["perpol"] = perpol
    if not perpol:
        s.update(Azy=s["Ayz"], by=s["b"], bz=s["b"], cy=s["c"], cz=s["c"], sigs_y=s["sigs"], sigs_z=s["sigs"],
                 sig_y=s["sig"], sig_z=s["sig"], Q0y=s["Q0"], Q0z=s["Q0"], v0y=s["v0"], v0z=s["v0"],
                 ReY_y=None, ReY_z=None)
        return
    if p.get("loss") != "valette" or p["lossless"]:
        raise ValueError("per-polarisation body drain (Yb_y / Yb_z) needs loss='valette' and lossless=False")
    T, L, dt, m, om, beta = s["T"], s["L"], s["dt"], s["m"], s["om"], s["beta"]
    f = om / (2 * np.pi)
    sig_str = T1.loss_sigma(dict(p, Yb=0.0), T, s["mu"], s["EI"], L, beta, om)
    ReY_y = re_y_smooth(f, p["Yb"], p["mb"], p["fb"]) + \
        float(p["Yb_y"]) * body_mobility_shape(f, p["drain_body"], p.get("drain_weights", "gain"))
    Ybz = p["Yb"] if p.get("Yb_z") is None else float(p["Yb_z"])
    ReY_z = re_y_smooth(f, Ybz, p["mb"], p["fb"])
    sig_y = sig_str + T * ReY_y / L
    sig_z = sig_str + T * ReY_z / L
    th = s["theta"]
    cs, sn = np.cos(th), np.sin(th)
    by, bz = np.exp(-2 * sig_y * dt), np.exp(-2 * sig_z * dt)
    sigs_y, sigs_z = (1 - by) / ((1 + by) * dt), (1 - bz) / ((1 + bz) * dt)
    ks = []
    for o, sk in ((om, cs * cs * sig_y + sn * sn * sig_z), (s["om2"], sn * sn * sig_y + cs * cs * sig_z)):
        omd = np.sqrt(np.maximum(o ** 2 - sk ** 2, 0.0))
        a = 2 * np.exp(-sk * dt) * np.cos(omd * dt)
        bb = np.exp(-2 * sk * dt)
        ks.append((2 - 2 * a / (1 + bb)) / dt ** 2)
    Kyy = cs * cs * ks[0] + sn * sn * ks[1]
    Kzz = sn * sn * ks[0] + cs * cs * ks[1]
    Kyz = cs * sn * (ks[0] - ks[1])
    if not s["use_z"]:
        Kyz = np.zeros(M); Kzz = np.zeros(M)
    hy, hz = 0.5 * (1 + by), 0.5 * (1 + bz)
    cy, cz = dt ** 2 * hy / m, dt ** 2 * hz / m
    Phi, dxb, gc = s["Phi"], s["dxb"], s["gc"]
    Q0y = dxb * (Phi * cy[None, :]) @ Phi.T
    Q0z = dxb * (Phi * cz[None, :]) @ Phi.T
    s.update(Ayy=hy * (2 - dt ** 2 * Kyy), Ayz=-hy * dt ** 2 * Kyz, Azy=-hz * dt ** 2 * Kyz, Azz=hz * (2 - dt ** 2 * Kzz),
             by=by, bz=bz, cy=cy, cz=cz, sigs_y=sigs_y, sigs_z=sigs_z, sig_y=sig_y, sig_z=sig_z,
             Kyy=Kyy, Kyz=Kyz, Kzz=Kzz, k_ax1=ks[0], k_ax2=ks[1],
             Q0y=np.ascontiguousarray(0.5 * (Q0y + Q0y.T)), Q0z=np.ascontiguousarray(0.5 * (Q0z + Q0z.T)),
             v0y=Phi @ (cy * gc), v0z=Phi @ (cz * gc), ReY_y=ReY_y, ReY_z=ReY_z, sig_str=sig_str)
    if not s["use_z"]:
        s["Azz"] = np.zeros(M)


def tune_tension(p, T0, mu, EI, L, beta):
    """tune='speaking': tension for which the lowest linear eigenfrequency of the DISCRETE scheme with the thread
    spring (contact off) of the nut's axis-1 series equals f1 (tune_ref='axis1'), or f1 2^(-detune/2400)
    (tune_ref='mean': axis 1 and axis 2 = axis 1 + detune_cents straddle f1).  tanpura1d.tune_tension does the work,
    so with pol2 off the tension is identical to tanpura1d's.  Exact for the axis-1 series when the thread is
    isotropic (kcz = kc, rcz = rc) or theta_nut = 0: the linear system then separates into the two principal axes."""
    ref = p.get("tune_ref", "axis1")
    if ref == "axis1" or not p["pol2"]:
        return T1.tune_tension(p, T0, mu, EI, L, beta)
    if ref == "mean":
        return T1.tune_tension(dict(p, f1=p["f1"] * 2.0 ** (-p["detune_cents"] / 2400.0)), T0, mu, EI, L, beta)
    raise ValueError(f"unknown tune_ref {ref!r}")


def stiffness_matrix(s):
    """Static stiffness of the discrete linear scheme (contact and TM off), unknowns x = [q (y), p (z)]:
    per mode m [[K*yy, K*yz], [K*yz, K*zz]] (the scheme's own omega*^2, rotated by theta_nut) plus the thread
    springs k_c g_c g_c^T in y and k_cz g_c g_c^T in z.  Size M (pol2 off) or 2M."""
    M = s["M"]; m = s["m"]; gc = s["gc"]
    n = 2 * M if s["use_z"] else M
    K = np.zeros((n, n))
    iy = np.arange(M)
    K[iy, iy] = m * s["Kyy"]
    K[:M, :M] += s["kc"] * np.outer(gc, gc)
    if s["use_z"]:
        iz = iy + M
        K[iy, iz] = m * s["Kyz"]; K[iz, iy] = m * s["Kyz"]
        K[iz, iz] = m * s["Kzz"]
        K[M:, M:] += s["kcz"] * np.outer(gc, gc)
    return K


def static_compliance(s):
    """Static pad displacement ALONG the pluck direction e = (cos phi, sin phi) per unit excitation force:
    (g_e x e)^T K^-1 (g_e x e) for the discrete scheme with the thread springs (contact off, TM off).
    pol2 off: exactly tanpura1d.static_compliance (Sherman-Morrison on diag(m omega*^2) + k_c g_c g_c^T)."""
    ge, gc, kc = s["ge"], s["gc"], s["kc"]
    if not s["use_z"]:
        D = s["m"] * s["Kyy"]
        a = np.sum(ge * ge / D)
        if kc == 0.0:
            return float(a)
        b = np.sum(ge * gc / D)
        g = np.sum(gc * gc / D)
        return float(a - kc * b * b / (1.0 + kc * g))
    F = np.concatenate([s["cphi"] * ge, s["sphi"] * ge])
    return float(F @ np.linalg.solve(stiffness_matrix(s), F))


def excitation_amp(p, s=None):
    """Force amplitude: Fe_amp, or pluck_disp / static_compliance if pluck_disp is set."""
    if p.get("pluck_disp") is None:
        return p["Fe_amp"]
    if s is None:
        s = build(p)
    return float(p["pluck_disp"]) / static_compliance(s)


def excitation(p, s=None):
    """Excitation force history F_e(t), applied through g_e along (cos phi, sin phi): sin^2 rise, cos^2 fall.
    Whole-string pulse (default): Fe_rise / Fe_fall, amplitude Fe_amp.  Finger pluck (pluck='finger'): pluck_pull /
    pluck_release, amplitude from pluck_disp if set (needs the set-up s; built if not given)."""
    fs = p["fs"]
    if p.get("pluck") is not None:
        rise, fall = p["pluck_pull"], p["pluck_release"]
    else:
        rise, fall = p["Fe_rise"], p["Fe_fall"]
    amp = excitation_amp(p, s)
    tr = np.arange(int(round(rise * fs))) / fs
    tf = np.arange(int(round(fall * fs))) / fs
    return np.concatenate([amp * np.sin(0.5 * np.pi * tr / rise) ** 2,
                           amp * np.cos(0.5 * np.pi * tf / fall) ** 2, [0.0]])


def static_equilibrium(p, s, maxit=200):
    """EXACT static equilibrium of the DISCRETE 2-D scheme: the fixed point x^{n+1} = x^n = x^{n-1} of _run with
    bridge contact, thread springs in y and z, the rotated nut stiffness and Kirchhoff-Carrier tension modulation.

    It is the minimiser of the convex potential
        V(x) = 1/2 x^T K x + (EA/8L) P(x)^2 + dxb k_b/(alpha+1) sum_j [h_j - Phi_j q]_+^(alpha+1),
        P = (L/2) sum beta_i^2 (q_i^2 + p_i^2),   K = stiffness_matrix(s)  (the scheme's omega*^2, not omega^2),
    because at rest the trapezoidal thread force is -k_c u, the discrete-gradient contact force is k_b eta^alpha,
    the linearly implicit TM force is -(EA/4) P beta_i^2 x_i, friction vanishes (no slip) and the update
    m [delta_tt x + 2 sigma* delta_t. x + K*_i x] = F reduces to m K*_i x_i = F_i.  Newton with backtracking
    (generalises tanpura1d.static_state; identical problem when pol2 and tm are off).  A vertical preload makes a
    small z deflection when the nut is anisotropic (theta_nut, detune_cents).  Returns (q, p, info)."""
    M = s["M"]
    use_z = bool(s["use_z"])
    q = np.zeros(M); pz = np.zeros(M)
    info = dict(static_force=0.0, thread_force=0.0, thread_force_z=0.0, n_active=0, max_pen=0.0, iters=0,
                residual=0.0, max_z=0.0, kind="exact")
    if not p["bridge"] or np.max(s["hj"]) <= 0.0:
        return q, pz, info
    if s.get("groove"):
        return _static_equilibrium_groove(p, s, maxit)
    K = stiffness_matrix(s)
    n = K.shape[0]
    gc = s["gc"]
    hj, dxb = s["hj"], s["dxb"]
    PhiF = np.zeros((hj.size, n)); PhiF[:, :M] = s["Phi"]
    kb, al = float(p["kb"]), float(p["alpha"])
    ktm = s["EA"] / 8.0 if p["tm"] else 0.0
    L = s["L"]
    B2 = np.concatenate([s["beta"] ** 2] * (2 if use_z else 1))

    def P_of(x):
        return 0.5 * L * np.sum(B2 * x * x)

    def energy(x):
        eta = np.maximum(hj - PhiF @ x, 0.0)
        return 0.5 * x @ (K @ x) + ktm / L * P_of(x) ** 2 + dxb * kb / (al + 1) * np.sum(eta ** (al + 1))

    def grad(x):
        eta = hj - PhiF @ x
        act = eta > 0
        g = K @ x + 2.0 * ktm * P_of(x) * B2 * x - dxb * PhiF[act].T @ (kb * eta[act] ** al)
        return g, eta, act

    x = np.zeros(n)
    it = 0
    for it in range(1, maxit + 1):
        g, eta, act = grad(x)
        Hs = K + 2.0 * ktm * (P_of(x) * np.diag(B2) + L * np.outer(B2 * x, B2 * x))
        if act.any():
            wgt = dxb * kb * al * eta[act] ** (al - 1)
            Hs += (PhiF[act].T * wgt) @ PhiF[act]
        dx = np.linalg.solve(Hs, -g)
        E0 = energy(x)
        tstep = 1.0
        while energy(x + tstep * dx) > E0 + 1e-4 * tstep * (g @ dx) + 1e-14 * abs(E0) and tstep > 1e-12:
            tstep *= 0.5
        x = x + tstep * dx
        if np.max(np.abs(tstep * dx)) < 1e-18 + 1e-15 * np.max(np.abs(x)):
            break
    g, eta, act = grad(x)
    q = x[:M].copy()
    if use_z:
        pz = x[M:].copy()
    fa = kb * np.maximum(eta, 0.0) ** al
    info = dict(static_force=float(dxb * fa.sum()), thread_force=float(-s["kc"] * (gc @ q)),
                thread_force_z=float(-s["kcz"] * (gc @ pz)), n_active=int(act.sum()), max_pen=float(np.max(eta)),
                iters=it, residual=float(np.max(np.abs(g))), max_z=float(np.max(np.abs(pz))) if use_z else 0.0,
                x_active=s["xj"][act].copy(), kind="exact")
    return q, pz, info


def _static_equilibrium_groove(p, s, maxit=200):
    """static_equilibrium with the transverse groove: the gap is eta_j = h_j + kg Z_j^2 + sg Z_j - Y_j (Y = Phi q,
    Z = Phi p), so the contact potential dxb k_b/(alpha+1) sum [eta_j]_+^(alpha+1) also depends on z.  It is still
    the scheme's fixed point: at rest the discrete-gradient contact forces are k_b eta^alpha (+y) and
    -k_b eta^alpha (2 kg Z + sg) (z), exactly the gradient used here.  Convex for kg >= 0 (eta convex in (y, z),
    V convex non-decreasing); Newton with backtracking as in static_equilibrium."""
    M = s["M"]
    K = stiffness_matrix(s)
    n = K.shape[0]
    gc, hj, dxb, Phi = s["gc"], s["hj"], s["dxb"], s["Phi"]
    kg, sg = s["kg"], s["sg"]
    kb, al = float(p["kb"]), float(p["alpha"])
    ktm = s["EA"] / 8.0 if p["tm"] else 0.0
    L = s["L"]
    B2 = np.concatenate([s["beta"] ** 2] * 2)

    def P_of(x):
        return 0.5 * L * np.sum(B2 * x * x)

    def eta_of(x):
        Z = Phi @ x[M:]
        return hj + kg * Z * Z + sg * Z - Phi @ x[:M], Z

    def energy(x):
        eta, _ = eta_of(x)
        eta = np.maximum(eta, 0.0)
        return 0.5 * x @ (K @ x) + ktm / L * P_of(x) ** 2 + dxb * kb / (al + 1) * np.sum(eta ** (al + 1))

    def grad(x):
        eta, Z = eta_of(x)
        act = eta > 0
        fa = kb * np.where(act, eta, 0.0) ** al
        g = K @ x + 2.0 * ktm * P_of(x) * B2 * x
        g[:M] -= dxb * Phi.T @ fa
        g[M:] += dxb * Phi.T @ (fa * (2 * kg * Z + sg))
        return g, eta, act, Z, fa

    x = np.zeros(n)
    it = 0
    for it in range(1, maxit + 1):
        g, eta, act, Z, fa = grad(x)
        Hs = K + 2.0 * ktm * (P_of(x) * np.diag(B2) + L * np.outer(B2 * x, B2 * x))
        if act.any():
            Pa = Phi[act]
            G = np.concatenate([-Pa, (2 * kg * Z[act] + sg)[:, None] * Pa], axis=1)   # d eta / d x
            wgt = dxb * kb * al * eta[act] ** (al - 1)
            Hs += (G.T * wgt) @ G
            Hs[M:, M:] += dxb * 2 * kg * (Pa.T * fa[act]) @ Pa
        dx = np.linalg.solve(Hs, -g)
        E0 = energy(x)
        tstep = 1.0
        while energy(x + tstep * dx) > E0 + 1e-4 * tstep * (g @ dx) + 1e-14 * abs(E0) and tstep > 1e-12:
            tstep *= 0.5
        x = x + tstep * dx
        if np.max(np.abs(tstep * dx)) < 1e-18 + 1e-15 * np.max(np.abs(x)):
            break
    g, eta, act, Z, fa = grad(x)
    q = x[:M].copy(); pz = x[M:].copy()
    info = dict(static_force=float(dxb * fa.sum()), static_force_z=float(-dxb * np.sum(fa * (2 * kg * Z + sg))),
                thread_force=float(-s["kc"] * (gc @ q)), thread_force_z=float(-s["kcz"] * (gc @ pz)),
                n_active=int(act.sum()), max_pen=float(np.max(eta)), iters=it, residual=float(np.max(np.abs(g))),
                max_z=float(np.max(np.abs(pz))), x_active=s["xj"][act].copy(), kind="exact+groove")
    return q, pz, info


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


@njit(cache=True)
def _run3(N, Ayy, Ayz, Azy, Azz, by, bz, cy, cz, beta, beta2, ge, Fe, cphi, sphi,
          gc, kapy, lamy, kcy, rcy, kapz, lamz, kcz, rcz,
          use_z, use_bridge, use_fric, use_tm, joint, kg, sg, perpol,
          Phi, hj, dxb, Q0y, Q0z, v0y, v0z, kb, alpha, mu1, mu2,
          ktm, L, w0, wn, sgn,
          tol, maxit, ftol, fmaxit, guess,
          rec_energy, m, Kyy, Kyz, Kzz, sigsy, sigsz, dt, q_init, p_init, rec_every, rec_modes):
    """Round-3 time loop (MODEL_2D_ROUND3.md).  Same scheme as _run, plus
      * per-polarisation linear coefficients (non-proportional damping): y forces act through cy, z forces through cz,
        x^{n+1} = A x^n - diag(by, bz) x^{n-1} + diag(cy, cz) F^n with A = (I + S* dt)^-1 (2 I - dt^2 K*) (not symmetric);
      * transverse groove: eta_j = h_j + kg z_j^2 + sg z_j - y_j; discrete gradient f = [V(eta^{n+1}) - V(eta^{n-1})] /
        (eta^{n+1} - eta^{n-1}); force density +f on y and -f (kg (z^{n+1} + z^{n-1}) + sg) on z (exact discrete chain rule);
      * joint=True: one Newton on (s_a = eta_a^{n+1} - eta_a^{n-1}, Z_a = z_a^{n+1}) for the candidate points, friction
        -mu(w) f along z included (w = (Z - z^{n-1}) / 2dt), backtracking on the residual norm.
    With joint=False, kg = sg = 0 and equal y/z coefficients the arithmetic is that of _run (block-triangular solve)."""
    M = by.shape[0]
    Nb = hj.shape[0]
    q0 = q_init.copy(); q1 = q_init.copy(); q2 = np.zeros(M)
    p0 = p_init.copy(); p1 = p_init.copy(); p2 = np.zeros(M)
    qh = np.zeros(M); ph = np.zeros(M); cpy = np.zeros(M); cpz = np.zeros(M); qs = np.zeros(M); ps = np.zeros(M)
    Fby = np.zeros(N); Fbz = np.zeros(N); Fny = np.zeros(N); Fnz = np.zeros(N)
    Fcont = np.zeros(N); Ffric = np.zeros(N)
    H = np.zeros(N); Dcum = np.zeros(N); Dfric = np.zeros(N); Wcum = np.zeros(N); pen = np.zeros(N)
    NR = N // rec_every + 1
    rEy = np.zeros(NR); rEz = np.zeros(NR); rdT = np.zeros(NR); rslip = np.zeros(NR)
    NM = N // rec_modes + 1 if rec_modes > 0 else 1
    mEy = np.zeros((NM, M)); mEz = np.zeros((NM, M)); mEx = np.zeros((NM, M))
    eta_p = np.zeros(Nb); eta_c = np.zeros(Nb); eta_n = np.zeros(Nb)
    grv = kg != 0.0 or sg != 0.0
    for j in range(Nb):
        acc = 0.0; accz = 0.0
        for i in range(M):
            acc += Phi[j, i] * q_init[i]
            accz += Phi[j, i] * p_init[i]
        eta_p[j] = hj[j] - acc + (kg * accz + sg) * accz
        eta_c[j] = eta_p[j]
    yt = np.zeros(Nb); zt = np.zeros(Nb); zn = np.zeros(Nb); bv = np.zeros(Nb)
    inS = np.zeros(Nb, dtype=np.bool_)
    sidx = np.zeros(Nb, dtype=np.int64)
    s = np.zeros(Nb); f = np.zeros(Nb); fp = np.zeros(Nb); r = np.zeros(Nb); dl = np.zeros(2 * Nb)
    act = np.zeros(Nb, dtype=np.int64)
    vSy = np.zeros(Nb); vSz = np.zeros(Nb)
    QB = np.zeros((Nb, Nb)); QBz = np.zeros((Nb, Nb)); QY = np.zeros((Nb, Nb)); QZ = np.zeros((Nb, Nb))
    J = np.zeros((2 * Nb, 2 * Nb)); rhs = np.zeros(2 * Nb)
    fA = np.zeros(Nb); hA = np.zeros(Nb); wA = np.zeros(Nb); whA = np.zeros(Nb); wt = np.zeros(Nb)
    muA = np.zeros(Nb); mupA = np.zeros(Nb); rz = np.zeros(Nb); dz = np.zeros(Nb)
    jA = np.zeros(Nb, dtype=np.int64)
    # joint-Newton work arrays (candidate-set indexed)
    Zc = np.zeros(Nb); Zp = np.zeros(Nb); ztS = np.zeros(Nb); gg = np.zeros(Nb); gs_ = np.zeros(Nb)
    gZ = np.zeros(Nb); r2 = np.zeros(Nb); dsv = np.zeros(Nb); dZv = np.zeros(Nb)
    s_t = np.zeros(Nb); Z_t = np.zeros(Nb); f_t = np.zeros(Nb); g_t = np.zeros(Nb)
    tolz = ftol * 2.0 * dt
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
    thcc0y = 0.0; thcc0z = 0.0
    for i in range(M):
        thcc0y += cy[i] * gc[i] * gc[i]
        thcc0z += cz[i] * gc[i] * gc[i]
    inv2dt = 1.0 / (2.0 * dt)
    max_pen = 0.0; n_contact = 0; tot_it = 0; max_it = 0; n_fail = 0; max_res = 0.0
    max_Fcy = 0.0; max_Fcz = 0.0; edge_pen = 0.0; max_nact = 0; n_viol = 0
    n_fric = 0; f_tot_it = 0; f_max_it = 0; f_fail = 0; f_max_res = 0.0; max_w = 0.0; max_dT = 0.0
    max_mu = 0.0; n_ls = 0; max_zres = 0.0
    Dacc = 0.0; Facc = 0.0; Wacc = 0.0
    Eyacc = 0.0; Ezacc = 0.0; slipacc = 0.0
    nFe = Fe.shape[0]
    for n in range(N):
        fe = Fe[n] if n < nFe else 0.0
        fy = fe * cphi; fz = fe * sphi
        # ---- free (linear) prediction + TM-modified compliance (per polarisation)
        thccy = 0.0; thccz = 0.0; uchy = 0.0; uchz = 0.0
        for i in range(M):
            qt = Ayy[i] * q1[i] - by[i] * q0[i] + cy[i] * ge[i] * fy
            pt = 0.0
            if use_z:
                qt += Ayz[i] * p1[i]
                pt = Azy[i] * q1[i] + Azz[i] * p1[i] - bz[i] * p0[i] + cz[i] * ge[i] * fz
            if use_tm:
                eps = cy[i] * ktm * P_c * beta2[i]
                d = 1.0 / (1.0 + eps)
                qh[i] = d * (qt - eps * q0[i])
                cpy[i] = cy[i] * d
                eps = cz[i] * ktm * P_c * beta2[i]
                d = 1.0 / (1.0 + eps)
                ph[i] = d * (pt - eps * p0[i])
                cpz[i] = cz[i] * d
            else:
                qh[i] = qt; ph[i] = pt; cpy[i] = cy[i]; cpz[i] = cz[i]
            uchy += gc[i] * qh[i]
            uchz += gc[i] * ph[i]
        if use_tm:
            for i in range(M):
                thccy += cpy[i] * gc[i] * gc[i]
                thccz += cpz[i] * gc[i] * gc[i]
        else:
            thccy = thcc0y; thccz = thcc0z
        den = 1.0 + kapy * thccy
        g0y = -(kapy * uchy + lamy * ucy_p) / den
        g1y = -kapy / den
        den = 1.0 + kapz * thccz
        g0z = -(kapz * uchz + lamz * ucz_p) / den
        g1z = -kapz / den
        for i in range(M):
            qs[i] = qh[i] + cpy[i] * gc[i] * g0y
            ps[i] = ph[i] + cpz[i] * gc[i] * g0z
        nA = 0
        Fb_tot = 0.0; Fh_tot = 0.0; vf = 0.0; vh = 0.0; pfric = 0.0
        Fcy = g0y; Fcz = g0z
        it = 0; res = 0.0; nact = 0
        contact = False
        if use_bridge:
            anyS = False
            if grv:
                zt[:] = np.dot(Phi, ps)
            for j in range(Nb):
                acc = 0.0
                for i in range(M):
                    acc += Phi[j, i] * qs[i]
                yt[j] = acc
                bv[j] = hj[j] - acc - eta_p[j]
                gpred = 0.0
                if grv:
                    gpred = (kg * zt[j] + sg) * zt[j]
                inS[j] = (eta_p[j] > 0.0) or (eta_p[j] + bv[j] + gpred > 0.0)
                if inS[j]:
                    anyS = True
            if anyS:
                contact = True
                for attempt in range(6):
                    nS = 0
                    for j in range(Nb):
                        if inS[j]:
                            sidx[nS] = j; nS += 1
                    # compliances on the candidate set (y through cpy, z through cpz)
                    for a_ in range(nS):
                        ja = sidx[a_]
                        if use_tm:
                            acc = 0.0; accz = 0.0
                            for i in range(M):
                                acc += Phi[ja, i] * cpy[i] * gc[i]
                                accz += Phi[ja, i] * cpz[i] * gc[i]
                            vSy[a_] = acc; vSz[a_] = accz
                        else:
                            vSy[a_] = v0y[ja]; vSz[a_] = v0z[ja]
                    for a_ in range(nS):
                        ja = sidx[a_]
                        for b_ in range(a_, nS):
                            jb = sidx[b_]
                            if use_tm:
                                acc = 0.0; accz = 0.0
                                for i in range(M):
                                    acc += Phi[ja, i] * cpy[i] * Phi[jb, i]
                                    accz += Phi[ja, i] * cpz[i] * Phi[jb, i]
                                acc *= dxb; accz *= dxb
                            else:
                                acc = Q0y[ja, jb]; accz = Q0z[ja, jb]
                            QB[a_, b_] = acc; QB[b_, a_] = acc
                            QBz[a_, b_] = accz; QBz[b_, a_] = accz
                    for a_ in range(nS):
                        for b_ in range(nS):
                            QY[a_, b_] = QB[a_, b_] + g1y * dxb * vSy[a_] * vSy[b_]
                            QZ[a_, b_] = QBz[a_, b_] + g1z * dxb * vSz[a_] * vSz[b_]
                    if not joint:
                        # ---- block-triangular solve of _run: vertical contact Newton, then friction in z
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
                        nA = 0
                        for a_ in range(nS):
                            if f[a_] != 0.0:
                                jA[nA] = a_; fA[nA] = f[a_]; nA += 1
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
                        vf = 0.0; vh = 0.0; Fb_tot = 0.0; Fh_tot = 0.0; pfric = 0.0
                        for k in range(nA):
                            vf += vSy[jA[k]] * fA[k]
                            vh += vSz[jA[k]] * hA[k]
                            Fb_tot += fA[k]; Fh_tot += hA[k]
                            pfric += muA[k] * wA[k] * fA[k]
                    else:
                        # ---- joint Newton on (s_a, Z_a), a in the candidate set
                        for a_ in range(nS):
                            ja = sidx[a_]
                            acc = 0.0
                            for i in range(M):
                                acc += Phi[ja, i] * p0[i]
                            Zp[a_] = acc
                            if grv:
                                ztS[a_] = zt[ja]
                            else:
                                acc = 0.0
                                for i in range(M):
                                    acc += Phi[ja, i] * ps[i]
                                ztS[a_] = acc
                            Zc[a_] = ztS[a_]
                            gpred = (kg * Zc[a_] + sg) * Zc[a_]
                            s[a_] = bv[ja] + gpred
                            if alpha == 1.0 and guess:
                                s[a_] = _scalar_root(eta_p[ja], eta_p[ja] + bv[ja] + gpred,
                                                     0.5 * kb * QY[a_, a_]) - eta_p[ja]
                        it = 0
                        res = 0.0
                        resz = 0.0
                        nact = 0
                        while True:
                            # forces at the current iterate
                            nact = 0
                            for a_ in range(nS):
                                fj, fpj = _dgrad(eta_p[sidx[a_]], s[a_], kb, alpha)
                                f[a_] = fj; fp[a_] = fpj
                                tg = kg * (Zc[a_] + Zp[a_]) + sg
                                mupj = 0.0
                                if use_fric:
                                    wv = (Zc[a_] - Zp[a_]) * inv2dt
                                    tg += mu1 * np.arctan(mu2 * wv)
                                    mupj = mu1 * mu2 / (1.0 + (mu2 * wv) ** 2)
                                gg[a_] = -fj * tg
                                gs_[a_] = -fpj * tg
                                gZ[a_] = -fj * (kg + mupj * inv2dt)
                                if fj != 0.0 or fpj != 0.0:
                                    act[nact] = a_; nact += 1
                            res = 0.0; resz = 0.0; base = 0.0
                            for a_ in range(nS):
                                acc = s[a_] - bv[sidx[a_]] - (kg * Zc[a_] + sg) * Zc[a_]
                                accz = Zc[a_] - ztS[a_]
                                for kk in range(nact):
                                    k = act[kk]
                                    acc += QY[a_, k] * f[k]
                                    accz -= QZ[a_, k] * gg[k]
                                r[a_] = acc; r2[a_] = accz
                                base += acc * acc + accz * accz
                                if abs(acc) > res:
                                    res = abs(acc)
                                if abs(accz) > resz:
                                    resz = abs(accz)
                            if res <= tol and resz <= tolz:
                                break
                            if it >= maxit:
                                n_fail += 1
                                break
                            # Newton step on the active points: unknowns [s_A ; Z_A]
                            for ii in range(nact):
                                jr = act[ii]
                                for kk in range(nact):
                                    k = act[kk]
                                    J[ii, kk] = QY[jr, k] * fp[k]
                                    J[ii, nact + kk] = 0.0
                                    J[nact + ii, kk] = -QZ[jr, k] * gs_[k]
                                    J[nact + ii, nact + kk] = -QZ[jr, k] * gZ[k]
                                J[ii, ii] += 1.0
                                J[ii, nact + ii] = -(2.0 * kg * Zc[jr] + sg)
                                J[nact + ii, nact + ii] += 1.0
                                rhs[ii] = -r[jr]
                                rhs[nact + ii] = -r2[jr]
                            _gauss(J, rhs, 2 * nact, dl)
                            for a_ in range(nS):
                                dsv[a_] = 0.0; dZv[a_] = 0.0
                            for kk in range(nact):
                                dsv[act[kk]] = dl[kk]; dZv[act[kk]] = dl[nact + kk]
                            # inactive candidates: their rows are explicit (unit diagonal, no force of their own)
                            for a_ in range(nS):
                                if f[a_] != 0.0 or fp[a_] != 0.0:
                                    continue
                                accz = -r2[a_]
                                acc = -r[a_]
                                for kk in range(nact):
                                    k = act[kk]
                                    accz += QZ[a_, k] * (gs_[k] * dsv[k] + gZ[k] * dZv[k])
                                    acc -= QY[a_, k] * fp[k] * dsv[k]
                                dZv[a_] = accz
                                dsv[a_] = acc + (2.0 * kg * Zc[a_] + sg) * accz
                            # backtracking on |r|^2
                            lam_ = 1.0
                            for ls in range(40):
                                for a_ in range(nS):
                                    s_t[a_] = s[a_] + lam_ * dsv[a_]
                                    Z_t[a_] = Zc[a_] + lam_ * dZv[a_]
                                for a_ in range(nS):
                                    fj, fpj = _dgrad(eta_p[sidx[a_]], s_t[a_], kb, alpha)
                                    f_t[a_] = fj
                                    tg = kg * (Z_t[a_] + Zp[a_]) + sg
                                    if use_fric:
                                        tg += mu1 * np.arctan(mu2 * (Z_t[a_] - Zp[a_]) * inv2dt)
                                    g_t[a_] = -fj * tg
                                trial = 0.0; tr1 = 0.0; tr2 = 0.0
                                for a_ in range(nS):
                                    acc = s_t[a_] - bv[sidx[a_]] - (kg * Z_t[a_] + sg) * Z_t[a_]
                                    accz = Z_t[a_] - ztS[a_]
                                    for b_ in range(nS):
                                        if f_t[b_] != 0.0:
                                            acc += QY[a_, b_] * f_t[b_]
                                            accz -= QZ[a_, b_] * g_t[b_]
                                    trial += acc * acc + accz * accz
                                    if abs(acc) > tr1:
                                        tr1 = abs(acc)
                                    if abs(accz) > tr2:
                                        tr2 = abs(accz)
                                if trial < base or (tr1 <= tol and tr2 <= tolz) or ls == 39:
                                    break
                                lam_ *= 0.5
                                n_ls += 1
                            for a_ in range(nS):
                                s[a_] = s_t[a_]; Zc[a_] = Z_t[a_]
                            it += 1
                        if resz > max_zres:
                            max_zres = resz
                        # force-active points, friction bookkeeping (z force = groove + friction)
                        nA = 0
                        for a_ in range(nS):
                            if f[a_] != 0.0:
                                jA[nA] = a_; fA[nA] = f[a_]; hA[nA] = gg[a_]
                                wA[nA] = (Zc[a_] - Zp[a_]) * inv2dt
                                muA[nA] = mu1 * np.arctan(mu2 * wA[nA]) if use_fric else 0.0
                                nA += 1
                        if use_fric and nA > 0:
                            n_fric += 1
                            f_tot_it += it
                            if it > f_max_it:
                                f_max_it = it
                            if resz > f_max_res:
                                f_max_res = resz
                        vf = 0.0; vh = 0.0; Fb_tot = 0.0; Fh_tot = 0.0; pfric = 0.0
                        for k in range(nA):
                            vf += vSy[jA[k]] * fA[k]
                            vh += vSz[jA[k]] * hA[k]
                            Fb_tot += fA[k]; Fh_tot += hA[k]
                            pfric += muA[k] * wA[k] * fA[k]
                    # ---- thread forces and modal update
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
                        q2[i] = qh[i] + cpy[i] * accy
                        p2[i] = ph[i] + cpz[i] * accz
                    viol = False
                    if grv:
                        zn[:] = np.dot(Phi, p2)
                    for j in range(Nb):
                        acc = 0.0
                        for i in range(M):
                            acc += Phi[j, i] * q2[i]
                        eta_n[j] = hj[j] - acc
                        if grv:
                            eta_n[j] += (kg * zn[j] + sg) * zn[j]
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
                    if grv:
                        eta_n[j] += (kg * zt[j] + sg) * zt[j]
        if not contact:
            Fcy = g0y; Fcz = g0z
            for i in range(M):
                q2[i] = qs[i]; p2[i] = ps[i]
        # ---- outputs
        ucy_n = 0.0; ucz_n = 0.0; uey_n = 0.0; uez_n = 0.0; F0y = 0.0; F0z = 0.0; Fny_ = 0.0; Fnz_ = 0.0
        P_n = 0.0; by_ = 0.0; bz_ = 0.0; byn = 0.0; bzn = 0.0
        for i in range(M):
            ucy_n += gc[i] * q2[i]; ucz_n += gc[i] * p2[i]
            uey_n += ge[i] * q2[i]; uez_n += ge[i] * p2[i]
            F0y += w0[i] * q2[i]; F0z += w0[i] * p2[i]
            Fny_ += wn[i] * q2[i]; Fnz_ += wn[i] * p2[i]
            P_n += 0.5 * L * beta2[i] * (q2[i] * q2[i] + p2[i] * p2[i])
            by_ += beta[i] * q2[i]; bz_ += beta[i] * p2[i]
            byn += sgn[i] * beta[i] * q2[i]; bzn += sgn[i] * beta[i] * p2[i]
        dT = 0.0
        if use_tm:
            dT = 4.0 * ktm / L * P_n
            F0y += dT * by_; F0z += dT * bz_; Fny_ += dT * byn; Fnz_ += dT * bzn
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
                if perpol:
                    dloss += 2.0 * m * (sigsy[i] * vdy * vdy + sigsz[i] * vdz * vdz)
                else:
                    dloss += 2.0 * m * sigsy[i] * (vdy * vdy + vdz * vdz)
            vu = (ucy_n - ucy_p) * inv2dt
            vz_ = (ucz_n - ucz_p) * inv2dt
            dloss += rcy * vu * vu + rcz * vz_ * vz_
            Dacc += dt * dloss
            Wacc += dt * (fy * (uey_n - uey_p) + fz * (uez_n - uez_p)) * inv2dt
            Dcum[n] = Dacc; Wcum[n] = Wacc
        Dfric[n] = Facc
        ey = 0.0; ez = 0.0
        for i in range(M):
            vy = (q2[i] - q1[i]) / dt
            vz = (p2[i] - p1[i]) / dt
            ey += 0.5 * m * (vy * vy + Kyy[i] * q2[i] * q1[i])
            ez += 0.5 * m * (vz * vz + Kzz[i] * p2[i] * p1[i])
        Eyacc += ey; Ezacc += ez
        if rec_modes > 0 and n % rec_modes == 0:
            km = n // rec_modes
            for i in range(M):
                vy = (q2[i] - q1[i]) / dt
                vz = (p2[i] - p1[i]) / dt
                mEy[km, i] = 0.5 * m * (vy * vy + Kyy[i] * q2[i] * q1[i])
                mEz[km, i] = 0.5 * m * (vz * vz + Kzz[i] * p2[i] * p1[i])
                mEx[km, i] = 0.5 * m * Kyz[i] * (q2[i] * p1[i] + p2[i] * q1[i])
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
                      Facc, n_ls, max_zres])
    return (Fby, Fbz, Fny, Fnz, Fcont, Ffric, H, Dcum, Dfric, Wcum, pen, rEy, rEz, rdT, rslip, stats,
            mEy, mEz, mEx)


def simulate(over=None, energy=False, init=None):
    """Run the model.  init=(q, p): explicit initial modal state (x^0 = x^-1; testing only; default None)."""
    p = dict(DEFAULT)
    if over:
        p.update(over)
    s = build(p)
    N = int(round(p["dur"] * p["fs"]))
    Fe = excitation(p, s)
    q_init = np.zeros(s["M"]); p_init = np.zeros(s["M"]); stat_info = None
    if init is not None:
        q_init = np.ascontiguousarray(init[0], float); p_init = np.ascontiguousarray(init[1], float)
    elif p.get("static_init"):
        q_init, p_init, stat_info = static_equilibrium(p, s)
    else:   # original behaviour (unchanged): approximate preload state only for the parabola with h0 > 0
        init_static = p["init_static"] if p["init_static"] is not None else (p["bridge"] and p["h0"] > 0)
        if init_static and p["bridge"]:
            q_init, p_init, stat_info = static_state(p, s)
    use_fric = bool(p["friction"]) and s["use_z"] and bool(p["bridge"])
    ktm = s["EA"] / 8.0 if p["tm"] else 0.0
    if uses_round3(p, s):
        return _simulate3(p, s, N, Fe, q_init, p_init, stat_info, use_fric, ktm, energy)
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


def uses_round3(p, s=None):
    """True if any round-3 option is on (then simulate runs _run3; otherwise the original _run, bit-identical)."""
    joint = p.get("solver") == "joint"
    if p.get("solver") not in (None, "joint"):
        raise ValueError(f"unknown solver {p.get('solver')!r}")
    return bool(joint or p.get("groove_kappa") is not None or p.get("Yb_y") or p.get("Yb_z") is not None
                or p.get("rec_modes") or (p.get("groove_slope") or 0.0) != 0.0)


def _simulate3(p, s, N, Fe, q_init, p_init, stat_info, use_fric, ktm, energy):
    if (p.get("groove_slope") or 0.0) != 0.0 and not s["groove"]:
        raise ValueError("groove_slope needs groove_kappa (use groove_kappa=0.0 for a pure tilt)")
    joint = bool(s["groove"] or p.get("solver") == "joint")
    if joint and not s["use_z"]:
        raise ValueError("the joint (y, z) contact solver needs pol2=True")
    rec_modes = int(p.get("rec_modes") or 0)
    t0 = time.time()
    out = _run3(N, s["Ayy"], s["Ayz"], s["Azy"], s["Azz"], s["by"], s["bz"], s["cy"], s["cz"], s["beta"],
                s["beta"] ** 2, s["ge"], Fe, float(s["cphi"]), float(s["sphi"]),
                s["gc"], s["kapy"], s["lamy"], float(s["kc"]), float(s["rc"]), s["kapz"], s["lamz"],
                float(s["kcz"]), float(s["rcz"]),
                bool(s["use_z"]), bool(p["bridge"]), use_fric, bool(p["tm"]), joint, float(s["kg"]), float(s["sg"]),
                bool(s["perpol"]),
                s["Phi"], s["hj"], float(s["dxb"]), s["Q0y"], s["Q0z"], s["v0y"], s["v0z"], float(p["kb"]),
                float(p["alpha"]), float(p["mu_c1"]), float(p["mu_c2"]),
                float(ktm), float(s["L"]), s["w0"], s["wn"], s["sgn"],
                float(p["tol"]), int(p["maxit"]), float(p["ftol"]), int(p["fmaxit"]), bool(p["guess"]),
                bool(energy), float(s["m"]), s["Kyy"], s["Kyz"], s["Kzz"], s["sigs_y"], s["sigs_z"], float(s["dt"]),
                q_init, p_init, int(p["rec_every"]), rec_modes)
    cpu = time.time() - t0
    (Fby, Fbz, Fny, Fnz, Fcont, Ffric, H, D, Df, W, pen, rEy, rEz, rdT, rslip, st, mEy, mEz, mEx) = out
    nc = st[1]
    stats = dict(max_pen=st[0], contact_frac=nc / N, newton_mean=(st[2] / nc if nc > 0 else 0.0),
                 newton_max=int(st[3]), newton_fail=int(st[4]), max_res=st[5], max_Fcy=st[6], max_Fcz=st[7],
                 edge_pen=st[8], max_active=int(st[9]), n_resolve=int(st[10]), fric_steps=int(st[11]),
                 fric_newton_mean=(st[12] / st[11] if st[11] > 0 else 0.0), fric_newton_max=int(st[13]),
                 fric_fail=int(st[14]), fric_max_res=st[15], max_slip=st[16], max_dT=st[17], max_mu=st[18],
                 E_friction=st[19], static=stat_info, n_linesearch=int(st[20]), max_zres=st[21], kernel="_run3",
                 joint=joint)
    tr = np.arange(len(rEy)) * p["rec_every"] / p["fs"]
    res = dict(Fy=Fby, Fz=Fbz, Fny=Fny, Fnz=Fnz, Fcont=Fcont, Ffric=Ffric, H=H, D=D, Dfric=Df, W=W,
               pen=pen, Ey=rEy, Ez=rEz, dT=rdT, slip=rslip, t_rec=tr, stats=stats, cpu=cpu, N=N)
    if rec_modes > 0:
        res.update(mEy=mEy, mEz=mEz, mEx=mEx, t_modes=np.arange(mEy.shape[0]) * rec_modes / p["fs"])
    return p, s, res


def thread_lift(p):
    """height of the string above the bridge surface at the thread position (the 'thread height')."""
    if p.get("profile") is not None:
        return T1.thread_lift(p)
    return -(p["h0"] - p["curv"] * (p["xc"] - p["xb"]) ** 2)
