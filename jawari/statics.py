"""Rest state (statics) of a tanpura string over a jawari bridge with a finite-diameter thread (jiva).

Problem (small-slope beam-string, bridge frame):
    minimise  E[y] = 1/2 int (T y'^2 + EI y''^2) dx  +  E_tail + E_nut
    subject to y(x) >= o(x) = max(bridge surface b(x), thread top t(x))
  * x along the string, x = 0 at the bridge's back face, heights y in metres, +y away from the bridge.
  * bridge surface: BridgeProfile (parabolic / power-law / measured points via smoothing spline) with a
    rounded back edge (radius r_e, vertical back face at x = 0) so the string can wrap it.
  * thread: rigid cylinder of (effective, compressed) diameter d_t whose centre is at x = x_c, resting on
    the profile (lowest centre height for which the circle does not penetrate the bridge).
  * tail side: straight free span to the tailpiece anchor at (-L_tail, b(0) - L_tail tan(phi)).
    nut side: straight free span to the nut at (x_ref + L_speak, b(x_ref) - L_speak tan(psi)) (or y_nut).
    Both are EXACT far-field terms (Dirichlet-to-Neumann): for a free beam-string span of length D ending
    in a pin, the minimum exterior energy given y0 = y(x_b) and p0 = y'(x_b) (slope pointing away from
    the grid) is  E_ext = T/(2(D-l)) (y_anchor - y0 - l p0)^2 + (T l/2) p0^2,  l = sqrt(EI/T)
    (exponentially small terms e^{-D/l} dropped).
  * discretisation: uniform grid (default 10 um), FD energy
        sum T/(2h) (y_{i+1}-y_i)^2 + sum_{i=1}^{N-2} EI/(2h^3) (y_{i+1}-2y_i+y_{i-1})^2,
    a sparse SPD pentadiagonal system; obstacle constraint at every node.
  * solver: plain PDAS (semismooth Newton for min(lam, y-o) = 0) CYCLES on this 4th-order obstacle problem
    (adjacent nodes on the thread / edge arcs alternate between +-1e6 N multipliers).  Used instead: a Mehrotra
    primal-dual interior-point method on the banded SPD system (~20 iterations, 2 banded Cholesky solves each),
    started from the exact EI = 0 solution (upper convex hull of the obstacle) lifted by 1 um; then an exact
    active-set polish (equality-constrained QP) verified by PDAS (1 iteration).  Contact multipliers
    lam_i = (K y - f)_i are nodal contact forces in N.
  * precision: cond(K) ~ 5e11 (h = 10 um) .. 8e12 (5 um).  Rounded float64 entries break the exact zero row sums
    of the difference operators, which perturbs the low modes at the 1e-4 level; the polish therefore uses
    iterative refinement with the residual evaluated in long double from the difference stencils themselves
    (make_grad), the float64 Cholesky acting only as preconditioner.
  * grid alignment: with EI > 0 the string touches the thread and the edge arc at single points and the far field
    depends to FIRST order on the x-position of such a point force (~ e F/(2T) for an offset e): the spacing and
    offset are chosen (iteratively) so that both tangent points are grid nodes.

Outputs (StaticSolution): rest shape, contact sets (back termination / thread / front bridge) and
forces, thread force, front clearance g = y - b, crest gap (smallest interior local minimum of the THREAD-ONLY
shape: negative = string would be pressed onto the bridge), grazing point, termination point x_term.

profile_for_dynamics(): maps to tanpura1d coordinates (x_rel = x - x_term; y = 0 is the thread-only
static shape; bridge surface relative to it, h_rel = b - y_thread_only, positive = pressed).

All lengths SI (m), forces N.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import solveh_banded
from scipy.optimize import brentq, minimize_scalar

BACK, THREAD, FRONT = 1, 2, 3
DEBUG = False


# ----------------------------------------------------------------------------------------------- profile
class BridgeProfile:
    """Bridge-top height b(x) in the bridge frame (x from the back face, metres), with a rounded back edge.

    top(x)/dtop(x): the profile function and its slope (no edge rounding); surface(x): obstacle height
    including the edge arc, -inf where there is no bridge (x < 0 or x > length).
    """

    def __init__(self, top, dtop, length, edge_radius=0.5e-3, x_crest=None, name="profile", meta=None):
        self._top, self._dtop = top, dtop
        self.length = float(length)
        self.r_e = float(edge_radius)
        self.name = name
        self.meta = meta or {}
        if x_crest is None:
            xs = np.linspace(0.0, self.length, 200001)
            x_crest = float(xs[np.argmax(self._top(xs))])
        self.x_crest = float(x_crest)
        self._make_edge()

    # -- constructors
    @classmethod
    def parabolic(cls, c=3.0, x_crest=10e-3, length=20e-3, edge_radius=0.5e-3, h_crest=0.0):
        """b = h_crest - c (x - x_crest)^2 (DAFx-16 shape, c in 1/m)."""
        return cls(lambda x: h_crest - c * (np.asarray(x, float) - x_crest) ** 2,
                   lambda x: -2.0 * c * (np.asarray(x, float) - x_crest),
                   length, edge_radius, x_crest, name=f"parabola c={c:g}/m crest {x_crest*1e3:g} mm",
                   meta=dict(kind="parabolic", c=c, p=2.0))

    @classmethod
    def power(cls, c=3.0, p=4.0, x_crest=10e-3, x_ref=5e-3, length=20e-3, edge_radius=0.5e-3, h_crest=0.0):
        """Flatter (p > 2) / sharper (p < 2) family b = h_crest - c x_ref^2 (|x - x_crest|/x_ref)^p.
        Same drop c x_ref^2 as the parabola at |x - x_crest| = x_ref; p = 2 is the parabola."""
        A = c * x_ref ** 2

        def top(x):
            return h_crest - A * (np.abs(np.asarray(x, float) - x_crest) / x_ref) ** p

        def dtop(x):
            u = np.asarray(x, float) - x_crest
            return -A * p * (np.abs(u) / x_ref) ** (p - 1) * np.sign(u) / x_ref

        return cls(top, dtop, length, edge_radius, x_crest,
                   name=f"power p={p:g} (c_ref={c:g}/m at {x_ref*1e3:g} mm) crest {x_crest*1e3:g} mm",
                   meta=dict(kind="power", c=c, p=p, x_ref=x_ref))

    @classmethod
    def from_points(cls, x_mm, h_um, smoothing=0.0, edge_radius=0.5e-3, length=None, name="measured"):
        """Measured profile: points (x in mm from the back face, height in um).  smoothing = expected rms
        measurement noise in um (0 -> interpolating cubic spline; >0 -> cubic smoothing spline with
        residual sum of squares N*smoothing^2).  The profile is defined on [0, max x]."""
        from scipy.interpolate import CubicSpline, UnivariateSpline
        x = np.asarray(x_mm, float) * 1e-3
        h = np.asarray(h_um, float) * 1e-6
        o = np.argsort(x)
        x, h = x[o], h[o]
        if smoothing and smoothing > 0:
            spl = UnivariateSpline(x, h, k=3, s=len(x) * (smoothing * 1e-6) ** 2)
            dspl = spl.derivative()
        else:
            spl = CubicSpline(x, h, bc_type="natural")
            dspl = spl.derivative()
        L = float(length) if length is not None else float(x[-1])
        if x[0] > 1e-9:
            raise ValueError("measured profile must start at the back face (x = 0)")
        return cls(lambda xx: spl(np.asarray(xx, float)), lambda xx: dspl(np.asarray(xx, float)),
                   L, edge_radius, None, name=name,
                   meta=dict(kind="points", n=len(x), smoothing_um=smoothing))

    @classmethod
    def from_csv(cls, path, smoothing=0.0, edge_radius=0.5e-3, **kw):
        """CSV with columns x_mm,height_um (header optional)."""
        xs, hs = [], []
        with open(path) as fh:
            for row in csv.reader(fh):
                if not row or row[0].strip().startswith("#"):
                    continue
                try:
                    xs.append(float(row[0])); hs.append(float(row[1]))
                except ValueError:
                    continue      # header
        return cls.from_points(xs, hs, smoothing, edge_radius, name=str(path), **kw)

    # -- geometry
    def top(self, x):
        return self._top(x)

    def dtop(self, x):
        return self._dtop(x)

    def _make_edge(self):
        r = self.r_e
        if r <= 0:
            self.x_a, self.edge_c = 0.0, None
            return
        xa = r
        for _ in range(100):
            th = float(np.arctan(self._dtop(np.array([xa]))[0]))
            xn = r * (1.0 - np.sin(th))
            if abs(xn - xa) < 1e-16:
                break
            xa = xn
        th = float(np.arctan(self._dtop(np.array([xa]))[0]))
        self.x_a = xa
        self.edge_c = (r, float(self._top(np.array([xa]))[0]) - r * np.cos(th))

    def surface(self, x):
        """Obstacle height of the bridge (with the rounded back edge); -inf off the bridge."""
        x = np.asarray(x, float)
        out = np.full(x.shape, -np.inf)
        on = (x >= 0.0) & (x <= self.length)
        out[on] = self._top(x[on])
        if self.edge_c is not None:
            e = (x >= 0.0) & (x < self.x_a)
            cx, cy = self.edge_c
            out[e] = cy + np.sqrt(np.maximum(self.r_e ** 2 - (x[e] - cx) ** 2, 0.0))
        return out


# ----------------------------------------------------------------------------------------------- string etc.
@dataclass
class String:
    T: float
    EI: float
    mu: float
    name: str = "string"
    rho: float = 7850.0
    A: float = 0.0
    E: float = 2e11
    I: float = 0.0
    L: float = 1.0
    f1: float = 0.0

    @property
    def l(self):
        return float(np.sqrt(self.EI / self.T)) if self.EI > 0 else 0.0

    @classmethod
    def from_material(cls, name, L, rho, A, E, I, f1):
        mu = rho * A
        EI = E * I
        T = mu * (2 * L * f1) ** 2 - EI * (np.pi / L) ** 2      # same convention as tanpura1d.build
        return cls(T=T, EI=EI, mu=mu, name=name, rho=rho, A=A, E=E, I=I, L=L, f1=f1)

    @classmethod
    def c3_dafx16(cls):
        return cls.from_material("C3 (DAFx-16)", 1.0, 7850.0, 6.16e-8, 2.0e11, 3.02e-16, 130.81)

    @classmethod
    def kharaj(cls, d=0.7e-3, L=1.05, f1=69.4, rho=7800.0, E=2.0e11):
        A = np.pi * d ** 2 / 4
        I = np.pi * d ** 4 / 64
        return cls.from_material(f"kharaj ({d*1e3:g} mm steel)", L, rho, A, E, I, f1)

    def with_EI(self, EI):
        s = String(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        s.EI = float(EI)
        return s


@dataclass
class Thread:
    x_c: float               # x of the thread centre (bridge frame)
    d_t: float               # effective (compressed) diameter
    point: bool = False      # True: point obstacle of height b(x_c) + d_t at x_c (validation only)


@dataclass
class Setup:
    profile: BridgeProfile
    string: String
    thread: Thread
    phi_deg: float = 10.0          # tail break angle below the bridge datum
    psi_deg: float = 0.0           # speaking-length angle below the bridge datum (nut lower = presses)
    L_tail: float = 0.15           # back face -> tailpiece anchor (assumed)
    L_speak: float = 1.0           # x_ref -> nut
    y_nut: float | None = None     # overrides psi
    x_ref: float | None = None     # reference for the nut distance/height; default: profile crest

    def anchors(self):
        pr = self.profile
        xT = -self.L_tail
        yT = float(pr.top(np.array([0.0]))[0]) - self.L_tail * np.tan(np.deg2rad(self.phi_deg))
        xr = pr.x_crest if self.x_ref is None else self.x_ref
        xN = xr + self.L_speak
        if self.y_nut is not None:
            yN = float(self.y_nut)
        else:
            yN = float(pr.top(np.array([xr]))[0]) - self.L_speak * np.tan(np.deg2rad(self.psi_deg))
        return (xT, yT), (xN, yN)

    def replace(self, **kw):
        d = dict(profile=self.profile, string=self.string, thread=self.thread, phi_deg=self.phi_deg,
                 psi_deg=self.psi_deg, L_tail=self.L_tail, L_speak=self.L_speak, y_nut=self.y_nut,
                 x_ref=self.x_ref)
        if "x_c" in kw or "d_t" in kw:
            th = self.thread
            kw["thread"] = Thread(kw.pop("x_c", th.x_c), kw.pop("d_t", th.d_t), th.point)
        d.update(kw)
        return Setup(**d)


def thread_centre(profile: BridgeProfile, th: Thread):
    """(cx, cy, r): lowest centre height at x_c for which the circle does not penetrate the bridge."""
    r = 0.5 * th.d_t
    cx = th.x_c
    if th.point:
        return cx, float(profile.top(np.array([cx]))[0]) + th.d_t, 0.0

    def neg(x):
        return -(float(profile.top(np.array([x]))[0]) + np.sqrt(max(r * r - (x - cx) ** 2, 0.0)))

    res = minimize_scalar(neg, bounds=(cx - r, cx + r), method="bounded", options=dict(xatol=1e-12))
    # guard against the bounded search landing on a local edge: compare with a dense scan
    xs = np.linspace(cx - r, cx + r, 2001)
    v = profile.top(xs) + np.sqrt(np.maximum(r * r - (xs - cx) ** 2, 0.0))
    cy = max(-res.fun, float(v.max()))
    return cx, cy, r


# ----------------------------------------------------------------------------------------------- assembly
def assemble(x, T, EI, left, right):
    """Stiffness K (sparse csr) and load f for the FD energy on grid x plus boundary terms.
    left/right: ('dtn', x_anchor, y_anchor)  or  ('pinned', y_value) (left only: node 0 eliminated;
    the caller must drop row/col 0)."""
    N = x.size
    h = x[1] - x[0]
    l = np.sqrt(EI / T) if EI > 0 else 0.0
    d0 = np.zeros(N); d1 = np.zeros(N - 1); d2 = np.zeros(N - 2)
    kt = T / h
    d0[:-1] += kt; d0[1:] += kt; d1 -= kt
    if EI > 0:
        kb = EI / h ** 3
        # sum_{i=1}^{N-2} kb (y_{i-1} - 2 y_i + y_{i+1})^2 / 2
        d0[:-2] += kb; d0[1:-1] += 4 * kb; d0[2:] += kb
        d1[:-1] -= 2 * kb; d1[1:] -= 2 * kb
        d2 += kb
    K = sp.diags([d2, d1, d0, d1, d2], [-2, -1, 0, 1, 2], shape=(N, N), format="lil")
    f = np.zeros(N)
    left = tuple(left) + ((x[0],) if left[0] == "dtn" else ())
    right = tuple(right) + ((x[-1],) if right[0] == "dtn" else ())
    for side, bc in (("L", left), ("R", right)):
        if bc[0] != "dtn":
            continue
        xa, ya = bc[1], bc[2]
        if side == "R":
            D = xa - x[-1]
            idx = [N - 2, N - 1]
            c = np.array([-l / h, 1 + l / h])
        else:
            D = x[0] - xa
            idx = [0, 1]
            c = np.array([1 + l / h, -l / h])
        if D <= l:
            raise ValueError("anchor too close to the grid boundary")
        dvec = np.array([-1 / h, 1 / h])
        kD = T / (D - l)
        Hb = kD * np.outer(c, c) + T * l * np.outer(dvec, dvec)
        for a in range(2):
            f[idx[a]] += kD * ya * c[a]
            for b in range(2):
                K[idx[a], idx[b]] += Hb[a, b]
    return K.tocsr(), f, make_grad(h, T, EI, left, right)


def make_grad(h, T, EI, left, right):
    """Energy gradient K y - f evaluated in long double from the difference stencils themselves.

    Rounding the float64 matrix entries (EI/h^3 ~ 1e11-1e13 N/m) destroys the exact zero row sums of the
    difference operators and injects spurious low-mode stiffness ~eps*EI/h^3 that is NOT small against the
    physical lowest stiffness (~1 N/m for the long free spans): the float64 matrix is then a different
    problem at the 1e-4 level.  This structure-preserving gradient (translations/rotations annihilated
    exactly) is used as the residual of the iterative refinement, so the refined solution is that of the
    intended discrete problem; the float64 Cholesky only acts as a preconditioner."""
    hL, TL, EIL = LD(h), LD(T), LD(EI)
    lL = np.sqrt(EIL / TL) if EI > 0 else LD(0)
    r = lL / hL

    def grad(y):
        y = np.asarray(y, LD)
        g = np.zeros(y.size, LD)
        t = (TL / hL) * (y[1:] - y[:-1])
        g[:-1] -= t
        g[1:] += t
        if EI > 0:
            m = (EIL / hL ** 3) * ((y[2:] - y[1:-1]) - (y[1:-1] - y[:-2]))
            g[:-2] += m
            g[1:-1] -= 2 * m
            g[2:] += m
        for side, bc in (("L", left), ("R", right)):
            if bc[0] != "dtn":
                continue
            xa, ya = LD(bc[1]), LD(bc[2])
            if side == "R":
                dy = y[-1] - y[-2]
                kD = TL / (LD(xa) - LD(bc[3]) - lL) if len(bc) > 3 else None
                q = y[-1] + r * dy
                pz = dy / hL
                e1 = -kD * (ya - q)
                g[-2] += e1 * (-r) + TL * lL * pz * (-1 / hL)
                g[-1] += e1 * (1 + r) + TL * lL * pz * (1 / hL)
            else:
                dy = y[0] - y[1]
                kD = TL / (LD(bc[3]) - LD(xa) - lL) if len(bc) > 3 else None
                q = y[0] + r * dy
                pz = -dy / hL
                e1 = -kD * (ya - q)
                g[0] += e1 * (1 + r) + TL * lL * pz * (-1 / hL)
                g[1] += e1 * (-r) + TL * lL * pz * (1 / hL)
        return g

    return grad


# ----------------------------------------------------------------------------------------------- linear algebra
LD = np.longdouble


class SPDSystem:
    """SPD stiffness matrix: LAPACK banded Cholesky when the bandwidth is <= 2 (the beam-string FD matrix),
    sparse LU otherwise (dynamic-frame problem with the thread-window spring).  Matrix-vector products are
    available in extended precision (long double) for iterative refinement."""

    def __init__(self, K, grad=None):
        K = sp.csr_matrix(K)
        self.K = K
        self.grad = grad
        self.N = K.shape[0]
        coo = K.tocoo()
        self.bw = int(np.abs(coo.row - coo.col).max()) if coo.nnz else 0
        self.banded = self.bw <= 2
        if self.banded:
            self.ab = np.zeros((3, self.N))
            for u in range(3):
                self.ab[2 - u, u:] = K.diagonal(u)
        else:
            self.coo = (coo.row, coo.col, coo.data)

    def matvec(self, y, dtype=float):
        y = np.asarray(y, dtype)
        if self.banded:
            ab = self.ab.astype(dtype)
            out = ab[2] * y
            out[1:] += ab[1, 1:] * y[:-1]; out[:-1] += ab[1, 1:] * y[1:]
            out[2:] += ab[0, 2:] * y[:-2]; out[:-2] += ab[0, 2:] * y[2:]
            return out
        r, c, d = self.coo
        out = np.zeros(self.N, dtype)
        np.add.at(out, r, d.astype(dtype) * y[c])
        return out

    def solve(self, rhs, diag_add=None, active=None):
        """Solve (K + diag(diag_add)) u = rhs with the rows/columns of `active` replaced by the identity."""
        if self.banded:
            ab = self.ab.copy()
            if diag_add is not None:
                ab[2] += diag_add
            if active is not None and active.any():
                A = active
                ab[2, A] = 1.0
                ab[1, 1:][A[1:] | A[:-1]] = 0.0
                ab[0, 2:][A[2:] | A[:-2]] = 0.0
            return solveh_banded(ab, rhs, lower=False, check_finite=False)
        K = self.K
        if diag_add is not None:
            K = K + sp.diags(diag_add)
        if active is not None and active.any():
            keep = sp.diags((~active).astype(float))
            K = keep @ K @ keep + sp.diags(active.astype(float))
        return spla.splu(sp.csc_matrix(K)).solve(rhs)


def eqp(sysm: SPDSystem, f, o, A, refine=6):
    """Equality-constrained QP: y_A = o_A, (K y - f)_I = 0.  Iterative refinement with the residual evaluated
    in long double (from the exact structured gradient when available).  Returns y, lam (= K y - f on A,
    0 elsewhere), last correction size."""
    oA = np.where(A, o, 0.0)
    rhs = f - sysm.matvec(oA)
    rhs[A] = o[A]
    y = sysm.solve(rhs, active=A)
    y[A] = o[A]
    yl = y.astype(LD)
    fl = f.astype(LD)
    if sysm.grad is not None:
        gfun = sysm.grad
    else:
        def gfun(v):
            return sysm.matvec(v, LD) - fl
    dmax = 0.0
    for _ in range(refine):
        r = -gfun(yl)
        r[A] = 0.0
        d = sysm.solve(r.astype(float), active=A)
        d[A] = 0.0
        yl += d
        dmax = float(np.abs(d).max())
        if dmax < 1e-22:
            break
    lam = gfun(yl).astype(float)
    lam[~A] = 0.0
    return yl.astype(float), lam, dmax


def _step(v, dv):
    neg = dv < 0
    return float(np.min(-v[neg] / dv[neg])) if neg.any() else np.inf


def ipm(sysm: SPDSystem, f, o, y0, maxit=80, mu_tol=1e-26, z0=1e-3):
    """Mehrotra predictor-corrector primal-dual interior point method for min 1/2 y'Ky - f'y s.t. y >= o
    (finite o only).  Each iteration solves two SPD banded systems.  y0 must be strictly feasible."""
    C = np.isfinite(o)
    m = int(C.sum())
    y = y0.astype(float).copy()
    s = y[C] - o[C]
    if s.min() <= 0:
        raise ValueError("IPM start not strictly feasible")
    z = np.full(m, z0)
    it = 0
    for it in range(1, maxit + 1):
        rd = sysm.matvec(y) - f
        rd[C] -= z
        rp = y[C] - o[C] - s
        mu = float(s @ z) / m
        # (the dual residual has a rounding floor ~1e-7 N; the exact polish (eqp) removes it, so the IPM
        #  only has to identify the active set: stop on the complementarity measure)
        if mu < mu_tol:
            break
        Dg = np.zeros(sysm.N)
        Dg[C] = z / s
        # predictor
        rc = s * z
        rhs = -rd
        rhs[C] -= (rc + z * rp) / s
        dy = sysm.solve(rhs, diag_add=Dg)
        ds = dy[C] + rp
        dz = (-rc - z * ds) / s
        ap = min(1.0, _step(s, ds)); ad = min(1.0, _step(z, dz))
        mu_aff = float((s + ap * ds) @ (z + ad * dz)) / m
        sigma = (mu_aff / mu) ** 3
        # corrector
        rc = s * z + ds * dz - sigma * mu
        rhs = -rd
        rhs[C] -= (rc + z * rp) / s
        dy = sysm.solve(rhs, diag_add=Dg)
        ds = dy[C] + rp
        dz = (-rc - z * ds) / s
        ap = min(1.0, 0.995 * _step(s, ds)); ad = min(1.0, 0.995 * _step(z, dz))
        y += ap * dy
        s += ap * ds
        z += ad * dz
    zf = np.zeros(sysm.N); zf[C] = z
    sf = np.full(sysm.N, np.inf); sf[C] = s
    return y, zf, sf, it


def pdas(sysm: SPDSystem, f, o, A0, maxit=100, ytol=1e-14, ltol=1e-12):
    """Primal-dual active set iterations from A0 (used as a polish / safeguard after the IPM).
    Returns y, lam, iterations, converged, active set."""
    has = np.isfinite(o)
    A = np.asarray(A0, bool) & has
    seen = set()
    damped = False
    for it in range(1, maxit + 1):
        y, lam, _ = eqp(sysm, f, o, A)
        I = ~A
        viol = I & has & (y < o - ytol)
        neg = A & (lam < -ltol)
        if not viol.any() and not neg.any():
            return y, lam, it, True, A
        if damped:
            newA = A.copy()
            if neg.any():
                newA[np.argmin(np.where(neg, lam, np.inf))] = False
            if viol.any():
                newA[np.argmax(np.where(viol, o - y, -np.inf))] = True
        else:
            newA = (A & ~neg) | viol
        key = newA.tobytes()
        if key in seen:
            damped = True
        seen.add(key)
        A = newA
    return y, lam, maxit, False, A


def obstacle_qp(K, f, o, y_start=None, A_start=None, use_ipm=True, grad=None):
    """Solve min 1/2 y'Ky - f'y  s.t. y >= o.  Strategy: IPM from a strictly feasible start (the EI = 0
    hull solution lifted by 1 um), identify the active set, exact polish with refinement, verify with PDAS.
    Returns y, lam, info dict (ipm_iters, pdas_iters, converged, refine_last)."""
    sysm = SPDSystem(K, grad)
    info = {}
    if use_ipm:
        y0 = y_start + 1e-6
        y, z, s, nit = ipm(sysm, f, o, y0)
        A = np.isfinite(o) & (z > 1e9 * s)
        info["ipm_iters"] = nit
    else:
        A = np.asarray(A_start, bool)
        info["ipm_iters"] = 0
    y, lam, nit2, ok, A = pdas(sysm, f, o, A)
    info["pdas_iters"] = nit2
    info["converged"] = ok
    _, _, info["refine_last"] = eqp(sysm, f, o, A)
    info["active"] = A
    return y, lam, info


# ----------------------------------------------------------------------------------------------- EI = 0 start
def upper_hull_active(x, o, pa, pb):
    """Exact EI = 0 active set: nodes on the upper convex hull of {pa, (x_i, o_i), pb}."""
    fin = np.isfinite(o)
    idx = np.where(fin)[0]
    P = [(pa[0], pa[1], -1)] + [(x[i], o[i], i) for i in idx] + [(pb[0], pb[1], -2)]
    hull = []
    for p in P:
        while len(hull) >= 2:
            (x1, y1, _), (x2, y2, _) = hull[-2], hull[-1]
            # remove hull[-1] if it lies on or below the segment hull[-2] -> p
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    A = np.zeros(x.size, bool)
    hx = np.array([p[0] for p in hull]); hy = np.array([p[1] for p in hull])
    for _, _, i in hull:
        if i >= 0:
            A[i] = True
    return A, np.interp(x, hx, hy)


# ----------------------------------------------------------------------------------------------- solution
@dataclass
class StaticSolution:
    setup: Setup
    front: bool
    x: np.ndarray
    y: np.ndarray
    o: np.ndarray
    b: np.ndarray            # bridge surface on the grid (with edge), -inf off the bridge
    kind: np.ndarray         # obstacle kind per node: 0 none, BACK, THREAD, FRONT
    lam: np.ndarray
    active: np.ndarray
    iters: list
    converged: bool
    h: float
    thread_c: tuple
    anchors: tuple
    info: dict = field(default_factory=dict)

    def forces(self):
        a = self.active
        return {k: float(self.lam[a & (self.kind == v)].sum()) for k, v in
                (("back", BACK), ("thread", THREAD), ("front", FRONT))}

    def contact_x(self, which):
        v = dict(back=BACK, thread=THREAD, front=FRONT)[which]
        return self.x[self.active & (self.kind == v)]

    def x_term(self):
        xb = self.contact_x("back")
        return float(xb.max()) if xb.size else np.nan

    def front_clearance(self):
        """g = y - b over the front bridge region (x > thread edge, on the bridge)."""
        cx, cy, r = self.thread_c
        m = (self.x > cx + r) & (self.x <= self.setup.profile.length)
        return self.x[m], self.y[m] - self.b[m]

    def crest_gap(self):
        """(crest gap, grazing point): the smallest INTERIOR local minimum of the front clearance g = y - b
        (parabolic sub-grid refinement).  A minimum at the thread edge or at the bridge's front end is not a
        graze (the string rises away from the bridge there); if no interior minimum exists -> (nan, nan)."""
        xg, g = self.front_clearance()
        k = np.where((g[1:-1] <= g[:-2]) & (g[1:-1] < g[2:]))[0] + 1
        if k.size == 0:
            return np.nan, np.nan
        k = int(k[np.argmin(g[k])])
        a, b_, c = g[k - 1], g[k], g[k + 1]
        den = a - 2 * b_ + c
        if den > 0:
            d = 0.5 * (a - c) / den
            return float(b_ - 0.25 * (a - c) * d), float(xg[k] + d * (xg[1] - xg[0]))
        return float(g[k]), float(xg[k])

    def min_front_clearance(self):
        """Global minimum of the front clearance (may sit at the thread edge)."""
        xg, g = self.front_clearance()
        k = int(np.argmin(g))
        return float(g[k]), float(xg[k])

    def slopes_far(self):
        """Far-field slopes of the tail and nut spans (from the exact exterior solution)."""
        (xT, yT), (xN, yN) = self.anchors
        s = self.setup.string
        l = s.l
        h = self.h
        pL = (self.y[1] - self.y[0]) / h
        pR = (self.y[-1] - self.y[-2]) / h
        DL = self.x[0] - xT
        DR = xN - self.x[-1]
        # exterior: y = A + B xi + C e^{-xi/l};  B = p0 + w/(D - l) with w = y_anchor - y0 - p0 D
        BR = pR + (yN - self.y[-1] - pR * DR) / (DR - l)
        BL = pL - (yT - self.y[0] + pL * DL) / (DL - l)
        return BL, BR


def _obstacles(x, setup: Setup, front: bool):
    pr = setup.profile
    b = pr.surface(x)
    cx, cy, r = thread_centre(pr, setup.thread)
    t = np.full(x.shape, -np.inf)
    if setup.thread.point:
        k = int(np.argmin(np.abs(x - cx)))
        t[k] = cy
        under = np.zeros(x.shape, bool); under[k] = True
    else:
        under = np.abs(x - cx) <= r
        t[under] = cy + np.sqrt(np.maximum(r * r - (x[under] - cx) ** 2, 0.0))
    o = b.copy()
    kind = np.where(np.isfinite(b), np.where(x < cx, BACK, FRONT), 0)
    if not front:
        o[(x > cx)] = -np.inf
        kind[(x > cx)] = 0
    useth = under & (t >= o)
    o[useth] = t[useth]
    kind[useth] = THREAD
    return o, b, kind, (cx, cy, r)


def _tangent_x(cx, r, theta):
    """x of the tangent point on a circle (centre x cx, radius r) of a line with slope theta (from above)."""
    return cx - r * theta / np.sqrt(1.0 + theta * theta)


def solve(setup: Setup, h=10e-6, front=True, align=True, margin=None, max_align=8):
    """Static rest state on a uniform grid.  front=False: thread-only state (front bridge removed).

    Grid alignment (align=True): for EI > 0 the string touches the thread and the rounded back edge at single
    points, and the far field depends to FIRST order on the x-position of such a point force (the far-field
    lines meet above it): a contact on a node a distance e from the true tangent point costs ~ e F/(2T)
    (0.3 um for e = 5 um).  The grid spacing h' (~h) and offset are therefore chosen so that BOTH tangent
    points (thread and edge arc) are nodes; they depend on the solution, so this is iterated (2-4 solves).
    For EI = 0 the thread's front tangent point is aligned (the hull/wrap is otherwise exact)."""
    s = setup.string
    l = s.l
    pr = setup.profile
    if margin is None:
        margin = max(4.0 * l, 1.0e-3)
    pa, pb = setup.anchors()
    cx, cy, r = thread_centre(pr, setup.thread)
    xt = cx                 # thread tangent point (target node)
    xe = None               # edge-arc tangent point (target node), EI > 0 only
    iters = []
    sol = None
    prev_x = None
    hh = h
    for _ in range(max_align):
        if xe is not None and xt - xe > h:
            n = max(1, int(round((xt - xe) / h)))
            hh = (xt - xe) / n
        else:
            hh = h
        x_lo, x_hi = -margin, pr.length + margin
        i_lo = int(np.floor((x_lo - xt) / hh))
        i_hi = int(np.ceil((x_hi - xt) / hh))
        x = xt + hh * np.arange(i_lo, i_hi + 1)
        o, b, kind, tc = _obstacles(x, setup, front)
        K, f, gr = assemble(x, s.T, s.EI, ("dtn",) + tuple(pa), ("dtn",) + tuple(pb))
        A_h, y_h = upper_hull_active(x, o, pa, pb)
        info = None
        if s.EI == 0:
            # the discrete EI = 0 solution IS the upper hull; PDAS only verifies it (1 iteration)
            y, lam, info = obstacle_qp(K, f, o, A_start=A_h, use_ipm=False, grad=gr)
        elif prev_x is not None:
            # warm start from the previous (shifted-grid) active set
            A1 = np.zeros(x.size, bool)
            idx = np.round((prev_x - x[0]) / hh).astype(int)
            A1[idx[(idx >= 0) & (idx < x.size)]] = True
            y, lam, info = obstacle_qp(K, f, o, A_start=A1, use_ipm=False, grad=gr)
            if not info["converged"]:
                info = None
        if info is None:
            y, lam, info = obstacle_qp(K, f, o, y_start=y_h, grad=gr)
        act = info["active"]
        ok = info["converged"]
        prev_x = x[act]
        iters.append((info["ipm_iters"], info["pdas_iters"]))
        sol = StaticSolution(setup, front, x, y, o, b, kind, lam, act, iters, ok, hh, tc, (pa, pb))
        sol.info["refine_last"] = info["refine_last"]
        if not align:
            break
        xt_new, xe_new = xt, xe
        th = np.where(act & (kind == THREAD))[0]
        if th.size and not setup.thread.point and r > 0:
            if s.EI > 0:
                k = th[np.argmax(lam[th])]
                xt_new = _tangent_x(cx, r, (y[k + 1] - y[k - 1]) / (2 * hh))
            else:
                k = th.max()          # front-most contact on the thread: align the front tangent point
                xt_new = _tangent_x(cx, r, (y[k + 1] - y[k]) / hh)
        if s.EI > 0 and pr.edge_c is not None:
            ed = np.where(act & (kind == BACK) & (x < pr.x_a))[0]
            if ed.size:
                k = ed[np.argmax(lam[ed])]
                xe_c = _tangent_x(pr.edge_c[0], pr.r_e, (y[k + 1] - y[k - 1]) / (2 * hh))
                xe_new = xe_c if xe_c < pr.x_a else None
            else:
                xe_new = None
        moved = abs(xt_new - xt) > 1e-4 * hh or ((xe_new is None) != (xe is None)) or \
            (xe_new is not None and xe is not None and abs(xe_new - xe) > 1e-4 * hh)
        if DEBUG:
            print('  align', hh, xt, xe, '->', xt_new, xe_new, 'contacts', x[act], lam[act])
        xt, xe = xt_new, xe_new
        if not moved:
            break
    sol.info["x_tangent_thread"] = xt
    sol.info["x_tangent_edge"] = xe
    return sol


@dataclass
class RestState:
    """Thread-only and full static solutions plus derived quantities."""
    setup: Setup
    to: StaticSolution          # thread-only (front bridge removed)
    full: StaticSolution | None

    @property
    def crest_gap(self):
        return self.to.crest_gap()[0]

    @property
    def x_graze(self):
        return self.to.crest_gap()[1]

    @property
    def x_term(self):
        return self.to.x_term()

    def summary(self):
        f_to = self.to.forces()
        g, xg = self.to.crest_gap()
        BL, BR = self.to.slopes_far()
        d = dict(crest_gap=g, x_graze=xg, x_term=self.to.x_term(), back_contact=bool(np.isfinite(self.to.x_term())),
                 F_thread_TO=f_to["thread"], F_back_TO=f_to["back"], iters_TO=list(self.to.iters),
                 tail_angle_deg=float(np.rad2deg(np.arctan(BL))), nut_angle_deg=float(np.rad2deg(np.arctan(-BR))),
                 converged=bool(self.to.converged))
        if self.full is not None:
            ff = self.full.forces()
            d.update(F_thread_full=ff["thread"], F_back_full=ff["back"], F_front=ff["front"],
                     iters_full=list(self.full.iters), converged=d["converged"] and bool(self.full.converged))
            xf = self.full.contact_x("front")
            d["front_contact"] = (float(xf.min()), float(xf.max())) if xf.size else None
        return d


def rest_state(setup: Setup, h=10e-6, full=True, **kw):
    to = solve(setup, h=h, front=False, **kw)
    fu = solve(setup, h=h, front=True, **kw) if full else None
    return RestState(setup, to, fu)


def crest_gap(setup: Setup, h=10e-6, **kw):
    """Crest gap of the thread-only shape (m): negative = string would be pressed onto the bridge."""
    return solve(setup, h=h, front=False, **kw).crest_gap()[0]


def auto_place(setup: Setup, target_gap=0.0, x_lo=None, x_hi=None, h=10e-6, xtol=1e-7, n_scan=17, **kw):
    """Thread position x_c (bridge frame) that gives the target crest gap ('setting the jawari').
    Scans [x_lo, x_hi] for a sign change of gap - target and refines with Brent's method.
    Returns (x_c, RestState)."""
    pr = setup.profile
    r = 0.5 * setup.thread.d_t
    x_lo = r + 0.2e-3 if x_lo is None else x_lo
    x_hi = pr.x_crest + 5e-3 if x_hi is None else x_hi
    x_hi = min(x_hi, pr.length - r - 0.5e-3)

    def F(xc):
        return crest_gap(setup.replace(x_c=xc), h=h, **kw) - target_gap

    xs = np.linspace(x_lo, x_hi, n_scan)
    vals = [F(xc) for xc in xs]
    for k in range(len(xs) - 1):
        if vals[k] == 0:
            xc = xs[k]
            break
        if np.sign(vals[k]) != np.sign(vals[k + 1]):
            xc = brentq(F, xs[k], xs[k + 1], xtol=xtol)
            break
    else:
        raise ValueError(f"target crest gap {target_gap*1e6:.2f} um not bracketed in "
                         f"[{x_lo*1e3:.2f}, {x_hi*1e3:.2f}] mm; gaps {np.array(vals)*1e6}")
    return xc, rest_state(setup.replace(x_c=xc), h=h, **kw)


def design_psi(setup: Setup, x_c0, target_gap=0.0, psi_lo=0.0, psi_hi=12.0, h=10e-6):
    """Instrument set-up helper: speaking-length angle psi (deg; equivalently the bridge tilt) for which the
    thread at x_c0 gives the target crest gap.  Returns psi_deg."""
    su = setup.replace(x_c=x_c0)

    def F(psi):
        g = crest_gap(su.replace(psi_deg=psi), h=h)
        return (g if np.isfinite(g) else 1e-3) - target_gap

    ps = np.linspace(psi_lo, psi_hi, 25)
    v = [F(p_) for p_ in ps]
    for k in range(len(ps) - 1):
        if np.sign(v[k]) != np.sign(v[k + 1]):
            return brentq(F, ps[k], ps[k + 1], xtol=1e-6)
    raise ValueError("no psi in range gives the target crest gap")


# ----------------------------------------------------------------------------------------------- dynamics map
def profile_for_dynamics(rs: RestState, h_floor=None):
    """Map a rest state to tanpura1d coordinates.

    y = 0 of the dynamic model is the THREAD-ONLY static shape (termination + thread, front bridge removed,
    a linear static deflection -> superposition with the dynamic displacement is exact for the linear
    string).  Returns dict with
      x_rel, h_rel   bridge surface relative to that shape over the front region (h_rel = b - y_TO;
                     h_rel > 0 where the string is pressed onto the bridge), x_rel = x - x_term
      xc_rel, L_rel  thread centre and nut position in the dynamic frame
      F_thread       static thread preload (thread-only), F_thread_full (with front bridge), crest_gap,
      x_term, T, EI and the tanpura1d override dict `t1` (L, xc, profile, rho, A, E, I, T, xe, we).
    The dynamic model pins the string at x_rel = 0, which requires a back termination contact."""
    to = rs.to
    xt = to.x_term()
    if not np.isfinite(xt):
        raise ValueError("no back termination: the string lifts off the back of the bridge and hangs from the "
                         "thread (tail break angle too small for this thread/bridge) - not representable by the "
                         "pinned-termination dynamic model")
    xg, g = to.front_clearance()
    x_rel = xg - xt
    h_rel = -g
    if h_floor is not None:
        keep = h_rel >= h_floor
        last = np.where(keep)[0].max()
        x_rel, h_rel = x_rel[:last + 1], h_rel[:last + 1]
    (xT, yT), (xN, yN) = to.anchors
    cx = to.thread_c[0]
    s = rs.setup.string
    L_rel = xN - xt
    F_full = rs.full.forces()["thread"] if rs.full is not None else np.nan
    out = dict(x_rel=x_rel, h_rel=h_rel, xc_rel=cx - xt, L_rel=L_rel, x_term=xt,
               F_thread=to.forces()["thread"], F_thread_full=F_full, crest_gap=to.crest_gap()[0],
               T=s.T, EI=s.EI)
    out["t1"] = dict(L=L_rel, xc=cx - xt, profile=(x_rel.copy(), h_rel.copy()), rho=s.rho, A=s.A, E=s.E, I=s.I,
                     T=s.T, xe=0.5 * L_rel, we=L_rel)
    return out


# ----------------------------------------------------------------------------------------------- dynamic-frame statics
def sin2win(x, xc, w):
    out = np.zeros_like(x)
    m = np.abs(x - xc) < w / 2
    out[m] = (2.0 / w) * np.sin(np.pi * (x[m] - xc + w / 2) / w) ** 2
    return out


def dynamic_frame_statics(T, EI, L, xc, wc, kc, x_rel, h_rel, h=10e-6, x_end=None, kb=None):
    """Fine-grid statics of EXACTLY the tanpura1d continuum problem: string pinned at 0 and L, thread spring
    k_c acting through the sin^2 window, rigid bridge obstacle h_rel (linear interpolation of the profile).
    Used to measure the modal-truncation error of tanpura1d's static_init.  kb is ignored (rigid contact).
    Returns dict(x, y, lam, F_bridge, F_thread, active x, iters)."""
    l = np.sqrt(EI / T) if EI > 0 else 0.0
    x_end = (x_rel[-1] + max(4 * l, 1e-3)) if x_end is None else x_end
    x = h * np.arange(0, int(np.ceil(x_end / h)) + 1)
    K, f, gr = assemble(x, T, EI, ("pinned", 0.0), ("dtn", L, 0.0))
    w = h * sin2win(x, xc, wc)
    wl, kcl = w.astype(LD), LD(kc)

    def grad(yred):
        yf = np.concatenate([np.zeros(1, LD), np.asarray(yred, LD)])
        return (gr(yf) + kcl * wl * (wl @ yf))[1:]

    wi = np.where(w != 0)[0]
    Kth = sp.coo_matrix((kc * np.outer(w[wi], w[wi]).ravel(),
                         (np.repeat(wi, wi.size), np.tile(wi, wi.size))), shape=K.shape)
    K = (K + Kth).tocsr()[1:, 1:]
    f = f[1:]
    xx = x[1:]
    o = np.full(xx.shape, -np.inf)
    m = (xx >= x_rel[0]) & (xx <= x_rel[-1])
    o[m] = np.interp(xx[m], x_rel, h_rel)
    # strictly feasible start: lift the string above the (positive part of the) profile
    y0 = np.full(xx.shape, max(float(np.max(np.where(np.isfinite(o), o, 0.0))), 0.0))
    y, lam, info = obstacle_qp(K, f, o, y_start=y0, grad=grad)
    act = info["active"]
    it, ok = (info["ipm_iters"], info["pdas_iters"]), info["converged"]
    Fth = -kc * float(w[1:] @ y)
    return dict(x=xx, y=y, lam=lam, F_bridge=float(lam.sum()), F_thread_change=Fth,
                x_active=xx[act], iters=it, converged=ok)
