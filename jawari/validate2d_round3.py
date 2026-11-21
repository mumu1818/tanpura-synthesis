"""Round-3 physics of the 2-D model (MODEL_2D_ROUND3.md): validation and the quick kharaj demo.

    cd jawari
    python validate2d_round3.py                 # A-H (about 15 min CPU, 2 processes for the renders)
    python validate2d_round3.py A B C D E F G   # validation only
    python validate2d_round3.py H --jobs 2      # demo only (renders cached in out/round3/runs/)
    python validate2d_round3.py A --ref <npz>   # also compare with a dump saved before editing (--bitdump)
    python validate2d_round3.py --bitdump DIR OUT.npz   # internal: dump default-path outputs of the modules in DIR

Tests
  A  defaults bit-identical: tanpura1d, tanpura2d, presets, presets2d of the working tree vs git HEAD (every output
     array of 13 default-path runs and every numeric entry of 7 preset dicts), each set computed in its own process.
  B  energy with groove + friction + per-polarisation (non-proportional) damping + TM + nut coupling.
  C  the round-3 kernel without new physics is _run bit for bit; the joint (y, z) Newton with kappa_g = 0 or 1e-6
     reproduces the old block-triangular solver to round-off until the chaotic contact dynamics separate them.
  D  static_init with the groove: the rest state, click-free starts (grazing and pressed crest; offset groove).
  E  non-proportional damping: discrete poles vs the exact continuous eigenvalues of every mode pair, and the
     kernel's free decay of a strongly non-proportional mode pair vs scipy solve_ivp (rtol 1e-12).
  F  pitch design (presets design_f1): reference pitches, the designed psi / thread for f1 below 69.1 Hz, tuning.
  G  CPU seconds per second of audio.
  H  quick demo: kharaj 5 s, FIT2D best own-kharaj parameters, baseline vs groove vs body drain vs both, through a
     PHYSICAL output chain (body 'tanpura_male', 150 Hz 2nd-order high-pass, gz 0.1), against the user's recording.
Outputs: out/round3/ (results.txt/.json, runs/, png/, wav/).  The references are read in place, never copied.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] == "--bitdump":      # isolated import of the modules in DIR (A)
    _mdir = str(Path(sys.argv[2]).resolve())
    _here = Path(__file__).resolve().parent
    sys.path = [_mdir] + [q for q in sys.path if q and Path(q).resolve() != _here]

import numpy as np

PKG = Path(__file__).resolve().parent
OUT = Path(os.environ.get("TANPURA_OUT", PKG / "out")) / "round3"
SR = 44100


# ================================================================================ A. bit identity
def bitdump(out_path):
    """Every output array of fixed default-path configurations (imports whatever tanpura1d/... are on sys.path)."""
    import tanpura1d as T1
    import tanpura2d as T2
    import presets as P1
    import presets2d as P2
    out = {}

    def put(prefix, r):
        for k, v in r.items():
            if isinstance(v, np.ndarray) and v.dtype != object:
                out[f"{prefix}|{k}"] = np.ascontiguousarray(v)
            elif k == "stats":
                for a, b in v.items():
                    if np.isscalar(b):
                        out[f"{prefix}|stats.{a}"] = np.array([b], float)

    def put_dict(prefix, p):
        for k, v in p.items():
            if k == "meta":
                for a, b in v.items():
                    if isinstance(b, (int, float)) and not isinstance(b, bool):
                        out[f"{prefix}|meta.{a}"] = np.array([b], float)
                continue
            if isinstance(v, tuple):
                for i, a in enumerate(v):
                    out[f"{prefix}|{k}{i}"] = np.asarray(a, float)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out[f"{prefix}|{k}"] = np.array([v], float)
            elif isinstance(v, (str, bool)) or v is None:
                out[f"{prefix}|{k}"] = np.frombuffer(json.dumps(v).encode(), np.uint8)

    C1 = {"d1_default": (dict(dur=0.3), True), "d1_h0": (dict(dur=0.2, h0=5e-6), False)}
    C2 = {"d2_default": (dict(dur=0.3), True), "d2_lossless": (dict(dur=0.2, lossless=True), True),
          "d2_h0": (dict(dur=0.2, h0=5e-6), False), "d2_pol2off": (dict(dur=0.2, pol2=False, tm=False), False),
          "d2_45_5": (dict(dur=0.2, theta_nut=45.0, detune_cents=5.0), True),
          "d2_softz": (dict(dur=0.2, kcz=1.2e3, rcz=0.05), False), "d2_nofric": (dict(dur=0.2, friction=False), False)}
    for k, (ov, en) in C1.items():
        put(k, T1.simulate(ov, energy=en)[2])
    for k, (ov, en) in C2.items():
        put(k, T2.simulate(ov, energy=en)[2])
    for name in ("kharaj", "pa", "sa", "kharaj_steel"):
        p = P1.preset(name, dur=0.1)
        put_dict(f"p1_{name}", p)
        if name == "kharaj":
            put("p1run_kharaj", T1.simulate({k: v for k, v in p.items() if k != "meta"}, energy=True)[2])
    for name in ("kharaj", "pa", "sa"):
        put_dict(f"p2_{name}", P2.preset2d(name, dur=0.1))
    pk = P2.preset2d("kharaj", dur=0.1)
    sig = P2.render2d(pk, energy=True)
    for k in ("Fy", "Fz", "Fny", "Fnz", "mono"):
        out[f"p2run_kharaj|{k}"] = sig[k]
    put("p2run_kharaj_raw", sig["r"])
    pf = P2.preset2d("kharaj", f1=69.127, pluck_deg=90.0, theta_nut=45.0, detune_cents=7.0, pluck_disp=2.7e-3,
                     dur=0.1)
    pf["kcz"] = 2.5 * pf["kc"]
    put("p2run_fit", T2.simulate(P2.sim_params(pf), energy=True)[2])
    out["radiate|y"] = P2.radiate(sig["mono"])
    np.savez(out_path, **out)
    return len(out)


def _run_dump(mdir, path, single_thread=True):
    """Dump in a fresh process.  BLAS threading changes round-off (multi-threaded LU in static_compliance), so two
    dumps are only comparable with the same threading: single_thread=False keeps the caller's environment."""
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    if not single_thread:
        env = {k: v for k, v in os.environ.items() if k not in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--bitdump", str(mdir), str(path)], check=True,
                   env=env, cwd=str(mdir), stdout=subprocess.DEVNULL)
    d = np.load(path)
    return {k: d[k] for k in d.files}


def compare_dumps(ref, new):
    same, diff, missing = 0, [], []
    for k, a in ref.items():
        if k not in new:
            missing.append(k)
            continue
        b = new[k]
        if a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
            same += 1
        else:
            diff.append(k)
    extra = sorted(set(new) - set(ref))
    return same, diff, missing, extra


def test_A(ref_npz=None):
    log("=== A. Defaults bit-identical to git HEAD (tanpura1d, tanpura2d, presets, presets2d) ===")
    work = OUT / "bit"
    work.mkdir(parents=True, exist_ok=True)
    head = Path(tempfile.mkdtemp(prefix="head_", dir=work))
    tar = subprocess.run(["git", "-C", str(PKG), "archive", "HEAD", "."], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(head)], input=tar, check=True)
    t0 = time.time()
    d_head = _run_dump(head, work / "head.npz")
    d_work = _run_dump(PKG, work / "work.npz")
    same, diff, missing, extra = compare_dumps(d_head, d_work)
    ok = not diff and not missing
    log(f"  HEAD vs working tree: {same} arrays bitwise identical, {len(diff)} different, {len(missing)} missing "
        f"({time.time() - t0:.0f} s)")
    if diff:
        log(f"  DIFFERENT: {diff[:20]}")
    new_keys = sorted({k.split('|')[1] for k in extra})
    log(f"  new entries in the working tree (new DEFAULT keys in preset dicts, values off): {new_keys}")
    row = dict(head_same=same, head_diff=diff, head_missing=missing, new_keys=new_keys)
    if ref_npz and Path(ref_npz).exists():
        d_ref = dict(np.load(ref_npz))
        d_work2 = _run_dump(PKG, work / "work_default_threads.npz", single_thread=False)
        s2, df2, m2, _ = compare_dumps(d_ref, d_work2)
        log(f"  pre-edit dump {ref_npz} (made with default BLAS threading) vs the working tree with default threading: "
            f"{s2} identical, {len(df2)} different, {len(m2)} missing")
        ok = ok and not df2 and not m2
        row.update(pre_edit_same=s2, pre_edit_diff=df2)
    sha = {}
    for f in ("tanpura1d.py", "body.py", "statics.py"):
        h0 = hashlib.sha256((head / f).read_bytes()).hexdigest()
        h1 = hashlib.sha256((PKG / f).read_bytes()).hexdigest()
        sha[f] = h0 == h1
    log(f"  unchanged files (SHA-256 equal to HEAD): {sha}")
    import shutil
    shutil.rmtree(head, ignore_errors=True)
    ok = ok and all(sha.values())
    log(f"  PASS={ok}")
    save("A", dict(test="A. default outputs bitwise identical to git HEAD", passed=bool(ok), **row, sha_equal=sha))


# ================================================================================ shared set-up
FIT_X = dict(pluck_deg=90.0, theta_nut=45.0, detune_cents=7.0, target_gap=0.0, pluck_disp=2.7e-3, kcz_x=2.5)
FIT_F1 = 69.127


def fitted_kharaj(dur, f1=FIT_F1, **geom):
    """FIT2D best own-kharaj parameters (final_kharaj.json 'best' own_kharaj) as a tanpura2d dict."""
    import presets2d as P2
    if not geom:
        import fit2d as F
        p = F.make_params("kharaj", FIT_X, dur, f1=f1)
    else:
        p = P2.preset2d("kharaj", f1=f1, target_gap=0.0, **geom)
        p.update(dur=float(dur), pluck_deg=FIT_X["pluck_deg"], theta_nut=FIT_X["theta_nut"],
                 detune_cents=FIT_X["detune_cents"], pluck_disp=FIT_X["pluck_disp"])
        p["kcz"] = p["kc"] * FIT_X["kcz_x"]
    return P2.sim_params(p)


def energy_check(sp, label):
    import tanpura2d as T2
    p, s, r = T2.simulate(sp, energy=True)
    H, D, Df, W = r["H"], r["D"], r["Dfric"], r["W"]
    n0 = int(round((p["pluck_pull"] + p["pluck_release"]) * p["fs"])) + 2
    Hm = float(H.max())
    bal = H + D + Df - W - (H[0] + D[0] + Df[0] - W[0])
    res = float(np.max(np.abs(bal)) / Hm)
    dH = np.diff(H[n0:])
    st = r["stats"]
    row = dict(res=res, drift=float(np.max(np.abs(H[n0:] / H[n0] - 1))), max_up=float(dH.max() / Hm),
               step=float(np.max(np.abs(dH)) / Hm), fric_neg=int(np.sum(np.diff(Df) < -1e-30 * Hm)),
               fric_frac=float(Df[-1] / H[n0]), H_ratio=float(H[-1] / H[n0]), contact=float(st["contact_frac"]),
               fails=int(st["newton_fail"] + st["fric_fail"]), newton=float(st["newton_mean"]),
               max_pen_nm=float(st["max_pen"] * 1e9), n_static=int((st["static"] or {}).get("n_active", 0)))
    log(f"  {label}: balance {res:.1e}, H drift {row['drift']:.1e}, largest dH {row['max_up']:+.1e} Hmax, friction "
        f"{100 * row['fric_frac']:.1f} % (power < 0 in {row['fric_neg']} steps), H(end)/H(pluck) {row['H_ratio']:.4f}, "
        f"contact {100 * row['contact']:.0f} %, Newton {row['newton']:.2f}, fails {row['fails']}, "
        f"max pen {row['max_pen_nm']:.0f} nm, static contact pts {row['n_static']}")
    return row


# ================================================================================ B. energy
def test_B():
    log("=== B. Energy: groove + friction + non-proportional damping + TM + nut coupling (kharaj, FIT2D best) ===")
    dur = 0.5
    base = fitted_kharaj(dur)
    LL = dict(lossless=True, rc=0.0, rcz=0.0, friction=False)
    rows = {}
    rows["B1 lossless kappa 30"] = energy_check(dict(base, groove_kappa=30.0, **LL), "B1 lossless, kappa 30 /m")
    rows["B1 lossless kappa 1e3"] = energy_check(dict(base, groove_kappa=1e3, **LL), "B1 lossless, kappa 1e3 /m")
    rows["B1 lossless offset"] = energy_check(dict(base, groove_kappa=30.0, groove_z0=1e-3, **LL),
                                             "B1 lossless, kappa 30 /m, z0 1 mm")
    rows["B1 lossless crown"] = energy_check(dict(base, groove_kappa=-30.0, **LL), "B1 lossless, crown -30 /m")
    eps = max(r["step"] for r in rows.values())
    fr = dict(lossless=True, rc=0.0, rcz=0.0)
    rows["B2 friction"] = energy_check(dict(base, groove_kappa=1e3, groove_z0=0.1e-3, **fr),
                                       "B2 friction 2.84 s/m, kappa 1e3, z0 0.1 mm")
    rows["B2 near-Coulomb"] = energy_check(dict(base, groove_kappa=1e3, groove_z0=0.1e-3, mu_c2=2840.0, **fr),
                                           "B2 friction 2840 s/m (near Coulomb)")
    rows["B3 full lossy"] = energy_check(dict(base, groove_kappa=1e3, groove_z0=0.1e-3, Yb_y=0.3, Yb_z=1e-3),
                                         "B3 Valette + drain (gain, Yb_y 0.3, Yb_z 1e-3) + groove + friction")
    rows["B3 axisym drain"] = energy_check(dict(base, Yb_y=0.3, Yb_z=1e-3, drain_weights="axisym"),
                                           "B3 Valette + drain (axisym), no groove")
    pr = fitted_kharaj(dur, thread_shift=0.1e-3)
    rows["B4 pressed"] = energy_check(dict(pr, groove_kappa=1e3, groove_z0=0.3e-3, Yb_y=0.3, Yb_z=1e-3),
                                      "B4 pressed crest (+0.1 mm), static_init, groove z0 0.3 mm, drain, friction")
    ok_l = all(r["res"] < 1e-10 and r["drift"] < 1e-10 for k, r in rows.items() if k.startswith("B1"))
    ok_f = all(r["res"] < 1e-10 and r["max_up"] <= 2 * eps and r["fric_neg"] == 0 for k, r in rows.items()
               if not k.startswith("B1"))
    ok = ok_l and ok_f and all(r["fails"] == 0 for r in rows.values())
    log(f"  lossless round-off per step (eps) {eps:.1e}; PASS={ok}")
    save("B", dict(test="B. energy conservation / dissipation with the round-3 options", passed=bool(ok), eps=eps,
                   rows=rows))


# ================================================================================ C. kappa -> 0
def _sep(a, b, fs, tt):
    e = np.abs(a - b)
    sc = np.max(np.abs(a))
    out = {}
    for th in (1e-9, 1e-6, 1e-3):
        i = np.where(e > th * sc)[0]
        out[f"{th:g}"] = float(i[0] / fs) if i.size else None
    i_rel = int(tt * fs)
    return float(np.max(e[:i_rel]) / max(np.max(np.abs(a[:i_rel])), 1e-300)), out


def test_C():
    log("=== C. Round-3 kernel limits: no new physics == _run; joint Newton with kappa -> 0 == old solver ===")
    import tanpura2d as T2
    keys = ("Fy", "Fz", "Fny", "Fnz", "Fcont", "Ffric", "H", "D", "Dfric", "W", "pen", "Ey", "Ez", "dT", "slip")
    rows = {}
    ok = True
    for lab, sp in (("C3 default", dict(dur=0.3)), ("kharaj fit", fitted_kharaj(0.3))):
        _, _, a = T2.simulate(sp, energy=True)
        _, _, b = T2.simulate(dict(sp, rec_modes=1000), energy=True)
        eq = all(np.array_equal(a[k], b[k]) for k in keys)
        ok = ok and eq
        rows[f"C1 {lab}"] = dict(identical=bool(eq))
        log(f"  C1 {lab}: _run3 (joint off, no groove, common damping) vs _run: all 15 arrays identical {eq}")
    sp = fitted_kharaj(0.5)
    fs = 176400.0
    tr = sp["pluck_pull"] + sp["pluck_release"]
    _, _, a = T2.simulate(sp)
    for lab, ov in (("C2 joint, kappa 0", dict(solver="joint")), ("C3 groove kappa 1e-6", dict(groove_kappa=1e-6)),
                    ("C3 groove kappa 1e-8", dict(groove_kappa=1e-8)),
                    ("C4 old solver, tol 1e-21 (chaos twin)", dict(tol=1e-21)),
                    ("C5 per-pol path, Yb_y 0, Yb_z = Yb", dict(Yb_z=sp["Yb"]))):
        _, _, b = T2.simulate(dict(sp, **ov))
        rel, sep = _sep(a["Fy"], b["Fy"], fs, tr)
        relz, _ = _sep(a["Fz"], b["Fz"], fs, tr)
        rows[lab] = dict(rel_before_release=rel, rel_Fz=relz, separation_s=sep, newton=float(b["stats"]["newton_mean"]),
                         fails=int(b["stats"]["newton_fail"] + b["stats"]["fric_fail"]))
        log(f"  {lab}: max |dF| / max |F| up to the release ({1e3 * tr:.0f} ms): F_y {rel:.1e}, F_z {relz:.1e}; "
            f"F_y separation to 1e-9/1e-6/1e-3 of max |F_y| at {sep} s")
        ok = ok and rows[lab]["fails"] == 0
    ok = ok and rows["C2 joint, kappa 0"]["rel_before_release"] < 1e-10
    r6, r8 = rows["C3 groove kappa 1e-6"]["rel_before_release"], rows["C3 groove kappa 1e-8"]["rel_before_release"]
    lin = r8 < 1e-10 and r6 / max(r8, 1e-300) > 20.0
    log(f"  C3: the kappa 1e-6 -> 1e-8 difference shrinks by {r6 / max(r8, 1e-300):.0f}x (linear in kappa = a physical "
        f"effect of the groove, not a solver error); kappa 1e-8 is at round-off: {lin}")
    # C5: the per-polarisation coefficients equal the common ones to round-off (~1e-16 in sigma, ~4e-11 in omega*^2
    # of the lowest modes through 2 - dt^2 k*); F_y of a sideways pluck is the y-z beat, which amplifies a relative
    # frequency error by omega/d_omega ~ 250 (7 cents), so F_y agrees to ~1e-9 and F_z (not amplified) closer.
    c5 = rows["C5 per-pol path, Yb_y 0, Yb_z = Yb"]
    ok = ok and lin and c5["rel_before_release"] < 1e-8 and c5["rel_Fz"] < 1e-10
    log(f"  PASS={ok}")
    save("C", dict(test="C. round-3 kernel limits (no new physics, kappa -> 0, proportional damping)",
                   passed=bool(ok), rows=rows))


# ================================================================================ D. static_init with the groove
def test_D():
    log("=== D. static_init with the groove ===")
    import tanpura2d as T2
    rows = {}
    ok = True
    gr = fitted_kharaj(0.3)
    p = dict(T2.DEFAULT); p.update(gr); p["groove_kappa"] = 1e3
    s = T2.build(p)
    q, pz, info = T2.static_equilibrium(p, s)
    z0ok = (np.max(np.abs(q)) == 0.0 and np.max(np.abs(pz)) == 0.0)
    log(f"  D1 grazing kharaj (crest gap 0, max h_j {1e6 * np.max(s['hj']):.2e} um): rest state with groove = 0 "
        f"exactly: {z0ok}")
    ok = ok and z0ok
    pr = fitted_kharaj(0.3, thread_shift=0.1e-3)
    pr["pluck_disp"] = 0.0
    full = dict(Yb_y=0.3, Yb_z=1e-3)
    for lab, ov in (("no groove", {}), ("kappa 1e3", dict(groove_kappa=1e3)),
                    ("kappa 1e3 z0 0.3 mm", dict(groove_kappa=1e3, groove_z0=0.3e-3))):
        pp = dict(T2.DEFAULT); pp.update(pr); pp.update(full); pp.update(ov)
        s = T2.build(pp)
        q, pz, info = T2.static_equilibrium(pp, s)
        rows[lab] = dict(q=q, pz=pz, info=info)
        _, _, r = T2.simulate(dict(pr, **full, **ov))
        dFy = float(np.max(np.abs(r["Fy"] - r["Fy"][0])))
        dFz = float(np.max(np.abs(r["Fz"] - r["Fz"][0])))
        rows[lab].update(dFy=dFy, dFz=dFz, Fy0=float(r["Fy"][0]), Fz0=float(r["Fz"][0]))
        log(f"  D2 pressed crest (+0.1 mm), {lab}: {info['n_active']} contact pts, static force "
            f"{1e3 * info['static_force']:.3f} mN (z {1e3 * info.get('static_force_z', 0.0):+.4f} mN), max |p| "
            f"{1e9 * info['max_z']:.3f} nm, residual {info['residual']:.1e}, {info['iters']} it; 0.3 s with "
            f"drain + friction + TM + nut, no pluck: max |F_y - F_y(0)| {dFy:.1e} N, |F_z - F_z(0)| {dFz:.1e} N")
        ok = ok and dFy < 1e-11 and dFz < 1e-11
    a, b = rows["no groove"], rows["kappa 1e3"]
    dq = float(np.max(np.abs(a["q"] - b["q"])) / np.max(np.abs(a["q"])))
    dp = float(np.max(np.abs(a["pz"] - b["pz"])))
    pen = a["info"]["max_pen"]
    log(f"  D3 rest state with vs without the centred groove (anisotropic nut): max |dq|/max|q| {dq:.1e}, max |dp| "
        f"{dp:.1e} m.  The rest z is {1e9 * a['info']['max_z']:.2f} nm (nut anisotropy), so the groove changes the "
        f"static gap by kappa z^2 ~ {1e3 * a['info']['max_z'] ** 2:.1e} m = {1e3 * a['info']['max_z'] ** 2 / pen:.1e} of "
        f"the static penetration {1e9 * pen:.3f} nm: a physical O(kappa z^2) effect")
    # with an isotropic nut (theta_nut = 0) the rest state has z = 0 exactly and the groove must not change it at all
    iso = {}
    for lab, ov in (("iso", {}), ("iso+groove", dict(groove_kappa=1e3))):
        pp = dict(T2.DEFAULT); pp.update(pr); pp.update(full); pp.update(theta_nut=0.0); pp.update(ov)
        s = T2.build(pp)
        iso[lab] = T2.static_equilibrium(pp, s)
    dqi = float(np.max(np.abs(iso["iso"][0] - iso["iso+groove"][0])) / np.max(np.abs(iso["iso"][0])))
    zi = float(np.max(np.abs(iso["iso+groove"][1])))
    log(f"  D4 isotropic nut (theta_nut 0): rest z = {zi:.1e} m with the groove; rest q with vs without the groove: "
        f"max |dq|/max|q| {dqi:.1e}")
    ok = ok and dq < 1e-5 and zi == 0.0 and dqi < 1e-12
    log(f"  PASS={ok}")
    save("D", dict(test="D. static_init with the groove", passed=bool(ok), grazing_zero=bool(z0ok), dq_rel=dq, dp=dp,
                   rows={k: dict(dFy=v["dFy"], dFz=v["dFz"], n_active=v["info"]["n_active"],
                                 static_force=v["info"]["static_force"],
                                 static_force_z=v["info"].get("static_force_z", 0.0), max_z=v["info"]["max_z"])
                         for k, v in rows.items()}))


# ================================================================================ E. non-proportional damping
def pair_poles(s, i):
    """Discrete poles (as continuous-time rates) of mode pair i of the scheme, and the exact continuous eigenvalues
    of m [x'' + 2 diag(sig_y, sig_z) x' + R diag(w1^2, w2^2) R^T x] = 0."""
    dt = s["dt"]
    A = np.array([[s["Ayy"][i], s["Ayz"][i]], [s["Azy"][i], s["Azz"][i]]])
    B = np.diag([s["by"][i], s["bz"][i]])
    C = np.block([[A, -B], [np.eye(2), np.zeros((2, 2))]])
    z = np.linalg.eigvals(C)
    ld = np.log(z.astype(complex)) / dt
    th = s["theta"]
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    K = R @ np.diag([s["om"][i] ** 2, s["om2"][i] ** 2]) @ R.T
    Sg = np.diag([s["sig_y"][i], s["sig_z"][i]])
    S = np.block([[np.zeros((2, 2)), np.eye(2)], [-K, -2 * Sg]])
    lc = np.linalg.eigvals(S)
    ld = ld[np.argsort(ld.imag)][2:]
    lc = lc[np.argsort(lc.imag)][2:]
    return ld, lc


def test_E():
    log("=== E. Non-proportional damping update vs exact eigenvalues and an independent integrator ===")
    import tanpura2d as T2
    from scipy.integrate import solve_ivp
    from scipy.linalg import expm
    rows = {}
    ok = True
    for lab, ov in (("gain Yb_y 0.3", dict(Yb_y=0.3, Yb_z=1e-3)), ("gain Yb_y 1.0", dict(Yb_y=1.0, Yb_z=1e-3)),
                    ("axisym Yb_y 0.3", dict(Yb_y=0.3, Yb_z=1e-3, drain_weights="axisym")),
                    ("proportional limit (Yb_z = Yb)", dict(Yb_z=5e-3))):
        p = dict(T2.DEFAULT); p.update(fitted_kharaj(0.1)); p.update(ov)
        s = T2.build(p)
        er_s, er_f = 0.0, 0.0
        for i in range(s["M"]):
            ld, lc = pair_poles(s, i)
            er_s = max(er_s, float(np.max(np.abs(ld.real - lc.real) / np.abs(lc.real))))
            er_f = max(er_f, float(np.max(np.abs(ld.imag - lc.imag) / np.abs(lc.imag))))
        rows[lab] = dict(rel_err_decay=er_s, rel_err_freq=er_f, max_sig_y=float(s["sig_y"].max()))
        log(f"  E1 {lab}: {s['M']} mode pairs, discrete vs exact continuous eigenvalues: max rel. error of the decay "
            f"rates {er_s:.1e}, of the frequencies {er_f:.1e} (max sigma_y {s['sig_y'].max():.1f} /s)")
        ok = ok and er_s < 1e-5 and er_f < 1e-8
    # E2: the kernel's free decay of one strongly non-proportional mode pair vs solve_ivp
    for lab, ov in (("69 Hz, 45 deg, 7 c, Yb_y 1", dict(Yb_y=1.0)), ("69 Hz, 30 deg, 40 c, Yb_y 0.3", dict(
            Yb_y=0.3, theta_nut=30.0, detune_cents=40.0))):
        sp = dict(fitted_kharaj(2.0), M=1, thread=False, bridge=False, tm=False, Yb_z=1e-3, pluck_disp=None,
                  Fe_amp=0.0, **ov)
        p = dict(T2.DEFAULT); p.update(sp)
        s = T2.build(p)
        x0 = np.array([0.0, 1e-3])                      # 1 mm along z (a sideways start)
        th = s["theta"]
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        K = R @ np.diag([s["om"][0] ** 2, s["om2"][0] ** 2]) @ R.T
        Sg = np.diag([s["sig_y"][0], s["sig_z"][0]])
        Sm = np.block([[np.zeros((2, 2)), np.eye(2)], [-K, -2 * Sg]])
        dt = s["dt"]
        Phi = expm(-Sm * dt)                              # x(-dt) = x(0) fixes v(0) (the kernel starts at rest: x^-1 = x^0)
        v0 = np.linalg.solve(Phi[:2, 2:], x0 - Phi[:2, :2] @ x0)
        _, _, r = T2.simulate(sp, init=(x0[:1], x0[1:]))
        N = r["N"]
        t = (np.arange(N) + 1) * dt
        q = r["Fy"] / s["w0"][0]; pz = r["Fz"] / s["w0"][0]
        sol = solve_ivp(lambda tt, u: Sm @ u, (0.0, t[-1]), np.concatenate([x0, v0]), method="DOP853", rtol=1e-12,
                        atol=1e-20, t_eval=t[::50], dense_output=False)
        e = max(np.max(np.abs(q[::50] - sol.y[0])), np.max(np.abs(pz[::50] - sol.y[1]))) / 1e-3
        Ey = q ** 2; Ez = pz ** 2
        rows[f"E2 {lab}"] = dict(max_err_rel=float(e), sig_y=float(s["sig_y"][0]), sig_z=float(s["sig_z"][0]))
        log(f"  E2 {lab} (sigma_y {s['sig_y'][0]:.2f} /s, sigma_z {s['sig_z'][0]:.3f} /s, M = 1, 2 s): kernel vs "
            f"solve_ivp DOP853 (rtol 1e-12): max |x_kernel - x_ivp| = {e:.1e} of the 1 mm start; y-share of q^2 at "
            f"0.5/1/2 s {[round(float(Ey[int(x / dt) - 1] / (Ey[int(x / dt) - 1] + Ez[int(x / dt) - 1] + 1e-300)), 3) for x in (0.5, 1.0, 2.0)]}")
        ok = ok and e < 1e-6
    # E3: T60 per partial and polarisation (continuous linear system incl. the thread springs/dampers in y and z)
    t60 = {}
    for lab, ov in (("no drain (FIT2D best)", {}), ("drain gain 0.1", dict(Yb_y=0.1, Yb_z=1e-3)),
                    ("drain gain 0.3", dict(Yb_y=0.3, Yb_z=1e-3)),
                    ("drain axisym 0.3", dict(Yb_y=0.3, Yb_z=1e-3, drain_weights="axisym"))):
        f, T60, ysh, s = linear_modes(dict(fitted_kharaj(0.1), **ov))
        rows_l = []
        for n in range(1, 13):
            sel = np.where(np.abs(f - n * s["om"][0] / (2 * np.pi)) < 0.3 * s["om"][0] / (2 * np.pi))[0]
            sel = sel[np.argsort(-ysh[sel])][:2]
            rows_l.append(dict(n=n, f=float(f[sel[0]]), T60_ymode=float(T60[sel[0]]), y_ymode=float(ysh[sel[0]]),
                               T60_zmode=float(T60[sel[1]]), y_zmode=float(ysh[sel[1]]),
                               T60_sig_y=float(6.9078 / s["sig_y"][n - 1]), T60_sig_z=float(6.9078 / s["sig_z"][n - 1])))
        t60[lab] = rows_l
        log(f"  E3 {lab}: T60 (s) per partial n = 1..12: drain rates 6.91/sigma_y | 6.91/sigma_z | coupled eigenmodes "
            f"(y-share) incl. nut and thread")
        for r_ in rows_l:
            log(f"      n{r_['n']:2d} {r_['f']:7.1f} Hz  sigma_y {r_['T60_sig_y']:6.2f} sigma_z {r_['T60_sig_z']:6.2f} | "
                f"modes {r_['T60_ymode']:6.2f} (y {r_['y_ymode']:.2f}), {r_['T60_zmode']:6.2f} (y {r_['y_zmode']:.2f})")
    log(f"  PASS={ok}")
    save("E", dict(test="E. non-proportional damping: poles and independent integrator", passed=bool(ok), rows=rows,
                   t60=t60))


def linear_modes(sp):
    """Eigenmodes of the continuous linear string (contact off): per mode pair m [x'' + 2 diag(sig_y, sig_z) x' +
    R diag(w1^2, w2^2) R^T x] plus the thread springs/dampers k_c, r_c (y) and k_cz, r_cz (z) on g_c.  Returns
    frequencies (Hz), T60 (s), y-share of each eigenvector and the set-up dict."""
    import tanpura2d as T2
    p = dict(T2.DEFAULT); p.update(sp)
    s = T2.build(p)
    M, m, gc = s["M"], s["m"], s["gc"]
    th = s["theta"]
    c, sn = np.cos(th), np.sin(th)
    w1, w2 = s["om"] ** 2, s["om2"] ** 2
    K = np.zeros((2 * M, 2 * M)); C = np.zeros((2 * M, 2 * M)); i = np.arange(M)
    K[i, i] = m * (c * c * w1 + sn * sn * w2); K[i + M, i + M] = m * (sn * sn * w1 + c * c * w2)
    K[i, i + M] = K[i + M, i] = m * c * sn * (w1 - w2)
    K[:M, :M] += s["kc"] * np.outer(gc, gc); K[M:, M:] += s["kcz"] * np.outer(gc, gc)
    C[i, i] = 2 * m * s["sig_y"]; C[i + M, i + M] = 2 * m * s["sig_z"]
    C[:M, :M] += s["rc"] * np.outer(gc, gc); C[M:, M:] += s["rcz"] * np.outer(gc, gc)
    A = np.block([[np.zeros((2 * M, 2 * M)), np.eye(2 * M)], [-K / m, -C / m]])
    lam, V = np.linalg.eig(A)
    k = lam.imag > 0
    lam, V = lam[k], V[:2 * M, k]
    o = np.argsort(lam.imag)
    lam, V = lam[o], V[:, o]
    ysh = np.sum(np.abs(V[:M]) ** 2, 0) / np.sum(np.abs(V) ** 2, 0)
    return lam.imag / (2 * np.pi), 6.9078 / (-lam.real), ysh, s


# ================================================================================ F. pitch design
def test_F():
    log("=== F. Pitch design: presets.preset(..., design_f1=...) ===")
    import presets as P1
    import presets2d as P2
    import tanpura1d as T1
    import fit2d as F
    rows = {}
    for r in ("own_kharaj", "itanpura_kharaj", "own_pa", "itanpura_pa", "own_sa", "itanpura_sa"):
        R = F.ref_features(r)
        rows[r] = dict(f0=R["f0"], B=R["B"])
    log("  F1 reference pitch (sequential partial tracking 0.3-3 s): " +
        ", ".join(f"{k} {v['f0']:.3f} Hz" for k, v in rows.items()))
    ok = True
    try:
        P2.preset2d("kharaj", f1=68.95)
        fixed = "ok"
    except ValueError as e:
        fixed = f"fails: {e}"
    log(f"  F2 kharaj f1 = 68.95 Hz with the fixed psi {P1.PSI_DESIGN['bronze']} deg: {fixed}")
    des = {}
    for name, f1 in (("kharaj", 68.95), ("kharaj", 68.75), ("pa", 103.66), ("sa", 138.14)):
        t0 = time.time()
        p = P2.preset2d(name, f1=f1, design_f1=True)
        m = p["meta"]
        q = {k: v for k, v in P2.to_1d(p).items()}
        pp = dict(T1.DEFAULT); pp.update(q)
        s1 = T1.build(pp)
        f_lin = T1.lowest_linear_freq(s1["T"], s1["mu"], s1["EI"], s1["L"], s1["beta"], s1["gc"], pp["kc"], s1["M"],
                                      1.0 / pp["fs"])
        cents = 1200 * np.log2(f_lin / f1) if np.isfinite(f_lin) else np.nan
        des[f"{name} {f1}"] = dict(psi=m["psi_designed"], x_c=m["x_c"], crest_gap=m["crest_gap"], T=float(p["T"]),
                                   tune_cents=float(cents), t=time.time() - t0)
        log(f"  F3 {name} f1 {f1} Hz, design_f1=True: psi {m['psi_designed']:.4f} deg, thread at "
            f"{1e3 * m['x_c']:.3f} mm, crest gap {1e6 * m['crest_gap']:+.4f} um, T {p['T']:.3f} N; contact-off "
            f"lowest linear frequency {f_lin:.5f} Hz ({cents:+.4f} cents); {time.time() - t0:.0f} s")
        ok = ok and abs(m["crest_gap"]) < 1e-8 and (not np.isfinite(cents) or abs(cents) < 0.01)
    p0 = P1.preset("kharaj")
    p1 = P1.preset("kharaj", design_f1=None)
    same = all(np.array_equal(np.asarray(p0[k]), np.asarray(p1[k])) if not isinstance(p0[k], tuple) else
               all(np.array_equal(a, b) for a, b in zip(p0[k], p1[k])) for k in p0 if k != "meta")
    log(f"  F4 design_f1=None leaves the preset unchanged: {same}")
    ok = ok and same and fixed != "ok"
    log(f"  PASS={ok}")
    save("F", dict(test="F. pitch design", passed=bool(ok), refs=rows, fixed_psi=fixed, designs=des))


# ================================================================================ G. CPU
def test_G():
    log("=== G. CPU per second of audio (single thread, kharaj FIT2D best, 2 s) ===")
    import tanpura2d as T2
    sp = fitted_kharaj(2.0)
    T2.simulate(dict(sp, dur=0.01)); T2.simulate(dict(sp, dur=0.01, groove_kappa=30.0))
    rows = {}
    for lab, ov in (("original kernel _run", {}), ("_run3 joint Newton, kappa 0", dict(solver="joint")),
                    ("groove kappa 30", dict(groove_kappa=30.0)), ("groove kappa 1e3", dict(groove_kappa=1e3)),
                    ("body drain (gain, 0.1)", dict(Yb_y=0.1, Yb_z=1e-3)),
                    ("groove 1e3 + drain", dict(groove_kappa=1e3, Yb_y=0.1, Yb_z=1e-3))):
        _, _, r = T2.simulate(dict(sp, **ov))
        rows[lab] = float(r["cpu"] / 2.0)
        log(f"  {lab:32s} {rows[lab]:.2f} s/s  (Newton {r['stats']['newton_mean']:.2f}/{r['stats']['newton_max']}, "
            f"line-search halvings {r['stats'].get('n_linesearch', 0)}, contact {100 * r['stats']['contact_frac']:.0f} %)")
    log(f"  load average {os.getloadavg()[0]:.2f} on {os.cpu_count()} CPUs")
    save("G", dict(test="G. CPU", passed=True, cpu_s_per_s=rows))


# ================================================================================ H. demo
DEMO_DUR = 5.0
REF = "own_kharaj"
CHAIN = dict(gz=0.1, hp_hz=150.0, hp_order=2, mode_gain_db=0.0)     # physical output chain (hp <= 150 Hz, order <= 2)
CHAIN_GRID = dict(gz=(0.0, 0.02, 0.05, 0.1, 0.2, 0.35), hp_hz=(60.0, 80.0, 100.0, 125.0, 150.0), hp_order=(1, 2),
                  mode_gain_db=(-6.0, 0.0, 6.0))
FIT_CHAIN = dict(gz=0.1, hp_hz=400.0, hp_order=2, mode_gain_db=-6.0)  # FIT2D's fitted chain for the baseline
DEMO_CASES = {   # label: (round-3 overrides, f1 or None)
    "baseline (FIT2D best)": ({}, None),
    "groove kappa 10": (dict(groove_kappa=10.0), None),
    "groove kappa 30": (dict(groove_kappa=30.0), None),
    "groove kappa 1e3": (dict(groove_kappa=1e3), None),
    "groove kappa 30, z0 1 mm": (dict(groove_kappa=30.0, groove_z0=1e-3), None),
    "drain gain 0.1": (dict(Yb_y=0.1, Yb_z=1e-3), None),
    "drain axisym 0.3": (dict(Yb_y=0.3, Yb_z=1e-3, drain_weights="axisym"), None),
    "both: z0 groove + drain gain 0.1": (dict(groove_kappa=30.0, groove_z0=1e-3, Yb_y=0.1, Yb_z=1e-3), None),
    "both: kappa 1e3 + drain axisym 0.3": (dict(groove_kappa=1e3, Yb_y=0.3, Yb_z=1e-3, drain_weights="axisym"), None),
    "drain gain 0.03": (dict(Yb_y=0.03, Yb_z=1e-3), None),
    "both: z0 groove + drain gain 0.03": (dict(groove_kappa=30.0, groove_z0=1e-3, Yb_y=0.03, Yb_z=1e-3), None),
    "both: z0 groove + drain 0.1, split 10 c": (dict(groove_kappa=30.0, groove_z0=1e-3, Yb_y=0.1, Yb_z=1e-3,
                                                     detune_cents=10.0), None),
}
PITCH_CASE = "both: z0 groove + drain gain 0.1"


def demo_key(label, ov, f1):
    return hashlib.sha1(json.dumps([label, ov, f1, DEMO_DUR, FIT_X, "r3demo-1"], sort_keys=True).encode()).hexdigest()[:12]


def _render_demo(args):
    label, ov, f1 = args
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    path = OUT / "runs" / f"{demo_key(label, ov, f1)}.npz"
    if path.exists():
        return label, str(path)
    import tanpura2d as T2
    import fit2d as F
    from common import to_audio
    if f1 is None:
        sp = fitted_kharaj(DEMO_DUR)
    else:
        sp = fitted_kharaj(DEMO_DUR, f1=f1, design_f1=True)
    sp.update(ov)
    sp["rec_modes"] = 882
    pp, s, r = T2.simulate(sp)
    Fy, Fz = to_audio(r["Fy"], pp["fs"]), to_audio(r["Fz"], pp["fs"])
    i0 = int(round((pp["pluck_pull"] + pp["pluck_release"] - 0.005) * SR))
    f0, B = F.fit_f0_B(Fy[i0:], pp["f1"])
    st = r["stats"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, Fy=Fy, Fz=Fz, i0=i0, f0=f0, B=B, f1=pp["f1"], cpu=r["cpu"], mEy=r["mEy"].astype(np.float32),
             mEz=r["mEz"].astype(np.float32), mEx=r["mEx"].astype(np.float32), t_modes=r["t_modes"],
             Ey=r["Ey"], Ez=r["Ez"], t_rec=r["t_rec"],
             stats=np.array([st["newton_fail"], st["fric_fail"], st["max_pen"], st["contact_frac"],
                             st["newton_mean"]]), sig_y=s["sig_y"], sig_z=s["sig_z"], om=s["om"])
    return label, str(path)


def qf_features(y, f0, nmax=60):
    """quickfeat.features with B = 0 (the recordings and renders are harmonic): 180 ms / 20 ms STFT, partial n at
    n f0 +- 0.3 f0; level re max at 0.05/0.3/1/2 s, rise = max - level(0.05 s), t_max, H4-10 share, H31-60."""
    import scipy.signal as ss
    nper = int(0.18 * SR)
    f, t, Z = ss.stft(y, SR, nperseg=nper, noverlap=nper - int(0.02 * SR), boundary=None)
    S = np.abs(Z) ** 2
    A = np.zeros((nmax, len(t)))
    for n in range(1, nmax + 1):
        sel = (f > n * f0 - 0.3 * f0) & (f < n * f0 + 0.3 * f0)
        A[n - 1] = S[sel].max(0)
    tot = A.sum(0) + 1e-30
    lev = 10 * np.log10(tot)
    lev -= lev.max()

    def i(x):
        return int(np.argmin(np.abs(t - x)))
    return dict(level_db={x: float(lev[i(x)]) for x in (0.05, 0.3, 1.0, 2.0)}, rise_db=float(lev.max() - lev[i(0.05)]),
                t_max=float(t[np.argmax(lev)]), share_H4_10={x: float(A[3:10, i(x)].sum() / tot[i(x)])
                                                            for x in (0.05, 1.0, 3.0)},
                hf_31_60_db={x: float(10 * np.log10(A[30:60, i(x)].sum() / tot[i(x)] + 1e-30)) for x in (1.0, 3.0)})


BANDS4 = ((1, 3), (4, 10), (16, 31), (31, 60))


def band_db(A, t, tt):
    tot = A.sum(0) + 1e-30
    i = int(np.argmin(np.abs(t - tt)))
    return [float(10 * np.log10(A[a - 1:b, i].sum() / tot[i] + 1e-30)) for a, b in BANDS4]


def late_metrics(A, t):
    """second bloom: largest rise of the total level within 2-3.8 s above its running minimum; decay 1.5-3.5 s."""
    L = 10 * np.log10(A.sum(0) + 1e-30)
    L -= L.max()
    sel = (t >= 2.0) & (t <= 3.8)
    Ls = L[sel]
    rr = float(np.max(Ls - np.minimum.accumulate(Ls)))
    s2 = (t >= 1.5) & (t <= 3.5)
    slope = float(np.polyfit(t[s2], L[s2], 1)[0])
    return rr, slope, L


def hf_burst(y):
    """reference.py's attack HF burst: energy above 4 kHz in the first 20 ms vs 20-200 ms after the anchor (the
    signals start 5 ms before the onset)."""
    import scipy.signal as ss
    h = ss.sosfiltfilt(ss.butter(4, 4000.0, "highpass", fs=SR, output="sos"), y)
    a = int(0.005 * SR)
    ea = np.mean(h[max(0, a - int(0.002 * SR)):a + int(0.020 * SR)] ** 2)
    eb = np.mean(h[a + int(0.020 * SR):a + int(0.200 * SR)] ** 2)
    return float(10 * np.log10(ea / eb))


def analyse_demo(d, R_own, R_it):
    import body
    import fit2d as F
    Fy, Fz, i0, f0 = d["Fy"], d["Fz"], int(d["i0"]), float(d["f0"])
    y = body.apply_body(Fy + CHAIN["gz"] * Fz, SR, "tanpura_male", hp_hz=CHAIN["hp_hz"], hp_order=CHAIN["hp_order"],
                        mode_gain_db=CHAIN["mode_gain_db"])[i0:]
    f, t, Z = F.stft(y)
    sl, _ = F.partial_slices(f, f0, 0.0)
    A = F.penv(np.abs(Z) ** 2, sl)
    fe = F.features(t, A, R_own["tcut"])
    D_phys = F.distance(R_own["feat"], fe)[:2]
    run = dict(Fy=Fy, Fz=Fz, i0=i0, f0=f0, B=0.0)
    tt, c, envs = F.model_envelopes(run, CHAIN_GRID["mode_gain_db"], CHAIN_GRID["gz"])
    best = {}
    for rn, R in (("own", R_own), ("itanpura", R_it)):
        bb = None
        for (mgd, gz), Ae in envs.items():
            for o in CHAIN_GRID["hp_order"]:
                for fc in CHAIN_GRID["hp_hz"]:
                    D, Db, _ = F.distance(R["feat"], F.features(tt, Ae * F.hp_gain(c, fc, o)[:, None], R["tcut"]))
                    if bb is None or D < bb[0]:
                        bb = (D, Db, dict(gz=gz, hp_hz=fc, hp_order=o, mode_gain_db=mgd))
        best[rn] = bb
    fitc = None
    Ae = F.model_envelopes(run, (FIT_CHAIN["mode_gain_db"],), (FIT_CHAIN["gz"],))[2].popitem()[1]
    fitc = F.distance(R_own["feat"], F.features(tt, Ae * F.hp_gain(c, FIT_CHAIN["hp_hz"], FIT_CHAIN["hp_order"])[:, None],
                                                 R_own["tcut"]))[:2]
    rr, slope, L = late_metrics(A, t)
    mE = d["mEy"] + d["mEz"] + d["mEx"]
    tm = d["t_modes"]

    def share13(x):
        k = int(np.argmin(np.abs(tm - (x + i0 / SR))))
        return float(mE[k, :3].sum() / mE[k].sum())
    ez = d["Ez"] / np.maximum(d["Ey"] + d["Ez"], 1e-30)
    zs = {x: float(ez[int(np.argmin(np.abs(d["t_rec"] - x)))]) for x in (0.05, 1.0, 3.0)}
    return dict(qf=qf_features(y, f0), feat=fe, bands={x: band_db(A, t, x) for x in (0.09, 1.0, 3.0)},
                hfb=hf_burst(y), rerise=rr, slope=slope, D_phys=D_phys, D_best=best, D_fitchain=fitc,
                E13={x: share13(x) for x in (0.09, 1.0, 1.5, 3.0)}, zshare=zs, f0=f0,
                cents=float(1200 * np.log2(f0 / R_own["f0"])), L=L, t=t, y=y,
                stats=[float(v) for v in d["stats"]], cpu=float(d["cpu"]))


def ref_analysis(R):
    import fit2d as F
    y = F.load_ref(R["name"])
    f, t, Z = F.stft(y)
    sl, _ = F.partial_slices(f, R["f0"], 0.0)
    A = F.penv(np.abs(Z) ** 2, sl)
    rr, slope, L = late_metrics(A, t)
    return dict(qf=qf_features(y, R["f0"]), feat=R["feat"], bands={x: band_db(A, t, x) for x in (0.09, 1.0, 3.0)},
                hfb=hf_burst(y), rerise=rr, slope=slope, L=L, t=t, f0=R["f0"], y=y)


def _fmt_row(lab, a):
    q = a["qf"]
    b = a["bands"]
    bs = " | ".join("/".join(f"{v:5.1f}" for v in b[x]) for x in (0.09, 1.0, 3.0))
    sh = "/".join(f"{v:.2f}" for v in q["share_H4_10"].values())
    hf = "/".join(f"{v:5.1f}" for v in q["hf_31_60_db"].values())
    return (f"  {lab:36s} rise {q['rise_db']:5.1f} t_max {q['t_max']:4.2f} H4-10 {sh} H31-60 {hf} | bands(1-3/4-10/"
            f"16-31/31-60) .09: {bs} | HFburst {a['hfb']:+5.1f} | 2nd bloom {a['rerise']:4.1f} dB, decay "
            f"{a['slope']:+5.1f} dB/s")


def test_H(jobs=2):
    log("=== H. Quick demo: kharaj 5 s, FIT2D best own-kharaj, physical chain (body tanpura_male, HP 150 Hz/2, gz 0.1) ===")
    import fit2d as F
    from concurrent.futures import ProcessPoolExecutor
    R_own, R_it = F.ref_features(REF), F.ref_features("itanpura_kharaj")
    t0 = time.time()
    tasks = [(lab, ov, f1) for lab, (ov, f1) in DEMO_CASES.items()]
    paths = {}
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            for lab, pth in ex.map(_render_demo, tasks):
                paths[lab] = pth
    else:
        for tk in tasks:
            lab, pth = _render_demo(tk)
            paths[lab] = pth
    # pitch-designed variant: f1 so that the PITCH_CASE sounds at the recording's pitch (one correction step)
    d0 = dict(np.load(paths[PITCH_CASE]))
    c0 = 1200 * np.log2(float(d0["f0"]) / float(d0["f1"]))
    f1p = round(R_own["f0"] * 2 ** (-c0 / 1200), 3)
    lab_p = f"{PITCH_CASE}, tuned (f1 {f1p})"
    DEMO_CASES[lab_p] = (DEMO_CASES[PITCH_CASE][0], f1p)
    paths[lab_p] = _render_demo((lab_p, DEMO_CASES[PITCH_CASE][0], f1p))[1]
    wall = time.time() - t0
    res = {}
    ra = ref_analysis(R_own)
    rit = ref_analysis(R_it)
    log(_fmt_row("recording: own kharaj", ra))
    log(_fmt_row("recording: iTanpura kharaj", rit))
    cpu = 0.0
    for lab in DEMO_CASES:
        d = dict(np.load(paths[lab]))
        a = analyse_demo(d, R_own, R_it)
        res[lab] = a
        cpu += a["cpu"]
        log(_fmt_row(lab, a))
        bo, bi = a["D_best"]["own"], a["D_best"]["itanpura"]
        log(f"      D own: physical chain {a['D_phys'][0]:.1f} (bloom {a['D_phys'][1]:.1f}); best chain with HP <= 150 Hz "
            f"{bo[0]:.1f} ({bo[2]['hp_hz']:.0f} Hz/{bo[2]['hp_order']}, gz {bo[2]['gz']}, modes {bo[2]['mode_gain_db']:+.0f}"
            f" dB); FIT2D chain (400 Hz/2, -6 dB) {a['D_fitchain'][0]:.1f} | D iTanpura best {bi[0]:.1f} | string "
            f"energy in partials 1-3 at 0.09/1/1.5/3 s {'/'.join(f'{v:.2f}' for v in a['E13'].values())} | z share "
            f"{'/'.join(f'{v:.2f}' for v in a['zshare'].values())} | sounds {a['f0']:.3f} Hz ({a['cents']:+.1f} c re "
            f"recording) | fails {a['stats'][0]:.0f}/{a['stats'][1]:.0f} | cpu {a['cpu']:.0f} s")
    log(f"  renders: {cpu:.0f} s CPU (string), {wall:.0f} s wall with {jobs} processes")
    _plot_demo(res, ra, rit)
    _write_wavs(res, ra)
    save("H", dict(test="H. quick demo", passed=True, cpu=cpu, wall=wall, f1_tuned=f1p,
                   rows={lab: dict(qf=a["qf"], bands={str(k): v for k, v in a["bands"].items()}, hfb=a["hfb"],
                                   rerise=a["rerise"], slope=a["slope"], D_phys=a["D_phys"],
                                   D_best={k: [v[0], v[1], v[2]] for k, v in a["D_best"].items()},
                                   D_fitchain=a["D_fitchain"], E13={str(k): v for k, v in a["E13"].items()},
                                   zshare={str(k): v for k, v in a["zshare"].items()}, f0=a["f0"], cents=a["cents"],
                                   stats=a["stats"], cpu=a["cpu"], feat=a["feat"]) for lab, a in res.items()},
                   refs={k: dict(qf=v["qf"], bands={str(x): y for x, y in v["bands"].items()}, hfb=v["hfb"],
                                 rerise=v["rerise"], slope=v["slope"], f0=v["f0"]) for k, v in
                         (("own_kharaj", ra), ("itanpura_kharaj", rit))}))


def _plot_demo(res, ra, rit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (OUT / "png").mkdir(parents=True, exist_ok=True)
    show = ["baseline (FIT2D best)", "groove kappa 1e3", "drain gain 0.1", "both: z0 groove + drain gain 0.1"]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#8a3fd1"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    ax[0].plot(ra["t"], ra["L"], color="#0b0b0b", lw=2.2, label="recording: own kharaj")
    ax[0].plot(rit["t"], rit["L"], color="#8a8984", lw=1.4, ls="--", label="recording: iTanpura kharaj")
    for lab, c in zip(show, cols):
        a = res[lab]
        ax[0].plot(a["t"], a["L"], color=c, lw=1.4, label=lab)
    ax[0].set_xlim(0, 4.0); ax[0].set_ylim(-30, 2)
    ax[0].set_xlabel("time after onset (s)"); ax[0].set_ylabel("total partial level re max (dB)")
    ax[0].set_title("Level curve, physical chain (HP 150 Hz / 2)", loc="left", fontsize=10)
    ax[0].grid(alpha=0.25); ax[0].legend(fontsize=7, frameon=False)
    xs = np.arange(4)
    w = 0.16
    for k, (lab, a, c) in enumerate([("own recording", ra, "#0b0b0b")] + [(lb, res[lb], cc) for lb, cc in
                                                                          zip(show, cols)]):
        ax[1].bar(xs + (k - 2) * w, a["bands"][1.0], w, color=c, label=lab)
    ax[1].set_xticks(xs, ["1-3", "4-10", "16-31", "31-60"])
    ax[1].set_xlabel("partials"); ax[1].set_ylabel("band level re total at 1 s (dB)")
    ax[1].set_title("Band shares at 1 s", loc="left", fontsize=10)
    ax[1].grid(alpha=0.25, axis="y"); ax[1].legend(fontsize=7, frameon=False)
    fig.savefig(OUT / "png" / "round3_demo.png", dpi=130)
    plt.close(fig)


def _write_wavs(res, ra):
    import soundfile as sf
    import fit2d as F
    (OUT / "wav").mkdir(parents=True, exist_ok=True)
    L_ref = None
    for lab, a in res.items():
        y = a["y"]
        tag = "".join(ch if ch.isalnum() else "_" for ch in lab)[:60]
        sf.write(OUT / "wav" / f"{tag}.wav", F.peak_norm(y), SR)
        if lab.startswith("baseline"):
            L_ref = F.loudness_db(y)
    yr = ra["y"] * 10 ** ((L_ref - F.loudness_db(ra["y"])) / 20)          # loudness-matched local copy (git-ignored)
    sf.write(OUT / "wav" / "own_kharaj_reference_loudness_matched.wav", yr / max(1.0, np.max(np.abs(yr)) / 0.89), SR)


# ================================================================================ driver
RESULTS = {}


def log(s=""):
    print(s, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "results.txt", "a") as fh:
        fh.write(s + "\n")


def save(key, entry):
    RESULTS[key] = entry
    path = OUT / "results.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old[key] = entry
    path.write_text(json.dumps(old, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def main(argv):
    if argv and argv[0] == "--bitdump":
        n = bitdump(argv[2])
        print(f"{n} arrays -> {argv[2]}")
        return
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    jobs = 2
    ref = None
    if "--jobs" in argv:
        jobs = int(argv[argv.index("--jobs") + 1])
    if "--ref" in argv:
        ref = argv[argv.index("--ref") + 1]
    sel = [a for a in argv if a in "ABCDEFGH" and len(a) == 1] or list("ABCDEFGH")
    log(f"\n==== validate2d_round3 {' '.join(sel)}  {time.strftime('%Y-%m-%d %H:%M:%S')} ====")
    for k in sel:
        t0 = time.time()
        if k == "A":
            test_A(ref)
        elif k == "H":
            test_H(jobs)
        else:
            globals()[f"test_{k}"]()
        log(f"  ({time.time() - t0:.0f} s)")
    log("summary: " + ", ".join(f"{k} {'PASS' if v.get('passed') else 'FAIL'}" for k, v in RESULTS.items()))


if __name__ == "__main__":
    main(sys.argv[1:])
