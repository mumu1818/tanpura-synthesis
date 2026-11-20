"""Validation of the tanpura2d feature-parity extensions (statics geometry, static_init, tuning, directed finger
pluck, Valette loss; see MODEL_2D_PLUCK.md) and the sideways-pluck 'bloom' demonstration.

    cd jawari
    python validate2d_parity.py              # tests A-F and the demo G  (~10 min wall with 2 processes)
    python validate2d_parity.py A C          # selected tests
    python validate2d_parity.py G --jobs 2   # demo only (renders are cached in out/2d_pluck/runs/*.npz)

A  defaults bit-identical: SHA-256 of 10 default-path configurations == hashes recorded from the original code,
   and (if git history is available) a bitwise comparison with tanpura2d.py of commit ORIG_COMMIT
B  energy with the new options (kharaj 2-D, pressed thread so the static state has contact, finger pluck at 88 deg,
   nut coupling, TM): lossless drift; friction only dissipates; lossy (Valette) balance
C  1-D parity: preset2d('kharaj', pol2=False, tm=False, pluck_deg=0) vs tanpura1d on presets.preset('kharaj'):
   set-up arrays, waveforms before the chaotic divergence, averaged metrics; pressed-thread static state
D  static_init: no start-up click (no pluck, pressed thread, full 2-D model) vs the straight-string start and the
   older approximate init_static; static solver residual and agreement with tanpura1d.static_state
E  tuning: contact off, lossless, TM off: axis-1 speaks at f1, axis 2 detune_cents sharper; tune_ref='mean'
F  CPU seconds per second of audio, 2-D vs 1-D kharaj
G  demo: kharaj, pluck_deg {0,45,80,90} x detune {1,5} cents, + 86/90 deg at 3.5 cents, 90 deg with theta_nut 45,
   the 1-D kharaj and the 2-D preset default (88 deg),
   5 s; bloom features on Fy and on the mono mix after body + 250 Hz radiation high-pass vs the real recordings
Outputs: out/2d_pluck/parity_results.txt/.json, runs/*.npz, wav/*.wav, png/*.png.  The reference recordings are only
READ from REF_DIR (never copied); if they are absent the stored feature table REF_TABLE is printed instead.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy.signal as ss

from common import OUT_ROOT, PKG, analyze, mean_metric, to_audio
import presets as P1
import presets2d as P2
import tanpura1d as T1
import tanpura2d as T2

OUT = OUT_ROOT / "2d_pluck"
for d in ("", "runs", "wav", "png"):
    os.makedirs(OUT / d, exist_ok=True)
RES_PATH = OUT / "parity_results.json"
TXT_PATH = OUT / "parity_results.txt"
LOG = []
ORIG_COMMIT = "f4a483a"          # last commit with the original tanpura2d.py (before the parity extensions)
REF_DIR = os.environ.get("TANPURA_REFS", "/home/user/tanpura-app/assets")
F_REF = 68.56                    # f0 of the kharaj references as given by quickfeat (with B 4.3e-4; see fit_f0_B)


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def save(key, entry):
    d = {}
    if RES_PATH.exists():
        d = json.loads(RES_PATH.read_text())
    d[key] = entry
    RES_PATH.write_text(json.dumps(d, indent=1, default=float))
    with open(TXT_PATH, "a") as fh:
        fh.write("\n".join(LOG) + "\n")
    LOG.clear()


def peak_freqs(y, fs, f_guess, search, nfft=1 << 22):
    """Hann-windowed FFT peak + log-parabolic interpolation near each guess -> (freqs, amps)."""
    w = np.hanning(len(y))
    Y = np.abs(np.fft.rfft(y * w, nfft))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    df = f[1]
    fo, ao = [], []
    for fg in np.atleast_1d(f_guess):
        sel = np.where(np.abs(f - fg) < search)[0]
        k = sel[np.argmax(Y[sel])]
        a, b, c = np.log(Y[k - 1]), np.log(Y[k]), np.log(Y[k + 1])
        d = 0.5 * (a - c) / (a - 2 * b + c)
        fo.append((k + d) * df)
        ao.append(2 * np.exp(b - 0.25 * (a - c) * d) / w.sum())
    return np.array(fo), np.array(ao)


# ================================================================================ A. bit-identity of the defaults
A_CASES = {   # name: (overrides, energy)
    "def1s": (dict(dur=1.0), False),
    "def_energy": (dict(dur=0.3), True),
    "lossless_energy": (dict(dur=0.3, lossless=True, rc=0.0, rcz=0.0), True),
    "h0pos_legacy_static": (dict(dur=0.3, h0=5e-6), True),
    "nobridge": (dict(dur=0.3, bridge=False), False),
    "pol2off_tmoff": (dict(dur=0.3, pol2=False, tm=False), False),
    "p45_d5": (dict(dur=0.3, pluck_deg=45.0, detune_cents=5.0), True),
    "nofric": (dict(dur=0.3, friction=False), False),
    "c2coul": (dict(dur=0.3, mu_c2=2840.0), False),
    "soft_z": (dict(dur=0.3, kcz=1.2e3, rcz=0.05), False),
}
A_KEYS = ("Fy", "Fz", "Fny", "Fnz", "Fcont", "Ffric", "H", "D", "Dfric", "W", "pen", "Ey", "Ez", "dT", "slip")
# SHA-256 of the A_KEYS arrays (float64, in order) of A_CASES, computed with the ORIGINAL tanpura2d.py (commit
# f4a483a) on this machine (numpy 2.x, numba, x86-64) before any edit.  Platform-dependent; on another platform use
# the git comparison (A2), which is platform-independent.
A_HASHES = {
    "def1s": "881391100e114943cf3865f2903f010afd8d9a5719774ae18901ebc38351770c",
    "def_energy": "2489bf5891e45e7632061408b92fd63cbd55b05ec62f9170539a73e3a554cdc4",
    "lossless_energy": "8ccbc8d0bfeccd1a4829280c2fe81f7cacd87d8d8b20d0c3ae62e6bdf0b227b3",
    "h0pos_legacy_static": "cea8e5f11059b5b83ef8f684bee634ee591cba00c0d70e481f439bfbc159e0fa",
    "nobridge": "9bde518cb323e3bb3015a103900a6e2f5d16cb88df50410805d4ca2c43c55f4a",
    "pol2off_tmoff": "452d644733e04e3c64fd93ca9a7105c84354ebaa4d6999ecc445e8ad8dde92df",
    "p45_d5": "4c95f254a8d0d7ca1d9261aa3b9c59c4b551ff9b96ef8d76dc7c372db775d507",
    "nofric": "bb1f72d6d20dcbbc0941bd0f974a4c568baf8197ed0f8a4dbe661282ce7f3aa5",
    "c2coul": "3dbaeaee9891187c0b2c201772827e6ccc8fa29e7475fabc183f91c83042353d",
    "soft_z": "e3130f16c18603d185562926939d24a534af623971a2dd1087bd9e2e93189ecf",
}


def case_hash(mod, ov, energy):
    p, s, r = mod.simulate(ov, energy=energy)
    h = hashlib.sha256()
    for k in A_KEYS:
        h.update(np.ascontiguousarray(r[k], dtype=np.float64).tobytes())
    return h.hexdigest(), r


def default_hashes(mod):
    return {k: case_hash(mod, ov, en)[0] for k, (ov, en) in A_CASES.items()}


def load_original():
    """tanpura2d.py of ORIG_COMMIT as a separate module (None if git history is unavailable)."""
    try:
        src = subprocess.run(["git", "-C", str(PKG), "show", f"{ORIG_COMMIT}:jawari/tanpura2d.py"], check=True,
                             capture_output=True, text=True).stdout
    except Exception as e:   # noqa: BLE001
        log(f"  [git original unavailable: {e}]")
        return None
    d = tempfile.mkdtemp(prefix="t2orig_")
    path = os.path.join(d, "tanpura2d_orig.py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("tanpura2d_orig", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_A():
    log("=== A. Defaults bit-identical to the original tanpura2d ===")
    ok_h = True
    rows = {}
    for k, (ov, en) in A_CASES.items():
        hsh, _ = case_hash(T2, ov, en)
        ref = A_HASHES.get(k)
        same = (ref == hsh)
        ok_h = ok_h and same
        rows[k] = dict(hash=hsh, match=same)
        log(f"  A1 {k:22s} sha256 {hsh[:16]}...  recorded {str(ref)[:16]}...  {'identical' if same else 'DIFFERENT'}")
    orig = load_original()
    ok_g = None
    worst = None
    if orig is not None:
        ok_g = True; worst = 0.0
        for k, (ov, en) in A_CASES.items():
            _, _, r0 = orig.simulate(ov, energy=en)
            _, _, r1 = T2.simulate(ov, energy=en)
            d = max(float(np.max(np.abs(r0[a] - r1[a]))) for a in A_KEYS)
            eq = all(np.array_equal(r0[a], r1[a]) for a in A_KEYS)
            st_eq = all((r0["stats"][a] == r1["stats"][a]) for a in r0["stats"] if np.isscalar(r0["stats"][a]))
            ok_g = ok_g and eq and st_eq
            worst = max(worst, d)
            log(f"  A2 {k:22s} vs git {ORIG_COMMIT}: max abs diff {d:.1e}, arrays identical {eq}, stats identical {st_eq}")
    passed = ok_h and (ok_g is not False)
    log(f"  PASS={passed} (hashes {ok_h}, git comparison {ok_g})")
    save("A", dict(test="A. defaults bit-identical (10 default-path configurations: SHA-256 vs the original code, and a "
                        f"bitwise comparison with tanpura2d.py of commit {ORIG_COMMIT})", passed=bool(passed),
                   result=f"hashes identical: {ok_h}; git comparison: {ok_g}, max abs diff {worst}", rows=rows))


# ================================================================================ B. energy
def pressed_kharaj(**ov):
    """kharaj 2-D with the thread 0.1 mm towards the nut from grazing: the crest is pressed 1.2 um into the bridge,
    so the static state has contact."""
    return P2.preset2d("kharaj", thread_shift=+0.1e-3, **ov)


def energy_check(p, label, eps=None):
    """Discrete energy balance of one run.  eps = per-step round-off level of H (max |dH|/Hmax of the lossless twin);
    'H increased' counts steps with dH > 2 eps Hmax (with eps=None: 1e-13)."""
    p, s, r = T2.simulate(P2.sim_params(p), energy=True)
    H, D, Df, W = r["H"], r["D"], r["Dfric"], r["W"]
    rise = p["pluck_pull"] if p.get("pluck") else p["Fe_rise"]
    fall = p["pluck_release"] if p.get("pluck") else p["Fe_fall"]
    n0 = int(round((rise + fall) * p["fs"])) + 2
    Hm = float(np.max(H))
    bal = H + D + Df - W - (H[0] + D[0] + Df[0] - W[0])
    res = float(np.max(np.abs(bal)) / Hm)
    dH = np.diff(H[n0:])
    step_rel = float(np.max(np.abs(dH)) / Hm)
    thr = 2.0 * eps if eps is not None else 1e-13
    n_up = int(np.sum(dH > thr * Hm))
    max_up = float(np.max(dH) / Hm)
    dDf = np.diff(Df)
    st = r["stats"]
    drift = float(np.max(np.abs(H[n0:] / H[n0] - 1)))
    log(f"  {label}: {r['N']} steps, contact {100 * st['contact_frac']:.0f}% (max {st['max_active']} pts), Newton "
        f"{st['newton_mean']:.2f}/{st['newton_max']} fail {st['newton_fail']}, friction Newton fail {st['fric_fail']}; "
        f"static: {st['static']['n_active'] if st['static'] else 0} pts, {1e3 * (st['static'] or {}).get('static_force', 0):.2f} mN")
    log(f"      max|H + D + D_fric - W - const|/max H = {res:.2e}; H drift after pluck {drift:.2e}; H(end)/H(pluck) "
        f"{H[-1] / H[n0]:.6f}; friction loss / H {Df[-1] / H[n0]:.2e}; max per-step |dH|/Hmax {step_rel:.1e}; "
        f"steps with dH > {thr:.1e} Hmax: {n_up} (largest increase {max_up:+.1e}); friction power < 0 in "
        f"{int(np.sum(dDf < 0))} steps")
    return dict(res=res, drift=drift, n_up=n_up, max_up=max_up, step=step_rel, fric=float(Df[-1] / H[n0]),
                H_ratio=float(H[-1] / H[n0]), fric_neg=int(np.sum(dDf < 0)),
                fails=int(st["newton_fail"] + st["fric_fail"]))


def test_B():
    log("\n=== B. Energy balance with the new options (statics profile + static_init with contact + finger pluck "
        "88 deg + nut coupling 30 deg / 3.5 cents + TM) ===")
    LL = dict(lossless=True, rc=0.0, rcz=0.0)
    b1 = energy_check(pressed_kharaj(dur=1.0, friction=False, **LL), "B1 lossless, friction off, 1 s")
    eps = b1["step"]
    log(f"      -> round-off level of H per step: {eps:.1e} Hmax (used as the noise floor below)")
    b2 = energy_check(pressed_kharaj(dur=1.0, friction=True, **LL), "B2 lossless + friction (c2 2.84 s/m), 1 s", eps)
    b3 = energy_check(pressed_kharaj(dur=0.5, friction=True, mu_c2=2840.0, **LL),
                      "B3 lossless + near-Coulomb friction (c2 2840 s/m), 0.5 s", eps)
    b4 = energy_check(P2.preset2d("kharaj", dur=2.0), "B4 nominal lossy preset (Valette loss, friction), 2 s", eps)
    p1 = b1["drift"] < 1e-9 and b1["fails"] == 0
    p2 = all(b["res"] < 1e-9 and b["n_up"] == 0 and b["fails"] == 0 and b["fric_neg"] == 0 for b in (b2, b3, b4)) \
        and b2["fric"] > 0 and b3["fric"] > 0
    passed = p1 and p2
    log(f"  PASS={passed} (lossless drift < 1e-9: {p1}; balance < 1e-9, friction power >= 0 every step and H never "
        f"increases by more than 2x the lossless round-off: {p2})")
    save("B", dict(test="B. energy with profile + static_init + finger pluck 88 deg + nut coupling + TM", passed=bool(passed),
                   result=f"lossless drift {b1['drift']:.1e} (round-off {eps:.0e} per step); friction: residual "
                          f"{b2['res']:.1e}/{b3['res']:.1e}, H increased (> 2x round-off) in {b2['n_up']}/{b3['n_up']} "
                          f"steps, friction removed {100 * b2['fric']:.2f}%/{100 * b3['fric']:.1f}%; lossy preset residual "
                          f"{b4['res']:.1e}, {b4['n_up']} increases", rows=dict(B1=b1, B2=b2, B3=b3, B4=b4)))


# ================================================================================ C. 1-D parity
def one_d_limit(**ov):
    return P2.preset2d("kharaj", pol2=False, tm=False, pluck_deg=0.0, **ov)


def err_profile(a, b, fs, win=0.002):
    w = int(win * fs)
    mx = np.max(np.abs(a))
    err = np.array([np.max(np.abs(a[i:i + w] - b[i:i + w])) / mx for i in range(0, len(a) - w, w)])
    return np.arange(len(err)) * win, err


def first_above(tt, err, th):
    return float(tt[np.argmax(err > th)]) if np.any(err > th) else np.inf


def test_C():
    log("\n=== C. 1-D parity: preset2d('kharaj', pol2=False, tm=False, pluck_deg=0) vs tanpura1d on preset('kharaj') ===")
    p2 = one_d_limit(dur=3.0)
    p1 = P1.preset("kharaj", dur=3.0)
    s1 = T1.build(p1); s2 = T2.build(P2.sim_params(p2))
    pairs = [("T", s1["T"], s2["T"]), ("M", s1["M"], s2["M"]), ("sigma", s1["sig"], s2["sig"]),
             ("a / Ayy", s1["a"], s2["Ayy"]), ("b", s1["b"], s2["b"]), ("c", s1["c"], s2["c"]),
             ("om*^2 / Kyy", s1["om2s"], s2["Kyy"]), ("g_e", s1["ge"], s2["ge"]), ("g_c", s1["gc"], s2["gc"]),
             ("x_j", s1["xj"], s2["xj"]), ("h_j", s1["hj"], s2["hj"]), ("Phi", s1["Phi"], s2["Phi"]),
             ("w0", s1["w0"], s2["w0"]), ("F_e(t)", T1.excitation(p1, s1), T2.excitation(P2.sim_params(p2), s2))]
    setup_ok = True
    for lab, a, b in pairs:
        a = np.asarray(a, float); b = np.asarray(b, float)
        eq = a.shape == b.shape and np.array_equal(a, b)
        setup_ok = setup_ok and eq
    log(f"  C1 set-up arrays identical (T, M, sigma, a, b, c, omega*^2, g_e, g_c, x_j, h_j, Phi, w0, F_e): {setup_ok}")
    _, _, r1 = T1.simulate(p1)
    _, _, r2 = T2.simulate(P2.sim_params(p2))
    _, _, r1t = T1.simulate(dict(p1, tol=1e-21))          # 1-D twin: Newton tolerance x 1e-3
    fs = p1["fs"]
    tt, err = err_profile(r1["Fbridge"], r2["Fy"], fs)
    _, errt = err_profile(r1["Fbridge"], r1t["Fbridge"], fs)
    t_rel = p1["pluck_pull"] + p1["pluck_release"]
    e_pull = float(np.max(err[tt < p1["pluck_pull"] - 0.002]))
    thr = (1e-12, 1e-9, 1e-6, 1e-3)
    td = [first_above(tt, err, t) for t in thr]
    tdt = [first_above(tt, errt, t) for t in thr]
    m1 = analyze(to_audio(r1["Fbridge"], fs), 44100); m2 = analyze(to_audio(r2["Fy"], fs), 44100)
    mt = analyze(to_audio(r1t["Fbridge"], fs), 44100)
    mb = [(mean_metric(m, "buzz_db", 0.3, 2.9), mean_metric(m, "hf_frac", 0.3, 2.9)) for m in (m1, m2, mt)]
    log(f"  C2 grazing kharaj, 3 s, |F_1D - F_y| / max F: {e_pull:.1e} during the pull (0-{p1['pluck_pull'] * 1e3:.0f} ms, "
        f"release at {t_rel * 1e3:.0f} ms); first exceeds " + ", ".join(f"{t:.0e} at {d:.3f} s" for t, d in zip(thr, td)))
    log(f"     1-D twin (Newton tol 1e-21 instead of 1e-18) vs 1-D: first exceeds "
        + ", ".join(f"{t:.0e} at {d:.3f} s" for t, d in zip(thr, tdt))
        + "  -> the separation is the chaotic amplification of solver-tolerance/round-off differences")
    log(f"     mean buzz 0.3-2.9 s: 1-D {mb[0][0]:.2f}, 2-D {mb[1][0]:.2f}, 1-D twin {mb[2][0]:.2f} dB; hf "
        f"{mb[0][1]:.3f} / {mb[1][1]:.3f} / {mb[2][1]:.3f}")
    # pressed thread: static state with contact
    q1 = P1.preset("kharaj", dur=0.3, thread_shift=+0.1e-3)
    q2 = one_d_limit(dur=0.3, thread_shift=+0.1e-3)
    sq1 = T1.build(q1); sq2 = T2.build(P2.sim_params(q2))
    qs1, i1 = T1.static_state(q1, sq1)
    qs2, _, i2 = T2.static_equilibrium(P2.sim_params(q2), sq2)
    dq = float(np.max(np.abs(qs1 - qs2)) / np.max(np.abs(qs1)))
    _, _, rq1 = T1.simulate(q1)
    _, _, rq2 = T2.simulate(P2.sim_params(q2))
    tq, eq = err_profile(rq1["Fbridge"], rq2["Fy"], fs)
    eq_pull = float(np.max(eq[tq < q1["pluck_pull"] - 0.002]))
    log(f"  C3 pressed thread (+0.1 mm): static contact {i1['n_active']} / {i2['n_active']} points, static bridge force "
        f"{1e3 * i1['static_force']:.4f} / {1e3 * i2['static_force']:.4f} mN, max |q_1D - q_2D| / max|q| = {dq:.1e}; "
        f"waveforms {eq_pull:.1e} during the pull, first > 1e-6 at {first_above(tq, eq, 1e-6):.3f} s")
    dbuzz = abs(mb[0][0] - mb[1][0])
    passed = setup_ok and e_pull < 1e-12 and dbuzz < 1.5 and dq < 1e-12 and eq_pull < 1e-12 and td[3] > 0.5 * tdt[3]
    log(f"  PASS={passed} (set-up identical; < 1e-12 before the release; static states < 1e-12; |d mean buzz| < 1.5 dB; "
        f"divergence not faster than half the 1-D twin's)")
    save("C", dict(test="C. 1-D parity (kharaj: statics profile, static_init, tune, finger pluck, Valette loss)",
                   passed=bool(passed),
                   result=f"set-up arrays identical: {setup_ok}; waveforms agree to {e_pull:.0e} up to the release, "
                          f"then separate like the 1-D twin (>1e-3 after {td[3]:.2f} s vs twin {tdt[3]:.2f} s; chaotic "
                          f"contact dynamics); mean buzz {mb[0][0]:.2f} vs {mb[1][0]:.2f} dB (twin {mb[2][0]:.2f}); pressed "
                          f"thread: static state agrees to {dq:.0e}, waveform {eq_pull:.0e} before the release",
                   t_div=td, t_div_twin=tdt, buzz=mb))


# ================================================================================ D. static_init: no click
def test_D():
    log("\n=== D. static_init: no start-up click (pressed thread, no pluck, full 2-D model: nut coupling, TM, friction, "
        "Valette loss), 0.3 s ===")
    base = pressed_kharaj(dur=0.3, pluck_disp=0.0)
    rows = {}
    for lab, ov in (("exact static_init", dict(static_init=True)),
                    ("older init_static (approx.)", dict(static_init=False, init_static=True)),
                    ("straight string (no init)", dict(static_init=False, init_static=False))):
        p, s, r = T2.simulate(P2.sim_params(dict(base, **ov)))
        dFy = float(np.max(np.abs(r["Fy"] - r["Fy"][0])))
        dFz = float(np.max(np.abs(r["Fz"] - r["Fz"][0])))
        dFn = float(np.max(np.abs(r["Fny"] - r["Fny"][0])))
        st = r["stats"].get("static") or {}
        rows[lab] = dict(dFy=dFy, dFz=dFz, dFn=dFn, Fstat=st.get("static_force", 0.0), res=st.get("residual", np.nan))
        log(f"  {lab:28s} max |F_y(t) - F_y(0)| {dFy:.2e} N, |F_z| change {dFz:.2e} N, nut {dFn:.2e} N; static bridge "
            f"force {1e3 * st.get('static_force', 0.0):.4f} mN" + (f", solver residual {st['residual']:.1e} N, "
                                                                     f"{st['iters']} its, max z {1e9 * st['max_z']:.3f} nm"
                                                                     if st.get('kind') == 'exact' else ""))
    # plucked reference level for scale
    p, s, r = T2.simulate(P2.sim_params(dict(base, pluck_disp=2e-3, dur=0.3)))
    peak = float(np.max(np.abs(r["Fy"] - r["Fy"][0])))
    click = rows["exact static_init"]["dFy"]
    log(f"  plucked (2 mm, 88 deg) peak |F_y - F_y(0)| over 0.3 s: {peak:.3e} N -> exact start-up transient is "
        f"{20 * np.log10(max(click, 1e-300) / peak):.0f} dB below the pluck; straight-string start "
        f"{20 * np.log10(rows['straight string (no init)']['dFy'] / peak):+.0f} dB")
    # the C3 parabola with h0 > 0 (pol2 off): the older closed-form static_state and the exact solver agree
    pc = dict(T2.DEFAULT, h0=5e-6, pol2=False, tm=False, dur=0.01)
    sc = T2.build(pc)
    qa, _, _ = T2.static_state(pc, sc)
    qb, _, ib = T2.static_equilibrium(pc, sc)
    dqc = float(np.max(np.abs(qa - qb)) / np.max(np.abs(qa)))
    log(f"  C3 parabola h0 = +5 um, pol2 off: older static_state vs exact solver: max |dq|/max|q| = {dqc:.1e}")
    passed = click < 1e-9 * max(rows["exact static_init"]["Fstat"], 1e-3) + 1e-12 and dqc < 1e-10
    log(f"  PASS={passed}")
    save("D", dict(test="D. static_init: no start-up click; exactness of the static solver", passed=bool(passed),
                   result=f"exact: max |dF_y| {click:.1e} N ({20 * np.log10(max(click, 1e-300) / peak):.0f} dB re pluck); "
                          f"older init_static {rows['older init_static (approx.)']['dFy']:.1e} N; straight string "
                          f"{rows['straight string (no init)']['dFy']:.1e} N; parabola old vs exact {dqc:.0e}",
                   rows=rows))


# ================================================================================ E. tuning
def test_E():
    log("\n=== E. Tuning (tune='speaking'): contact off, lossless, TM off, 8 s at 176.4 kHz, finger pluck 88 deg ===")
    rows = {}
    ok = True
    for lab, ov in (("axis1, isotropic thread", {}), ("tune_ref='mean'", dict(tune_ref="mean")),
                    ("axis1, soft lateral thread", dict(kcz=1.2e3, rcz=0.05))):
        p = P2.preset2d("kharaj", dur=8.0, bridge=False, lossless=True, tm=False, **ov)
        pp, s, r = T2.simulate(P2.sim_params(p))
        th = np.deg2rad(pp["theta_nut"])
        u1 = np.cos(th) * r["Fny"] + np.sin(th) * r["Fnz"]
        u2 = -np.sin(th) * r["Fny"] + np.cos(th) * r["Fnz"]
        f1 = pp["f1"]
        fa, _ = peak_freqs(u1[int(0.05 * pp["fs"]):], pp["fs"], [f1], search=0.2 * f1)
        fb, _ = peak_freqs(u2[int(0.05 * pp["fs"]):], pp["fs"], [f1], search=0.2 * f1)
        fy, _ = peak_freqs(r["Fny"][int(0.05 * pp["fs"]):], pp["fs"], [f1], search=0.2 * f1)
        fz, _ = peak_freqs(r["Fnz"][int(0.05 * pp["fs"]):], pp["fs"], [f1], search=0.2 * f1)
        e1 = 1200 * np.log2(fa[0] / f1); e2 = 1200 * np.log2(fb[0] / f1); split = e2 - e1
        ey = 1200 * np.log2(fy[0] / f1); ez = 1200 * np.log2(fz[0] / f1)
        emean = 0.5 * (e1 + e2)
        rows[lab] = dict(axis1=e1, axis2=e2, split=split, mean=emean, y=ey, z=ez, T=s["T"])
        log(f"  {lab:28s} T = {s['T']:.6f} N: axis 1 {fa[0]:.5f} Hz ({e1:+.4f} cents re f1 = {f1} Hz), axis 2 "
            f"{fb[0]:.5f} Hz ({e2:+.4f} cents), split {split:.4f} cents (detune_cents {pp['detune_cents']}), mean "
            f"{emean:+.4f}; strongest peak in F_nut,y {ey:+.3f}, in F_nut,z {ez:+.3f} cents")
    ok = abs(rows["axis1, isotropic thread"]["axis1"]) < 0.01 and abs(rows["tune_ref='mean'"]["mean"]) < 0.05
    # pol2 off: the tension is tanpura1d's, bit for bit
    pa = one_d_limit(dur=0.01); s2 = T2.build(P2.sim_params(pa)); s1 = T1.build(P1.preset("kharaj", dur=0.01))
    same_T = s1["T"] == s2["T"]
    log(f"  pol2 off: T_2D == T_1D bit for bit: {same_T} ({s2['T']!r})")
    ok = ok and same_T
    log(f"  PASS={ok} (axis 1 within 0.01 cents of f1; 'mean' within 0.05 cents)")
    save("E", dict(test="E. tune='speaking' in 2-D (contact off): axis-1 series at f1, axis 2 detune_cents sharper",
                   passed=bool(ok), rows=rows,
                   result="; ".join(f"{k}: axis1 {v['axis1']:+.4f}, axis2 {v['axis2']:+.4f} cents, split {v['split']:.3f}"
                                    for k, v in rows.items())))


# ================================================================================ F. CPU
def test_F():
    log("\n=== F. CPU seconds per second of audio (single thread, 3 s, kharaj presets) ===")
    p1 = P1.preset("kharaj", dur=3.0)
    p2 = P2.preset2d("kharaj", dur=3.0)
    T1.simulate(dict(p1, dur=0.01)); T2.simulate(P2.sim_params(dict(p2, dur=0.01)))   # load the JIT caches
    _, _, r1 = T1.simulate(p1)
    _, _, r2 = T2.simulate(P2.sim_params(p2))
    _, _, r3 = T2.simulate(P2.sim_params(dict(p2, tm=False)))
    c1, c2, c3 = r1["cpu"] / 3, r2["cpu"] / 3, r3["cpu"] / 3
    load = os.getloadavg()[0]
    log(f"  1-D kharaj {c1:.2f} s/s; 2-D kharaj (pluck 88 deg, TM, friction) {c2:.2f} s/s = {c2 / c1:.2f} x 1-D; "
        f"2-D without TM {c3:.2f} s/s  (load average {load:.1f} on {os.cpu_count()} CPUs)")
    save("F", dict(test="F. CPU per second of audio", passed=True, cpu_1d=c1, cpu_2d=c2, cpu_2d_notm=c3, load=load,
                   result=f"1-D {c1:.2f} s/s, 2-D {c2:.2f} s/s ({c2 / c1:.2f}x), 2-D without TM {c3:.2f} s/s"))


# ================================================================================ G. demo: the sideways pluck bloom
# Bloom features of a single pluck (quickfeat): per-partial STFT envelopes (180 ms window, 20 ms hop, partials 1..60
# at n f0 sqrt(1 + B n^2), +-0.3 f0), total level re its maximum, rise = max - level(0.05 s), t_max, energy share of
# partials 4-10 at 0.05/1/3 s, energy of partials 31-60 re total at 1/3 s.
REF_FILES = {   # name: (path relative to REF_DIR, string, measured f0)
    "itanpura_kharaj": ("iTanpura/Tanpura_CS3_Kharaj.mp3", "kharaj", 68.56),
    "own_kharaj": ("Mumu_Tanpura/Male/tt_CS3_kharaj.mp3", "kharaj", 68.56),
    "itanpura_pa": ("iTanpura/Tanpura_CS3_Pa.mp3", "pa", 103.8),
    "itanpura_sa": ("iTanpura/Tanpura_CS3_Sa.mp3", "sa", 138.5),
    "own_pa": ("Mumu_Tanpura/Male/tt_CS3_pa.mp3", "pa", 103.8),
    "own_sa": ("Mumu_Tanpura/Male/tt_CS3_sa.mp3", "sa", 138.5),
}
# the same features as measured on 2026-11-20 (used when the recordings are not available)
REF_TABLE = {
    "itanpura_kharaj": dict(rise_db=17.3, t_max=1.29, share=(0.26, 0.97, 0.95), hf=(-20.2, -15.2)),
    "own_kharaj": dict(rise_db=13.7, t_max=1.01, share=(0.57, 0.96, 0.72), hf=(-24.2, -17.9)),
    "itanpura_pa": dict(rise_db=14.4, t_max=1.17), "itanpura_sa": dict(rise_db=17.1, t_max=1.01),
    "own_pa": dict(rise_db=17.7, t_max=1.15), "own_sa": dict(rise_db=17.2, t_max=0.71),
}
FEAT_TIMES = (0.05, 0.3, 1.0, 2.0, 4.0)


def trim_onset(y, sr, rel_db=-30):
    i0 = int(np.argmax(np.abs(y) > 10 ** (rel_db / 20) * np.abs(y).max()))
    return y[max(0, i0 - int(0.005 * sr)):]


def partial_env(y, sr, f0, B, nmax=60, win=0.18, hop=0.02):
    nper = int(win * sr)
    f, t, Z = ss.stft(y, sr, nperseg=nper, noverlap=nper - int(hop * sr), boundary=None)
    S = np.abs(Z) ** 2
    A = np.zeros((nmax, len(t)))
    for n in range(1, nmax + 1):
        c = n * f0 * np.sqrt(1 + B * n * n)
        sel = (f > c - 0.3 * f0) & (f < c + 0.3 * f0)
        if sel.any():
            A[n - 1] = S[sel].max(0)
    return t, A


def bloom_features(y, sr, f0, B):
    t, A = partial_env(y, sr, f0, B)
    tot = A.sum(0) + 1e-30
    lev = 10 * np.log10(tot); lev -= lev.max()
    i = lambda x: int(np.argmin(np.abs(t - x)))   # noqa: E731
    dur = t[-1]
    return dict(t=t, lev=lev,
                level_db={x: float(lev[i(x)]) for x in FEAT_TIMES if x < dur},
                rise_db=float(lev.max() - lev[i(0.05)]), t_max=float(t[np.argmax(lev)]),
                share=tuple(float(A[3:10, i(x)].sum() / tot[i(x)]) for x in (0.05, 1.0, 3.0) if x < dur),
                hf=tuple(float(10 * np.log10(A[30:60, i(x)].sum() / tot[i(x)] + 1e-30)) for x in (1.0, 3.0) if x < dur),
                h16_40=tuple(float(10 * np.log10(A[15:40, i(x)].sum() / tot[i(x)] + 1e-30)) for x in (0.05, 1.0, 2.0)
                             if x < dur))


def fit_f0_B(y, sr, f_guess, t0=0.3, t1=3.0, nmax=40, B0=0.0):
    """Sounding f0 and inharmonicity B (f_n = n f0 sqrt(1 + B n^2)) by SEQUENTIAL partial tracking: Hann FFT of
    t0..t1, partial n searched within +-0.2 f_guess of the prediction from the robust (soft-L1, cents) fit to partials
    1..n-1.  Picking all partials at once around a guessed stretched series mis-numbers them above n ~ 15 (a 69 Hz
    string has only 69 Hz between partials): that is how a harmonic kharaj (f0 69.32 Hz, B ~ 0) can be fitted as
    f0 68.56 Hz, B 4.3e-4.  Returns (f0, B, f_n)."""
    from scipy.optimize import least_squares
    seg = y[int(t0 * sr):int(t1 * sr)]
    w = np.hanning(len(seg)); nfft = 1 << 21
    Y = np.abs(np.fft.rfft(seg * w, nfft)); f = np.fft.rfftfreq(nfft, 1 / sr); df = f[1]
    ns, fn = [], []
    f0, B = float(f_guess), float(B0)
    for n in range(1, nmax + 1):
        fg = n * f0 * np.sqrt(1 + B * n * n)
        sel = np.where(np.abs(f - fg) < 0.2 * f_guess)[0]
        k = sel[np.argmax(Y[sel])]
        a, b, c = np.log(Y[k - 1]), np.log(Y[k]), np.log(Y[k + 1])
        ns.append(n); fn.append((k + 0.5 * (a - c) / (a - 2 * b + c)) * df)
        if n >= 3:
            nn, ff = np.array(ns), np.array(fn)
            r = least_squares(lambda x: 1200 * np.log2(ff / (nn * x[0] * np.sqrt(1 + x[1] * nn ** 2))), [f0, B],
                              loss="soft_l1", f_scale=3.0)
            f0, B = float(r.x[0]), float(r.x[1])
    return f0, B, np.array(fn)


def ref_features():
    """Bloom features of the references: 'qf' with the quickfeat settings (f0 from REF_FILES, B 4.3e-4 / 1e-4: the
    task's table), 'fit' with the sequentially fitted f0, B (correct partial numbering).  Read-only."""
    out = {}
    for k, (rel, string, f0) in REF_FILES.items():
        path = os.path.join(REF_DIR, rel)
        if not os.path.exists(path):
            continue
        import soundfile as sf
        y, sr = sf.read(path, always_2d=True)
        y = trim_onset(y.mean(1), sr)
        f0f, Bf, _ = fit_f0_B(y, sr, f0)
        out[k] = dict(qf=bloom_features(y, sr, f0, 4.3e-4 if string == "kharaj" else 1e-4),
                      fit=bloom_features(y, sr, f0f, Bf), f0=f0f, B=Bf, string=string)
    return out


def demo_cases():
    """name -> (model, params).  Kharaj presets at their own pitch (69.4 Hz; the references are at 69.32 Hz once
    their partials are numbered correctly, see fit_f0_B), 5 s.  The bloom features are per partial number and use
    each signal's own fitted f0 and B."""
    cases = {"1d": ("1d", P1.preset("kharaj", dur=5.0))}
    for deg in (0.0, 45.0, 80.0, 90.0):
        for det in (1.0, 5.0):
            cases[f"2d_p{deg:g}_d{det:g}"] = ("2d", P2.preset2d("kharaj", dur=5.0, pluck_deg=deg, detune_cents=det))
    for deg in (86.0, 90.0):
        cases[f"2d_p{deg:g}_d3.5"] = ("2d", P2.preset2d("kharaj", dur=5.0, pluck_deg=deg, detune_cents=3.5))
    cases["2d_p90_t45_d3.5"] = ("2d", P2.preset2d("kharaj", dur=5.0, pluck_deg=90.0, theta_nut=45.0, detune_cents=3.5))
    cases["2d_preset"] = ("2d", P2.preset2d("kharaj", dur=5.0))     # pluck 88 deg, 30 deg / 3.5 cents, gz 0.1
    return cases


def _render_case(args):
    name, model, p = args
    path = OUT / "runs" / f"{name}.npz"
    if path.exists():
        return name, float(np.load(path)["cpu"])
    if model == "1d":
        pp, s, r = T1.simulate(p)
        y = to_audio(r["Fbridge"], pp["fs"]); z = np.zeros_like(y)
        cpu = r["cpu"]
    else:
        sig = P2.render2d(p)
        y, z, cpu = sig["Fy"], sig["Fz"], sig["cpu"]
        r = sig["r"]
    np.savez(path, Fy=y, Fz=z, cpu=cpu, Ey=r.get("Ey", np.zeros(1)), Ez=r.get("Ez", np.zeros(1)),
             t_rec=r.get("t_rec", np.zeros(1)))
    return name, float(cpu)


def test_G(jobs=2):
    log("\n=== G. Demo: does the sideways pluck reproduce the bloom on the real (statics) kharaj model? ===")
    log(f"  kharaj presets (f1 69.4 Hz), 5 s; features use each signal's sequentially fitted f0, B.  Radiation: "
        "presets2d.radiate = body.apply_body(y, 44100, 'tanpura_male', hp_hz=250, hp_order=2).  mono = F_y + gz F_z.")
    cases = demo_cases()
    todo = [(k, m, p) for k, (m, p) in cases.items()]
    t0 = time.time()
    if jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(jobs) as pool:
            cpu = dict(pool.map(_render_case, todo, chunksize=1))
    else:
        cpu = dict(map(_render_case, todo))
    log(f"  renders: {time.time() - t0:.0f} s wall, {sum(cpu.values()):.0f} s CPU total")
    refs = ref_features()
    rows = {}
    hdr = (f"  {'case':22s} {'signal':5s} {'rise dB':>7s} {'t_max s':>7s}  {'H4-10 share 0.05/1/3 s':>23s}  "
           f"{'H31-60 dB 1/3 s':>15s}  {'H16-40 dB 0.05/1/2 s':>20s}  level 0.05/0.3/1/2/4 s")
    log(hdr)
    for k, (rel, string, f0) in REF_FILES.items():
        f = refs.get(k)
        if f is None:
            if string == "kharaj":
                g = REF_TABLE[k]
                log(f"  {k:22s} ref   {g['rise_db']:7.1f} {g['t_max']:7.2f}  "
                    f"{'/'.join(f'{v:.2f}' for v in g['share']):>23s}  {'/'.join(f'{v:.1f}' for v in g['hf']):>15s}  "
                    "(stored table, quickfeat settings)")
            continue
        if string == "kharaj":
            log(_feat_line(k, "refQF", f["qf"]) + f"   [quickfeat: f0 {f0}, B 4.3e-4]")
        log(_feat_line(k, "ref", f["fit"]) + f"   [fitted: f0 {f['f0']:.3f} Hz, B {f['B']:.1e}]")
        rows[k] = dict(_feat_row(f["fit"]), f0=f["f0"], B=f["B"], quickfeat=_feat_row(f["qf"]))
    import body  # noqa: F401  (radiate imports it)
    curves = {}
    for name, (model, p) in cases.items():
        d = np.load(OUT / "runs" / f"{name}.npz")
        s = T1.build(p) if model == "1d" else T2.build(P2.sim_params(p))
        B0 = float(np.pi ** 2 * s["EI"] / (s["T"] * s["L"] ** 2))
        f0, B, _ = fit_f0_B(d["Fy"], 44100, p["f1"])
        gz = p.get("gz", 0.3) if model == "2d" else 0.0
        for lab, x in (("Fy", d["Fy"]), ("mono", d["Fy"] + gz * d["Fz"])):
            if model == "1d" and lab == "mono":
                continue
            y = P2.radiate(x, 44100)
            f = bloom_features(trim_onset(y, 44100), 44100, f0, B)
            rows[f"{name}/{lab}"] = dict(_feat_row(f), f0=f0, B=B, cents=1200 * np.log2(f0 / p["f1"]), cpu=cpu[name])
            curves[f"{name}/{lab}"] = (f["t"], f["lev"])
            log(_feat_line(name, lab, f))
            if lab == "mono" or model == "1d":
                import soundfile as sf
                sf.write(OUT / "wav" / f"{name}_radiated.wav", 0.9 * y / np.max(np.abs(y)), 44100)
        dur = 5.0
        Ez = d["Ez"] / np.maximum(d["Ey"] + d["Ez"], 1e-30) if model == "2d" else None
        if Ez is not None:
            tr = d["t_rec"]
            sh = [float(Ez[np.argmin(np.abs(tr - x))]) for x in (0.05, 1.0, 3.0)]
            rows[f"{name}/Fy"]["Ez_share"] = sh
        log(f"  {'':22s}       sounding f0 {f0:.3f} Hz ({1200 * np.log2(f0 / p['f1']):+.1f} cents re f1), fitted B "
            f"{B:.1e} (string alone pi^2 EI/(T L^2) = {B0:.2e}); {cpu[name] / dur:.2f} CPU s/s" +
            (f"; z share of string energy 0.05/1/3 s {'/'.join(f'{v:.2f}' for v in sh)}" if Ez is not None else ""))
    _plot_demo(curves, refs)
    save("G", dict(test="G. sideways-pluck bloom demo on the statics kharaj", passed=True, rows=rows,
                   result="see table (rise, t_max, H4-10 share, H31-60) for Fy and mono after body + 250 Hz HP"))


def _feat_row(f):
    return dict(rise_db=f["rise_db"], t_max=f["t_max"], share=f["share"], hf=f["hf"], h16_40=f["h16_40"],
                level_db=f["level_db"])


def _feat_line(name, lab, f):
    return (f"  {name:22s} {lab:5s} {f['rise_db']:7.1f} {f['t_max']:7.2f}  "
            f"{'/'.join(f'{v:.2f}' for v in f['share']):>23s}  {'/'.join(f'{v:.1f}' for v in f['hf']):>15s}  "
            f"{'/'.join(f'{v:.1f}' for v in f['h16_40']):>20s}  " + "/".join(f"{v:.1f}" for v in f["level_db"].values()))


PLOT_CASES = (  # (case, label), fixed colour order (categorical slots 1-6 of the dataviz reference palette)
    ("1d", "1-D kharaj (vertical pluck)"), ("2d_p0_d5", "2-D, pluck 0 deg"), ("2d_p80_d5", "2-D, pluck 80 deg"),
    ("2d_p86_d3.5", "2-D, pluck 86 deg"), ("2d_preset", "2-D preset, pluck 88 deg"), ("2d_p90_d3.5", "2-D, pluck 90 deg"))
PLOT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")


def _plot_demo(curves, refs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, lab in zip(axs, ("Fy", "mono")):
        for k, f in refs.items():
            if k.endswith("kharaj"):
                f = f["fit"]
                ax.plot(f["t"], f["lev"], color="#222222", lw=2.6 if k.startswith("own") else 1.6,
                        ls="-" if k.startswith("own") else "--",
                        label="recording: own kharaj" if k.startswith("own") else "recording: iTanpura kharaj")
        for (case, name), col in zip(PLOT_CASES, PLOT_COLORS):
            key = f"{case}/{lab}" if case != "1d" else "1d/Fy"
            if key in curves:
                t, lev = curves[key]
                ax.plot(t, lev, color=col, lw=2.0 if case == "2d_preset" else 1.4, label=name)
        ax.set_title("bridge force F_y" if lab == "Fy" else "mono mix F_y + 0.1 F_z", fontsize=11)
        ax.set_xlabel("time after onset (s)"); ax.set_xlim(0, 5); ax.set_ylim(-32, 1)
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("level of partials 1-60 re maximum (dB)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("Kharaj bloom: radiated level vs time (model through body + 250 Hz high-pass)", fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "png" / "bloom_demo.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    jobs = 2
    if "--jobs" in sys.argv:
        jobs = int(sys.argv[sys.argv.index("--jobs") + 1])
        args = [a for a in args if a != str(jobs)]
    which = args or ["A", "B", "C", "D", "E", "F", "G"]
    with open(TXT_PATH, "a") as fh:
        fh.write(f"\n==== validate2d_parity run {time.strftime('%Y-%m-%d %H:%M:%S')} tests {which} ====\n")
    for w in which:
        t0 = time.time()
        if w == "G":
            test_G(jobs)
        else:
            globals()[f"test_{w}"]()
        log(f"  [{w} took {time.time() - t0:.0f} s]")
        with open(TXT_PATH, "a") as fh:
            fh.write("\n".join(LOG) + "\n")
        LOG.clear()
