# STATICS: the string's rest state over a jawari bridge and thread (`statics.py`)

The 1-D/2-D jawari models used to *assume* the rest geometry: the string straight through a pinned
termination and the thread, with the bridge parabola grazing it (crest gap h0 = 0). Since §4 of
`docs/EVALUATION.md` showed that the sound depends on that gap at the **micrometre** level, this module
*derives* it from statics. It covers a finite-diameter thread on a parametrised or measured bridge profile,
the tail break angle, the speaking-length angle and bending stiffness. The result is fed into
`tanpura1d.py`.

Run everything from `jawari/`. Outputs go to `out/statics/`.

```
python validate_statics.py     # 8 validation tests, ~2 min  -> out/statics/validate_statics.txt
python study_statics.py        # statics sweeps + C3 dynamic sweep, ~5 min (2 processes)
                               #   -> out/statics/study_statics_part{1,2}.txt, *.png, *.npz
```

## 1. Model

**Frame and sign conventions.** These are bridge-frame coordinates:

- x runs along the string, with x = 0 at the bridge's **back face** (tailpiece side) and x increasing towards the nut.
- y is the height, with +y away from the bridge, in metres throughout.
- The **crest gap** is the minimum clearance between the thread-only rest shape and the bridge in front of the thread.
  - **Negative** means the string would be **pressed** onto the bridge, i.e. it rests on it with a static force.
  - (`EVALUATION.md` §4 used the opposite sign, h0 > 0 = pressed. Here crest gap = −h0.)

**Bridge** (`BridgeProfile`). The top surface is b(x) on [0, length], with a **rounded back edge**: a circular
arc of radius r_e (default 0.5 mm) tangent to the profile and to the vertical back face. Constructors:

- `parabolic(c, x_crest, length)` gives b = −c (x − x_crest)².
- `power(c, p, x_crest, x_ref, length)` gives b = −c x_ref² (|x − x_crest|/x_ref)^p. The case p > 2 is flatter at the crest.
- `from_points(x_mm, h_um, smoothing)` builds a cubic spline. With smoothing = σ > 0 (µm rms noise) it is a
  smoothing spline with residual N σ²; with smoothing = 0 it interpolates.
- `from_csv(path, smoothing)` reads columns `x_mm,height_um`, header optional.

**Thread** (`Thread(x_c, d_t)`). A rigid cylinder of effective (compressed) diameter d_t:

- Its centre is at x = x_c, at the lowest height for which it does not penetrate the profile, so it rests on the
  possibly sloped surface.
- Its top arc is an obstacle.

**String** (`String`). It has T, EI and μ:

- `String.c3_dafx16()` gives T = 33.10 N and EI = 6.04e-5 N m², so l = √(EI/T) = 1.35 mm.
- `String.kharaj()` (0.7 mm steel, L = 1.05 m, f1 = 69.4 Hz) gives T = 63.74 N and EI = 2.36e-3 N m², so l = 6.08 mm.
- In both cases T is computed from f1 and L exactly as `tanpura1d.build` does.

**Anchors** (`Setup`):

- **Tail:** a straight free span to the tailpiece at (−L_tail, b(0) − L_tail tan φ). L_tail = 0.15 m is an assumption.
- **Nut:** at x_ref + L_speak, with height b(x_ref) − L_speak tan ψ (or `y_nut`). x_ref defaults to the profile crest.

**Statics problem.** Minimise the small-slope beam-string energy

  E = ½∫(T y′² + EI y″²) dx + E_tail + E_nut   subject to   y ≥ o(x) = max(bridge, thread top).

The far-field spans are **exact Dirichlet-to-Neumann terms**. For a free beam-string span of length D ending in
a pin, the minimum exterior energy, given y0 and the outward slope p0 at the grid end, is

  E_ext = T/(2(D − l)) · (y_anchor − y0 − l p0)² + (T l/2) p0²,

dropping terms of order e^{−D/l}. For EI → 0 this reduces to T(y_anchor − y0)²/(2D).

**Thread-only state and the dynamic model.** The *thread-only* state is the same problem with the front bridge
removed, i.e. termination plus thread. Because the string is linear, its static shape is the exact
superposition base for the dynamics. `profile_for_dynamics()` maps it to `tanpura1d` coordinates:

- x_rel = x − x_term, where x_term is the last back contact before the string lifts onto the thread.
- y = 0 is the thread-only shape.
- h_rel = b − y_thread-only, which is > 0 where the string is pressed. h_rel carries the boundary-layer detail at
  µm resolution, rather than asking 133 modes to represent it.

It also returns:

- xc_rel and L_rel;
- the static thread preload (thread-only and full);
- a ready-made `tanpura1d` override dict `t1` with keys L, xc, profile, rho, A, E, I, T, xe and we.

## 2. Numerics: what was needed, and why

1. **Grid.** The grid is uniform, h ≤ 10 µm. It covers the bridge plus max(4 l, 1 mm) on each side, which is
   ~3–7 k nodes. The FD energy uses the operators T/h·D1ᵀD1 + EI/h³·D2ᵀD2 plus the 2×2 DtN blocks: an SPD
   pentadiagonal matrix, factorised by LAPACK banded Cholesky.
2. **Solver.**
   - The prescribed primal-dual active set (PDAS) method **cycles** on this 4th-order obstacle problem: adjacent
     nodes on the high-curvature thread and edge arcs alternate between ±10⁶ N multipliers. That is a known
     non-M-matrix failure.
   - The solver therefore uses a **Mehrotra primal-dual interior-point method**, ~20 iterations with two banded
     solves each, started from the exact EI = 0 solution (the upper convex hull of the obstacle, lifted by 1 µm).
   - It then does an **exact active-set polish**: the equality-constrained QP on the identified set.
   - PDAS (with a cycling safeguard) is kept only to *verify* KKT, and it converges in 1 iteration.
   - For EI = 0 the hull **is** the discrete solution.
   - The contact multipliers λ = K y − f are nodal contact forces in N.
3. **Extended-precision refinement.**
   - The matrix has cond ≈ 5e11 at h = 10 µm and ≈ 8e12 at 5 µm. Rounding the float64 entries (EI/h³ ≈ 1e11–1e13)
     breaks the exact zero row sums of the difference operators. That injects spurious low-mode stiffness
     ≈ eps·EI/h³, which is **not** small against the physical lowest stiffness (~1 N/m for the free spans).
   - The float64 matrix is therefore a different problem: a 1e-16 m change of h moved the thread force by 5e-4 N.
   - The refinement residual is evaluated in long double **from the difference stencils themselves**, so that
     translations and rotations are annihilated exactly. The float64 Cholesky acts only as preconditioner.
   - After the fix, perturbations of 1e-16 m change the thread force and the total force by < 1e-8 N.
4. **Grid alignment of point contacts.**
   - With EI > 0 the string touches the thread and the edge arc at single points. The far-field lines of a
     beam-string meet *above* a point force, so the far field depends to **first order** on the x-position of the
     contact.
   - A contact on a node a distance e from the true tangent point costs ≈ e·F/(2T), up to 0.3 µm at h = 10 µm.
     This is not the e²/(2r) one might expect.
   - The spacing h′ ≈ h and the grid offset are therefore chosen so that **both** tangent points (thread and edge)
     are nodes. This is iterated 2–4 times.
   - Result: the crest gap changes by 0.0003–0.0007 µm when h is halved, against 0.1–0.25 µm before.
5. **Crest gap.** It is the smallest **interior** local minimum of the clearance, with sub-grid parabolic
   refinement. A minimum at the thread edge or at the bridge's front end is not a graze; if no interior minimum
   exists, the crest gap is NaN.
6. **Cost.** 0.03–0.2 s per rest state (thread-only + full), single-threaded.

## 3. API

```python
import statics as S
pr = S.BridgeProfile.parabolic(c=3.0, x_crest=2e-3, length=40e-3)       # or .from_csv("profile.csv", smoothing=0.5)
st = S.String.c3_dafx16()
su = S.Setup(pr, st, S.Thread(x_c=4e-3, d_t=0.2e-3), phi_deg=10, psi_deg=4.05, L_speak=st.L)
rs = S.rest_state(su)                    # thread-only (rs.to) and full (rs.full) StaticSolution
rs.summary()                             # crest_gap, x_graze, x_term, back_contact, thread/back/front forces, iterations
rs.to.y, rs.to.x, rs.to.forces(), rs.full.contact_x("front"), rs.to.front_clearance()
xc, rs = S.auto_place(su, target_gap=0.0)            # "set the jawari": thread position for a target crest gap
psi = S.design_psi(su, x_c0=4e-3)                    # instrument set-up: psi for grazing at x_c0
dyn = S.profile_for_dynamics(rs, h_floor=-0.3e-3)    # tanpura1d coordinates (+ dyn["t1"] override dict)

import tanpura1d as T1
p, s, r = T1.simulate(dict(dyn["t1"], dur=3.0, static_init=True))
```

## 4. `tanpura1d.py` integration

All new parameters are off by default. The default behaviour is **bit-identical**:

- Before editing, I saved `simulate(dict(dur=1))` Fbridge/Fnut, the energy run's H, and an h0 ≠ 0 run and a
  no-bridge run. After editing, the max abs difference is exactly 0 for all six arrays.
- `validate1d.py` reproduces the reference `validation_results.txt` (tests A–E) **line for line**, apart from CPU timings.
- `validate2d.py` test D passes: 2.1e-9 agreement over 0–0.3 s; mean buzz −11.25 vs −11.24 dB.
- `study_thread_geometry.py` (through the updated `jlib.py`) reproduces the EVALUATION §4 table exactly.

| new parameter | effect |
|---|---|
| `profile=(x_rel, h_rel)` | Bridge points at spacing `dxb` (0.2 mm) over the profile support, or over `br_x0`/`br_x1`, shifted so that the profile maximum is a point. `h0` is added as an offset. |
| `static_init=True` | `static_state()`: Newton with backtracking on the convex potential, for the **discrete** scheme's fixed point. This uses ω\*² (which depends on dt and σ) and not ω², so that q⁰ = q⁻¹ = q_s is an exact rest state. It includes the trapezoidal thread spring and the discrete-gradient contact. `_run` takes a new last argument `q_init`. |
| `T=…` | Tension override (the default is from f1 and L, as before). |
| `tune='speaking'` | T chosen so that the lowest linear eigenfrequency **of the discrete scheme** with the thread spring (contact off) equals f1. It solves the exact secular equation 1 + k_c cos θ Σ c_i g_i²/(2cos θ − a_i) = 0, with the mode count kept consistent. Measured in the time domain: **+0.0002 cents**. The untuned model is **+9.9 cents** sharp at 176.4 kHz (+8.8 at 44.1 kHz) from the thread spring alone; the remainder of the "+12 cents" comes from the nonlinear bridge contact, which tuning does not address. |

`jlib.simulate_sig` now computes `q_init` via `T1.initial_state(p, s)` *after* its damping override, because
the static fixed point depends on ω\*².

## 5. Validation (`validate_statics.py`, all PASS)

| test | result |
|---|---|
| (i) EI = 0, point thread, sharp corner vs analytic kinked string | shape error 7e-17 m; corner and thread forces to 5e-14 N; crest gap to 2e-11 µm |
| (ii) beam-string boundary layer vs analytic Green's function y = A + B(x−x0) − F/(2T)\|x−x0\| − F l/(2T) e^{−\|x−x0\|/l} | error 1.3–1.6 nm on boundary-layer (BL) lifts of 191 µm (C3) and 873 µm (kharaj); 14.6 nm (1.7e-5 relative) with a grid margin of only 1.5 l, which confirms the exact far-field terms; thread force to 3e-8 relative |
| (iii) Σ contact forces vs T × total slope change | equal to 3e-10 N. Against the nominal anchor angles T(tan φ + tan ψ): +0.2 % (C3) and +1.5 % (kharaj), because the anchors are not at exactly the datum angles once the string sits on the bridge |
| (iv) grid convergence h = 20 / 10 / 5 µm | \|gap(10 µm) − gap(5 µm)\| ≤ **0.0007 µm** for both strings, EI on and off (criterion 0.05 µm) |
| (v) dg/dx_c vs 2√(c d_t) for EI → 0 | **0.02–0.05 %** agreement once the exact effective thread height and a finite-nut-distance factor (1 − u/D), with u = √(d/c), are included. The bare formula is off by −0.5 to −0.9 %. With real EI the sensitivity departs from the rule by **−26…+23 % (C3)** and **−4…+105 % (kharaj)** |
| (vi) modal static (`static_init`) vs fine statics, C3 pressed by 2 µm | fine-grid statics of the same continuum problem: bridge force 7.62 mN. **133 modes: +6.3 %**; 222 modes: +1.6 %; 345: +0.4 %; 512: −0.1 %. Penetration 0.4 nm, Newton 5 iterations. Mapping error (rigid thread and real termination vs pinned + 1 mm spring thread): −4.5 % |
| (vii) tune / static_init | tuned pitch +0.0002 cents. A 3 µm pressed crest with no pluck gives a 8.6 N peak-to-peak start-up "click" without `static_init`, 5e-15 N with it. Energy drift 4e-12, balance 6e-12 |
| (viii) measured profile via CSV and spline | a noise-free CSV reproduces the crest gap to 0.000 µm. **0.5 µm rms profile noise → crest-gap error −1.9 µm unsmoothed (grazing position off by 50 µm), +0.11 µm with smoothing.** The power-law family runs |

## 6. Findings (`study_statics.py`)

> **Note on materials.** The kharaj numbers in this section are for the **steel** string (`String.kharaj()` defaults: ρ 7800, E 2e11). The user's kharaj is **phosphor bronze**, and `presets.preset('kharaj')` uses bronze: T 71.39 N, l = 4.26 mm, ψ = 5.79°. For the bronze preset at grazing, the EI-induced crest-gap shift is **+471 µm**, confirmed by the independent solver in `crosscheck_statics.py` (steel: +803 µm; C3: +91 µm). The qualitative findings are the same.

**Set-up.** The bridge is a parabola with c = 3/m and its crest 2 mm from the back face, so the surface falls
towards the nut; it is 40 mm long with r_e = 0.5 mm. Each string gets its own speaking-length angle ψ, chosen so
that the nominal set-up (d_t = 0.2 mm, φ = 10°, real EI) grazes with the thread at x_c = 4 mm:

- ψ = 4.05° for C3;
- ψ = 6.92° for the kharaj.

All other cases run on that fixed instrument. Every geometric value here is an assumption, not a measurement.

1. **Bending stiffness lifts the string at the crest by tens to hundreds of µm.** At fixed thread position
   (crest gap with real EI minus crest gap with EI = 0):
   - **C3: +48 … +164 µm.** This is close to the single-kink estimate l·Δθ_thread/2 (85 vs 91 µm at x_c = 4 mm),
     because the termination is 2–4 l away.
   - **Kharaj: +500 … +860 µm.** That is 30–50 % *more* than l·Δθ/2 (539 vs 803 µm at x_c = 4 mm), because the
     termination is only 0.6–0.9 l from the thread, so the edge boundary layer adds to the thread's.
   - In thread-position terms, EI moves the C3 grazing position **1.2–2.5 mm** towards the termination.
   - For the kharaj the EI = 0 analysis on this instrument predicts the string pressed 500–1000 µm into the
     bridge everywhere. An EI = 0 rest state is useless for the kharaj.
2. **The break angle leaks into the crest gap.** At fixed x_c, with EI on and the string resting on the back:
   - **C3:** going from φ = 5° to 10° changes the gap by **−8 … −12 µm**. That is 2–4 times the ~3–5 µm per 0.1 mm
     thread move that already costs 8–60 dB of buzz.
   - **Kharaj:** **−42 … +250 µm** for the same change; 2° → 10° gives **+58 … +400 µm**. At φ = 5° the kharaj
     mostly lifts off the back, so most of these involve the transition in finding 3.
   - With EI = 0 the break angle has no effect at all, which the solver reproduces exactly.
   - So the tailpiece geometry is part of the jawari adjustment, and for the kharaj it dominates.
3. **Lift-off of the back termination.**
   - When φ is too small for the thread's kink, the string no longer rests on the back of the bridge and hangs
     from the thread: tan φ + tan ψ must exceed ≈ 4√(c d_t) (EI = 0, parabola).
   - At the *grazing* thread position this happens for C3 with d_t = 0.2 mm at φ ≤ 5°, and with d_t = 0.3 mm at
     every φ (the graze then sits only 1.4–2.7 mm from the back face).
   - For the kharaj it happens for every (d_t, φ) except d_t = 0.1 mm at φ ≥ 5° and the second and third crossings at
     d_t = 0.2 mm, φ = 10°.
   - The crest gap jumps at the transition, so a thread position can have **several** grazing solutions (kharaj,
     φ = 10°, d_t = 0.2: x_c* = 3.52, 4.00 and 6.64 mm).
   - The pinned-termination dynamic model cannot represent the lifted state, and `profile_for_dynamics` refuses it.
4. **Sensitivity.** dg/dx_c at grazing is 36–73 µm/mm for C3 and 22–107 µm/mm for the kharaj (EI on; one kharaj
   crossing has −40 µm/mm). The EI = 0 values are 34–59 µm/mm.
5. **Dynamics (C3, 3 s; mean buzz 0.3–2.8 s; nut force; 0.2 / 0.05 N pluck).**

   | thread shift from grazing | new: crest gap | new: buzz 0.2 / 0.05 N | old sweep: buzz 0.2 / 0.05 N (no static init) | old + static_init |
   |---|---|---|---|---|
   | −0.20 mm | −7.0 µm (pressed) | −24.1 / −46.3 dB | −21.9 / **−8.1** | −29.7 / −124.5 |
   | −0.10 mm | −3.5 µm | −18.3 / −30.1 | −17.1 / −16.0 | −18.7 / −52.7 |
   | 0 | 0 | **−11.9 / −11.9** | −10.7 / −10.7 | −10.7 / −10.7 |
   | +0.10 mm | +3.6 µm | −29.3 / −22.9 | −18.3 / −65.5 | same as old |
   | +0.20 mm | +7.3 µm | −26.1 / **−128.6 (silent)** | −23.9 / −127.9 | same as old |

   - The statics-derived model keeps the sharp optimum at grazing.
   - A **0.05 mm** move already costs 5–15 dB, and the soft-pluck jawari switches off beyond ~+7 µm.
   - Over ±0.1 mm the 0.2 N buzz range is 17 dB (old: 8 dB). The larger dg/dx_c (36 vs 30 µm/mm) and the different
     h_rel shape make the new model *more* position-sensitive.
   - **Correction to EVALUATION §4(a):** the pressed-crest results for soft plucks there (e.g. "−8.1 dB at +6 µm
     pressed, 0.05 N") are **start-up artefacts**. A pressed crest started from a straight string releases an
     ~8 N click. Started from its static equilibrium, the same geometry gives −124.5 dB: the string stays in
     permanent contact, and permanent contact does not buzz.
   - The artefact grows with the press:
     - at 1 µm (validate1d E5, 0.05 N) it is small: −20.1 dB without static init, −23.0 dB with it; 0.8 N is
       unchanged at −9.2/−9.3 dB;
     - at 3–6 µm it dominates the soft-pluck results of the §4(a) table.
   - `study_thread_geometry_2d.py` already used static initialisation and is not affected.
   - Checks for these runs:
     - edge penetration 0 for the new runs;
     - 0 Newton failures;
     - max dynamic thread force 0.23 N against a static preload ≥ 3.99 N, so the bilateral thread is valid;
     - 96–101 bridge points per run; ~3.6 s CPU per 3 s run.

## 7. Limitations

- **Small-slope theory.** Forces are T × slope, not T × sin. The tail and nut angles are 4–10° (1.5 % force
  error). Large-rotation (elastica) corrections to the boundary-layer shape are O(Δθ²/6) ≈ 0.2–0.5 % of BL
  amplitudes of 100–900 µm. That is 0.2–4 µm, comparable to the effects studied for the kharaj.
- **Thread.**
  - The thread is rigid with a given *effective* diameter; compression under the preload (4–19 N) is absorbed
    into d_t, not modelled.
  - In the dynamics it is a 1.2e5 N/m spring on a 1 mm window. That is a different idealisation; the (a) vs (b)
    mapping error in validation (vi) is 4.5 % of the static contact force.
- **Termination.**
  - `tanpura1d` pins the string at x_term. Rolling of the contact point on the edge arc, and dynamic lift-off from
    the back edge, are not modelled.
  - The thread-only rest shape is computed with the back termination as an obstacle, but the dynamic model only
    sees the front bridge.
- **Geometry is assumed, not measured:**
  - parabolic bridge, c = 3/m (DAFx-16 value), crest position, 0.5 mm edge radius;
  - tailpiece distance 0.15 m;
  - the per-string ψ design.

  The kharaj in particular needed ψ ≈ 7° to graze at all on a c = 3/m bridge (no interior graze at ψ = 3.8°).
  Real jawari bridges are filed per string. **The method is ready for a measured profile** (`from_csv`), but the
  numbers above are for these assumed instruments.
- **Modal truncation.** With 133 modes the static contact force is 6 % too high, and the string cannot follow
  the µm h_rel shape between bridge points except through the static h_rel itself. The dynamic sweep keeps the
  default 133 modes.
- **Material.** The kharaj is 0.7 mm steel as specified. A phosphor-bronze kharaj (E ≈ 1.1e11, ρ ≈ 8800) would
  have l ≈ 4.2 mm, and the EI effects would shrink roughly in proportion. Not run.
- **Dynamic study scope.** Only C3, 3 s per run, one instrument. The kharaj dynamic sweep is not done.
