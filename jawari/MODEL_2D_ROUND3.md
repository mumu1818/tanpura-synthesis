# Round 3: a bridge that feels sideways motion, a body that drains y, and a tunable kharaj

> Run every script from inside `jawari/`. Outputs go to `jawari/out/round3/` (git-ignored). The recordings are only **read** from `/home/user/tanpura-app/assets`; the one loudness-matched copy is written to the git-ignored `out/round3/wav/`.

This round adds the physics the judge named as missing after `FIT2D.md`, so that the string itself (not the output filter) would drain its low partials, buzz in the attack and not bloom twice:

1. a **transverse bridge profile** (groove, crown, offset groove), solved jointly in y and z;
2. a **polarisation-dependent, frequency-dependent body drain**, with an energy-consistent non-proportional damping update;
3. **pitch design** in the presets, so that the kharaj can be tuned to the recording.

## 0. Bottom line

* **All three are implemented, energy-consistent and validated.**
  * The defaults are bit-identical to git HEAD: 935 output arrays of 13 default-path runs and 7 preset dicts.
  * Lossless energy is conserved to 1.6–2.7e-11. With friction, damping and the new options, energy only decreases, and the balance closes to ≤ 2.7e-11.
  * The joint Newton with κ_g → 0 reproduces the old solver to round-off (5.6e-12) until the chaotic contact separates them. It separates at the same time as a twin of the old solver with a tighter tolerance.
  * The damping update matches an independent integrator to 6e-8.
* **What the new physics does to the kharaj** (5 s, FIT2D best own-kharaj parameters, physical output chain with HP 150 Hz / 2nd order):

| target | result |
|---|---|
| **buzz in the attack** | **Partly.** A groove of plausible size gives an earlier click but nothing else: the HF burst rises from −24.5 to −21.0 / −17.6 dB at κ_g = 10 / 30 /m. A string resting on the groove wall (κ_g 30 /m, z0 1 mm, wall slope 0.06) gives **−4.0 dB**, against −1.4 dB (own recording) and +1.6 dB (iTanpura). But partials 16–31 in the first 180 ms stay at −27 dB re total, against −8 dB in your recording. The click is above 4 kHz and shorter than 20 ms; it is not a bright 1–2 kHz attack. |
| **no second bloom** | **Yes, with the body drain.** The re-rise between 2 and 3.8 s falls from 4.6 dB to 0.9 dB (Yb_y 0.1) or 0.2 dB (with the offset groove). The recordings re-rise 1.1 / 0.8 dB. **The cost: the main bloom shrinks from 10.9 to 8.0–8.4 dB** (recording 13.9 dB on the same measure), because the body modes at 320–860 Hz also drain partials 4–12, the ones the jawari pumps. |
| **drain partials 1–3** | **No.** The string keeps 0.89–0.97 of its energy in partials 1–3 at 1 s in every case (0.92 before). Through the physical chain, partials 1–3 sit at −1.5 to −3 dB re total at 1 s, against −15.5 dB in your recording. The 400 Hz high-pass is still needed. |

  **Why the drain cannot reach partials 1–3.** A drain in y empties a mode pair only as fast as the nut coupling carries energy from z (where the sideways pluck puts it) to y.
  * For partial 1 (7 cent split, 45°) the slow, z-like eigenmode keeps T60 ≈ 15 s (Yb_y 0.1), 16 s (0.3) and 34 s (1.0). More drain decouples the polarisations further (the overdamped limit).
  * Your recordings lose partial 1 with T60 ≈ 3–9 s after the bloom.
  * Draining the low partials needs an in-plane loss (Yb_z), a much stronger y–z coupling, or contact acting on z at low frequency. The groove does the last of these, but only weakly at plausible depths: 0.92 → 0.89 at κ_g = 1e3 /m.
* **Feature distance D against your recording** (own kharaj):
  * Physical chain: 223 for the baseline; 222–234 for the groove cases and the weak-drain cases; 231 for groove + drain 0.1; 273–399 for strong drains.
  * Best chain allowed (HP ≤ 150 Hz, order ≤ 2): 155 for the baseline, 154–163 for the groove and weak-drain combinations.
  * With FIT2D's fitted 400 Hz chain, the baseline (49) stays best, because it was fitted with that chain and nothing was refitted here.
  * **None of these one-at-a-time additions improves the match on its own.** They change the attack and the late decay in the right direction, at the cost of the bloom height.
* **Pitch.**
  * With `design_f1`, the bronze kharaj can now be tuned below 69.1 Hz. The ψ design is re-run at the new tension: ψ 5.8049° for f1 = 69.043 Hz.
  * The model then sounds at **69.326 Hz, −0.0 cents from your recording** (69.327 Hz), instead of +4.4 cents.
  * The contact sharpening depends on the physics: +9.4 cents for the baseline, +6.8 cents with groove + drain. So f1 has to be chosen per configuration (one correction step was enough).

## 1. What was added

All new options default to off (`None`/`0`). With all of them off, `simulate()` runs the original kernel `_run`, unchanged.

| option | file | meaning |
|---|---|---|
| `groove_kappa` κ_g [1/m] | tanpura2d | Transverse profile: gap η_j = h_b(x_j) + κ_g z_j² + s z_j − y_j. κ_g > 0 is a groove whose walls rise as the string swings sideways; κ_g < 0 is a crowned face. |
| `groove_z0` [m], `groove_slope` | tanpura2d | s = groove_slope − 2 κ_g z0. z0 is the offset of the groove bottom from the string: the string rests on the groove wall with slope −2 κ_g z0. `groove_slope` is a face tilt across the string. |
| `Yb_y` [s/kg], `drain_body`, `drain_weights` | tanpura2d | Bridge-normal modal mobility: Re Y_y(f) = Re Y_smooth(Yb) + Yb_y · shape(f). shape(f) is the sum of the single-DOF modal mobilities of body.py's explicit modes (`drain_body`, default `tanpura_male`): A0 65 Hz, T(0,1) 120 Hz, T(1,1) 216 Hz, …, A1 521 Hz, up to 1.5 kHz. Each mode's peak is weighted by its body.py gain relative to T(0,1); T(0,1)'s peak equals Yb_y. `drain_weights='axisym'` keeps only A0 and the T(0,n) modes, the only ones a normal force at the centre of a circular plate drives. Needs `loss='valette'`. |
| `Yb_z` [s/kg] | tanpura2d | In-plane smooth mobility. It replaces Yb in the z drain, with the same fb / mb shape. `None` means = Yb, i.e. isotropic. |
| `solver='joint'`, `rec_modes` | tanpura2d | Force the joint (y, z) Newton (for the κ_g → 0 test); record per-mode energies every N steps. |
| `simulate(..., init=(q, p))` | tanpura2d | An explicit initial modal state, for test E only. |
| `design_f1` | presets (a statics key, passed through by `preset2d`) | Re-design ψ so the thread grazes at x_c0 = 4 mm at the tension of f1 = design_f1 (`True` = the preset's own f1). Cached. Combining it with `psi_deg` raises. |

Any round-3 option switches `simulate()` to the new kernel `_run3`. Without new physics, `_run3` reproduces `_run` bit for bit (test C1).

## 2. Transverse bridge profile

**Contact law and discretisation.** The gap of bridge point j is η_j = h_j + κ z_j² + s z_j − y_j. The contact potential is unchanged: V(η) = k_b/(α+1) [η]₊^(α+1) per unit length. The discrete gradient is taken over the two-step interval:

  f_j = [V(η_j^{n+1}) − V(η_j^{n−1})] / (η_j^{n+1} − η_j^{n−1}),

and the generalised force densities are

  on y: **+f_j**;  on z: **−f_j (κ (z_j^{n+1} + z_j^{n−1}) + s)**.

**Proof of the energy identity.**

- η^{n+1} − η^{n−1} = (κ (z^{n+1} + z^{n−1}) + s)(z^{n+1} − z^{n−1}) − (y^{n+1} − y^{n−1}), exactly, because z² differences factor.
- So the work of these forces over the step is dx Σ_j [f_j δy_j − f_j (κ(z^{n+1}+z^{n−1}) + s) δz_j] = −dx Σ_j [V(η_j^{n+1}) − V(η_j^{n−1})].
- With the stored energy ½ dx Σ_j [V(η^{n+1}) + V(η^n)] in H^{n+1/2}, the contact is exactly conservative.
- Signs: the y force is +V′(η) because ∂η/∂y = −1. The z force is −V′(η)(2κz + s): the groove wall pushes the string back towards the bottom.
- f_j ≥ 0 because V is non-decreasing.

**Friction** stays along z: h_j = −μ(w_j) f_j, with w_j = (z_j^{n+1} − z_j^{n−1})/2Δt and μ(w) = 0.39 arctan(2.84 w).
- Its power is −dx Σ μ(w_j) w_j f_j ≤ 0, strictly dissipative, whatever the groove.
- The local face tangent is tilted from z by atan(2κz + s). That is ≤ 0.5° for κ ≤ 30 /m at the 90 µm crest excursion, 3.4° for the offset groove (s = 0.06) and 10° for κ = 1e3 /m.
- At these angles the tangential and z slip velocities differ by < 2 %. The friction law itself (VT-2015) is uncertain by far more, so friction along z is kept. Exact dissipation holds either way.

**Joint Newton.** The contact now couples y and z at every active point. So the block-triangular solve (y contact, then friction in z) is replaced, for the candidate points a, by one Newton on (s_a = η_a^{n+1} − η_a^{n−1}, Z_a = z_a^{n+1}):

  r1_a = s_a − b_a − (κ Z_a + s) Z_a + Σ_b QY_ab f_b(s_b) = 0
  r2_a = Z_a − z̃_a − Σ_b QZ_ab g_b = 0,  g_b = −f_b [κ (Z_b + z_b^{n−1}) + s + μ((Z_b − z_b^{n−1})/2Δt)]

Here QY and QZ are the per-polarisation compliances, with the thread eliminated and tension modulation included; z̃ is the free z prediction.
- The Jacobian is assembled on the active points; inactive candidate rows are explicit, with a unit diagonal and no force of their own, as in the old solver.
- Backtracking on ‖r‖² guards the kink of f and the inflection of arctan.
- Candidate points are chosen from the predicted gap h + κ z̃² + s z̃ − ỹ. Any non-candidate point that ends up penetrating enlarges the set and the step is re-solved.

**Static state.**
- `static_init` minimises the same convex potential, now with η depending on z (`_static_equilibrium_groove`). It is convex for κ ≥ 0, because η is convex in (y, z) and V is convex and non-decreasing.
- At rest the discrete-gradient forces are k_b η^α and −k_b η^α (2κZ + s), exactly this gradient, so the result is the scheme's fixed point.
- With a grazing crest, or with z = 0 at rest, the state is unchanged (test D).

**Scale.**
- The kharaj contact point is the crest graze at 17.9 mm from the termination, 14 mm in front of the thread.
- During the pull, the sideways excursion there is 91 µm (2.7 mm pad displacement). So κz² = 0.08 / 0.24 / 8.3 µm for κ = 10 / 30 / 1000 /m.
- κ = 10–30 /m is the guidance range: a worn groove a few tens of µm deep across 1–3 mm.
- κ = 1e3 /m is a groove of 0.5 mm radius, i.e. a string seated in its own worn groove. That is outside the guidance; it is run as an upper bound.
- The offset groove (s = 0.06) adds s·z ≈ 5 µm, more than any plausible κz²: a string resting on the groove wall couples z into the contact linearly.

## 3. Polarisation-dependent body drain

**Model.** σ_y,i = σ_str,i + (T/L) Re Y_y(f_i) and σ_z,i = σ_str,i + (T/L) Re Y_z(f_i).
- σ_str is the Valette string loss (air, visco-elastic, internal), with Yb = 0.
- Re Y_y adds the modal body mobility to the old smooth drain; Re Y_z is the smooth in-plane drain with Yb_z.
- f_i are the axis-1 frequencies, as before.

**Update (non-proportional damping across the rotated nut pair).** Per mode pair x = (q_i, p_i):

  m [δ_tt x + 2 S* δ_t· x + K* x^n] = F^n,
  S* = diag(σ*_y, σ*_z), σ* = tanh(σΔt)/Δt,  K* = R(θ) diag(k*_1, k*_2) R(θ)ᵀ,

i.e. x^{n+1} = (I + S*Δt)⁻¹ [(2I − Δt² K*) x^n − (I − S*Δt) x^{n−1} + Δt² F^n/m].
- k*_k is the exact-pole ω*² of axis k, evaluated with the axis-projected damping σ_1 = c²σ_y + s²σ_z and σ_2 = s²σ_y + c²σ_z.
- For σ_y = σ_z this is algebraically the old exact-pole update (tests C5, E1).
- I + S*Δt is diagonal, so a y force still reaches only q^{n+1} and a z force only p^{n+1}. The thread and contact eliminations keep their structure, with per-polarisation compliances c_y, c_z.

**Energy.** Multiply by δ_t· xᵀ:

  δ_t+ H_lin = −2m δ_t·xᵀ S* δ_t·x + δ_t·xᵀ F,  H_lin^{n+1/2} = (m/2)[|δ_t+ x^n|² + x^{n+1}ᵀ K* x^n].

- This needs K* symmetric (it is: R diag Rᵀ) and S* ≥ 0 (diagonal, σ ≥ 0). So the damping term is a non-negative loss.
- H_lin = (m/2)[δ_t+xᵀ(I − Δt²K*/4)δ_t+x + μ_t+xᵀ K* μ_t+x] ≥ 0, because k*_k Δt² = 2 − 2cos(ω_d Δt)/cosh(σΔt) < 4. So the scheme is stable.
- The poles are not exact when S* and K* do not commute. Test E bounds the error: ≤ 3.6e-7 in decay rate and 3e-9 in frequency over all 141 pairs.

**Mobility values (assumptions).**
- Yb_y = 0.1 s/kg at T(0,1) is a modest peak. The bare-plate estimate for a centre-driven 4 mm, 40 cm toon plate is Q/(Mω) ≈ 0.4–0.6 s/kg.
- Yb_z = 1e-3 s/kg (in-plane, −14 dB re the old smooth Yb = 5e-3) is a guess. No mobility was measured.

**T60 per partial and polarisation** (test E3, continuous linear system with nut 45° / 7 c and thread kcz = 2.5 kc). The drain T60 columns are 6.91/σ; the eigenmode columns show each coupled mode's T60 and, in brackets, its y-share.

| n | f (Hz) | no drain: σ_y = σ_z | drain gain 0.1: σ_y / σ_z | drain gain 0.1: eigenmodes | drain gain 0.3: σ_y | drain gain 0.3: eigenmodes |
|---|---|---|---|---|---|---|
| 1 | 69.2 | 41.4 | 6.5 / 58.8 | 9.8 (0.63), **14.7** (0.37) | 2.4 | 2.7 (0.89), **16.2** (0.11) |
| 2 | 138.3 | 21.1 | 12.4 / 34.5 | 16.4, 20.2 | 6.8 | 9.8, 13.3 |
| 3 | 207.6 | 15.3 | 7.5 / 25.2 | 10.2, 13.0 | 3.7 | 5.5, 7.8 |
| 4 | 277.0 | 12.6 | 11.0 / 20.1 | 13.0, 14.9 | 8.8 | 11.1, 13.2 |
| 5 | 346.5 | 10.9 | 4.4 / 16.7 | 6.1, 7.9 | 2.0 | 3.0, 4.3 |
| 6 | 416.1 | 9.7 | 6.9 / 14.3 | 8.3, 9.7 | 4.3 | 5.8, 7.4 |
| 8 | 556.1 | 7.9 | 2.9 / 10.8 | 4.0, 5.1 | 1.3 | 1.9, 2.8 |
| 10 | 697.2 | 6.6 | 2.1 / 8.4 | 2.9, 3.8 | 0.9 | 1.4, 2.0 |
| 12 | 839.6 | 5.5 | 2.6 / 6.6 | 3.2, 4.0 | 1.3 | 1.8, 2.5 |

The slow mode of partial 1 is the point: more y-drain decouples the pair, and the z-like mode then keeps ringing (34 s at Yb_y = 1). With body.py's gains, the drain is strongest at partials 5, 8, 10 and 12, next to the cluster of tabli modes at 320–860 Hz.

## 4. Pitch design

**References** (sequential partial tracking, 0.3–3 s; all harmonic, |B| < 2e-6):

| reference | f0 (Hz) |
|---|---|
| own kharaj | 69.327 |
| iTanpura kharaj | 69.344 |
| own Pa | 103.962 |
| iTanpura Pa | 103.873 |
| own Sa | 138.541 |
| iTanpura Sa | 138.530 |

**Before:** `preset2d('kharaj', f1=68.95)` raises "no thread position near x_c0 = 4.00 mm gives crest gap 0 … (psi = 5.7911 deg)".

**Now:** `preset2d('kharaj', f1=68.95, design_f1=True)` re-runs `design_instrument` at the tension of 68.95 Hz.

| string, f1 | designed ψ | thread / crest gap | contact-off tuning |
|---|---|---|---|
| kharaj 68.95 Hz | 5.8086° | 4.000 mm, 0 | 0.0000 c |
| kharaj 68.75 Hz | 5.8164° | 4.000 mm, 0 | 0.0000 c |
| Pa 103.66 Hz | 4.5416° | 4.000 mm, 0 | 0.0000 c |
| Sa 138.14 Hz | 4.5113° | 4.000 mm, 0 | 0.0000 c |

The design is cached and takes 20–35 s.

**Sounding pitch.** For the "groove z0 + drain 0.1" kharaj, one correction step gives f1 = 69.043 Hz (ψ 5.8049°). It sounds at **69.326 Hz** against 69.327 Hz in your recording. The sharpening is +6.8 to +7.1 cents with this physics, and +9.4 cents for the FIT2D baseline. A real jawari-on tuning therefore has to be done per configuration.

## 5. Validation (`python validate2d_round3.py A B C D E F G`; results in `out/round3/results.txt/.json`)

| test | result | pass |
|---|---|---|
| **A. Defaults bit-identical** | Working tree vs `git archive HEAD`: each set is computed in its own process with the same BLAS threading. 935 arrays, 0 different. They cover 13 default-path runs (1-D and 2-D: default, energy, lossless, h0 > 0, pol2 off, 45°/5 c, soft z thread, friction off, the kharaj presets, the FIT2D best, `render2d`, `radiate`) and the preset dicts of kharaj, pa, sa and kharaj_steel (1-D) and kharaj, pa and sa (2-D). The dump saved **before any edit** also matches all 935 arrays. tanpura1d.py, body.py and statics.py are unchanged (SHA-256). The only new entries are the new DEFAULT keys, all off. **Caveat:** two dumps differ at round-off (30 arrays of the 2-D kharaj render) if one is made with single-threaded BLAS and the other with multi-threaded BLAS, through the LU in `static_compliance`. That is an environment effect, not a code change: HEAD vs HEAD differs the same way. The existing suites also pass: `validate2d.py` A–E, `validate2d_parity.py` A–F (including its hashes against the original code) and `validate_statics.py` 8/8. | yes |
| **B. Energy** (FIT2D best kharaj: TM, nut 45° / 7 c, kcz 2.5×) | **B1, lossless** (r_c = 0, friction off) with κ = 30, 1e3, offset z0 = 1 mm, crown −30: H drift 1.6–2.7e-11, balance 1.2–2.7e-11, per-step round-off 1.9e-13. **B2, friction** 2.84 / 2840 s/m with κ = 1e3, z0 0.1 mm: balance 1.2 / 1.6e-11, friction removes 2.1 / 15 %, friction power < 0 in 0 steps, largest H rise 1.6e-13 H_max (round-off). **B3, Valette + drain** (gain 0.3; axisym 0.3) + groove + friction: balance 0.8e-11; H never rises (largest dH −4e-9 H_max). **B4, pressed crest** (+0.1 mm, static contact) with static_init, offset groove, drain and friction: balance 1.3e-11. 0 Newton or friction failures. | yes |
| **C. Kernel limits** | **C1.** `_run3` without new physics vs `_run`: all 15 arrays bit-identical (C3 default and kharaj fit). **C2.** Joint Newton with κ = 0 vs the old block-triangular solver: 5.6e-12 (F_y) and 4.7e-13 (F_z) up to the release. Separation reaches 1e-3 at 0.213 s; a twin of the **old** solver with tolerance 1e-21 separates at 0.213 s too, so this is chaos, not a solver difference. **C3.** κ = 1e-6 /m: 2.3e-10. κ = 1e-8: 4.6e-12, 51× less, i.e. a linear, physical effect of κ down to round-off. **C5.** Per-polarisation path with σ_y = σ_z: F_z 4.7e-12, F_y 2.5e-9. F_y of a sideways pluck is the y–z beat, which amplifies the ~4e-11 round-off of the low-mode ω*² by ω/Δω ≈ 250. | yes |
| **D. static_init with the groove** | **Grazing kharaj:** the rest state is exactly 0 with the groove. **Pressed crest:** 3.922 mN at 1 point; the rest z is 1.54 nm from the nut anisotropy, so a centred groove changes q by 6.3e-9 (κz² = 2.4e-15 m, 1.2e-5 of the 0.196 nm static penetration). **Isotropic nut:** rest z = 0 exactly and q identical to 1.8e-16. **Offset groove** (z0 0.3 mm, pressed): the tilted wall pushes sideways with 1.76 mN, giving 182 nm of static z. **No pluck, 0.3 s** with drain, friction, TM and nut: \|F_y − F_y(0)\| ≤ 4.6e-13 N and \|F_z − F_z(0)\| ≤ 1.8e-14 N. | yes |
| **E. Damping vs independent references** | **E1.** Discrete poles vs exact continuous eigenvalues of all 141 pairs (σ_y up to 229 /s; gain 0.3 / 1.0, axisym 0.3): decay rates ≤ 3.6e-7 relative, frequencies ≤ 3.0e-9. Proportional limit: 9.8e-10 / 7.8e-11. **E2.** The kernel's free decay of one mode pair started 1 mm sideways vs `scipy solve_ivp` DOP853 (rtol 1e-12), 2 s: 5.9e-8 (69 Hz, 45°, 7 c, σ_y = 9.1, σ_z = 0.12 /s) and 3.0e-9 (30°, 40 c) of the start amplitude. **E3.** T60 table (§3). | yes |
| **F. Pitch design** | See §4. `design_f1=None` leaves the preset unchanged. | yes |
| **G. CPU** (single thread, 2 s, load ≈ 1) | Original kernel 3.45 s/s; `_run3` joint κ = 0 3.68; groove 30 3.87; groove 1e3 4.33; drain 3.27; groove 1e3 + drain 3.63 s/s. The joint Newton averages 4.5–4.8 iterations per contact step, against 4.0 for the old y-Newton plus its friction iterations. Its backtracking halves the step about once per step, a cheap O(n²) residual. | – |

## 6. Quick demo (`python validate2d_round3.py H --jobs 2`, 262 s of string CPU)

**Set-up.**
- **String:** kharaj, 5 s, the FIT2D best own-kharaj parameters (pluck 90°, nut 45° / 7 c, kcz 2.5×, 2.7 mm, f1 69.127 Hz, gap 0, L/3).
- **Output:** the **physical chain** `body.apply_body(F_y + 0.1 F_z, 'tanpura_male', hp_hz=150, hp_order=2)`, trimmed at the release − 5 ms.
- **Features** (on both renders and recordings):
  - quickfeat with B = 0 (`qf_features`, identical to quickfeat's numbers on the recording);
  - band levels re total (partials 1–3, 4–10, 16–31, 31–60) at 0.09 / 1 / 3 s;
  - reference.py's attack HF burst (> 4 kHz, first 20 ms vs 20–200 ms);
  - the second bloom, i.e. the largest re-rise of the level within 2–3.8 s;
  - the decay slope over 1.5–3.5 s.

**Spectral and level features.**

| signal | rise dB / t_max s | H4-10 share 0.05/1/3 s | H31-60 dB 1/3 s | bands 1-3/4-10/16-31/31-60 at 0.09 s | at 1 s | at 3 s | HF burst dB | 2nd bloom dB | decay dB/s |
|---|---|---|---|---|---|---|---|---|---|
| **own recording** | **13.9 / 1.01** | 0.57/0.92/0.69 | −17.4/−14.4 | −6.3/−2.4/−8.2/−16.5 | −15.5/−0.4/−15.0/−17.4 | −6.8/−1.6/−12.2/−14.4 | **−1.4** | **1.1** | −3.8 |
| iTanpura recording | 17.7 / 1.27 | 0.26/0.83/0.89 | −12.8/−12.2 | −1.8/−5.9/−19.9/−13.7 | −20.0/−0.8/−10.3/−12.8 | −26.1/−0.5/−14.2/−12.2 | +1.6 | 0.8 | −6.5 |
| **baseline (FIT2D best)** | 10.9 / 0.93 | 0.28/0.40/0.23 | −30.3/−27.8 | −1.5/−5.5/−27.7/−36.7 | −2.3/−3.9/−24.8/−30.3 | −1.4/−6.3/−20.5/−27.8 | −24.5 | 4.6 | −3.4 |
| groove κ 10 | 10.9 / 0.93 | 0.28/0.40/0.23 | −30.3/−27.7 | −1.5/−5.5/−27.7/−37.2 | −2.3/−3.9/−24.9/−30.3 | −1.4/−6.4/−20.5/−27.7 | −21.0 | 4.6 | −3.4 |
| groove κ 30 | 10.9 / 0.93 | 0.29/0.41/0.23 | −30.4/−27.7 | −1.5/−5.4/−27.4/−38.0 | −2.3/−3.9/−25.1/−30.4 | −1.4/−6.4/−20.5/−27.7 | −17.6 | 4.6 | −3.4 |
| groove κ 1e3 (outside guidance) | 9.8 / 0.83 | 0.30/0.46/0.22 | −31.3/−34.2 | −1.6/−5.3/−28.0/−38.2 | −2.8/−3.3/−28.0/−31.3 | −1.1/−6.6/−24.2/−34.2 | −5.8 | 4.2 | −2.0 |
| **offset groove κ 30, z0 1 mm** | 10.6 / 0.85 | 0.33/0.34/0.24 | −29.7/−30.2 | −1.8/−4.8/−26.8/−36.9 | −1.9/−4.6/−28.9/−29.7 | −1.3/−6.3/−18.4/−30.2 | **−4.0** | 3.7 | −4.5 |
| drain gain 0.03 | 10.1 / 0.93 | 0.29/0.36/0.13 | −29.8/−34.9 | −1.5/−5.4/−27.7/−36.8 | −2.0/−4.4/−25.5/−29.8 | −0.6/−8.9/−26.5/−34.9 | −24.4 | 3.8 | −2.7 |
| **drain gain 0.1** | 8.0 / 0.91 | 0.28/0.30/0.25 | −30.5/−31.3 | −1.5/−5.5/−27.9/−36.4 | −1.6/−5.2/−26.9/−30.5 | −1.3/−6.0/−24.0/−31.3 | −24.8 | **0.9** | −2.5 |
| drain axisym 0.3 | 5.2 / 0.61 | 0.28/0.29/0.72 | −31.9/−32.8 | −1.5/−5.6/−28.1/−35.5 | −1.5/−5.4/−27.9/−31.9 | −5.6/−1.4/−26.5/−32.8 | −24.9 | 0.0 | −2.5 |
| offset groove + drain 0.03 | 10.0 / 0.83 | 0.33/0.32/0.32 | −30.4/−27.4 | −1.8/−4.8/−26.7/−36.5 | −1.7/−5.0/−29.7/−30.4 | −1.8/−5.0/−19.8/−27.4 | −4.0 | 3.2 | −3.7 |
| **offset groove + drain 0.1** | 8.4 / 0.83 | 0.33/0.28/0.58 | −32.2/−30.8 | −1.8/−4.8/−26.6/−36.2 | −1.5/−5.6/−31.1/−32.2 | −4.0/−2.4/−21.5/−30.8 | **−4.4** | **0.2** | −3.0 |
| offset groove + drain 0.1, split 10 c | 7.6 / 0.73 | 0.34/0.50/0.50 | −28.9/−34.3 | −1.9/−4.6/−26.9/−39.0 | −3.1/−3.0/−23.6/−28.9 | −3.2/−3.0/−24.2/−34.3 | −4.8 | 0.9 | −2.2 |
| κ 1e3 + drain axisym 0.3 | 4.0 / 0.31 | 0.29/0.43/0.68 | −30.5/−29.0 | −1.5/−5.4/−26.8/−37.6 | −2.5/−3.7/−23.4/−30.5 | −5.1/−1.7/−24.6/−29.0 | −8.4 | 0.0 | −2.9 |
| offset groove + drain 0.1, **tuned** (f1 69.043, `design_f1`) | 8.3 / 0.81 | 0.33/0.28/0.60 | −32.2/−30.7 | −1.8/−4.8/−26.8/−36.3 | −1.5/−5.5/−31.4/−32.2 | −4.1/−2.3/−19.7/−30.7 | −4.4 | 0.2 | −3.1 |

**Distances, string energy and pitch.**
- **D own:** FIT2D distance to your recording, with the physical chain fixed or with the best chain allowed (HP ≤ 150 Hz, order ≤ 2, gz and body-mode gain fitted). Bloom part in brackets.
- **FIT2D chain:** 400 Hz / 2nd order, −6 dB.
- **E 1-3:** string energy share of partials 1–3 at 0.09 / 1 / 1.5 / 3 s.

| case | D own, physical chain | D own, best chain HP ≤ 150 | D own, FIT2D chain | D iTanpura, best HP ≤ 150 | E 1-3 | sounds at (Hz, cents re own) |
|---|---|---|---|---|---|---|
| baseline | 223 (147) | 155 | **49** | 261 | 0.96/0.92/0.91/0.96 | 69.503 (+4.4) |
| groove κ 10 / 30 | 223 / 223 | 155 / 156 | 49 / 50 | 261 / 261 | unchanged | +4.4 |
| groove κ 1e3 | 232 | 161 | 61 | 277 | 0.96/0.89/0.89/0.94 | +3.7 |
| offset groove | **222** | 163 | 72 | 298 | 0.96/0.93/0.89/0.95 | +3.5 |
| drain gain 0.03 / 0.1 | 234 / 275 | 164 / 179 | 54 / 77 | 286 / 343 | 0.94 / 0.96 at 1 s | +3.9 / +2.0 |
| drain axisym 0.3 | 358 | 271 | 210 | 410 | 0.97 at 1 s | +0.8 |
| offset groove + drain 0.03 / 0.1 | 225 / 231 | **154** / 160 | 75 / 120 | 310 / 337 | 0.94 / 0.96 at 1 s | +3.2 / +2.1 |
| offset groove + drain 0.1, tuned | 232 | 158 | 118 | 339 | 0.95 at 1 s | **69.326 (−0.0)** |

Figure: `out/round3/png/round3_demo.png` shows the level curves and the band shares at 1 s.

Audio: `out/round3/wav/<case>.wav` holds every case through the physical chain, peak-normalised. `own_kharaj_reference_loudness_matched.wav` is your recording loudness-matched to the baseline; it is a local copy, git-ignored.

**Reading.**

1. **Groove at plausible depth (κ 10–30 /m): an earlier click, nothing else.**
   - The sideways stroke now touches the bridge from the start of the pull. The HF burst rises by 3.5 / 7 dB.
   - Bloom, band balance, partials 16–31 and late decay are unchanged to ≤ 0.5 dB, because κz² ≤ 0.24 µm is small next to the y motion the nut coupling delivers later.
   - **So the stroke "buzzes immediately" only as a faint click.**
2. **The string resting on the groove wall (offset groove) is the effective version.**
   - Its linear term s·z (~5 µm) gives a click within 3–6 dB of the recordings (−4.0 vs −1.4 / +1.6 dB) without hurting the bloom (10.6 vs 10.9 dB).
   - It lowers the second bloom (3.7 dB) and raises partials 16–31 late (−18.4 vs −20.5 dB at 3 s; recording −12.2).
   - It does not create the recordings' bright 1–2 kHz attack: partials 16–31 at 0.09 s stay at −27 dB, against −8 dB in your recording.
   - A conformal worn groove (κ 1e3) acts similarly on the click (−5.8 dB), drains partials 1–3 slightly (0.92 → 0.89) and weakens the high partials.
3. **The y-drain removes the second bloom but also part of the first.**
   - With Yb_y ≥ 0.1, energy returning to y after the first exchange is absorbed, so nothing re-rises: 0.2–0.9 dB against 1.1 / 0.8 dB in the recordings.
   - With body.py's mode gains, the same drain hits partials 5, 8, 10 and 12 (T60_y 2–4 s), which the jawari is pumping during the bloom. The rise falls from 10.9 to 8.0 dB; with `axisym` weights at 0.3 it falls to 5.2 dB.
   - A stronger coupling (10 c) does not restore it (7.6 dB).
   - The y-drain also lowers the contact sharpening: +4.4 → +2.0 cents, since less energy reaches the contact.
4. **Partials 1–3 are not drained by any of this** (§0). The radiated balance through a physical chain is still dominated by partials 1–3: −1.5 to −3 dB re total at 1 s, against −15.5 (own) and −20.0 dB (iTanpura). The output high-pass is still doing that job.
5. **Overall match.**
   - With a physical chain, no case beats the baseline by more than 1 unit of D (222 vs 223), and the drains make it worse.
   - The FIT2D optimum (D 49 with the 400 Hz chain) was found without these mechanisms. Adding them one at a time at fixed coupling moves away from it (54–210).
   - A refit with the new parameters free (κ_g, z0, Yb_y, Yb_z, nut coupling, pluck angle) is the obvious next step. It was not done here (15 min CPU budget).

## 7. Assumptions and limitations

- **Mobilities are not measured.** The modal shape uses body.py's assumed modes and gains as relative peak mobilities; Yb_y 0.03–0.3 s/kg and Yb_z 1e-3 s/kg are guesses. A measured bridge admittance, in two directions, would settle §3. The body output filter is still one-way: the drained power is not fed back into the radiated signal beyond the bridge force F_y that already drives the filter.
- **No cross-mobility.** The bridge probably couples y and z through rocking (Y_yz). Here the only linear y–z coupling is still the nut rotation plus the anisotropic thread.
- **The groove is a parabola** (small-slope). At κ = 1e3 /m and 90 µm excursion the wall slope reaches 0.18; above that the profile shape would matter. Friction acts along z, not along the tilted tangent (§2). The groove applies at every bridge point, uniformly along x.
- **The thread is still linear and bilateral in z.**
- **Chaos.** Individual trajectories differ between cases after ~0.1–0.2 s (test C). Only smoothed features are meaningful; differences under ~1 dB or ~5 D units are within the landscape's ruggedness (FIT2D.md).
- **One string, one recording.** The demo is the kharaj only. Pa and Sa were not rendered with the new physics; only their pitch design was checked.
- **Late decay.** The recordings fade out after about 4 s, so decay slopes are taken over 1.5–3.5 s only.

## 8. Files and commands

| file | content |
|---|---|
| `tanpura2d.py` | Groove options, per-polarisation drain (`_round3_setup`, `re_y_smooth`, `body_mobility_shape`), `_static_equilibrium_groove`, the kernel `_run3` with the joint Newton, the dispatch `uses_round3` / `_simulate3`, `simulate(..., init=)`. `_run` is unchanged. |
| `presets.py` | `design_f1` (`designed_psi`, `DESIGN_KEYS`); defaults unchanged. |
| `presets2d.py` | Docstring only (the new keys pass through). |
| `validate2d_round3.py` | Tests A–G and the demo H. |
| `out/round3/` | `results.txt/.json`, `runs/*.npz` (demo renders with per-mode energies), `png/round3_demo.png`, `wav/`. |

```bash
cd jawari
python validate2d_round3.py A B C D E F G     # validation (~9 min, A alone ~4-6 min)
python validate2d_round3.py H --jobs 2        # demo (~4.5 min CPU; cached)
python validate2d_round3.py A --ref <npz>     # also compare with a --bitdump made before an edit (same BLAS threading)
```

```python
import tanpura2d as T2, presets2d as P2
p = P2.preset2d("kharaj", f1=69.043, design_f1=True, pluck_deg=90.0, theta_nut=45.0, detune_cents=7.0,
                pluck_disp=2.7e-3, groove_kappa=30.0, groove_z0=1e-3, Yb_y=0.1, Yb_z=1e-3, dur=5.0)
p["kcz"] = 2.5 * p["kc"]
sig = P2.render2d(p)          # runs _run3; sounds at 69.33 Hz
```
