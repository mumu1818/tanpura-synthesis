# 1-D model (tanpura1d.py): a validated 1-D modal tanpura string with jawari

> Detailed notes written while building this model. Run all scripts from inside `jawari/`; outputs (WAV, PNG, result files) go to `jawari/out/`. The evaluation that motivates this work is in [`../docs/EVALUATION.md`](../docs/EVALUATION.md).


This is a working physical model of one tanpura string. It moves in a single polarisation, normal to the bridge. The string meets a curved bridge with a thread (jiva) placed on it. **The model produces jawari:**

- The buzz builds up from −35 dB to −5 dB within 0.5 s.
- It holds between −5 and −20 dB for about 8 s.
- hf_frac reaches 0.55 at 1 s.
- A spectral "formant" sweeps down from about 1.8 kHz to about 0.85 kHz in roughly 1 s. It then sits near 720 Hz until about 4 s and near 460 Hz around 4.5–5 s.

The linear string (no bridge) gives about −120 dB buzz and hf_frac = 0.000 with the same code. The model is energy-conserving and checked against analytic results (see Validation).

The model follows the structure of van Walstijn, Bridges and Mehes (DAFx-16) and Bridges and van Walstijn (DAFx-17). The bridge contact is **distributed**, solved with a Newton–Raphson discrete-gradient method. The string is integrated with exact modal poles.

## Files

| file | contents |
|---|---|
| `tanpura1d.py` | The model: `build()` sets up the modal and contact matrices. `_run()` is one `@njit` function containing the whole time loop. `simulate()` is the Python wrapper. |
| `validate1d.py` | Validation tests A–E. Writes `validation_results.txt` and `validation_results.json`. Takes about 100 s. |
| `render1d.py` | Renders the cases to WAV + PNG + metrics. Appends to `render_results.txt`. |
| `diag_bands.py` | Diagnostic for the fs/4 contact-chatter artefact (see Findings). |
| `common.py` | Shared helpers: DC block, resampling, and the shared `metrics.py` module. |
| `out/1d/*.wav` | Bridge force, 44.1 kHz, 24-bit, peak-normalised to 0.9. |
| `out/1d/*.png` | `metrics.spectrogram_png` output for each WAV. |

## How to run

```

cd jawari
python validate1d.py                       # all tests (≈100 s)
python render1d.py                         # all 10 s cases (≈1 min)
python render1d.py nominal xc8             # selected cases
python render1d.py best:nominal            # 20 s render of a case
python -c "from tanpura1d import simulate; p,s,r=simulate(dict(xc=0.006, dur=5)); print(r['stats'])"
```

`simulate(overrides, energy=False)` returns `(params, built, results)`. `results` holds:

- `Fbridge`, `Fnut`, `Fb`, `Fc`, `F0`: forces (N), at the internal sample rate;
- `y`: displacement at `x_out`;
- `H`, `D`, `W`: discrete energy, cumulative dissipation and cumulative work input, filled when `energy=True`;
- `pen`: penetration;
- `stats`: max penetration, contact fraction, Newton iterations and failures, max thread force, edge penetration.

Use `common.to_audio(y, fs)` to DC-block the output and decimate it to 44.1 kHz.

## Model

### Geometry (rest state)

- The string lies along x ∈ [0, L], with L = 1 m.
- Both ends are simply supported. x = 0 is the termination at the rear of the bridge; x = L is the nut.
- y is vertical, positive away from the bridge. At rest the string is straight at y = 0.

The **thread** at x_c = 5 mm holds the string up:

- It is a bilateral linear spring-damper, k_c = 1.2e5 N/m and r_c = 0.5 kg/s.
- It acts through a sin² window of width w_c = 1 mm.

The **bridge** is a rigid curved barrier:

- Surface: h_b(x) = h0 − 3·(x − x_b)², with x in metres. The crest is at x_b = 10 mm and h0 = 0.
- At rest the crest **grazes** the string with zero force.
- The bridge surface lies **below** the string everywhere else.
- At the thread the gap ("thread height") is 3(x_b − x_c)²: 75 µm for x_c = 5 mm, 108 µm for 4 mm, 37 µm for 6.5 mm and 12 µm for 8 mm.

```
 y^  x=0      x_c (thread)    x_b (crest)                               x=L (nut)
  |   o---------[S]-------------.------------------- ... -------------------o   y = 0
  |              75 um gap   _/ \_   h_b(x) = -3 (x - x_b)^2  (bridge below the string)
```

**Why the thread is bilateral.** In a real instrument the string is pressed onto the thread by the break angle over the bridge. The largest dynamic thread force in any run is 0.85 N, even with a pluck 4× the nominal one. That equals the static preload from a break angle of only 1.5°. Real break angles are several degrees, so the string never lifts off the thread, and the linearised bilateral spring is accurate. (The 5° figure used in the validation check is an assumption, not a measured value.)

### String (DAFx-16 Table 1, C3 steel string)

| quantity | value |
|---|---|
| density ρ | 7850 kg/m³ |
| cross-section A | 6.16e-8 m² |
| Young's modulus E | 2e11 Pa |
| second moment I | 3.02e-16 m⁴ |
| μ = ρA | 4.836e-4 kg/m |
| EI | 6.04e-5 N·m² |
| tension T | 33.10 N (chosen so f1 = 130.81 Hz including stiffness) |
| inharmonicity B | 1.8e-5 |

Equation of motion:

μ y_tt = T y_xx − EI y_xxxx − (loss) + f_bridge(x,t) + ψ_c(x)·F_c(t) + ψ_e(x)·F_e(t)

**Modal expansion.** y = Σ q_i sin(β_i x), with β_i = iπ/L and modal mass m = μL/2. All modes below 20 kHz are kept: M = 133, the same count DAFx-16 reports. Each mode obeys:

m (q̈_i + 2σ_i q̇_i + ω_i² q_i) = Σ_j Δx sin(β_i x_j) f_j + g_{c,i} F_c + g_{e,i} F_e,

with ω_i² = (T β_i² + EI β_i⁴)/μ.

**Losses.** Woodhouse form:

- η_i = [T(η_F + η_A/ω_i) + EI η_B β_i²] / (T + EI β_i²)
- σ_i = ω_i η_i / 2
- Values: η_F = 5e-5, η_A = 1.2, η_B = 1e-2.

These are engineering defaults, not values fitted to a real string. They give T60 ≈ 11 s at f1, 6 s at 2 kHz and 0.4 s at 8 kHz.

### Bridge contact

The contact law is a distributed one-sided power law:

f(x) = k_b ⌊h_b(x) − y(x)⌋^α (force per unit length), with k_b = 1e11 N/m² and α = 1, as in DAFx-17.

- The bridge is sampled at N_b points spaced Δx = 0.2 mm, covering x_c … x_b + 6 mm (41–61 points).
- At most 2–4 points are in contact at any one time.
- The contact never reaches the ends of the sampled region (validation E2).

### Time stepping

The step is 1/fs, with fs = 176.4 kHz internally (4 × 44.1 kHz). The output is decimated to 44.1 kHz.

**Linear part: exact modal poles.**

- q̃_i = a_i q_i^n − b_i q_i^{n−1}, with a_i = 2e^{−σΔt} cos(ω_d Δt) and b_i = e^{−2σΔt}.
- Forces enter as q_i^{n+1} = q̃_i + c_i F_i^n, with c_i = Δt²(1 + b_i)/(2m).
- This is the energy-consistent explicit-stiffness form m[δ_tt q + 2σ* δ_t· q + ω*² q^n] = F^n. Because ω*Δt < 2 for every mode, it is unconditionally stable and has exact modal frequencies and decay rates.

**Contact force: discrete gradient.**

- f_j^n = [V(η_j^{n+1}) − V(η_j^{n−1})] / (η_j^{n+1} − η_j^{n−1}), with V(η) = k_b/(α+1) ⌊η⌋^{α+1}.
- For α = 1 this has closed forms with no 0/0 case.

**Thread force: trapezoidal.**

- F_c = −k_c (u_c^{n+1} + u_c^{n−1})/2 − r_c (u_c^{n+1} − u_c^{n−1})/(2Δt).
- The thread is linear, so it is eliminated analytically.
- The effective contact compliance becomes Q' = Δx Φ (C⁻¹ + κ g_c g_cᵀ)⁻¹ Φᵀ, which is symmetric positive semi-definite.

**Solve per sample.**

- Unknowns: s_j = η_j^{n+1} − η_j^{n−1}. Residual: r(s) = s − b + Q' f(s).
- Newton step: J = I + Q' diag(f'(s)), solved only on the **active** points (a reduced dense system with pivoting).
- Initial guess: the exact solution of the decoupled single-point problem (the DAFx-16 four-case root).
- Tolerance: |r| < 1e-18 m.
- At 176.4 kHz the renders average 1.2–2.5 iterations per contact sample (about 6 in the lossless test), with a maximum of 10 and no failures.

**Energy.** The discrete energy is:

H^{n+½} = Σ (m/2)[((q^{n+1} − q^n)/Δt)² + ω*² q^{n+1} q^n] + Σ_j Δx ½(V_j^{n+1} + V_j^n) + (k_c/4)(u_c^{n+1}² + u_c^n²)

It satisfies H^{n+½} − H^{n−½} = −Δt·(modal + thread damping) + Δt·(excitation power).

**Excitation** (as in DAFx-16):

- Force density ψ_e(x)·F_e(t), with ψ_e a sin² window at x_e = L/2 of width L.
- F_e rises as sin² over 10 ms, then is released as cos² over 1 ms. Peak 0.2 N.
- The resulting string amplitude is about 1.15 mm at mid-string.

**Output.** The **total vertical force the string exerts on the bridge assembly**:

F_bridge = F_0 − F_c − Σ_j Δx f_j

Here F_0 = Σ q_i (Tβ_i + EIβ_i³) is the termination force. The nut force −Σ q_i cos(iπ)(Tβ_i + EIβ_i³) is also returned. The only post-processing applied:

- a 20 Hz 2nd-order high-pass (sub-audio DC block, because the time-averaged one-sided contact force is non-zero);
- band-limited decimation 176.4 → 44.1 kHz;
- peak normalisation.

There is no body filter, noise or distortion.

**Cost.** 0.55–0.8 s of CPU per second of audio at 176.4 kHz (single-threaded numba), or about 0.3 s per second at 44.1 kHz.

## What was wrong in the repo, and what replaces it here

| repo (measured by the audit) | here |
|---|---|
| Bridge 0.5–1.0 mm below a string pinned at the bridge end. The jiva peak sat 0.26 mm below the string, so the string never touched anything. | The string grazes the bridge crest at rest (gap 0 µm). The thread is a support that **holds** the string; the bridge surface curves away 12–108 µm below it at the thread. The geometry is right at the µm scale. |
| Soft contact: 0.42 mm penetration. | k_b = 1e11 N/m². Maximum penetration is 24–93 nm at the nominal pluck, which is 0.09–0.2 % of the thread height. Results do not change between k_b = 1e10 and 1e12. |
| Force projection missing 1/μ and the quadrature weight. | Modal force Σ_j Δx sin(β_i x_j) f_j and modal mass μL/2, checked by the energy balance to 4e-12. |
| Output was displacement at 0.9 L (low-passed, dull). | Output is the bridge force (DAFx-16), with the nut force also available. |
| 80 modes (< 10 kHz). | 133 modes (< 20 kHz). Tested up to 344 modes (< 80 kHz). |
| Python loop around per-step numba calls. | The whole loop is in one `@njit` function. |
| "Tension modulation" that scaled ω by the penetration. | Removed (it is not Kirchhoff–Carrier). Proper tension modulation is listed under next steps. |
| With and without bridge: about 1 % difference, buzz ≈ −105 dB. | With and without bridge: buzz −120 dB vs −5 dB. |

## Validation (`validation_results.txt`)

| test | result | pass |
|---|---|---|
| A. Contact off: partial frequencies vs analytic stiff string (8 s at 44.1 kHz, FFT peak interpolation) | max error 0.00001 cents for partials 1–20 and 1–100 | yes |
| B. Lossless, bridge contact and thread spring active, 1 s | relative energy drift 4.1e-12 over 174 458 steps. Lossy energy balance residual 4.2e-12. H never increases after the pluck. | yes |
| C. Convergence vs base (176.4 kHz, 133 modes): fs ×2, modes to 40 and 80 kHz, bridge Δx 0.1 and 0.5 mm, k_b ×0.1 and ×10, Newton tolerance 1e-15 | mean buzz (0.3–2.7 s) within 0.9 dB; mean hf_frac within 0.013; band levels 0–15 kHz within 1.9 dB; the 15–20 kHz band (−33 dB re total) changes by up to 4.2 dB when modes are extended to 80 kHz | **no** against my own 4 dB band criterion, by 0.2 dB, in the top band only |
| D. Maximum penetration / thread height, x_c = 4, 5, 6.5, 8 mm | nominal pluck: 0.09–0.20 %. A 4× harder pluck (4.6 mm amplitude): up to 0.84 % with k_b = 1e11, 0.27 % with k_b = 1e12 | yes |
| E1. Thread never lifts off | max \|F_c\| 0.85 N, equivalent to a 1.5° break angle | yes (assumes a real break angle ≥ 5°) |
| E2. Contact stays inside the sampled bridge region | edge penetration 0 | yes |
| E3. Outputs finite | 7 % of the raw 176.4 kHz bridge-force energy lies above 20 kHz (fs/4 chatter, removed by decimation); 1e-6 in the nut force | yes |
| C/E3. 44.1 kHz without oversampling | +2.7 dB at 5–10 kHz and **+6.8 dB at 10–15 kHz** vs the oversampled reference | (finding) |

**Trajectory sensitivity.** The time-averaged metrics converge, but the detailed buzz trajectory does not. At t = 2 s the instantaneous buzz ranges from about −8 to −18 dB across the refinements. The fluctuating fine structure of the jvari is sensitive to tiny changes in the partial structure, as DAFx-17 also notes. Its statistics are robust.

## Renders (10 s each; bridge force; metrics from `metrics.py`)

| case | buzz at 0.05 / 0.5 / 1 / 2 / 4 / 8 s (dB) | hf_frac at 1 s | peak_hz trace (median per 0.5 s) |
|---|---|---|---|
| `linear` (no bridge, no thread) | −117 / −120 / −122 / −128 / −136 / −139 | 0.000 | constant (no formant) |
| `bridge_only` (no thread) | −28 / **−0.8** / −15 / −13 / −22 / −23 | 0.10 (0.66 at 0.5 s, centroid 5.9 kHz) | 9.4 kHz at 0.5 s, then collapses. Harsh and short; level −17 dB at 1 s. |
| `nominal` (x_c = 5 mm) | −35 / −5.1 / −5.4 / −9.2 / −15 / −18 | 0.55 | 1.8 kHz (0.9 s) → 1.39 k (1 s) → 0.93 k → 0.86 k → 0.72 k (3–4 s) → 0.46 k (4.5–5 s) |
| `xc4` (x_c = 4 mm) | −33 / −3.4 / −7.0 / −7.5 / −20 / −19 | 0.51 | 1.39 k (1 s) → 0.85 k (1.5–3.5 s) → 0.59 k (4 s) → 0.46 k (5 s) |
| `xc6p5` (x_c = 6.5 mm) | −39 / −11 / −11 / −15 / −20 / −27 | 0.41 | 1.25 k (1.5 s) → 0.85 k → 0.72 k (2.5–5 s) |
| `xc8` (x_c = 8 mm) | −44 / −27 / −25 / −14 / −18 / −25 | 0.18 | 1.12 k (2 s) → 0.86 k → 0.72 k (3–5.5 s) → 0.59 k → 0.46 k (6.5–7 s); cleanest, slowest sweep |
| `no_stiffness` (EI × 1e-6) | −36 / −13 / −4.5 / −10 / −12 / −12 | 0.52 | fixed at 334 Hz the whole time: a static, unswept buzz |
| `best_nominal_20s` | identical to `nominal` for the first 10 s; buzz −20 to −29 dB from 10 to 20 s (level below −75 dB by then) | | |

(The peak tracker falls back to the 2nd–3rd harmonic, 323–334 Hz, whenever the low harmonics outweigh the formant.)

**Thread-placement sweep.** The 1-D model does respond to the thread position, but mainly in *timing*:

- Moving the thread toward the crest (smaller thread height) delays and weakens the build-up. At 0.5 s the buzz is −3.4 dB at 4 mm, −5 dB at 5 mm, −11 dB at 6.5 mm and −27 dB at 8 mm.
- It also pushes the formant plateau later: at 8 mm the 720 Hz plateau lasts until about 5.5 s.

This agrees qualitatively with the DAFx-16 examples (4 mm "shorter jvari", 8 mm "increased jvari length"). The strong timbral dependence players report is not obviously present. That is consistent with VT-2015's statement that a 1-D model fails to replicate it.

## Findings worth knowing

1. **Jawari needs µm-correct geometry and a force output.** With the crest grazing the string and the thread holding it, every bridge-side configuration buzzes. The repo's millimetre-scale gaps and its displacement output are why it heard nothing.
2. **The contact is effectively rigid.** Penetrations are tens of nm, and k_b anywhere from 1e10 to 1e12 gives the same sound. The repo's soft contact (0.42 mm penetration) was a different physical regime altogether.
3. **Oversample ×4.** The energy-conserving (discrete-gradient / trapezoidal) treatment of a near-rigid contact maps the unresolved contact oscillation to about fs/4. At 44.1 kHz that lands at 11 kHz, inside the audio band; it is measured as +6.8 dB at 10–15 kHz. Running at 176.4 kHz with the same 133 modes moves it to 44 kHz, where decimation removes it. The cost is still only 0.75 s of CPU per second of audio. `diag_bands.py` shows this.
4. **The formant sweep comes from string dispersion.** With EI → 0 the bridge still produces a strong buzz, but the spectral peak never moves. This matches DAFx-16's "no string stiffness: no jvari". The time-varying formant is the inharmonic partials drifting in phase relative to the periodic grazing contact.
5. **The model is amplitude-invariant.** With h0 = 0, α = 1, a linear thread and no tension modulation, it is positively homogeneous: plucks of 0.05, 0.2 and 0.8 N give a mean buzz of −10.5, −10.5 and −10.2 dB. The curved, distributed bridge barely breaks this. Only 2–4 points touch, because 133 modes have a spatial resolution of about 7.5 mm and cannot wrap a string around a bridge profile at mm scale. A crest offset of ±1 µm does create amplitude dependence: soft plucks (0.05 N) then buzz 9–10 dB less than with h0 = 0, while hard plucks (0.8 N) change by at most 1.3 dB. Real amplitude dependence needs tension modulation, a µm-scale offset or wear profile, or much finer spatial resolution near the bridge.
6. **Removing the thread** gives a harsh, bright, short buzz: centroid 5.9 kHz at 0.5 s, and the level falls 17 dB in 1 s. This matches DAFx-16's "harsher".

## Limitations (honest list)

- **Single polarisation.** There is no horizontal motion, no friction on the bridge and no polarisation coupling. VT-2015 argues these are needed for the strong thread-placement dependence.
- **No tension modulation (Kirchhoff–Carrier).** So there is no pitch glide and no amplitude-dependent jvari.
- **Unfitted damping.** The loss parameters, k_c and r_c are defaults. The decay is faster than a real tanpura: −33 dB at 4 s, because the energy pumped into the high partials is dissipated quickly.
- **Idealised pluck.** The excitation is the papers' whole-string force pulse (mostly mode 1), not a finger pluck.
- **No body.** The output is the raw bridge force, with no body or radiation filter, no other strings and no sympathetic coupling.
- **Modal truncation.** 133 modes cannot resolve the string shape on the bridge below about 7.5 mm, and wrapping on the bridge is not represented. The convergence test to 344 modes changed the averaged metrics by less than 1 dB, but at 1–4 mm scale the contact is still effectively near-lumped.
- **Unverified spec values.** Values marked uncertain in the spec (k_c, r_c, w_b, x_b for DAFx-16, the sign of the bridge profile) were taken from the research spec (search-result excerpts of the papers) and not checked against the original PDFs.

## Next steps

1. **Tension modulation.** Kirchhoff–Carrier with the energy-conserving form P̄ = (P^{n+1} + P^{n−1})/2. It adds one scalar unknown to the Newton solve, and the active-point Q' only needs rebuilding over 2–4 points.
2. **Second polarisation.** Add it with arctan friction (μ(v) = 0.39·atan(2.84 v)) and coupling at the termination.
3. **Fit losses.** Fit η_F, η_A, η_B, k_c and r_c to a recorded string decay, and the thread position to recorded jawari spectrograms (nut force and bridge force).
4. **Body and strings.** Add a body IR (impulse-hammer measurement) and four strings coupled through a common bridge admittance.
5. **Finer local resolution.** Try a residual (quasi-static) compliance for the truncated modes at the bridge points. This would let the string wrap the curved bridge without needing MHz-range modes.
