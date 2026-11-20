# 2-D kharaj with a sideways finger pluck (tanpura2d parity with tanpura1d)

> Run every script from inside `jawari/`. Outputs go to `jawari/out/2d_pluck/` (git-ignored). The reference recordings are only **read** from `/home/user/tanpura-app/assets` (or `$TANPURA_REFS`); nothing from them is copied into this repository.

This note covers three things:
- what was added to `tanpura2d.py` so that the user's real strings (statics geometry, tuning, finger pluck, Valette losses) run in the two-polarisation model;
- the new `presets2d.py`;
- the validation (`validate2d_parity.py`) and the demonstration that a **sideways** finger pluck reproduces the "bloom" of the real kharaj.

## 0. Bottom line

* **Parity is complete, and the defaults are untouched.**
  * `tanpura2d.py` now takes `profile`, `static_init`, `T`, `tune='speaking'`, `pluck='finger'` (with direction `pluck_deg`) and `loss='valette'`. All are off by default and change only the set-up; the time loop `_run` is unchanged.
  * The default outputs are **bit-identical** to the original code: 10 configurations, SHA-256 and a bitwise comparison with the committed file, max abs diff 0. `validate2d.py` still passes.
* **The 2-D model reduces exactly to the 1-D kharaj.** `preset2d('kharaj', pol2=False, tm=False, pluck_deg=0)` against tanpura1d on `presets.preset('kharaj')`:
  * every set-up array is identical, including the tuned tension, bit for bit;
  * the waveforms agree to 4e-14 up to the release, then separate like a 1-D "twin" with a tighter Newton tolerance (chaotic contact);
  * the mean buzz is −22.20 vs −22.19 dB.
* **A sideways pluck reproduces the bloom on the real (statics) kharaj model, not just on C3.**
  * With the finger stroke 88–90° from the bridge normal (parallel to the soundboard), the radiated level of F_y starts 15.6–18.2 dB below its peak and peaks at 1.1–1.35 s.
  * The recordings: 13.9–17.7 dB, peak at 1.01–1.27 s.
  * A vertical, 45° or 80° pluck gives 2–7 dB, with the peak at 0.2–0.7 s (only 80°/5 cents peaks late, at 1.5 s, and weakly: 7 dB). The 1-D model is the same: 5.3 dB, 0.73 s.
  * The bloom needs the pluck within about 3° of parallel. It is robust to the unmeasured nut coupling: every tested θ_nut of 30–45° and split of 2–5 cents gives 14–18 dB.
* **The mono mix only blooms if the in-plane force radiates weakly.** With the old assumption gz = 0.3 (mono = F_y + gz F_z), the strong initial in-plane force masks the bloom (0–8 dB). The references need gz ≲ 0.1, which is the new preset default (11.8–14.0 dB for splits ≥ 3.5 cents; 5.3 dB at 1 cent).
* **What does not match yet: the late high-partial shimmer.** After the bloom, partials 31–60 are 9–19 dB weaker than in the recordings, and partials 16–40 are 7–13 dB weaker at 1 s. A sideways pluck feeds the jawari only through the ~10 % of the string energy that sits in y, so the shimmer is weaker than in 1-D. This is the next thing to fit (coupling strength, high-frequency losses, gap); see §6.
* **Correction to the measurement that motivated this work.**
  * The reference kharaj partials are **harmonic**: f0 = 69.33 Hz, every partial up to n = 31 within 0.2 % of n·f0, |B| < 1e-5. They are not "f0 68.56 Hz, B 4.4e-4". That pair comes from numbering the partials wrongly above n ≈ 15.
  * The model renders are harmonic too (B ~ 1e-6): the jawari locks the partials, although the bare string has B = π²EI/(TL²) = 1.6e-4.
  * The recordings are therefore at A440 pitch, not 19 cents flat.
  * Corrected, the references' H31-60 is −12.8/−12.2 dB (iTanpura) and −17.5/−14.4 dB (own) at 1/3 s, instead of −20.2/−15.2 and −24.2/−17.9.
  * Rise and t_max barely change (17.3 → 17.7 and 13.7 → 13.9 dB).

## 1. What was added to `tanpura2d.py`

All options default to off (`None`/`False`). They act in `build()`, `excitation()` and the initial state; `_run` is unchanged.

| option | meaning in 2-D |
|---|---|
| `profile=(x_rel, h_rel)` | Bridge surface relative to the rest line, from `statics.profile_for_dynamics`. Same bridge points as 1-D (`tanpura1d.profile_points`). |
| `static_init=True` | `static_equilibrium()`: the **exact** fixed point of the discrete 2-D scheme. See below. |
| `T=<N>` | Tension override. |
| `tune='speaking'` (+ `tune_ref`) | Tension such that the lowest linear eigenfrequency of the **discrete** scheme with the thread spring (contact off) equals f1. `tanpura1d.tune_tension` does the work, so with pol2 off the tension is tanpura1d's bit for bit. |
| `pluck='finger'`, `xp` (L/3), `wp` 12 mm, `pluck_pull` 40 ms, `pluck_release` 2 ms, `pluck_disp` | The 1-D finger-pad force (raised-cosine pad, sin² pull, cos² release), applied along **(cos φ, sin φ)**, φ = `pluck_deg` from the bridge normal y towards z. φ = 90° is parallel to the soundboard. |
| `loss='valette'` (`eta_air`, `rho_air`, `delta_ve`, `Qte_inv`, `Yb`, `mb`, `fb`) | `tanpura1d.loss_sigma`: the same σ_i for both polarisations, evaluated at the axis-1 frequencies. Common damping keeps the rotated two-step update symmetric and energy-consistent. A per-polarisation damping option was not added: nothing measured motivates it, and it would break the symmetric form. |

**`static_init=True`: the exact static state.**
- It minimises the convex potential ½ xᵀK x + (EA/8L) P² + Σ_j dx k_b/(α+1) [h_j − Φ_j q]₊^(α+1).
  - x = [q (y); p (z)].
  - K holds the scheme's own ω*² rotated by θ_nut, plus the thread springs k_c in y and k_cz in z.
  - P = (L/2) Σ β_i² (q_i² + p_i²).
- Newton with backtracking finds the minimiser.
- At rest:
  - the trapezoidal thread force is −k_c u;
  - the discrete-gradient contact force is k_b η^α;
  - the linearly implicit tension-modulation force is −(EA/4) P β_i² x_i;
  - friction is zero, because nothing slips.
- So this minimiser is exactly the scheme's fixed point.
- A vertical preload makes a small z deflection when the nut is anisotropic: 1.5 nm for the pressed kharaj.
- The older `init_static` / `static_state()` path is kept unchanged for bit-identity. It covers the parabola with h0 > 0 only (α = 1) and neglects the z-thread reaction.

**`pluck_disp`** is the static pad displacement **along the pluck direction**, (g_e⊗e)ᵀK⁻¹(g_e⊗e), with the thread springs on and contact and tension modulation off. This is the 1-D convention. With pol2 off it is literally tanpura1d's Sherman–Morrison formula.

**Tuning: which polarisation?**
- `tune='speaking'` tunes the nut's **axis-1** series, which is y when θ_nut = 0 or pol2 is off. Axis 2 is then `detune_cents` sharper.
- With an isotropic thread (k_cz = k_c, the default), the linear system separates exactly into the two principal axes, so axis 1 is exact.
- `tune_ref='mean'` instead puts the two series ∓detune/2 around f1.
- With a laterally soft thread (k_cz ≪ k_c), the z polarisation loses the thread's stiffening and speaks 6.3 cents flat. The tuning then refers to y only.

## 2. API

```python
import tanpura2d as T2
from presets2d import preset2d, render2d, radiate, to_1d

p = preset2d("kharaj", dur=5.0)          # bronze kharaj, statics geometry, tuned, finger pluck at L/3, 88 deg, Valette
sig = render2d(p)                        # 44.1 kHz: sig["Fy"], ["Fz"], ["Fny"], ["Fnz"], ["mono"] = Fy + gz Fz;
                                         # also sig["cpu"], sig["p"], sig["s"], sig["r"] (raw tanpura2d results)
y = radiate(sig["mono"])                 # body.apply_body(y, 44100, 'tanpura_male', hp_hz=250, hp_order=2)
p = preset2d("kharaj", pluck_deg=90, theta_nut=45, detune_cents=5, gz=0.05, thread_shift=0.05e-3)
p = preset2d("kharaj", pol2=False, tm=False, pluck_deg=0)   # the 1-D limit
p = preset2d("pa"); p = preset2d("sa")   # the other strings (presets.STRING_SPECS) work the same way
p, s, r = T2.simulate(dict(xc=0.005, pluck="finger", pluck_disp=2e-3, pluck_deg=90, loss="valette", Yb=5e-3))
```

**How `preset2d(name, **overrides)` builds the dict:**
- **Statics geometry.** It calls `presets.preset(name, ...)` with the statics keys (`thread_shift`, `f1`, `psi_deg`, …) and every key the two models share. So thread auto-placement, statics and tuning see them exactly as in 1-D.
- **Translation.** It copies every 1-D key and drops only `x_out`, the 1-D displacement output point. Any other 1-D key that tanpura2d does not know raises `ValueError`, so nothing essential is dropped silently.
- **2-D keys.** It then applies the 2-D defaults in `PRESET_2D`, then the caller's 2-D overrides.
- `to_1d(p)` gives back the tanpura1d dict.

| 2-D preset key | value | status |
|---|---|---|
| `pluck_deg` | **88** | ASSUMED. The references need ≳ 87° (§4). |
| `theta_nut`, `detune_cents` | **30°, 3.5 cents** | Unmeasured; a **fit parameter**. Natural sources: different y/z boundary stiffness at the thread/bridge, bridge or soundboard anisotropy. |
| `friction`, `mu_c1`, `mu_c2` | on, 0.39, 2.84 s/m | VT-2015 kinetic law, as in MODEL_2D.md |
| `tm` | on | Kirchhoff–Carrier tension modulation |
| `kcz`, `rcz` | `None` (= k_c, r_c) | isotropic thread |
| `tune_ref` | `'axis1'` | |
| `gz` | **0.1** | ASSUMED (−20 dB): the in-plane force drives the soundboard only through bridge rocking. 0.3 masks the bloom (§4). |

## 3. Validation (`python validate2d_parity.py`, A–F about 3 min, results in `out/2d_pluck/parity_results.txt/.json`)

| test | result | pass |
|---|---|---|
| **A. Defaults bit-identical** | (1) SHA-256 of 15 output arrays for 10 default-path configurations equals the hashes recorded from the original code before any edit. The configurations: default, energy, lossless, h0 > 0 with the old init_static, no bridge, pol2 off, 45°/5 cents, friction off, near-Coulomb, soft lateral thread. (2) Bitwise comparison with `tanpura2d.py` from git commit f4a483a: max abs diff 0, and the stats are identical. `validate2d.py` (tests A–E) still passes. | yes |
| **B. Energy with the new options.** Kharaj 2-D with a pressed thread (+0.1 mm, so the static state has contact: 8.4 mN at 1 point), `static_init`, finger pluck at 88°, nut 30°/3.5 cents, tension modulation on | (1) Lossless, friction off, 1 s: H drift 6.6e-11; balance residual 4.2e-11; per-step round-off 2.1e-13 H_max. (2) Friction on (c2 = 2.84 / 2840 s/m): residual 2.6e-11 / 2.2e-11; friction power ≥ 0 at every step; H never rises by more than 2× the lossless round-off; friction removes 3.7 % / 10.4 % of the energy. (3) Nominal lossy preset (Valette loss), 2 s: residual 2.6e-11, no increases. 0 Newton or friction failures. | yes |
| **C. 1-D parity** (kharaj preset: profile, static_init, tune, finger pluck, Valette loss) | All 14 set-up arrays are identical: T, M, σ, a, b, c, ω*², g_e, g_c, x_j, h_j, Φ, w0 and F_e(t). The waveforms agree to **3.7e-14** during the 40 ms pull. After the release they separate (1e-9 at 46 ms, 1e-3 at 0.17 s). A 1-D twin with the Newton tolerance 1e-21 instead of 1e-18 separates from 1-D at the same rate (1e-3 at 0.25 s), so the separation is chaotic amplification, not a model difference. Mean buzz 0.3–2.9 s: −22.20 (1-D), −22.19 (2-D), −22.20 (twin). Pressed thread: static states agree to 6.7e-15 and the waveforms to 1.7e-13 before the release. | yes |
| **D. static_init, no click** (pressed thread, no pluck, full 2-D model with tension modulation, friction, Valette loss and nut coupling; 0.3 s) | Exact static_init: max \|F_y(t) − F_y(0)\| = **9.9e-14 N**, 241 dB below a 2 mm pluck; solver residual 4e-15 N; 6 iterations. The older init_static gives 5.4e-7 N in y and 2.1e-5 N in z, because it ignores the lateral thread reaction under nut coupling. Starting from the straight string gives an **8.8 N click** (+38 dB re the pluck). On the C3 parabola with h0 = +5 µm and pol2 off, the old and exact solvers agree to 2.2e-15. | yes |
| **E. Tuning** (contact off, lossless, TM off, 8 s at 176.4 kHz) | Axis 1 at 69.40004 Hz (**+0.0010 cents**). Axis 2 at +3.491 cents: the thread spring makes the split 3.490 cents for a 3.5 cent setting. `tune_ref='mean'`: the mean is −0.005 cents. With pol2 off, T_2D == T_1D bit for bit. Soft lateral thread (info only): the z series speaks 6.3 cents flat. | yes |
| **F. CPU** (single thread, 3 s, load ≈ 1.2 on 4 CPUs) | 1-D kharaj 2.11 s/s; 2-D kharaj (88°, TM, friction) **2.84 s/s = 1.35× 1-D**; 2-D without TM 2.60 s/s. The 13 demo renders took 2.6–3.0 s/s with 2 processes. | – |

## 4. Demonstration: the sideways pluck and the bloom (`python validate2d_parity.py G`)

**Set-up.**
- Kharaj presets at their own pitch (69.4 Hz), 5 s each; 13 renders took 180 s of CPU (97 s wall with 2 processes).
- **Radiation.** `presets2d.radiate`, i.e. `body.apply_body(y, 44100, 'tanpura_male', hp_hz=250, hp_order=2)`. The 2nd-order 250 Hz high-pass stands in for the low radiation efficiency of the soundboard at the lowest partials.
- **Features.** quickfeat's single-pluck features, applied to the onset-trimmed signal: per-partial STFT envelopes (180 ms window) for partials 1–60.
  - Partial n is located at n f0 √(1+Bn²). f0 and B are fitted to each signal by **sequential partial tracking** (§5), for references and renders alike.
  - rise = maximum of the total level minus the level at 0.05 s; t_max = time of that maximum.
  - H4-10 = energy share of partials 4–10; H31-60 = level of partials 31–60 relative to the total.

**Kharaj.** For the 2-D renders: Fy row, then mono row (F_y + 0.1 F_z). Each 2-D case also shows z_E, the share of the string energy in z at 0.05/1/3 s.

| case | rise dB | t_max s | H4-10 share 0.05/1/3 s | H31-60 dB 1/3 s | z_E 0.05/1/3 s |
|---|---|---|---|---|---|
| **recording: iTanpura kharaj** | **17.7** | **1.27** | 0.26 / 0.83 / 0.89 | −12.8 / −12.2 | |
| **recording: own kharaj** | **13.9** | **1.01** | 0.58 / 0.92 / 0.69 | −17.5 / −14.4 | |
| (same two, task's quickfeat numbering) | 17.3 / 13.7 | 1.29 / 1.01 | 0.26/0.97/0.95, 0.57/0.96/0.72 | −20.2/−15.2, −24.2/−17.9 | |
| 1-D kharaj (vertical) | 5.3 | 0.73 | 0.67 / 0.83 / 0.74 | −25.1 / −22.6 | |
| 2-D 0°, 1 cent | 5.1 / 5.0 | 0.73 / 0.73 | 0.69 / 0.85 / 0.80 | −25.2 / −23.0 | 0.00/0.03/0.07 |
| 2-D 0°, 5 cents | 3.4 / 3.6 | 0.37 / 0.37 | 0.70 / 0.89 / 0.80 | −26.1 / −24.9 | 0.00/0.70/0.80 |
| 2-D 45°, 1 cent | 4.9 / 4.4 | 0.71 / 0.69 | 0.69 / 0.89 / 0.82 | −23.9 / −26.3 | 0.50/0.69/0.93 |
| 2-D 45°, 5 cents | 2.4 / 2.3 | 0.21 / 0.21 | 0.71 / 0.66 / 0.73 | −19.1 / −30.1 | 0.50/0.94/0.82 |
| 2-D 80°, 1 cent | 3.1 / 1.8 | 0.49 / 0.21 | 0.71 / 0.63 / 0.52 | −21.3 / −33.5 | 0.97/0.99/0.99 |
| 2-D 80°, 5 cents | 7.0 / 5.6 | 1.53 / 1.63 | 0.76 / 0.59 / 0.64 | −32.8 / −25.4 | 0.97/0.84/0.86 |
| 2-D 86°, 3.5 cents | 12.4 / 9.4 | 1.57 / 1.57 | 0.77 / 0.68 / 0.54 | −31.2 / −24.3 | 0.99/0.90/0.91 |
| **2-D preset: 88°, 3.5 cents** | **15.6 / 11.8** | **1.35 / 1.35** | 0.79 / 0.65 / 0.56 | −32.0 / −23.3 | 1.00/0.90/0.92 |
| 2-D 90°, 1 cent | **18.2** / 5.3 | 1.31 / 1.33 | 0.59 / 0.66 / 0.53 | −30.0 / −26.8 | 1.00/0.99/0.98 |
| 2-D 90°, 3.5 cents | **17.8** / 13.4 | 1.17 / 1.19 | 0.65 / 0.69 / 0.56 | −28.0 / −26.4 | 1.00/0.89/0.92 |
| 2-D 90°, 5 cents | **16.4** / 14.0 | 1.17 / 1.19 | 0.65 / 0.70 / 0.83 | −27.9 / −31.2 | 1.00/0.81/0.86 |
| 2-D 90°, θ_nut 45°, 3.5 cents | **16.3** / 12.4 | 1.11 / 1.13 | 0.64 / 0.63 / 0.17 | −30.9 / −27.9 | 1.00/0.87/0.97 |

The H4-10 and H31-60 columns are for F_y.

Other references (rise / t_max, fitted f0):
- iTanpura Pa: 14.5 dB / 1.17 s (103.87 Hz)
- iTanpura Sa: 17.1 / 1.01 (138.53 Hz)
- own Pa: 17.7 / 1.15
- own Sa: 17.3 / 0.71

All references are harmonic (B < 2e-6).

Figure: `out/2d_pluck/png/bloom_demo.png` shows the level vs time of both recordings and six model cases. Audio: `out/2d_pluck/wav/<case>_radiated.wav`.

**Extra exploration** (scratch, same set-up; F_y rise dB / t_max s, then mono rise with gz = 0.1):

| pluck | θ_nut 30°: 2 / 3.5 / 5 cents | θ_nut 45°: 2 / 3.5 / 5 cents |
|---|---|---|
| 86° | 8.7/1.67 (4.6), 12.4/1.57 (9.4), 13.0/1.39 (10.9) | 7.9/1.61 (3.8), 10.6/0.97 (7.4), 10.9/0.89 (8.5) |
| 88° | – , 15.6/1.35 (11.8), 15.2/1.31 (12.8) | – |
| 90° | 18.1/1.37 (10.4), 17.8/1.17 (13.4), 16.4/1.17 (13.9) | 17.6/1.13 (10.2), 16.3/1.11 (12.4), 14.2/0.93 (12.0) |

**Mono mix weight** (90°, 5 cents; rise dB at gz = 0 / 0.03 / 0.1 / 0.2 / 0.3):
- 5 cents: 16.4 / 16.2 / 13.9 / 10.3 / 7.7
- 1 cent: 18.2 / 13.6 / 5.3 / 0.6 / 0.0
- At gz = 0.3, the 88° preset gives 0.8 dB.

**Reading.**

1. **Why the bloom happens.**
   - A pluck parallel to the soundboard moves the string in z, which cannot hit the bridge and drives the soundboard only weakly.
   - Through the nut anisotropy, partial n trades energy between z and y. The y share is sin²(2θ_nut) · sin²(π Δf_n t), with Δf_n = n f1 (2^(c/1200) − 1). For 3.5 cents the y maximum comes at 0.6 s for n = 6, and earlier for higher partials.
   - Energy reaching y hits the jawari, which pumps it into partials 4–10 and above. The radiated level rises by 15–18 dB to a peak at 1.1–1.4 s, as in the recordings (13.9–17.7 dB, 1.0–1.3 s).
   - A vertical pluck is loudest at once (rise 2–5 dB, t_max ≤ 0.7 s), whether in 1-D or 2-D.
2. **Why the pluck must be nearly parallel.**
   - The initial y energy is cos²φ of the pluck, and only ~10–20 % of the energy ever sits in y (z_E ≥ 0.8). So the rise is roughly 10 log(E_y,max / cos²φ).
   - With E_y,max ~ 0.1, a rise above 15 dB needs cos²φ ≲ 0.003, i.e. φ ≳ 87°; the renders give 12.4 dB at 86° and 15.6 dB at 88°.
   - At 80° (cos² = 3 %) the rise is only 3–7 dB.
   - A real finger stroke "across the strings" may well be this flat, but this angle is unmeasured.
3. **The coupling strength is not critical for the bloom itself.** Splits of 2–5 cents and nut angles of 30–45° all bloom at 90°. The split moves t_max (larger split, earlier peak) and the late decay, so these remain fit parameters.
4. **The mono weight is.** The in-plane force F_z is large from the start. With gz = 0.3 it sets the early level, and the bloom disappears from the mix. The recordings imply gz ≲ 0.1 in this one-way radiation model.
5. **What is still wrong.**
   - The jawari shimmer after the bloom is too weak: partials 31–60 are 9–19 dB below the recordings at 1–3 s, and partials 16–40 are 7–13 dB below at 1 s.
   - The H4-10 share at 1 s is 0.65 against 0.83–0.92.
   - With most of the energy kept in z (z_E 0.9 at 1–3 s), the jawari is fed by only ~10 % of the energy, so the sideways pluck makes this deficit worse than in 1-D (−25 / −23 dB).
   - The late level, though, is sustained by the z reservoir: −10.5 dB at 4 s for the preset, against −11.4 / −13.4 dB in the recordings.
   - Candidates to fit with `reference.py`:
     - a stronger or faster y–z exchange (θ_nut → 45°, larger split);
     - lower high-frequency losses (`delta_ve`; T60 at 2 kHz is 1.5 s);
     - the thread gap (`thread_shift`);
     - a bridge whose contact normal depends on z, so z also buzzes (MODEL_2D.md §6, EVALUATION §3.4).

## 5. The reference spectra are harmonic: f0 69.33 Hz and B ≈ 0, not 68.56 Hz and 4.4e-4

**How it was found.**
- Picking all partial peaks at once around a guessed stretched series n f0 √(1+Bn²) gives whatever peak lies in the window. For a 69 Hz string, partials are only 69 Hz apart.
- With a starting B of 4.3e-4 the fit "confirms" B ≈ 4.2e-4; started at 1.6e-4 it returns 1.3e-4.
- **Sequential tracking** removes this ambiguity: partial n is searched within ±0.2 f0 of the prediction fitted to partials 1…n−1 (`validate2d_parity.fit_f0_B`).

**What it gives.**
- **References.** own kharaj f0 = 69.327 Hz, iTanpura kharaj 69.344 Hz, both with |B| < 1e-5. The raw peak list of the own kharaj (0.3–3 s) has every partial up to n = 31 within 0.2 % of n × 69.32 Hz (largest deviation 0.056 of a partial spacing, at n = 31).
- **Model renders.** Also harmonic, |B| ≲ 8e-6. The sounding pitch is +4 to +13 cents above the contact-off f1: +4.8 cents for the 88° preset, +7.7 for 1-D.
- So the jawari **locks the partials into a harmonic series** in the model as in the instrument, even though the bare string has B = π²EI/(TL²) = 1.6e-4. This is the classic tanpura/sitar observation (Raman).

**Consequences.**
- The recordings are at A440 pitch, not ~19 cents flat.
- quickfeat with B = 4.3e-4 samples windows *between* the harmonics above n ≈ 15. Its H16-40 and H31-60 values for the references are unreliable; they are corrected in the table above.
- rise, t_max and H4-10 change by less than 0.5 dB / 0.05.
- To make the model's sounding pitch equal 69.33 Hz with jawari on, lower f1 by the measured contact sharpening, e.g. ~4.8 cents for the 88° preset.
- Lowering f1 itself to 68.56 Hz fails on the bronze bridge: its speaking-length angle ψ is designed for 69.4 Hz, and at the lower tension no grazing thread position exists within 1.5 mm of 4 mm (`presets` raises).

## 6. Assumptions and limitations

- **Pluck direction.** 88° and the 2 mm pad displacement are assumptions; the bloom data only says "≳ 87° in this model".
- **Nut coupling.** θ_nut and the split remain unmeasured. They are an effective stand-in for any linear y–z coupling: a real tanpura probably couples mostly through the soundboard-loaded bridge, not the nut.
- **gz = 0.1** is inferred from the bloom, not measured. The body is a one-way filter applied to a weighted sum of F_y and F_z. The 250 Hz high-pass is a crude radiation-efficiency stand-in and applies only to these comparisons, not to `body.py`'s defaults.
- **Common damping in y and z** (Valette σ_i at the axis-1 frequencies). Polarisation-dependent bridge or body damping, which a real soundboard has, is not modelled.
- **Contact normal fixed (+y).** z never buzzes by itself (MODEL_2D.md §6), and the thread is bilateral and linear in z. With a 2 mm sideways pluck the lateral thread force reaches 0.7 N; whether a real thread grips that strongly is unverified.
- **The demo compares 5 s single plucks** with 4–7 s recordings through a small set of features. Instantaneous trajectories are chaotic: only smoothed features are meaningful, and differences under ~1 dB in rise are within the run-to-run spread seen in MODEL_2D.md.
- **The `sa` and `pa` presets work in 2-D** (`preset2d('sa')`, `preset2d('pa')`: 0.3 s smoke runs, 0 solver failures, about 2.2 CPU s/s). The bloom demonstration was run for the kharaj only.

## 7. Files and how to run

**Code:**
- `tanpura2d.py`: the extended model (new: `tune_tension`, `stiffness_matrix`, `static_compliance`, `excitation_amp`, `static_equilibrium`; `excitation(p, s)`).
- `presets2d.py`: `preset2d`, `render2d`, `radiate`, `to_1d`, `sim_params`, `PRESET_2D`.
- `validate2d_parity.py`: tests A–F and the demo G.

**Commands:**

```bash
cd jawari
python validate2d_parity.py              # A-G (~6 min: renders are cached in out/2d_pluck/runs/*.npz)
python validate2d_parity.py A B C D E F  # validation only (~3 min)
python validate2d_parity.py G --jobs 2   # demo only (2 processes)
python validate2d.py                     # the original 2-D tests (still pass)
```

**Outputs in `out/2d_pluck/`:**
- `parity_results.txt` and `parity_results.json`;
- `runs/*.npz`: 44.1 kHz F_y, F_z and polarisation energies per demo case;
- `wav/*_radiated.wav`;
- `png/bloom_demo.png`.
