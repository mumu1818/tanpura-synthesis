# Round-3 refit with a physical output chain (`fit3.py`)

> Run every script from inside `jawari/`. Outputs go to `jawari/out/fit3/` (git-ignored). The recordings are only **read** from `/home/user/tanpura-app/assets`; the loudness-matched copies in `out/fit3/wav/` are local and git-ignored. Never commit them.

`fit3.py` refits the 2-D string (`tanpura2d`, with the round-3 physics of [`MODEL_2D_ROUND3.md`](MODEL_2D_ROUND3.md)) to the real plucks. Unlike [`FIT2D.md`](FIT2D.md), the output chain may not use the 250–400 Hz high-pass that hid the string's surplus in partials 1–3. The string has to do that work itself.

```bash
cd jawari
python fit3.py refs                        # reference features (own, iTanpura, held-out iTanpura CS3Alt)
python fit3.py probe                       # one-at-a-time physics probe on the kharaj
python fit3.py fit kharaj pa sa --jobs 2   # grid -> CMA-ES -> geometry screening -> CMA-ES polish (resumable)
python fit3.py ablate kharaj pa sa --jobs 2
python fit3.py final kharaj pa sa          # pitch-corrected 6 s renders, match tables, WAVs, PNGs
```

Every render, statics solution and score is cached in `out/fit3/cache/`, so each command resumes.

## 0. Bottom line

* **Under the same physical chain, the refit string beats the FIT2D string on all 9 references.** D3 (lower is closer), fit3 vs the FIT2D string through the physical chain:

  | string | own | iTanpura | held-out CS3Alt |
  |---|---|---|---|
  | kharaj | **186.7** vs 309.8 | **241.3** vs 265.2 | **316.4** vs 453.4 |
  | Pa | **166.9** vs 216.9 | **175.8** vs 210.5 | **360.1** vs 452.5 |
  | Sa | **269.1** vs 317.4 | **285.2** vs 327.3 | **305.1** vs 312.8 |

* **Against FIT2D's own fits, with their 196–400 Hz high-pass and −6 dB body modes, the result is mixed.**
  * Better: own kharaj 186.7 vs 201.2, own Pa 166.9 vs 177.0, iTanpura Pa 175.8 vs 177.7 (equal).
  * Worse: own Sa 269.1 vs 209.8, iTanpura Sa 285.2 vs 182.8, iTanpura kharaj 241.3 vs 173.8.
  * On FIT2D's old distance D2 alone, FIT2D's fits stay ahead on 5 of 6 fit references (own kharaj 49.2 vs 92.2).
  * Where fit3 wins, it wins on the new terms: the click and the second bloom on the own kharaj (new terms 94.5 vs 152.0), the level curve on the own Pa (level RMS 2.1 vs 4.2 dB).
* **Ablation: of the round-3 physics, only the groove wall clearly earns its place, and only on the kharaj.**
  * Kharaj: groove off ΔJ **+60** (the click goes from −2.0 to −12.4 dB). The coupling refit is worth **+88**.
  * Pa: the slower finger release (2.82 ms) **+50**, coupling +13, the 0.30 mm thread +10. Groove +4.4 and body drain +4.6 are at the noise level.
  * Sa: body drain −0.4 (the fit left it at 0.005 s/kg, which has no measurable effect); no groove.
  * Pitch design does not change D3 (−0.8 kharaj, −4.0 Sa). It is needed for the tuning, and the Pa's 0.30 mm thread has no valid rest state without it.
* **Pitch: all three final renders sound within 0.1 cents of the user's recordings** (69.329 / 103.962 / 138.540 Hz).
* **Still missing** (own recordings, 1 s after the onset):
  1. Partials 16–31 are 10–17 dB too weak, partials 31–60 6–13 dB.
  2. The attack lacks brightness: partials 16–31 at 0.09 s are 7–17 dB too weak. The Sa has no click at all (−35.2 vs −2.0 dB).
  3. The string keeps 0.92–0.99 of its energy in partials 1–3 at every time. Radiated, partials 1–3 are 8.6 dB (kharaj) and 11.2 dB (Sa) too strong.
  4. The held-out CS3Alt samples are matched poorly (D3 305–360; kharaj level RMS 11.7 dB).

> **Update (2026-11-21, [`MATCH_REPORT.md`](MATCH_REPORT.md)).**
> * Listening to the final renders, the user hears **Sa as close** and **Pa and kharaj as not close**. D3 ranks it the other way round (Sa worst).
> * The time-resolved analysis finds that the swell's timing and the pitch slide after the pluck follow the user's verdict; the tone balance does not.
>   * The kharaj and Pa models' pitch slides up during the swell, where real plucks slide down.
>   * D3 has no pitch term.
> * It also finds that the "own" files (`Mumu_Tanpura/Male/tt_CS3_*`) and the iTanpura CS3 files are largely the same plucks, processed differently, so sharing string parameters across them is not a test on two independent plucks.
> * D3 puts the kharaj model (187) closer to the own file than the iTanpura copy of the same pluck (222).
> * Treat the D3 numbers below as a measure of this fit's objective, not of closeness to the ear.

## 1. Design of the fit

### 1.1 Physical output chain

The radiated signal is body(F_y + gz F_z) followed by a radiation high-pass. The chain is fitted per recording in closed loop, for every render, from this grid only (`CHAIN_GRID`, `BODY_A0`, `BODY_T1`):

| item | values | FIT2D allowed |
|---|---|---|
| body | `body.py` `tanpura_male` explicit modes, A0 ∈ {55, 65, 80} Hz, T1 ∈ {100, 120, 150} Hz | A0 65, T1 120 |
| body-mode gain | 0 dB (as designed) | −6, 0, +6 dB |
| in-plane weight gz | 0, 0.02, 0.05, 0.1 | 0–1 |
| radiation high-pass | none (the body's 20 Hz guard), or 60 / 80 / 100 / 125 / 150 Hz at order 1 or 2 | 60–400 Hz, order 1–3 |

That is 396 chains per render and recording. One render serves all of them: the body is linear, and the high-pass is applied per partial as |H(n f0)|². The > 4 kHz burst is taken before the high-pass, which does not reach it. The logs also report `phys`, the D3 with the round-3 demo's fixed chain (A0 65, T1 120, gz 0.1, HP 150 Hz / 2). For the best kharaj string it is 381.4 instead of 183.9, so the choice of chain within the physical bounds matters a lot.

### 1.2 References and objective

| role | kharaj | Pa | Sa |
|---|---|---|---|
| own (weight 2) | `Mumu_Tanpura/Male/tt_CS3_kharaj.mp3` | `tt_CS3_pa.mp3` | `tt_CS3_sa.mp3` |
| iTanpura (weight 1) | `iTanpura/Tanpura_CS3_Kharaj.mp3` | `Tanpura_CS3_Pa.mp3` | `Tanpura_CS3_Sa.mp3` |
| held out | `iTanpura/Tanpura_CS3Alt_Kharaj.mp3` | `Tanpura_CS3Alt_Pa.mp3` | `Tanpura_CS3Alt_Sa.mp3` |

* **One string per string name**, shared by the own and the iTanpura recording. Only the chain differs per recording. FIT2D fitted a separate string to the iTanpura kharaj and Pa.
* **Objective** J = (2 D3_own + D3_iTanpura) / 3.
* **Held out:** the CS3Alt sample of each string is scored at every evaluation (string frozen, chain re-fitted), but never enters J or any selection.
* **Onsets:** recording, first sample above −30 dB of the peak, minus 5 ms; model, end of the finger release minus 5 ms.
* **Partials** are numbered from each signal's own f0 with B = 0: the recordings are harmonic (MODEL_2D_PLUCK.md §0).
* **Time window:** late-time terms end at t_end = 3.8 s, or 3.0 s for the 4 s files that fade from about 3.1 s (own and iTanpura Sa, iTanpura Pa, CS3Alt Sa).

### 1.3 String parameters (`SPACE`)

| parameter | meaning | bounds | searched in |
|---|---|---|---|
| `pluck_deg` | stroke direction from the bridge normal | 82–90° | CMA |
| `pluck_release` | finger roll-off time (attack, click) | 0.1–3 ms | grid, CMA |
| `pluck_disp` | static pad displacement | 1.5–3 mm | CMA |
| `xp_frac` | pluck point from the bridge | 0.30–0.36 L | screening |
| `theta_nut`, `detune_cents` | nut coupling: axis angle, split | 0–45°, 0.5–8 c | grid, CMA |
| `kcz_x` | lateral / vertical thread stiffness | 0.5–3 | CMA |
| `groove_kappa`, `groove_z0` | groove curvature, offset of its bottom (string on the groove wall) | 0–30 /m, 0–1 mm | grid, CMA (κ = 30 /m whenever z0 ≥ 0.05 mm, else off) |
| `Yb_y` | bridge-normal modal body drain, `gain` weights, drain body A0 65 / T1 120 Hz | 0–0.3 s/kg | CMA |
| `Ybz_x` | in-plane drain Yb_z / Yb | 0.2–5 | screening |
| `c`, `d_t`, `target_gap` | bridge curvature, thread diameter, crest gap (statics, with `design_f1`) | 1.5–6 /m, 0.1–0.4 mm, −1…+3 µm | screening |
| `delta_ve_x` | visco-elastic loss × preset | 0.5–2 | screening |

Every statics geometry is solved with `design_f1`: ψ is re-designed so that the thread grazes at x_c0 = 4 mm at the tension of f1.

### 1.4 Distance D3

D3 = FIT2D's distance D2 + new terms. Every item scores w · min((Δ/tol)², 25).

| group | items | definition | tolerance | weight per item |
|---|---|---|---|---|
| rise_db | 1 | level(max) − level(first frame, 0–0.18 s) | 1.5 dB | 4 |
| t_max | 1 | time of the maximum (log ratio) | × 1.25 | 2 |
| t_m3 | 1 | first time within 3 dB of the maximum | × 1.3 | 2 |
| lev | 9–10 | level re maximum at 0.09 … 3 (4) s | 2 dB | 0.6 |
| share | 5 | energy share of partials 4–10 at 0.09 / 0.3 / 1 / 2 / 3 s | 0.07 | 1.2 |
| oct | 30 | partials 1, 2–3, 4–7, 8–15, 16–31, 32–60 re total at 0.09 / 0.3 / 1 / 2 / 3 s | 3 dB | 0.25 |
| **new:** band | 25 | partials 1–3, 4–7, 8–15, 16–31, 31–60 re total at 0.09 / 0.5 / 1 / 2 / 3 s | 3 dB | 0.3 |
| att16 | 1 | partials 16–31 re total at 0.09 s (attack brightness) | 3 dB | 2 |
| burst | 1 | `reference.py` HF burst: > 4 kHz energy in the first 20 ms vs 20–200 ms (finger click) | 4 dB | 1.5 |
| levrms | 1 | RMS of the level-curve difference, 0.1 s – t_end | 1.5 dB | 4 |
| decay | 1 | slope of the level, 1.5 s – t_end | 1 dB/s | 2 |
| rerise | 1 | second bloom: re-rise after 2 s; only the excess over max(2 dB, the recording's own) counts | 1 dB | 2 |

D2 carries 27.5 weight units (20 of them on the bloom), the new terms 19. A difference of a few units is not meaningful: the landscape is rugged at the 5-unit level (FIT2D.md, MODEL_2D_ROUND3.md §7).

## 2. Search stages and cost

The probe (`probe_kharaj.txt`, 23 points, about 3 min wall) chose the grid. One step at a time from the FIT2D kharaj string (J 345.1):

| step | J | what it did |
|---|---|---|
| split 7 → 2 c | 296.8 | removes the second bloom (7.2 → 1.6 dB) |
| groove wall 30 /m, z0 1 mm | 296.0 | click: burst −24.4 → −4.1 dB |
| release 2 → 0.3 ms | 361.7 | click −4.0 dB and att16 −15.1 dB, but the rise falls 11.7 → 8.2 dB |
| modal drain Yb_y 0.1 | 371.0 | rise 11.7 → 8.2 dB |
| drain on A0 + T(0,1) only, tuned to partials 1–2 (A0 69.5, T1 139 Hz), 0.1 / 0.3 s/kg | 550.5 / 528.1 | kills the bloom (rise 6.1 / 3.6 dB, peak at 0.29 / 0.41 s); partials 1–3 are not drained |
| Yb_z × 4, kcz × 1, c 1.5 or 6 /m | 359–427 | all worse |
| d_t 0.1 or 0.4 mm | – | no valid rest state for the bronze kharaj |

Fit stages (`fit_<string>.log`, `history_<string>.jsonl`), 4.2 s renders, at most 2 processes:

| stage | what | evaluations (kharaj / Pa / Sa) | string CPU s (kharaj / Pa / Sa) | best J of the stage (kharaj / Pa / Sa) |
|---|---|---|---|---|
| s0 | FIT2D best string and the 2-D preset string, new physics off | 2 / 2 / 2 | 25 / 22 / 18 | 236.5 / 208.4 / 319.9 |
| s1 | grid: split {2, 3.5 (Sa 3.3), 7 c} × nut {30, 45°} (Pa {15, 35°}) × groove wall {off, 30 /m with z0 1 mm} × release {0.7, 2 ms}, plus the start | 25 / 25 / 25 | 371 / 309 / 250 | 217.0 / 213.6 / 309.6 |
| s2 | CMA-ES, λ 8, 5 generations, σ 0.18, on pluck angle, release, displacement, nut angle, split, kcz, Yb_y, groove z0 | 40 / 40 / 40 | 654 / 560 / 444 | 235.9 / 177.4 / 301.2 |
| s3 | screening, 2 steps each: c {1.5, 6}, d_t {0.15, 0.3 mm}, gap {−1, +2 µm}, xp {0.30, 0.36}, delta_ve × {0.5, 2}, Yb_z × {0.25, 4} | 12 / 12 / 12 | 192 / 178 / 142 | 207.7 / 171.8 / 297.6 |
| s3b | the improving steps combined (if ≥ 2 improved) | 1 / 1 / – | 17 / 20 / – | 209.7 / 207.5 / – |
| s4 | CMA-ES polish, λ 8, 2 generations, σ 0.08 | 16 / 16 / 16 | 263 / 254 / 167 | **203.4 / 168.1 / 277.2** |
| total | | 96 / 96 / 95 | 25.4 / 22.4 / 17.0 min | |

* Wall time with 2 processes: about 15 / 16 / 13 min.
* Each new statics geometry takes 12–33 s and is cached. One kharaj screening step (d_t 0.3 mm) has no valid rest state: the string lifts off the back of the bridge.
* Combining the improving screening steps made things worse on both strings where it ran (kharaj 209.7 vs 207.7; Pa 207.5 vs 171.8). The parameters interact.
* The 5-generation CMA-ES did not beat the grid on the kharaj. The best points are good points from about 95 renders each, not converged optima.
* Finals: one 6 s render per string (25 / 21 / 11 s CPU). Ablation: 18 switches, 3.0 min wall, mostly cached.

## 3. Best parameters

From `final_<string>.json` (`best_x`).

| parameter | kharaj | Pa | Sa |
|---|---|---|---|
| pluck angle | 89.5° | 88.75° | 90.0° |
| finger release | 2.15 ms | 2.82 ms | 1.87 ms |
| pad displacement | 2.70 mm | 1.70 mm | 1.75 mm |
| pluck point | L/3 | L/3 | L/3 |
| nut angle / split | 44.0° / 3.71 c | 35.0° / 2.36 c | 32.5° / 3.00 c |
| lateral thread stiffness | × 2.56 | × 1.57 | × 1.93 |
| groove | 30 /m, z0 0.90 mm | 30 /m, z0 0.14 mm | off |
| modal body drain Yb_y | off | 0.045 s/kg | 0.005 s/kg (no effect, §5) |
| in-plane drain | isotropic (off) | isotropic | isotropic |
| bridge c / thread d_t / crest gap | 3 /m / 0.20 mm / 0 | 3 /m / **0.30 mm** / 0 | 3 /m / 0.20 mm / +1 µm |
| visco-elastic loss | × 0.50 | × 1 | × 0.50 |
| f1, fit / final | 69.007 / 69.075 Hz | 103.722 / 103.783 Hz | 138.222 / 138.264 Hz |
| J, fit / final render | 203.4 / 204.9 | 168.1 / 169.9 | 277.2 / 274.9 |
| FIT2D best own fit, for comparison | 90°, 45° / 7.0 c, kcz × 2.5, 2.7 mm, gap 0 | 88°, 15° / 3.5 c, kcz × 1, 2.0 mm, gap 0 | 89°, 30° / 3.3 c, kcz × 2.03, 2.05 mm, +1 µm |

* With z0 > 0 the string rests on the groove wall, slope 2κ z0: 0.054 (3.1°) on the kharaj, 0.008 (0.5°) on the Pa.
* The kharaj split halved, from 7.0 to 3.7 cents. A 7 c split under the physical chain gives a second bloom (§5).
* The Pa release sits near its 3 ms bound.
* Final renders and fit renders agree to 3.4 units of D3 or better, although the finals are 6 s long, pitch-corrected and analysed through the exact time-domain chain.

Chains fitted to the final renders:

| reference | kharaj | Pa | Sa |
|---|---|---|---|
| own | A0 55, T1 100, gz 0.05, HP 150 Hz / 1 | A0 65, T1 150, gz 0.05, HP 150 Hz / 1 | A0 55, T1 100, gz 0, HP 150 Hz / 1 |
| iTanpura | A0 55, T1 100, gz 0, HP 150 Hz / 2 | A0 65, T1 150, gz 0.1, HP 150 Hz / 2 | A0 55, T1 100, gz 0, HP 150 Hz / 1 |
| held out | A0 55, T1 100, gz 0, HP 150 Hz / 2 | A0 65, T1 150, gz 0.1, HP 100 Hz / 2 | A0 80, T1 100, gz 0.1, no HP |

Every own and iTanpura chain uses the strongest high-pass allowed (150 Hz). On the kharaj and Sa, A0 and T1 also sit at their lowest values. The chain is pushing against its bounds, as in FIT2D.

## 4. Match tables

From `final_<string>.txt` / `.json`.
- **fit3:** the final 6 s render at the corrected f1, through its own fitted chain (§3).
- **FIT2D fit:** FIT2D's 5 s fit render through FIT2D's own chain, which was fitted for D2, not D3. For the iTanpura kharaj and Pa that is a separate string fitted to that recording.
- **FIT2D string, physical chain:** FIT2D's own-recording string, with the chain re-fitted under fit3's constraints. This is the like-for-like comparison.
- **held out, FIT2D own string + own chain:** FIT2D's own-recording string and chain, unchanged.

Bands are partials 1–3 / 4–10 / 16–31 / 31–60 in dB re the total partial energy. att16 is partials 16–31 at 0.09 s. Burst, decay and 2nd bloom are defined in §1.4. Level RMS is the RMS level-curve error, 0.1 s to t_end.

#### Kharaj

| signal | rise dB / t_max s | bands 0.09 s | bands 1 s | bands 3 s | att16 dB | burst dB | decay dB/s | 2nd bloom dB | D3 | level RMS dB |
|---|---|---|---|---|---|---|---|---|---|---|
| **own: recording** | 13.9 / 1.01 | −6.3/−2.4/−8.2/−16.5 | −15.5/−0.4/−15.0/−17.4 | −6.8/−1.6/−12.2/−14.4 | −8.2 | −1.4 | −3.5 | 1.1 |  |  |
| own: fit3 | 14.8 / 1.43 | −7.2/−1.0/−21.0/−29.6 | −6.9/−1.1/−25.3/−30.5 | −7.1/−1.1/−19.4/−31.0 | −21.0 | −2.0 | −3.8 | 0.2 | 186.7 | 2.7 |
| own: FIT2D fit (400 Hz / 2 HP) | 13.8 / 1.03 | −9.3/−0.9/−17.9/−25.0 | −11.8/−0.5/−17.7/−21.8 | −10.0/−1.9/−11.2/−17.6 | −17.9 | −24.5 | −3.0 | 7.6 | 201.2 | 2.5 |
| own: FIT2D string, physical chain | 11.8 / 0.99 | −8.0/−0.9/−24.5/−33.5 | −6.0/−1.3/−22.5/−28.1 | −6.4/−1.4/−18.5/−25.8 | −24.5 | −24.6 | −3.8 | 7.3 | 309.8 | 2.6 |
| **iTanpura: recording** | 17.7 / 1.27 | −1.8/−5.9/−19.9/−13.7 | −20.0/−0.8/−10.3/−12.8 | −26.1/−0.5/−14.2/−12.2 | −19.9 | +1.6 | −6.0 | 0.8 |  |  |
| iTanpura: fit3 | 15.8 / 1.43 | −10.2/−0.5/−20.2/−28.9 | −7.1/−1.0/−25.6/−30.8 | −7.4/−1.0/−19.4/−31.0 | −20.2 | −2.1 | −3.8 | 0.2 | 241.3 | 2.6 |
| iTanpura: FIT2D fit (400 Hz / 3 HP) | 17.6 / 1.19 | −17.8/−0.6/−15.3/−21.4 | −16.3/−0.4/−15.9/−21.3 | −12.5/−0.7/−13.6/−18.3 | −15.3 | −24.9 | −5.6 | 1.3 | 173.8 | 1.7 |
| iTanpura: FIT2D string, physical chain | 16.3 / 1.25 | −8.4/−0.8/−24.4/−32.3 | −5.7/−1.4/−23.3/−29.5 | −5.2/−1.7/−23.2/−28.7 | −24.4 | −24.9 | −5.1 | 0.5 | 265.2 | 1.4 |
| **held-out CS3Alt: recording** | 17.3 / 0.87 | −19.8/−0.1/−20.5/−25.9 | −42.2/−0.1/−20.6/−24.4 | −40.1/−0.2/−18.7/−15.4 | −20.5 | −3.3 | −3.4 | 7.3 |  |  |
| held-out CS3Alt: fit3 | 15.8 / 1.43 | −10.2/−0.5/−20.2/−28.9 | −7.1/−1.0/−25.6/−30.8 | −7.4/−1.0/−19.4/−31.0 | −20.2 | −2.1 | −3.8 | 0.2 | 316.4 | 11.7 |
| held-out CS3Alt: FIT2D own string + own chain | 13.8 / 1.03 | −9.3/−0.9/−17.9/−25.0 | −11.8/−0.5/−17.7/−21.8 | −10.0/−1.9/−11.2/−17.6 | −17.9 | −24.5 | −3.0 | 7.6 | 381.5 | 9.6 |
| held-out CS3Alt: FIT2D string, physical chain | 11.6 / 0.99 | −8.9/−0.7/−24.9/−33.9 | −6.4/−1.2/−22.8/−28.3 | −7.0/−1.3/−18.8/−26.2 | −24.9 | −24.6 | −3.9 | 7.5 | 453.4 | 11.0 |

#### Pa

| signal | rise dB / t_max s | bands 0.09 s | bands 1 s | bands 3 s | att16 dB | burst dB | decay dB/s | 2nd bloom dB | D3 | level RMS dB |
|---|---|---|---|---|---|---|---|---|---|---|
| **own: recording** | 17.7 / 1.15 | −3.1/−3.8/−14.2/−22.3 | −12.9/−1.3/−18.4/−24.3 | −19.6/−0.3/−23.6/−21.1 | −14.2 | −6.8 | −4.3 | 2.0 |  |  |
| own: fit3 | 17.7 / 0.85 | −2.0/−4.4/−31.1/−30.1 | −10.3/−0.8/−31.2/−30.5 | −12.1/−0.3/−32.9/−33.1 | −31.1 | −9.3 | −4.7 | 0.0 | 166.9 | 2.1 |
| own: FIT2D fit (400 Hz / 1 HP) | 17.5 / 1.21 | −3.4/−2.9/−29.2/−26.2 | −12.0/−0.6/−29.9/−26.0 | −12.8/−0.4/−32.2/−23.8 | −29.2 | −9.2 | −3.3 | 0.0 | 177.0 | 4.2 |
| own: FIT2D string, physical chain | 16.1 / 1.03 | −4.3/−2.2/−33.4/−31.9 | −12.9/−0.7/−31.9/−31.2 | −12.9/−0.5/−34.3/−28.2 | −33.4 | −9.3 | −3.3 | 0.0 | 216.9 | 4.1 |
| **iTanpura: recording** | 14.5 / 1.17 | −1.6/−6.5/−15.2/−22.5 | −8.1/−1.0/−13.9/−22.2 | −15.8/−0.2/−23.5/−26.5 | −15.2 | +0.4 | −6.9 | 0.0 |  |  |
| iTanpura: fit3 | 15.2 / 0.85 | −1.3/−6.1/−34.0/−33.0 | −10.0/−0.8/−31.6/−30.9 | −11.5/−0.3/−33.4/−33.7 | −34.0 | −9.3 | −5.0 | 0.0 | 175.8 | 1.6 |
| iTanpura: FIT2D fit (196 Hz / 3 HP) | 14.3 / 1.11 | −2.5/−3.6/−33.4/−29.4 | −7.6/−1.1/−33.5/−23.9 | −12.4/−0.3/−36.7/−28.7 | −33.4 | −11.0 | −5.8 | 0.0 | 177.7 | 1.7 |
| iTanpura: FIT2D string, physical chain | 14.8 / 1.13 | −2.1/−4.3/−35.8/−32.7 | −6.2/−1.5/−35.3/−28.1 | −10.3/−0.4/−39.1/−33.2 | −35.8 | −11.0 | −5.6 | 0.0 | 210.5 | 1.7 |
| **held-out CS3Alt: recording** | 13.8 / 0.43 | −9.8/−0.9/−16.2/−23.2 | −20.8/−2.2/−9.7/−9.7 | −40.9/−0.0/−37.1/−27.0 | −16.2 | −1.2 | −6.0 | 0.0 |  |  |
| held-out CS3Alt: fit3 | 14.5 / 0.85 | −1.1/−6.7/−34.7/−33.7 | −9.5/−0.9/−31.6/−31.0 | −10.9/−0.4/−33.5/−33.7 | −34.7 | −9.3 | −4.7 | 0.0 | 360.1 | 3.2 |
| held-out CS3Alt: FIT2D own string + own chain | 17.5 / 1.21 | −3.4/−2.9/−29.2/−26.2 | −12.0/−0.6/−29.9/−26.0 | −12.8/−0.4/−32.2/−23.8 | −29.2 | −9.2 | −3.3 | 0.0 | 479.0 | 5.8 |
| held-out CS3Alt: FIT2D string, physical chain | 15.8 / 1.01 | −4.1/−2.3/−33.8/−32.3 | −12.4/−0.7/−32.0/−31.4 | −12.4/−0.6/−34.4/−28.4 | −33.8 | −9.3 | −3.3 | 0.0 | 452.5 | 5.4 |

#### Sa

| signal | rise dB / t_max s | bands 0.09 s | bands 1 s | bands 3 s | att16 dB | burst dB | decay dB/s | 2nd bloom dB | D3 | level RMS dB |
|---|---|---|---|---|---|---|---|---|---|---|
| **own: recording** | 17.3 / 0.71 | −4.7/−2.4/−15.0/−26.1 | −14.2/−2.4/−17.2/−24.3 | −8.9/−1.8/−18.8/−28.8 | −15.0 | −2.0 | −8.1 | 0.0 |  |  |
| own: fit3 | 14.3 / 0.77 | −2.0/−4.6/−21.8/−38.4 | −3.0/−4.1/−34.4/−33.0 | −6.4/−1.2/−39.3/−37.7 | −21.8 | −35.2 | −4.4 | 0.0 | 269.1 | 1.7 |
| own: FIT2D fit (249 Hz / 3 HP) | 18.7 / 0.85 | −3.1/−3.2/−18.8/−29.1 | −11.2/−2.5/−30.9/−25.7 | −14.3/−0.3/−37.9/−33.1 | −18.8 | −28.1 | −3.5 | 0.0 | 209.8 | 2.3 |
| own: FIT2D string, physical chain | 15.0 / 0.75 | −0.8/−7.7/−26.0/−37.9 | −3.4/−4.3/−33.0/−30.6 | −6.1/−1.3/−39.4/−37.4 | −26.0 | −27.7 | −4.2 | 0.0 | 317.4 | 1.7 |
| **iTanpura: recording** | 17.1 / 1.01 | −5.1/−2.2/−14.4/−27.6 | −14.4/−2.7/−19.9/−26.5 | −9.4/−1.4/−20.1/−32.3 | −14.4 | +1.4 | −7.7 | 0.7 |  |  |
| iTanpura: fit3 | 14.3 / 0.77 | −2.0/−4.6/−21.8/−38.4 | −3.0/−4.1/−34.4/−33.0 | −6.4/−1.2/−39.3/−37.7 | −21.8 | −35.2 | −4.4 | 0.0 | 285.2 | 2.0 |
| iTanpura: FIT2D fit (316 Hz / 3 HP) | 18.1 / 0.87 | −4.0/−2.4/−19.4/−29.9 | −13.7/−2.3/−31.0/−26.0 | −15.8/−0.2/−38.0/−33.0 | −19.4 | −27.9 | −3.6 | 0.0 | 182.8 | 1.9 |
| iTanpura: FIT2D string, physical chain | 15.0 / 0.75 | −0.8/−7.7/−26.0/−37.9 | −3.4/−4.3/−33.0/−30.6 | −6.1/−1.3/−39.4/−37.4 | −26.0 | −27.7 | −4.2 | 0.0 | 327.3 | 1.9 |
| **held-out CS3Alt: recording** | 6.9 / 0.57 | −0.3/−12.0/−30.9/−30.2 | −6.3/−4.2/−6.2/−17.4 | −2.9/−3.2/−39.7/−39.6 | −30.9 | +1.5 | −10.0 | 0.0 |  |  |
| held-out CS3Alt: fit3 | 9.3 / 0.77 | −0.7/−8.4/−27.7/−44.4 | −2.4/−4.7/−35.3/−34.2 | −4.3/−2.0/−40.7/−39.2 | −27.7 | −35.1 | −4.4 | 0.0 | 305.1 | 3.6 |
| held-out CS3Alt: FIT2D own string + own chain | 18.7 / 0.85 | −3.1/−3.2/−18.8/−29.1 | −11.2/−2.5/−30.9/−25.7 | −14.3/−0.3/−37.9/−33.1 | −18.8 | −28.1 | −3.5 | 0.0 | 530.4 | 5.0 |
| held-out CS3Alt: FIT2D string, physical chain | 8.9 / 0.75 | −0.3/−12.3/−33.0/−45.1 | −2.8/−4.8/−34.0/−31.8 | −4.5/−2.0/−40.4/−38.1 | −33.0 | −27.5 | −4.3 | 0.0 | 312.8 | 3.6 |

#### D3 split into FIT2D's old distance and the new terms

D3 (D2 + new), same signals as above:

| reference | fit3 | FIT2D fit | FIT2D string, physical chain |
|---|---|---|---|
| kharaj own | 186.7 (92.2 + 94.5) | 201.2 (49.2 + 152.0) | 309.8 (101.8 + 208.0) |
| kharaj iTanpura | 241.3 (145.7 + 95.6) | 173.8 (87.6 + 86.2) | 265.2 (140.3 + 124.9) |
| kharaj held out | 316.4 (159.9 + 156.5) | 381.5 (188.7 + 192.9) | 453.4 (242.6 + 210.8) |
| Pa own | 166.9 (66.6 + 100.2) | 177.0 (57.9 + 119.1) | 216.9 (81.8 + 135.1) |
| Pa iTanpura | 175.8 (66.4 + 109.4) | 177.7 (70.6 + 107.1) | 210.5 (87.7 + 122.7) |
| Pa held out | 360.1 (196.3 + 163.8) | 479.0 (282.5 + 196.5) | 452.5 (241.7 + 210.9) |
| Sa own | 269.1 (134.0 + 135.1) | 209.8 (90.6 + 119.3) | 317.4 (161.6 + 155.8) |
| Sa iTanpura | 285.2 (152.9 + 132.2) | 182.8 (69.3 + 113.5) | 327.3 (174.9 + 152.4) |
| Sa held out | 305.1 (145.2 + 159.9) | 530.4 (322.5 + 207.9) | 312.8 (150.1 + 162.7) |

**What is left of D3** at the best points (fit renders, `ablation_<string>.json`, row `best`):

| string / reference | D3 | bloom terms | oct + band (spectral balance) | att16 | burst | level RMS | decay | 2nd bloom |
|---|---|---|---|---|---|---|---|---|
| kharaj / own | 183.9 | 49.6 | 87.8 (48 %) | 34.0 | 0.0 | 12.5 | 0.1 | 0.0 |
| kharaj / iTanpura | 242.3 | 65.5 | 154.3 (64 %) | 0.0 | 1.3 | 12.0 | 9.2 | 0.0 |
| kharaj / held out | 319.8 | 94.0 | 125.1 (39 %) | 0.1 | 0.2 | **100.0** (cap) | 0.4 | 0.0 |
| Pa / own | 165.3 | 18.9 | 88.2 (53 %) | **50.0** (cap) | 0.6 | 7.2 | 0.4 | 0.0 |
| Pa / iTanpura | 173.7 | 11.6 | 91.9 (53 %) | **50.0** (cap) | 8.8 | 4.3 | 7.2 | 0.0 |
| Pa / held out | 358.5 | 97.9 | 183.6 (51 %) | **50.0** (cap) | 6.1 | 17.9 | 3.0 | 0.0 |
| Sa / own | 272.4 | 61.5 | 129.6 (48 %) | 10.3 | **37.5** (cap) | 5.1 | 28.5 | 0.0 |
| Sa / iTanpura | 286.8 | 87.6 | 120.1 (42 %) | 12.2 | **37.5** (cap) | 7.2 | 22.1 | 0.0 |
| Sa / held out | 306.5 | 71.0 | 121.7 (40 %) | 2.3 | **37.5** (cap) | 24.1 | **50.0** (cap) | 0.0 |

The spectral balance is 39–64 % of what is left. The bloom terms are small on the own Pa (18.9).

## 5. Ablation

`python fit3.py ablate kharaj pa sa --jobs 2` (`ablation_<string>.txt` / `.json`). One switch at a time from each string's best point, 4.2 s renders, chain re-fitted, **no refit of the string**. ΔJ > 0 means the switch made the match worse, so the feature was helping. Rows appear only for features the best point uses. |ΔJ| ≲ 5 is noise.

| string | switch | J | ΔJ | D3 own | D3 iTanpura | D3 held out | pitch re own (c) |
|---|---|---|---|---|---|---|---|
| kharaj | best | 203.4 |  | 183.9 | 242.3 | 319.8 | −1.7 |
| kharaj | groove off | 263.6 | **+60.2** | 270.1 | 250.6 | 382.6 | −1.0 |
| kharaj | release back to 2 ms | 208.3 | +4.9 | 187.5 | 250.0 | 323.2 | −1.6 |
| kharaj | coupling back to FIT2D (45° / 7 c / kcz × 2.5) | 291.6 | **+88.2** | 263.3 | 348.1 | 403.1 | +0.5 |
| kharaj | pluck angle back to FIT2D (90°) | 209.4 | +6.1 | 198.0 | 232.3 | 320.2 | −1.3 |
| kharaj | pitch design off (fixed ψ, f1 69.127) | 202.6 | −0.8 | 183.8 | 240.1 | 317.0 | +1.5 |
| Pa | best | 168.1 |  | 165.3 | 173.7 | 358.5 | −1.0 |
| Pa | groove off | 172.6 | +4.4 | 171.2 | 175.4 | 365.2 | −1.0 |
| Pa | modal drain Yb_y off | 172.7 | +4.6 | 169.2 | 179.7 | 366.3 | −0.9 |
| Pa | geometry back to d_t 0.2 mm | 178.1 | **+10.0** | 170.4 | 193.7 | 396.7 | −1.1 |
| Pa | release back to 2 ms | 218.6 | **+50.4** | 224.8 | 206.1 | 331.3 | −1.0 |
| Pa | coupling back to FIT2D (15° / 3.5 c / kcz × 1) | 181.5 | **+13.3** | 170.1 | 204.1 | 412.9 | −0.4 |
| Pa | pluck angle back to FIT2D (88°) | 168.0 | −0.1 | 167.0 | 169.9 | 355.5 | −1.1 |
| Pa | pitch design off (fixed ψ, f1 103.663) | failed | | | | | |
| Sa | best | 277.2 |  | 272.4 | 286.8 | 306.5 | −0.5 |
| Sa | modal drain Yb_y off | 276.8 | −0.4 | 272.3 | 285.8 | 308.5 | −0.5 |
| Sa | geometry back to gap 0 | 314.2 | **+36.9** | 300.1 | 342.2 | 336.7 | −0.1 |
| Sa | release back to 2 ms | 277.2 | −0.0 | 272.5 | 286.7 | 308.9 | −0.6 |
| Sa | coupling back to FIT2D (30° / 3.3 c / kcz × 2.03) | 292.6 | **+15.4** | 287.9 | 302.0 | 310.1 | −0.1 |
| Sa | pluck angle back to FIT2D (89°) | 302.8 | **+25.6** | 296.3 | 315.9 | 301.2 | −0.6 |
| Sa | pitch design off (fixed ψ, f1 138.142) | 273.2 | −4.0 | 268.5 | 282.8 | 306.6 | −1.5 |

The kharaj best point has no drain and default geometry, and the Sa has no groove, so those rows do not exist.

**Reading.**

1. **The groove wall is the one round-3 mechanism that pays, and only on the kharaj.**
   * Switching it off costs J +60 (own D3 +86).
   * It supplies the click: burst −2.0 dB with it, −12.4 dB without (recording −1.4).
   * It also lifts the attack: att16 −20.6 vs −30.6 dB (recording −8.2).
   * On the Pa (z0 0.14 mm, a flatter wall) it is worth +4.4, i.e. nothing measurable. The Sa fit switched it off and has no click.
2. **The body drain earns nothing:** Pa +4.6, Sa −0.4. The kharaj search tried Yb_y up to 0.17 s/kg, and its ten best points use 0–0.03. The probe shows why: a drain strong enough to matter shrinks the bloom (§2).
3. **Pitch design does not move D3** (kharaj −0.8, Sa −4.0). Its value is the tuning (§6) and feasibility. With the preset ψ = 4.536° no thread position lets the Pa's 0.30 mm thread graze, so the row fails ("no thread position near x_c0 = 4.00 mm gives crest gap 0"). The thread itself is worth +10.
4. **Most of the gain over the FIT2D string comes from refitting old parameters under the new chain.**
   * Coupling: kharaj +88, Pa +13, Sa +15. With FIT2D's 7 c split, the kharaj blooms a second time (6.0 dB) under the physical chain.
   * Finger release on the Pa: +50 (rise 19.0 dB with 2.82 ms, 15.5 dB with 2 ms).
   * On the Sa, the crest gap and pluck angle (+37 / +26) are FIT2D-era parameters. The +1 µm gap was already in the FIT2D Sa fit.
   * The ΔJ do not add up: the parameters interact (§2, stage s3b).

## 6. Pitch

* **Fit renders** were tuned with a guessed jawari sharpening of 8 / 4 / 4 cents (`SHARPEN0`): f1 = 69.007 / 103.722 / 138.222 Hz. They sound −1.7 / −1.0 / −0.5 cents from the own recordings.
* **The measured sharpening** of the best points is +6.3 / +3.0 / +3.5 cents (kharaj / Pa / Sa). It depends on the physics: the round-3 demo found +9.4 c for the FIT2D baseline and +6.8 c with groove and drain.
* **One correction step** (f1 = f0_own × 2^(−sharpening/1200), with ψ re-designed at that tension) gives f1 = 69.075 / 103.783 / 138.264 Hz. The 6 s finals sound at:

  | string | own recording | final render | error |
  |---|---|---|---|
  | kharaj | 69.327 Hz | 69.329 Hz | +0.1 c |
  | Pa | 103.962 Hz | 103.962 Hz | −0.0 c |
  | Sa | 138.541 Hz | 138.540 Hz | −0.0 c |

* **Tuning costs no match:** J changes by ≤ 2.3 between the fit render and the tuned final (203.4 → 204.9, 168.1 → 169.9, 277.2 → 274.9).
* **Without pitch design** (FIT2D's fixed ψ and f1 = f0_own × 2^(−5/1200)) the kharaj sounds +1.5 c and the Sa −1.5 c, and the Pa geometry is infeasible (§5). FIT2D's own fits sounded −1.4 to +4.4 c.

## 7. Limitations

**What got worse than FIT2D.**
* **Sa: D3 269.1 vs 209.8 (own) and 285.2 vs 182.8 (iTanpura).**
  * Without the 249–316 Hz third-order high-pass, partials 1–3 at 1 s sit at −3.0 dB re total, against −14.2 dB in the recording (FIT2D −11.2).
  * The Sa has no click (burst −35.2 vs −2.0 dB).
  * It decays too slowly (−4.4 vs −8.1 dB/s) and rises too little (14.3 vs 17.3 dB).
* **iTanpura kharaj: 241.3 vs 173.8.** FIT2D fitted a separate string to it. fit3 shares one string with the own kharaj, which weighs twice.
* **On FIT2D's old distance D2, FIT2D's fits stay better on 5 of 6 fit references** (§4). The constrained chain costs bloom and band-balance accuracy. fit3 gains it back only on the own kharaj (click, second bloom) and the own Pa (level curve).
* **Bloom timing.** The kharaj peaks at 1.43 s against 1.01 s (FIT2D 1.03). The Pa peaks at 0.85 s against 1.15 s (FIT2D 1.21).

**What is still missing** (own recordings; gaps in dB).

| gap | kharaj | Pa | Sa |
|---|---|---|---|
| partials 16–31 at 1 s, model vs recording | −25.3 vs −15.0 (10.3 weak) | −31.2 vs −18.4 (12.8) | −34.4 vs −17.2 (17.2) |
| partials 31–60 at 1 s | −30.5 vs −17.4 (13.1) | −30.5 vs −24.3 (6.2) | −33.0 vs −24.3 (8.7) |
| attack: partials 16–31 at 0.09 s | −21.0 vs −8.2 (12.8) | −31.1 vs −14.2 (16.9) | −21.8 vs −15.0 (6.8) |
| click (HF burst) | −2.0 vs −1.4 (matched) | −9.3 vs −6.8 (2.5) | −35.2 vs −2.0 (33.2) |
| partials 1–3 at 1 s | −6.9 vs −15.5 (8.6 too strong) | −10.3 vs −12.9 (2.6) | −3.0 vs −14.2 (11.2) |
| string energy in partials 1–3 at 0.09 / 0.5 / 1 / 1.5 / 3 s | 0.95 / 0.98 / 0.96 / 0.93 / 0.93 | 0.99 / 0.99 / 0.98 / 0.97 / 0.98 | 0.98 / 0.99 / 0.94 / 0.93 / 0.92 |

* **The string still keeps over 0.9 of its energy in partials 1–3 at every time, on all three strings.** None of the round-3 mechanisms changes that within physical bounds. The chain compensates as far as it may: every own and iTanpura chain uses the maximum 150 Hz high-pass, and the kharaj and Sa also use the lowest A0 and T1.
* **The upper partials are 6–17 dB too weak at 1 s,** and the attack is 7–17 dB too dark. The groove wall gives the click (> 4 kHz, < 20 ms) but not the bright 1–2 kHz attack.
* **The held-out CS3Alt samples generalise poorly**: D3 316 / 360 / 305, kharaj level RMS 11.7 dB.
  * They differ strongly from the fit samples. The CS3Alt kharaj has partials 1–3 at −42 dB at 1 s and a 7.3 dB second bloom. The CS3Alt Pa peaks at 0.43 s. The CS3Alt Sa rises only 6.9 dB.
  * fit3 is still better than FIT2D there (kharaj 316 vs 381 / 453, Pa 360 vs 479 / 453, Sa 305 vs 530 / 313).

**Method limits.**
* **The recordings may be processed.** The user's assets include `alt-jawari/mumu_EQ_Tanpura_CS3_Kharaj.mp3`, so at least some of the user's recordings were EQ'd. Whether the `Mumu_Tanpura/Male/tt_CS3_*` references were is not known. If they carry a low cut or a presence boost, part of gaps (1) and (3) is processing, which the physical chain is deliberately not allowed to copy. The iTanpura samples may be processed as well. Ask the user which files are raw and what EQ was used.
* **Capped terms give the search no slope.** Pa att16 is at its cap (50) on all three references. So are the Sa burst (37.5), the Sa held-out decay (50) and the kharaj held-out level RMS (100). The search cannot tell "16 dB too dark" from "30 dB too dark".
* **The search is not converged.** About 95 renders per string; the first CMA-ES stage found nothing better than the grid on the kharaj; combining improving steps made things worse. Differences under about 5 units are within the landscape's ruggedness.
* **The ablation is one-at-a-time without a refit.** A ΔJ says how much the best point leans on a feature, not what a refit without it could reach.
* **κ_g was not free:** the groove is either off or 30 /m, the top of the plausible range. Only its offset z0 was fitted.
* **One pluck per reference.** Pluck-to-pluck variation is unknown.
* **Unmeasured physics.** The body, the drain mobility, the thread diameter, the bridge curvature and the coupling values are effective, fitted values. The body is still a one-way filter.
* **Late decay** is judged only to 3.8 s (3.0 s for the 4 s files), because the recordings fade out.

## 8. Files

| file | content |
|---|---|
| `fit3.py` | references, D3 features and distance, closed-loop physical chain, cached renders, probe, grid / CMA-ES / screening search, ablation, finals, plots |
| `out/fit3/probe_kharaj.txt` | the two one-at-a-time probes |
| `out/fit3/fit_<string>.log`, `history_<string>.jsonl` | every fit evaluation, with stage, parameters, D3 per reference and the fitted chain |
| `out/fit3/ablation_<string>.txt` / `.json` | ablation rows (§5), with the per-term D3 |
| `out/fit3/final_<string>.txt` / `.json` | pitch correction, string energy shares, match tables (§4), best parameters |
| `out/fit3/png/<string>_partials_ref_vs_model.png` | partial envelopes (recording, fit3, FIT2D) and the level / band curves |
| `out/fit3/wav/<string>_model_chain_{own,itanpura,alt}.wav` | the final render through each reference's chain, −1 dBFS peak |
| `out/fit3/wav/<string>_model_dry_bridge_force.wav`, `<string>_previous_fit2d_loudness_matched.wav` | dry bridge force; the previous FIT2D fit, loudness-matched |
| `out/fit3/wav/<string>_reference_*_loudness_matched.wav`, `<string>_ab_own_then_model.wav` | local copies of the recordings, loudness-matched, and an A/B (recording, then model). Git-ignored; never commit or upload them |
| `out/fit3/cache/` | statics geometries, renders, scores |

Model code was not changed: `tanpura2d.py`, `presets*.py`, `fit2d.py` and `body.py` are as committed in round 3.
