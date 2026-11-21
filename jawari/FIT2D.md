# Fitting the 2-D sideways-pluck model to the real plucks (`fit2d.py`)

`fit2d.py` fits the two-polarisation model (`tanpura2d` via `presets2d`, finger pluck nearly parallel to the soundboard) to six real single plucks: the user's own male C# tanpura and the iTanpura app, for the kharaj, Pa and Sa strings. It is analysis by synthesis: interpretable, physically bounded parameters and a feature distance (no spectrogram loss), with the bloom features weighted highest.

```bash
cd jawari
python fit2d.py refs            # reference features (the recordings are read in place, never copied)
python fit2d.py fit kharaj      # grid -> one-at-a-time screening -> CMA-ES (own, then iTanpura) -> pressed-crest check
python fit2d.py fit pa sa
python fit2d.py final           # 8 s renders, WAVs, PNGs, match tables, sensitivity, shared chain, reference.py check
python fit2d.py all             # all of the above (~70 min CPU with 2 processes; every render and score is cached)
```

Outputs go to `out/fit2d/` (git-ignored):
- `final_<string>.txt` and `.json`: match table, best parameters, sensitivity.
- `fit_<string>.log` and `history_<string>.jsonl`: every evaluation.
- `refs.txt`, `joint_chain.txt`, `reference_compare.txt` with `reference_compare/`.
- `wav/` and `png/`.
- `cache/`: renders, statics and scores, so every command can be resumed.

## Summary

- **The 2-D model with a sideways pluck reproduces the bloom on all three strings. The current 1-D model cannot.**

  Rise / time of the peak, recording vs best 2-D fit vs 1-D preset:

  | string | own recording | best 2-D fit | 1-D preset |
  |---|---|---|---|
  | kharaj | 13.9 dB / 1.01 s | 13.8 dB / 1.03 s | 4.8 dB / 0.73 s |
  | Pa | 17.7 dB / 1.15 s | 17.5 dB / 1.21 s | 4.7 dB / 0.41 s |
  | Sa | 17.3 dB / 0.71 s | 18.7 dB / 0.85 s | 2.4 dB / 0.31 s |

  The iTanpura plucks are matched as well (kharaj 17.7 → 17.6 dB, Pa 14.5 → 14.3 dB).
- **Feature distance D** (lower is better):

  | string | own: best 2-D | own: 1-D | iTanpura: best 2-D | iTanpura: 1-D |
  |---|---|---|---|---|
  | kharaj | 49 | 244 | 88 | 361 |
  | Pa | 58 | 343 | 71 | 349 |
  | Sa | 91 | 394 | 69 | 393 |

  Fitting the radiation chain to the 1-D model does not help it (rise 3.7–4.7 dB): **the bloom is a string effect, not a filter effect.**
- **What creates the bloom:** the pluck direction (≥ 87°, best 88–90°), then the strength of the polarisation coupling.
  - The size of the nut frequency split sets the time of the peak.
  - The crest gap, pluck strength and string losses matter little.
  - The pluck point L/3 beats 2L/3 on every string. This agrees with the user's own plucking point.
- **Fitted coupling values are physically plausible.** The split is 3.3–7 cents, i.e. 0.14–0.28 Hz at the fundamental, equivalent to a 2–4 mm difference in effective length between the polarisations. The lateral thread is 1–2.5 × as stiff as the vertical one.
- **The radiation chain needs a strong low cut.**
  - The radiation high-pass and body-mode gain sit at the edge of their bounds (HP 250–400 Hz, order 2–3, explicit body modes −6 dB).
  - The model has too much energy in partials 1–3 compared with the recordings.
  - Widening the bounds (HP up to 1.2 kHz, order 4) does not improve the own kharaj (D 49.1 on that coarser grid, vs 48.7) and improves the iTanpura kharaj only from 88.2 to 83.9. The bounds are not what limits the fit.
- **What still does not match:**
  1. In the recordings the high partials (16–60) are present from the first frame. In the model they build up with the bloom: it has no attack buzz and no finger click.
  2. The jawari formant sits on partials 8–15 in the model, while the recordings have their energy in 4–7 plus a broad 16–60 shelf. The shimmer (partials 31–60) is now 3–4 dB (own kharaj) or 6–9 dB (iTanpura) too weak, against 10–19 dB with the 2-D preset and its default chain.
  3. After the bloom the model rings longer than the recordings. The recordings are app samples that probably end in fade-outs.

  An independent check with `reference.py` (colour-EQ'd, decay-weighted loss) rates the 1-D preset **better** than the 2-D fits, for these three reasons (see [Cross-check](#cross-check-with-referencepy)).

## References and pitch

The references are single plucks of 4–7 s, mono MP3 (`/home/user/tanpura-app/assets`; `TANPURA_REFS` overrides). They are analysed read-only. Only loudness-matched WAV copies are written, to `out/fit2d/wav/`, which is git-ignored.

| reference | file | f0 (Hz) | B | rise (dB) | t_max (s) | H4–10 share 0.09/0.3/1/2/3 s | H31–60 (dB) 1/3 s |
|---|---|---|---|---|---|---|---|
| own_kharaj (primary) | Mumu_Tanpura/Male/tt_CS3_kharaj.mp3 | 69.327 | ~0 | 13.9 | 1.01 | 0.57/0.82/0.92/0.87/0.69 | −17.4/−14.4 |
| itanpura_kharaj | iTanpura/Tanpura_CS3_Kharaj.mp3 | 69.344 | ~0 | 17.7 | 1.27 | 0.26/0.63/0.83/0.87/0.89 | −12.8/−12.2 |
| own_pa (primary) | Mumu_Tanpura/Male/tt_CS3_pa.mp3 | 103.962 | ~0 | 17.7 | 1.15 | 0.41/0.66/0.75/0.90/0.94 | −24.3/−21.1 |
| itanpura_pa | iTanpura/Tanpura_CS3_Pa.mp3 | 103.873 | ~0 | 14.5 | 1.17 | 0.22/0.43/0.79/0.89/0.96 | −22.2/−26.5 |
| own_sa (primary) | Mumu_Tanpura/Male/tt_CS3_sa.mp3 | 138.541 | ~0 | 17.3 | 0.71 | 0.58/0.32/0.58/0.78/0.66 | −24.3/−28.8 |
| itanpura_sa | iTanpura/Tanpura_CS3_Sa.mp3 | 138.530 | ~0 | 17.1 | 1.01 | 0.61/0.51/0.54/0.80/0.73 | −26.5/−32.3 |

**The recordings are harmonic at A440 pitch**, with f0 from sequential partial tracking (`fit_f0_B`). The task's 68.56 Hz with B = 4.4e-4 came from mis-numbered partials; `MODEL_2D_PLUCK.md` confirms this.

**f1 is tuned so that the model sounds at the reference pitch.** f1 = f0(own) × 2^(−5/1200), because the jawari contact sharpens the tone by about +5 cents:
- kharaj 69.127 Hz, Pa 103.662 Hz, Sa 138.141 Hz;
- the statics re-place the thread to graze at each pitch;
- the best fits sound −1.4 to +4.4 cents from their recording.

**Onsets are aligned the same way for both.**
- Recording: the first sample above −30 dB of the peak, minus 5 ms (quickfeat).
- Model: the end of the finger release minus 5 ms. The −30 dB threshold of the radiated model output falls 0.8 ms after the release.

## Model and parameters

**String:** `presets2d.preset2d(string, f1, target_gap[, thread_shift])`. The statics are cached on disk per geometry. On top of that, the fitted parameters below; all others are the preset's.

| parameter | meaning | bounds | preset |
|---|---|---|---|
| `pluck_deg` | finger stroke direction from the bridge normal towards the soundboard plane | 45–90° (CMA-ES: 80–90°) | 88° |
| `theta_nut` | nut principal-axis angle (polarisation coupling) | 0–45° | 30° |
| `detune_cents` | axis-2 frequency split (polarisation coupling) | 0.2–10 c | 3.5 c |
| `target_gap` | crest gap of the thread-only rest shape | −2…+6 µm | 0 |
| `thread_shift` | pressed-crest check: thread moved towards the bridge (−0.05, −0.1 mm) | ±0.15 mm | 0 |
| `pluck_disp` | static pad displacement along the stroke | 1–3 mm | 2 mm |
| `xp_frac` | pluck point from the bridge | L/3, mirror 2L/3 | L/3 |
| `delta_ve_x`, `Yb_x` | visco-elastic loss, body-drain mobility (× preset) | 0.5–2 | 1 |
| `kcz_x` | lateral thread stiffness / vertical (`kcz` in tanpura2d) | 0.3–3 | 1 |

**Radiation**, fitted in closed loop for every render and every reference:

| parameter | meaning | grid |
|---|---|---|
| `gz` | weight of the in-plane bridge force in the mono mix F_y + gz F_z | 0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.6, 1 |
| `hp_hz`, `hp_order` | radiation/microphone high-pass after the body (`body.apply_body(..., hp_hz, hp_order)`) | 60–400 Hz (9 log steps), order 1–3 |
| `mode_gain_db` | level of the body's explicit low modes (`tanpura_male`) | −6, 0, +6 dB |

**One render serves every radiation setting.** The body and the high-pass are linear: body(F_y + gz F_z) = body(F_y) + gz body(F_z), and the high-pass multiplies each partial's power by |H(n f0)|².
- Agreement with the exact `body.apply_body` render: the kharaj D differs by 0.5 or less (49.2 vs 48.7, 88.2 vs 88.2).
- The 8 s final renders, analysed exactly, reproduce every fit-stage D to 0.5.

**Negative crest gaps are infeasible for the bronze kharaj.** At 69.127 Hz there is no thread position that gives a crest gap of −1 µm with the string still resting on the back of the bridge (statics). The pressed side was therefore probed by moving the thread towards the bridge (`thread_shift`). A −1 µm gap is feasible for Pa.

## Features and distance

The features follow quickfeat / `validate2d_parity` G, extended:
- per-partial STFT envelopes: 180 ms Hann window, 20 ms hop, partials 1–60 within ±0.3 f0, numbered from each signal's own fitted f0 and B;
- analysed to 4 s, or to 3.89 s for the 4 s files.

| item | definition | tolerance | weight |
|---|---|---|---|
| rise_db | level(max) − level(first frame, 0–0.18 s) | 1.5 dB | 4 |
| t_max | time of the maximum | ×1.25 | 2 |
| t_m3 | first time within 3 dB of the maximum | ×1.3 | 2 |
| lev@t | total level re max at 0.09, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3 (4) s | 2 dB | 0.6 each |
| share@t | energy share of partials 4–10 at 0.09, 0.3, 1, 2, 3 s | 0.07 | 1.2 each |
| oct b@t | partials 1, 2–3, 4–7, 8–15, 16–31, 32–60 re total at 0.09, 0.3, 1, 2, 3 s | 3 dB | 0.25 each |

D = Σ w · min((Δ/tol)², 25). The bloom items (rise, t_max, t_m3, lev, share) carry 20 of the 27.5 weight units; "D bloom" in the tables is that part. About 1 per item is "just noticeable", so differences of a few units are not meaningful.

**Chaos noise is negligible.** Twin renders (pluck_disp × (1 + 1e-9)) give |ΔD| ≤ 0.1 on all strings.

**The landscape is rugged in the parameters themselves:** neighbouring points differ by 20–60 because the y↔z exchange pattern shifts features in time.

## Search and budget

1. **Stage 0.** The 1-D preset and the 2-D preset at the tuned f1.
2. **Stage 1, grid.**
   - Kharaj: 27 points, pluck 84/87/90° × coupling (30°/1.5 c, 30°/3.5 c, 45°/7 c) × gap −1/0/+2 µm; the −1 µm points are infeasible.
   - Pa and Sa: 8 points, pluck 87/90° × (30°/3.5 c, 45°/7 c) × gap 0/+2 µm.
3. **Stage 2, screening.** One parameter at a time, two steps each, from the grid winner (or the preset, if better). It includes the 2L/3 mirror check. For the kharaj a resumed run re-centred it on the improved point (kcz × 2.5), so it ran twice.
4. **Stage 3, CMA-ES on the own reference.** λ = 6, 4 generations, on the 5 most influential screening parameters.
5. **Stage 4, CMA-ES on the iTanpura reference.** 2 generations from its own best point.
6. **Pressed-crest check** at each best point.

In total there were 235 renders (5 s fits and 8 s finals): 50.8 min of string CPU, plus about 10 min of statics (28 geometries) and about 10 min of scoring. That is about 70 min of CPU, on at most 2 processes. The CMA-ES stages improved only the Sa: the kharaj and Pa optima are screening points. The best values are "good points", not converged optima.

## Results

### Best parameters

Each row gives the string parameters, then the radiation chain fitted for that reference.

| string / reference | pluck | nut θ / split | gap | pluck_disp | kcz × | losses | radiation: gz, HP, modes | D (bloom) |
|---|---|---|---|---|---|---|---|---|
| kharaj / **own** | 90° | 45° / 7.0 c | 0 | 2.7 mm | 2.5 | preset | 0.10, 400 Hz / 2, −6 dB | 48.7 (27.1) |
| kharaj / iTanpura | 90° | 45° / 3.5 c | 0 | 2.0 mm | 2.5 | preset | 0, 400 Hz / 3, −6 dB | 88.2 (39.7) |
| Pa / **own** | 88° | 15° / 3.5 c | 0 | 2.0 mm | 1 | preset | 0.05, 400 Hz / 1, −6 dB | 58.0 (15.8) |
| Pa / iTanpura | 88° | 30° / 3.5 c | 0 | 2.0 mm | 1 | delta_ve × 0.5 | 0.10, 196 Hz / 3, −6 dB | 70.4 (14.6) |
| Sa / **own** | 89° | 30° / 3.3 c | +1 µm | 2.05 mm | 2.03 | preset | 0.02, 249 Hz / 3, −6 dB | 90.5 (38.4) |
| Sa / iTanpura | same string parameters as own | | | | | | 0.10, 316 Hz / 3, −6 dB | 69.2 (21.2) |

Every best fit plucks at L/3. Full parameter dicts are in `final_<string>.json` (`best`).

The 2-D preset with the radiation chain fitted scores, own / iTanpura:
- kharaj 70.9 / 123.7;
- Pa 72.5 / 71.3 (the preset is already near-optimal for Pa);
- Sa 111.0 / 96.7.

**One radiation chain for the whole instrument** (`joint_chain.txt`). Microphone and body are shared by all strings, so one chain should fit all three own recordings: gz 0.05, HP 316 Hz / order 2, modes −6 dB. The own D values become 55 / 76 / 93 (kharaj / Pa / Sa), against 49 / 58 / 91 with per-string chains.

### Match tables

All rows are 8 s renders analysed with the exact `body.apply_body` chain; `final_<string>.txt` has all rows.

**Kharaj**

| signal | rise dB | t_max s | t_m3 s | H4–10 .09/1/3 s | H31–60 dB 1/3 s | D | D bloom |
|---|---|---|---|---|---|---|---|
| own recording | 13.9 | 1.01 | 0.49 | 0.57/0.92/0.69 | −17.4/−14.4 | | |
| 2-D best (own) | 13.8 | 1.03 | 0.67 | 0.81/0.89/0.65 | −21.8/−17.6 | 49.2 | 27.4 |
| 2-D preset, default chain | 11.0 | 1.39 | 0.85 | 0.48/0.64/0.52 | −31.3/−24.4 | 145.6 | 99.5 |
| 1-D preset, default chain | 4.8 | 0.73 | 0.15 | 0.62/0.79/0.76 | −23.3/−22.8 | 244.1 | 197.6 |
| iTanpura recording | 17.7 | 1.27 | 0.83 | 0.26/0.83/0.89 | −12.8/−12.2 | | |
| 2-D best (iTanpura) | 17.6 | 1.19 | 0.73 | 0.87/0.92/0.85 | −21.3/−18.3 | 88.2 | 39.7 |
| 1-D preset, default chain | 4.8 | 0.73 | 0.15 | 0.62/0.79/0.76 | −23.3/−22.8 | 361.3 | 293.7 |

**Pa**

| signal | rise dB | t_max s | t_m3 s | H4–10 .09/1/3 s | H31–60 dB 1/3 s | D | D bloom |
|---|---|---|---|---|---|---|---|
| own recording | 17.7 | 1.15 | 0.85 | 0.41/0.75/0.94 | −24.3/−21.1 | | |
| 2-D best (own) | 17.5 | 1.21 | 0.77 | 0.52/0.88/0.91 | −26.0/−23.8 | 58.0 | 15.9 |
| 2-D preset, default chain | 15.2 | 1.13 | 0.63 | 0.51/0.80/0.95 | −28.6/−33.6 | 97.6 | 30.0 |
| 1-D preset, default chain | 4.7 | 0.41 | 0.13 | 0.59/0.92/0.81 | −23.3/−20.7 | 343.4 | 319.6 |
| iTanpura recording | 14.5 | 1.17 | 0.55 | 0.22/0.79/0.96 | −22.2/−26.5 | | |
| 2-D best (iTanpura) | 14.3 | 1.11 | 0.63 | 0.43/0.77/0.94 | −23.9/−28.7 | 70.6 | 14.8 |

**Sa**

| signal | rise dB | t_max s | t_m3 s | H4–10 .09/1/3 s | H31–60 dB 1/3 s | D | D bloom |
|---|---|---|---|---|---|---|---|
| own recording | 17.3 | 0.71 | 0.51 | 0.58/0.58/0.66 | −24.3/−28.8 | | |
| 2-D best (own) | 18.7 | 0.85 | 0.55 | 0.47/0.57/0.94 | −25.7/−33.1 | 90.6 | 38.5 |
| 2-D preset, default chain | 11.4 | 0.71 | 0.45 | 0.30/0.73/0.91 | −30.3/−36.0 | 203.5 | 117.7 |
| 1-D preset, default chain | 2.4 | 0.31 | 0.09 | 0.59/0.94/0.95 | −36.1/−29.8 | 393.8 | 341.8 |
| iTanpura recording | 17.1 | 1.01 | 0.55 | 0.61/0.54/0.73 | −26.5/−32.3 | | |
| 2-D best (iTanpura) | 18.1 | 0.87 | 0.57 | 0.57/0.60/0.95 | −26.0/−33.0 | 69.3 | 21.3 |

**1-D with its own radiation chain fitted:** rise 3.7–4.6 dB and D 236–374. A filter on the 1-D output cannot make the bloom.

## Sensitivity: what makes the bloom

From the grid marginals and the one-at-a-time steps (`final_<string>.txt`, "sensitivity"). ΔD is the change of the own-reference distance.

1. **Pluck direction is the switch.**
   - Kharaj grid means: rise 6.2 / 11.3 / 12.4 dB at 84 / 87 / 90°.
   - From the screening centres, 90 → 86° costs 3 dB of rise and ΔD +40…+72 (kharaj), 88 → 84° costs ΔD +49 (Pa), and 90 → 86° costs ΔD +35 (Sa).
   - Physics: a stroke parallel to the soundboard puts the energy into z, which cannot hit the bridge. The radiated level starts low and grows as the nut coupling moves energy into y, where the jawari converts it into partials 4–10 and up. In the best kharaj fit, the z share of string energy is 1.00 / 0.82 / 0.66 / 0.75 / 0.92 at 0.05 / 0.5 / 1 / 2 / 3 s: the radiated peak coincides with the most energy in y.
2. **The coupling sets the timing.**
   - The y↔z exchange period of partial n is about 1/(n f1 Δ), with Δ = 2^(split/1200) − 1.
   - Kharaj grid: t_max 1.34–1.38 s at 1.5–3.5 cents, 0.88 s at 7 cents. Pa: 1.35 vs 0.92 s. Sa: 0.78 vs 0.51 s.
   - The split moves D by up to +50, the angle θ by up to +48. The two trade off, so (θ, split) is constrained jointly, not separately.
3. **The lateral thread stiffness (kcz) is the third lever.**
   - kcz × 0.4: ΔD +30…+55 on every string.
   - kcz × 2–2.5 helps the kharaj (iTanpura ΔD −45) and the Sa.
   - An anisotropic thread adds a partial-dependent y/z split, so it acts as extra coupling.
4. **The pluck point matters.** 2L/3 instead of L/3: ΔD +25…+92, with the peak 0.24–0.54 s too early. **L/3 is confirmed for all three strings.**
5. **Crest geometry has a modest effect.**
   - Gap +2 µm: ΔD +7…+55 (kharaj: −3 dB of rise).
   - Pa at −1 µm: ΔD −1.
   - Pressing the crest via the thread (−0.05 / −0.1 mm) is worse in 9 of 10 cases: ΔD +2…+65. The exception, −2.3 for the own Pa from a non-best point, is within the landscape's ruggedness.
6. **Pluck strength and string losses barely matter.** pluck_disp 1.3–2.7 mm: |ΔD| ≤ 15. delta_ve × 0.5–2 and Yb × 0.5–2: |ΔD| ≤ 3.5, except Yb × 2 (up to +13). This agrees with REFERENCE.md: the jawari masks the string loss.
7. **The radiation chain cannot make a bloom, but it is essential for the spectral balance.** On the 2-D preset, fitting the chain halves D (kharaj own 145.6 → 70.9). The same fit leaves the 1-D model at a 4–5 dB rise.

## Cross-check with reference.py

`reference.py` (the other workflow's analyser) compares the 8 s body renders with the recordings (`reference_compare.txt`, `reference_compare/<ref>__{2d_best,1d}/`). It uses heterodyne partial envelopes and per-partial T60, takes "bloom" as the rise of BI10 (partials ≥ 10 over < 10), and removes the colour difference with a smooth EQ first.

**Its decay-weighted loss rates the 1-D preset better than the 2-D fits:**

| reference | 2-D best fit | 1-D preset |
|---|---|---|
| own kharaj | 231 | 84 |
| iTanpura kharaj | 350 | 129 |
| own Pa | 275 | 79 |
| iTanpura Pa | 210 | 93 |
| own Sa | 172 | 70 |
| iTanpura Sa | 187 | 98 |

Its top items explain why:
- **BI10.** The recordings' BI10 is flat from the attack: own kharaj +0.4 dB bloom, peak at 0 s. In the recordings the level bloom happens within partials 1–9, and the high partials are present from the first frame. The 2-D model's high partials build up with the bloom (BI10 +11.8 dB, peak 0.57 s).
  - The quickfeat-style features show the same thing at 0.09 s: partials 16–31 are at −8 dB re total in the own kharaj recording and −18 dB in the model, partials 32–60 at −17 vs −25 dB.
  - The recordings also show an attack HF burst (> 4 kHz, first 20 ms) of −1 dB vs −14 dB in the model: a finger or nail click, or an initial bridge contact, that a pure sideways stroke lacks.
- **Decay.** After the bloom the model's partials decay 2–15 × more slowly than the recordings' (late T60 20–35 s vs 5–11 s).
  - Part of this is the recordings. They are app assets that end in what looks like a fade-out: the own kharaj falls from −11 dB at 4 s to −37 dB at 6.5 s, the own Pa from −14 dB at 4 s to −41 dB at 5.5 s, and the 4 s Sa files from about 3 s.
  - The fit window (≤ 4 s) mostly avoids the fades, except the last second of the three 4 s files. Within the window the late levels match: own kharaj −10.3 / −11.2 dB at 3 s.

**What the conflict means:** the 2-D fit matches the level bloom and the redistribution from partials 1–3 into 4–10 that the task targets. It does not match the immediate high-partial content of the attack or the decay of the high partials. A fit with `reference.py`'s loss would trade the bloom away, which is why the 1-D model, which buzzes from the start, scores better there.

## Physical plausibility of the fitted values

- **Pluck direction 88–90°: plausible.** The tanpura is played upright, with the finger pad stroking across the strings in the plane of the tabli. The fit cannot separate 88 from 90°, but anything ≤ 86° is clearly worse. A real stroke surely has a few degrees of normal component. The missing attack buzz (above) may be exactly that, or the finger click, which the model has no term for.
- **Polarisation split 3.3–7 cents: plausible, though not measured.**
  - It is 0.14–0.28 Hz at the fundamental: 7 c = 0.28 Hz on the kharaj, 3.5 c = 0.21 Hz on Pa, 3.3 c = 0.26 Hz on Sa.
  - The model's split is a constant frequency ratio, like a difference in effective vibrating length: 2.1–4.3 mm on 1.05 m.
  - That is comparable to the contact zone itself (crest 2 mm from the back face, thread at about 4.2 mm). The vertical polarisation ends on the thread and crest; the sideways one can be held at a different point by friction and the groove. So a few-mm difference in effective length is physically reasonable.
  - Fractions of a Hz are also the order usually attributed to anisotropic supports in other string instruments. That is general knowledge, not verified here.
  - **A 5–10 cent split would sit at the high end.** The 7 c of the own-kharaj fit is reached with θ = 45°, which is the maximum-coupling angle. The iTanpura kharaj and the Pa and Sa prefer 3.3–3.5 c.
  - A boundary-stiffness difference would give a split that shrinks with partial number, unlike the model's. That is an alternative this model cannot express.
- **Nut angle θ 15–45°: unconstrained physically.** It trades off against the split.
- **Lateral thread stiffness 2–2.5 × the vertical (kharaj, Sa): possible but unverified.**
  - The cotton thread is compressed by the string vertically and held laterally by friction and the groove the string cuts into it. Neither stiffness has been measured.
  - Because kcz ≠ kc adds a partial-dependent split, the fitted value may be standing in for a missing mechanism, such as a bridge face that also catches sideways motion (the bridge in the model acts only vertically).
  - Pa fits best with an isotropic thread.
- **In-plane radiation weight gz 0–0.1 (≤ −20 dB): plausible.** A soundboard is much stiffer in its plane than normal to it.
- **High-pass 250–400 Hz, order 2–3, body modes −6 dB: at the edge.**
  - A 40 cm tabli radiates inefficiently below ka ≈ 1, about 270 Hz, so some low cut is physical. Phone microphones add more below about 100–200 Hz.
  - A 400 Hz 2nd-order cut, however, attenuates partial 1 of the kharaj by about 30 dB. The real bloom moves energy out of partials 1–3 more completely than the model does (in the iTanpura kharaj partial 1 falls from −2 dB re total at 0.09 s to −23 dB at 1 s and −41 dB at 2 s). A strong static high-pass is the fit's way to compensate.
  - Read the chain as "microphone + radiation + unmodelled transfer", not as a body measurement.
- **Crest gap 0 to +1 µm, pluck displacement 2–2.7 mm, losses at the preset values: plausible and unremarkable.** A pressed crest is infeasible on this bridge for the bronze kharaj, and worse wherever it was tried.

## Limitations and next steps

- **The search is not converged.** The landscape is rugged in the parameters, though not chaotic: twins agree to 0.1. 4 + 2 CMA-ES generations only improved the Sa. The best points are good points from about 60–90 renders per string.
- **The attack lacks high partials** (see the [Cross-check](#cross-check-with-referencepy)). Candidates:
  - a small normal component of the stroke, with an attack-specific measure added to the distance;
  - a finger-release click;
  - a bridge that also acts on sideways motion.
- **The formant is in the wrong place.** The model's energy sits in partials 8–15, the recordings' in 4–7 plus 16–60. The next geometry parameters to fit, all statics keys (`presets.KHARAJ_GEOM`), are:
  - bridge curvature `c` (assumed 3 /m);
  - thread diameter `d_t` (assumed 0.2 mm);
  - thread position `x_c0`.
- **Late decay.** The model sustains longer after the bloom. Judge this only with recordings that are not faded (≥ 10 s; REFERENCE.md §1). With these app assets the decays are not trustworthy beyond about 4 s (3 s for the 4 s files).
- **Only one pluck per reference**, so pluck-to-pluck variation is unknown. The iTanpura samples may be processed (EQ or compression), which the radiation chain absorbs.
- **The coupling is a nut-only, frequency-ratio model.** The fitted (θ, split, kcz) are effective values. Only a measurement can separate them, e.g. the beating of a sideways-plucked string with the thread removed.
- **The body is still a one-way filter** applied to F_y + gz F_z.
- **Model code was not changed.** fit2d.py does not modify `tanpura1d.py`, `tanpura2d.py`, `presets.py`, `presets2d.py` or `body.py`. Their SHA-256 sums are the same before and after this work, and git shows no change to the two model files.

## Files

| file | content |
|---|---|
| `fit2d.py` | references, features, distance, closed-loop radiation chain, cached rendering, grid / screening / CMA-ES / pressed check, final renders, sensitivity, shared chain, `reference.py` cross-check, plots |
| `out/fit2d/wav/<string>_fit_<ref>_model_{dryFy,mono,body}.wav` | 8 s best fit per reference: dry bridge force F_y, dry mono F_y + gz F_z, through body and high-pass; −1 dBFS peak |
| `out/fit2d/wav/<ref>_reference_loudness_matched.wav` | the recording, K-weighted-loudness matched (first 4 s) to the body render; local and git-ignored, never commit it |
| `out/fit2d/wav/<string>_1d_preset_body_matched_to_<ref>.wav` | the current 1-D preset through the default chain, loudness-matched; peak-limited to −1 dBFS when needed (0–4.7 dB quieter) |
| `out/fit2d/png/<string>_partials_ref_vs_model.png` | per reference: partial-envelope images (recording, 2-D best), level, share of partials 4–10, partials 31–60 vs time (recording, 2-D best, 2-D preset, 1-D) |
| `out/fit2d/final_<string>.{txt,json}`, `fit_<string>.log`, `history_<string>.jsonl` | match tables, sensitivity, every evaluation |
| `out/fit2d/refs.txt`, `joint_chain.txt`, `reference_compare.txt` | reference features, shared instrument chain, `reference.py` cross-check |
