# Pa, Sa and kharaj: the four strings of the user's male tanpura in C#

`presets.py` now has presets for the other strings (`pa`, `sa`, plus `pa_steel`). `render_strings.py` renders single plucks of Pa, Sa and kharaj and a first four-string drone (Pa – Sa – Sa – kharaj) into `out/strings/`. All numbers below come from `out/strings/strings_results.txt` unless a section says otherwise.

**What this is, and what it is not.** It extends the validated kharaj model to the other strings, built exactly as the kharaj is. It does not address the user's remark that the kharaj does not yet sound like a real pluck. Matching real plucks needs the reference recordings (iTanpura Sa, kharaj and Pa, or the user's own instrument), which have not been uploaded yet. Sections 4 and 9 list what to compare first; the thread setting is the biggest lever.

## 1. The strings

All four strings share the speaking length (1.05 m), the bridge (parabola c = 3/m, crest 2 mm from the back face, 40 mm long, 0.5 mm back-edge radius), the thread (0.2 mm), the break angle (10°) and tailpiece (0.15 m), the finger pluck (12 mm pad at L/3, 40 ms pull, 2 ms release) and the loss model (`loss='valette'`, same coefficients). Only the gauge, the material, the pitch and the jawari setting (ψ and thread position, section 2) differ.

| | Pa (pancham) | Sa (jodi, ×2) | kharaj |
|---|---|---|---|
| preset | `pa` | `sa` | `kharaj` (unchanged) |
| note, f1 (contact off) | G#2, **104.1 Hz** (3/2 × 69.4) | C#3, **138.8 Hz** (2 × 69.4) | C#2, 69.4 Hz (user) |
| diameter | **0.55 mm (assumed)** | **0.50 mm (assumed)** | 0.70 mm (user) |
| material | **phosphor bronze (assumed)**; `pa_steel` = steel | **steel (assumed)** | phosphor bronze (user) |
| tension T | 99.43 N | 129.51 N | 71.39 N |
| stress T/A | 418 MPa | 660 MPa | 186 MPa |
| bending length l = √(EI/T) | 2.23 mm | 2.18 mm | 4.26 mm |
| inharmonicity B = π²EI/(TL²) | 4.5e-5 | 4.2e-5 | 1.6e-4 |
| modes below 20 kHz (highest) | 140 (19 848 Hz) | 115 (19 856 Hz) | 140 (19 753 Hz) |
| bridge contact points | 105 | 106 | 124 |
| modal T60 at f1 / 2f1 / 1 kHz / 2 kHz / 4 kHz (linear, s) | 21.4 / 12.0 / 5.2 / 3.0 / 1.1 | 13.3 / 8.3 / 4.7 / 3.1 / 1.4 | 40.9 / 20.9 / 4.5 / 1.5 / 0.4 |
| CPU, single pluck (s per s of audio) | 1.64 | 1.36 | 1.91 |

`pa_steel` (0.55 mm steel): T = 88.01 N, l = 3.20 mm, 124 modes, ψ = 5.1236°, thread at 3.9992 mm. It was not rendered.

### Gauges and materials: assumptions to replace

The user's Pa and Sa gauges are not known; a micrometer reading of the real strings should replace these values (`preset('sa', d=..., f1=...)`; ψ then needs `design_instrument`, section 2). The defaults come from a web search. Only search-result excerpts were readable: the pages themselves could not be fetched from here.

- **Pyramid 706/4 "Tanpura strings gents"** ([Thomann listing](https://www.thomann.co.uk/pyramid_706_4_tanpura_strings_gents.htm)) lists 0.55 / 0.50 / 0.50 / 0.70 mm. The listing does not say which gauge goes on which string; the order Pa / Sa / Sa / kharaj is inferred, and the 0.70 mm matches the user's kharaj. A buyer's review on the same listing says brass for kharaj and pancham, steel for the jodi.
- **[sangtar.com (2007)](https://www.sangtar.com/2007/05/tanpura/)**, a 60" male tanpura: string 1 bronze .022" (0.56 mm), strings 2 and 3 steel .020" (0.51 mm), string 4 brass .028" (0.71 mm).
- **Materials.** Every source has steel for the jodi.
  - The pancham is brass or bronze in the sangtar set, in the Pyramid review and on [sikhiwiki](https://www.sikhiwiki.org/index.php/Tanpura). One other sikhiwiki page says steel.
  - [Wikipedia](https://en.wikipedia.org/wiki/Tanpura) says the octave strings are usually steel and the tonic, 4th or 5th strings brass or bronze.
  - So `pa` is bronze, with the user's kharaj material constants (ρ 8800, E 1.1e11). Brass (ρ ≈ 8500, E ≈ 1.0e11) is within the assumption. `pa_steel` is the alternative.
- The two sets agree to 0.01 mm. The resulting stresses are plausible: Sa at 660 MPa is about 30 % of music-wire strength, and Pa at 418 MPa is about half of spring-temper phosphor bronze.

### Tuning

Each string's nominal f1 is a **pure interval on the user's 69.4 Hz kharaj**: Sa = 2 × 69.4 = 138.8 Hz and Pa = 3/2 × 69.4 = 104.1 Hz. Then 2 Pa = 3 kharaj and 4 Pa = 3 Sa, which is how a tanpura is tuned by ear. The equal-tempered A440 values (138.59 Hz and 103.83 Hz) are 2.6 and 4.5 cents flat of these and would beat against the kharaj's harmonics. The user's kharaj (69.4 Hz) is itself not A440 C#2 (69.30 Hz).

## 2. Bridge and thread: each string gets its own jawari setting

A bridge filed for one string does not graze with another. Each string's bending stiffness lifts it over the crest by a different amount (+471 µm for the kharaj; see STATICS.md). So each string is handled **as the kharaj preset is**:

1. The speaking-length angle ψ of its track is designed so that its thread just grazes at x_c0 = 4 mm at the tuned tension (`design_instrument`; stored in `PSI_STRING`). ψ is equivalent to the tilt the jawari is filed to under that string.
2. The thread is auto-placed at the grazing root (`target_gap = 0`).
3. The run starts from static equilibrium, and the statics are re-solved at the tuned tension (fixed point, |dT|/T < 1e-8).

On a real tanpura each string does have its own jawari: the bridge surface under each string is filed separately and each string has its own thread.

| | ψ (°) | thread x_c (mm) | grazing roots (mm) | crest touch x (mm) | thread preload (N) | crest gap |
|---|---|---|---|---|---|---|
| Pa | 4.5361 | 4.0004 | 2.734, 4.000 | 15.22 | 14.17 | 0.0000 µm |
| Sa | 4.5052 | 3.9997 | 4.000 | 15.13 | 18.33 | 0.0000 µm |
| kharaj | 5.7911 | 3.9997 | 4.000, 4.929 | 18.38 | 13.92 | 0.0000 µm |

**Common-bridge alternative (not the default).** Suppose the Pa and Sa tracks had the kharaj's tilt (ψ = 5.79°). With the thread at 4 mm they would then be pressed about 260 µm into the bridge. They graze only with the thread at 8.82 mm (Sa) or 8.77 mm (Pa), where the crest touch point is 18.96 mm and the preload is 12.90 N / 9.90 N; T is 128.38 N / 98.53 N because the thread spring moves with it. Both designs are self-consistent; which one the user's instrument resembles is a measurement question. Use `preset('sa', psi_deg=presets.PSI_DESIGN['bronze'], x_c0=8.8e-3)`.

## 3. Checks

| | Pa | Sa | kharaj |
|---|---|---|---|
| f1, contact off (0.5–3 s, 3–6 s) | 104.1000 Hz (−0.000 c) both | 138.8000 Hz (−0.000 c) both | 69.4000 Hz (−0.000 c) both |
| f1 with jawari, single pluck, 0.5–3 / 3–6 / 6–11.5 s | +3.70 / +2.72 / +2.51 c | +2.90 / +2.51 / +2.39 c | +6.03 / +4.85 / +4.40 c |
| contact (share of samples) | 78.3 % | 71.4 % | 66.0 % |
| max penetration | 261 nm | 288 nm | 229 nm |
| Newton: mean / max iterations, failures | 1.31 / 8, 0 | 1.28 / 9, 0 | 1.61 / 8, 0 |
| bilateral thread: max dynamic force vs preload | 1.25 vs 14.17 N, holds | 1.49 vs 18.33 N, holds | 1.28 vs 13.92 N, holds |
| static_init active points | 0 (grazes) | 0 | 0 |
| onset, first 1 ms re file peak | −111 dB | −139 dB | −112 dB |

- The kharaj single pluck reproduces `render_demo.py` exactly: the same CPU, penetration, contact and pitch numbers.
- `preset('kharaj')`, `preset('kharaj_steel')`, `preset('dafx16_c3')` and `preset('kharaj', thread_shift=1e-4)` are unchanged: their full parameter dicts, `meta` included, are identical (arrays byte-identical) to those of the code before this change.
- `tanpura1d.py` was not edited (same SHA-256).

## 4. Single plucks (12 s, the same pluck on every string)

Every string gets the identical finger pluck: 2 mm static pad displacement at L/3. Through the body (0–2 s), the three strings then come out equally loud in RMS (Pa −0.2 dB, Sa +0.6 dB re kharaj), and Pa and Sa are +4.6 and +4.9 dB louder A-weighted.

The dry columns are the bridge force; the "body" column is `apply_body('tanpura_male')`. Buzz is metrics.py's 2–8 kHz energy relative to the energy below 1 kHz.

| | Pa | Sa | kharaj |
|---|---|---|---|
| buzz 0.3–3 s / 3–8 s, dry | −20.3 / −24.2 dB | −17.8 / −24.3 dB | −22.1 / −26.3 dB |
| buzz 0.3–3 s / 3–8 s, body | −21.3 / −24.3 dB | −23.4 / −27.8 dB | −24.7 / −28.3 dB |
| hf_frac 0.3–3 s, dry | 0.036 | 0.093 | 0.015 |
| spectral centroid 0.3–3 s, dry / body | 1042 / 883 Hz | 1083 / 890 Hz | 364 / 357 Hz |
| frame level at 1 / 2 / 4 / 8 s, dry | −10 / −20 / −37 / −67 dB | −13 / −27 / −48 / −90 dB | −5 / −9 / −19 / −34 dB |
| **fundamental T60, with jawari (contact off)** | **7.4 s (21.4)** | **5.6 s (13.3)** | **15.3 s (40.9)** |

The T60 row comes from a side check: 8 s runs with the same presets, fitting the decay slope of the first partial over 1–7 s. The script for it is not in the repo; reproduce any row with `T1.simulate(preset('sa', thread_shift=2e-5, dur=8.0))`.

**Main finding: at grazing the jawari drains the thinner strings fast.**
- With the thread exactly grazing, the bridge contact cuts the sustain of every string by a factor of 2.5–2.9.
- All partials then decay together: the contact locks them and keeps feeding energy to high, quickly damped modes.
- For the kharaj that still leaves T60 ≈ 15 s. For Sa it leaves 5.6 s: the Sa pluck is 90 dB down after 8 s.
- A real Sa string on a well-set tanpura probably rings longer than this. That is to be checked against the references.

The trade-off depends on the thread setting. The side check (8 s single plucks, `thread_shift` from grazing; positive values open the gap for Pa and Sa) gave:

| string, thread shift | crest gap | contact | T60 of h1 | buzz 0.3–3 s | buzz 3–7.5 s |
|---|---|---|---|---|---|
| Sa −0.05 mm | −1.02 µm (pressed) | 91 % | 10.0 s | −14.9 dB | −70.6 dB |
| Sa −0.02 mm | −0.41 µm | 88 % | 8.5 s | −13.8 dB | −66.4 dB |
| **Sa 0 (preset)** | 0 | 74 % | **5.6 s** | −17.8 dB | **−24.2 dB** |
| Sa +0.01 mm | +0.21 µm (open) | 26 % | 10.8 s | −19.6 dB | −81.5 dB |
| Sa +0.02 mm | +0.42 µm | 21 % | 11.0 s | −21.8 dB | −94.8 dB |
| Sa +0.05 mm | +1.07 µm | 16 % | 12.6 s | −26.8 dB | −112.8 dB |
| Sa +0.10 / +0.20 mm | +2.2 / +4.5 µm | 13 / 8 % | 13.4 / 13.3 s | −32.6 / −40.6 dB | −117 / −121 dB |
| **Pa 0 (preset)** | 0 | 76 % | **7.4 s** | −20.3 dB | −24.4 dB |
| Pa +0.02 / +0.05 mm | +0.40 / +1.02 µm | 35 / 27 % | 12.0 / 15.9 s | −22.3 / −24.2 dB | −67.3 / −86.1 dB |
| **kharaj 0 (preset)** | 0 | 64 % | **15.3 s** | −22.1 dB | −26.2 dB |
| kharaj +0.02 / +0.05 mm | −0.27 / −0.66 µm (pressed: the kharaj's root has the opposite slope) | 65 / 68 % | 14.1 / 12.9 s | −22.0 / −21.5 dB | −24.0 / −22.9 dB |

For Sa and Pa there is no thread setting in this model that gives both a sustained buzz and a long sustain:
- **At grazing:** the jawari lasts, but the string dies in about 6–7 s.
- **0.2 µm open:** the string sustains, but the jawari is gone after about 3 s.
- **Pressed:** strong early buzz, then permanent contact and no buzz.

A real tanpura presumably does both. Candidate causes, none tested here:
- the assumed bridge (a parabola; the real filed profile is unknown);
- the one polarisation of the 1-D model;
- high-frequency string damping that may be too strong (`delta_ve`; energy pumped up by the contact is lost before it can return);
- the soundboard-coupled bridge, which is missing.

Why the kharaj is much less affected is not established; it has twice the bending length (4.3 mm) and the lowest tension.

## 5. The four-string drone

- **Pattern:** Pa – Sa – Sa – kharaj with gaps 0.8 / 0.8 / 0.8 s, then 1.0 s after the kharaj before the next Pa. The cycle is 3.4 s, so the mean pluck interval is 0.85 s. There are 8 cycles; the last kharaj pluck is at 26.2 s.
- **Timing and level:** the timing has ±15 ms uniform jitter and the level ±0.75 dB, both seeded (seed 7). All of these are constants at the top of `render_strings.py`.
- **Four separate simulations:** Pa, Sa1, Sa2 and kharaj, 30 s each. Each is a running string re-plucked at its own times, by the `render_demo.py` method: copies of the finger-pluck force history in one excitation signal, so the string keeps its memory. Sa1 and Sa2 use the same preset and differ only in their pluck times and jitter.
- **One body:** the four bridge forces are summed at the model rate (176.4 kHz), decimated, faded out over 29–30 s with a 1.5 s silent tail, and passed through **one** `tanpura_male` body filter. The outputs are mono and stereo + room (0.3).
- **No coupling between strings.** Each string sees a rigid bridge. There is no energy exchange through the bridge or soundboard, so no sympathetic resonance, for example Sa ringing when the kharaj is plucked.
- **Balance: equal finger work per pluck (my choice).** Every pluck does the same work, ½·F·d = 0.624 mJ, the kharaj's 2 mm pluck. The pad displacement therefore scales as √(compliance): Pa 1.70 mm, Sa 1.49 mm, kharaj 2.00 mm. Through the body over 3–26 s this gives:
  - RMS re kharaj: Pa −1.0 dB, Sa1 −0.5 dB, Sa2 −1.1 dB;
  - A-weighted re kharaj: Pa +3.6 dB, Sa1 +3.5 dB, Sa2 +2.8 dB.

  `BALANCE = 'equal_disp'` gives every string 2 mm instead.
- **Tuning by ear to the kharaj (`TUNING = 'by_ear'`):**
  - **Reference:** the kharaj stays the validated preset (contact-off 69.4 Hz) and is the reference. It is not retuned, because its grazing thread position is very sensitive to tension: lowering it by 8 c re-grazes the thread at 4.26 mm instead of 4.00 mm, which would be a different jawari setting.
  - **Pass 1:** the whole drone at nominal tuning. The sounding pitches are Pa +3.28 c, Sa1 +2.96 c, Sa2 +2.84 c and kharaj +4.79 c, so the intervals are 1.5–2 c narrow (Sa/kharaj −1.83 and −1.95 c, Pa/kharaj −1.51 c).
  - **Pass 2:** Pa and Sa are re-run with contact-off f1 = 104.1909 Hz and 138.9517 Hz. The statics are re-solved, and the thread re-grazes at 4.0100 mm and 4.0108 mm.
  - **Result:** the intervals are pure within 0.2 c: Sa1/kharaj +0.06 c, Sa2/kharaj −0.12 c, Sa1/Sa2 +0.18 c, Pa/kharaj +0.07 c, Sa/Pa −0.01 c. The absolute pitch is the kharaj's sounding pitch, 69.59 Hz, which is +4.8 c re 69.4 Hz.
- **Pitch measurement in the drone.** Pitch is measured per pluck interval, by complex demodulation of each voice's own nut force (`seg_pitch`). The per-interval values agree within ±0.5 c. A single long-window FFT peak is unusable here: periodic re-plucking turns each partial into a comb of lines 0.29 Hz apart. An earlier version of the script used it, and its by-ear correction made the tuning worse (up to −4 c); that is why `seg_pitch` exists.
- **Per-voice checks:**

  | voice | contact | max penetration | max thread force vs preload | Newton failures |
  |---|---|---|---|---|
  | Pa | 69 % | 239 nm | 1.14 vs 14.16 N | 0 |
  | Sa1 | 66 % | 266 nm | 1.20 vs 18.32 N | 0 |
  | Sa2 | 63 % | 256 nm | 1.18 vs 18.32 N | 0 |
  | kharaj | 53 % | 258 nm | 1.35 vs 13.92 N | 0 |

- **Drone metrics (body, mono):** buzz −20.9 dB on average over 3–26 s, ranging from −34 to −12 dB within each cycle; frame level over 1–29 s has a median of −5.8 dB re its maximum, never below −20 dB; L/R correlation of the stereo + room file is 0.42 broadband and 0.82 below 300 Hz.

## 6. Files (`jawari/out/strings/`, git-ignored)

| file | content |
|---|---|
| `{pa,sa,kharaj}_single_dry.wav` | 12 s single pluck, bridge force (no body) |
| `{pa,sa,kharaj}_single_body.wav` | through `tanpura_male`, mono |
| `{pa,sa,kharaj}_single_body_stereo_room.wav` | two mic positions + light room (0.3) |
| `drone_dry.wav`, `drone_body.wav`, `drone_body_stereo_room.wav` | the 31.5 s four-string drone |
| `*.png` | `metrics.spectrogram_png` of each file |
| `strings_results.txt` | every number in this note |
| `cache/` | pickled simulation results, about 0.9 GB. Delete it freely: `--no-cache` or a deleted cache recomputes everything. |

The levels within each group are kept relative:
- the nine single-pluck files share one write gain, so the string-to-string levels are real;
- the three drone files share another gain;
- the loudest file of each group peaks at −1 dBFS;
- `apply_body` RMS-matches its output to its input over the first 2 s.

## 7. Timings

Times are wall-clock with 2 worker processes on this 4-CPU container; another agent was running at the same time.

| step | wall |
|---|---|
| presets: statics and tuning, 3 strings | 36 s |
| single plucks 3 × 12 s plus contact-off references 3 × 6 s | 41 s |
| drone pass 1, 4 voices × 30 s | 90 s |
| re-solved Pa and Sa statics | 14 s |
| drone pass 2, Pa, Sa1, Sa2 × 30 s | 93 s |
| body, metrics and spectrograms | about 15 s |
| **total without cache** | **about 4.8 min** (about 8 min of CPU) |

Per-voice CPU in the drone, in seconds of CPU per second of audio: Pa 1.7–1.9, Sa 1.4–1.6, kharaj 2.0. With the cache, re-running the script takes about 15 s.

## 8. Limitations

1. **No coupling between strings.** There is no sympathetic resonance and no exchange of energy or polarisation through the bridge or soundboard. In a real tanpura the open strings excite each other; that is part of the drone's "cloud".
2. **Identical pluck.** The pluck has the same shape on every string: pad width, 40 ms pull, 2 ms release, at L/3. In the drone only the amplitude differs (equal work), plus the jitter. Players commonly pluck Pa with the middle finger and the others with the index finger, with a different touch.
3. **No finger contact on re-pluck.** The finger is a prescribed force, so it does not damp the vibrating string when it touches it again.
4. **Assumed gauges and materials** for Pa and Sa (section 1), and the Pa material is uncertain. The bridge (parabola), thread (0.2 mm), break angle (10°) and the per-string ψ are assumed too, as for the kharaj.
5. **Fast decay at grazing** for Sa and Pa (section 4). The preset follows the kharaj's design rule (thread exactly grazing). Whether real strings sit there, slightly open or slightly pressed is unknown until the references are compared. The 1-D model has no setting that gives both a long-lasting buzz and a long sustain for these strings.
6. **The two Sa strings are identical in parameters.** There is no deliberate detuning between them. They differ by about 0.2 c only because of pluck timing and the amplitude-dependent jawari pitch.
7. **One-way body** with unmeasured `tanpura_male` values (A0 65 Hz, T1 120 Hz, between Pa and Sa). It was designed with the kharaj in mind; its colouring of Pa and Sa was not examined separately.
8. **1-D:** one polarisation; the pluck acts in the bridge-normal direction only.
9. **Pitch:** jawari makes each string +2.4 to +6 c sharp. The drone compensates Pa and Sa relative to the kharaj, but the absolute pitch is +4.8 c over the user's 69.4 Hz.

## 9. Next steps once the references are uploaded

- Measure the user's Pa and Sa gauges, then design ψ with `presets.design_instrument(material, 4e-3, geom=dict(d=..., f1=...))`, add it to `PSI_STRING` and re-render.
- From the iTanpura or recorded Sa, Pa and kharaj plucks, measure per string:
  - the fundamental T60;
  - buzz over time;
  - the start frequency and sweep rate of the formant;
  - the sounding pitch offset.

  Then place each string on the thread-setting table in section 4. That table is the most direct knob, and it shows that the current grazing setting probably over-damps Sa.
- Then fit the loss terms (`delta_ve`, the body-drain mobility `Yb`). Add inter-string coupling through a bridge or soundboard admittance only after that.
