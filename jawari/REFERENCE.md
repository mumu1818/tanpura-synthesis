# Reference recordings: analyse, compare, fit (`reference.py`)

`reference.py` measures one recorded tanpura pluck feature by feature, compares it with a model render, and fits a few model parameters to it. It works on a real instrument or on a sample app such as iTanpura. `validate_reference.py` checks it on synthetic signals and on model renders; the results are in [Validation](#validation).

```bash
cd jawari
python reference.py analyse ../references/my_kharaj_1.wav --string kharaj
python reference.py compare ../references/my_kharaj_1.wav --render kharaj          # renders the preset now (8 s)
python reference.py compare ../references/my_kharaj_1.wav out/demo/kharaj_jawari_body.wav --string kharaj
python reference.py fit ../references/my_kharaj_1.wav --string kharaj \
       --params target_gap,delta_ve,Yb,xp --budget 40 --jobs 3                       # ~8 min
```

Every command writes to `jawari/out/reference/<name>/` (git-ignored): `report.txt`, `features.json` and the figures listed under [Outputs](#outputs).

## 1. How to record a reference

**Recordings stay private.** iTanpura audio is copyrighted and your own recordings are private, so never commit them. Put them in `references/`; everything in it except `.gitkeep` is git-ignored. `jawari/out/` is git-ignored too.

### Your own instrument

- **One string at a time.** Damp the other three strings lightly with a cloth or the other hand so they do not ring in sympathy. The analyser measures one string; sympathetic strings add partials at 1.5× and 2× the kharaj frequency, exactly on top of its own.
- **One pluck per file**, or several plucks at least 12 s apart (pick one with `--pluck k`, counted from 0). Pluck the way you normally play: finger pad, at your usual point. Note where you plucked (fraction of the length from the bridge).
- **At least 10 s**, ideally until the sound has died away. Start recording 1 s before the pluck: the analyser takes the noise floor from that silence. Without it, decay fits cannot tell signal from floor.
- **Quiet room.** No fan, fridge or traffic hum; mains hum at 50/100 Hz sits right on the kharaj's partials.
- **Phone 30–50 cm from the gourd (tumba)**, pointing at the soundboard (tabli) near the bridge, at the same height. Don't move it during the recording. Use the same position for every string.
- **No processing.** Use a recorder app that writes WAV or FLAC, 44.1 or 48 kHz, 16 or 24 bit, with automatic gain, noise suppression and "enhance" turned off:
  - iPhone Voice Memos: set Audio Quality to Lossless and turn Enhance Recording off.
  - Android: use a recorder app with a raw/unprocessed or "music" mode.
  - Peaks should stay below about −6 dBFS, with no clipping.
  - `.m4a`, `.mp4`, `.mov` (voice memos, screen recordings) are decoded with `ffmpeg` automatically when it is installed (it is in this container); otherwise convert first: `ffmpeg -i in.m4a out.wav`.
- **Tune with a tuner** and write down the reading. `--f0` takes the nominal pitch if it isn't the kharaj/Pa/Sa default of 69.4 / 104.1 / 138.8 Hz.
- Optional: after ~10 s, **stop the string with your hand**. If the file contains this damping, the analyser estimates the decay of what follows, room plus body ringing (`abrupt end` line).
- Optional: 3–5 plucks per string, as separate files, show how much pluck-to-pluck variation there is.

### iTanpura (or another sample app)

- **Record the app's output digitally**, not through the phone speaker. Use the device's internal-audio screen recording, an audio interface, or the app's own export if it has one. A speaker-plus-microphone path adds its own colour, which the analyser cannot tell apart from the instrument.
- **One string alone if the app allows muting the others.** Otherwise use the slowest tempo, so each string rings as long as possible before the next pluck.
- **The analyser handles overlapping strings:**
  - It detects every pluck in the file (log-band spectral flux; the report's `plucks detected at` list).
  - It analyses the pluck you select (`--pluck k`) only until the next pluck is detected.
  - Strings still ringing from earlier plucks are measured as a per-partial floor in the 0.5 s before the onset, and partials within 20 dB of that floor are flagged (`contaminated`).
  - If partials ≤ 12 lie within 15 dB of the floor, f0 and B are refit without them.
  - Only partials above floor + 10 dB enter the decay fits.
- **Expect less from a pluck inside a drone cycle:**
  - The window is short, typically 1–1.5 s. Late decay times of the low partials, which take tens of seconds, are then extrapolations. `features.json` gives their standard errors (`T60_late_se_groups`).
  - Sa partials coincide with the kharaj's even partials and Pa's with its multiples of 3. Sa/Pa "contaminate" them, and frequencies within about 1 cent cannot be separated.
  - **Best: the first pluck of the file**, i.e. a cycle that starts with the string you want, preceded by silence.
  - **Check the detected onsets** (`plucks detected at` in the report). Inside a dense 4-string drone a new pluck raises the band-averaged flux by only 0.8–2 dB, against ≤ 0.6 dB for the jawari's own fluctuations. On the model drone of `render_strings.py` the detector found 28–30 of 32 plucks, with no false ones. A missed pluck means the analysis runs into it. Pass the pluck times yourself with `--onsets 0.10,0.90,1.69,...`; each one is still refined to the sample, inside the band where that pluck rose most.
- Name the files `references/itanpura_kharaj_1.wav`, `references/mine_pa_2.wav`, and so on. The name becomes the output folder name.

## 2. What the analyser measures

All times are relative to the **onset anchor**, which is t = 0. The anchor is the first time the high-passed (> max(100 Hz, 1.5 f0)) 1 ms envelope comes within 6 dB of the attack peak, found within 40 ms of the start. It is interpolated to a fraction of a sample and moves exactly with the signal (validated to 0.000 samples). Before thresholding, the power of whatever was already sounding is subtracted. When a pluck barely rises above strings that are still ringing, the envelope is taken in the band where that pluck's onset flux rose most.

Levels are normalised so that the power of the first second after the onset is 0 dB.

| report line / key | meaning |
|---|---|
| `onset`, `start`, `plucks detected`, `analysed` | anchor time; where the envelope first leaves the floor (causal); every detected pluck; seconds analysed, "(cut at the next pluck)" if truncated |
| `noise floor`, `contaminated` | where the per-partial floor came from (pre-onset noise, or only a 100 dB range for clean renders); partials within 20 dB of it |
| `f0` | robust fit of f_n = n f0 √(1 + B n²) to partials ≤ 12 over 0.1–2.5 s (Blackman-Harris, 8× zero-padded peaks); cents re the nominal. Also given: partial 1 alone, and all partials |
| `B` | inharmonicity of partials ≤ 12, of partials > 12, and of all partials. A plain stiff string has one B, e.g. 1.6e-4 for the 0.7 mm bronze kharaj. A grazing jawari **locks the upper partials harmonic** (B > 12 ≈ 0), which is a jawari signature in itself |
| `glide`, `drift` | f0(t) is the power-weighted median of partials 4–12, each de-stretched. Partials 1–3 are left out because phones roll them off and body modes pull them: with partials 2–8 the same kharaj render read −3.3 to +6.3 c of glide depending only on the body preset, with 4–12 the spread is 1.6 c. glide = mean over 30–150 ms minus mean over 0.5–1.5 s (+ = starts sharp, as tension modulation does); drift = robust slope after 1 s, in cents/s |
| `T60 late / early by partial group` | per partial, from the filter-bank envelope (below): **early** = Huber line fit from the partial's peak over ≤ 0.5 s / 15 dB (EDT-like); **late** = from 1 s after the peak (or 15 dB down, if sooner) to 10 dB above the floor. Group medians over partials 1, 2–3, 4–6, 7–10, 11–16, 17–25, 26–40, 41–60, 61–100 |
| `pluck comb` | early (0–40 ms) partial levels fitted as \|sin(nπβ)\| (notch depth free) × a smooth trend: **β, or 1 − β**, of the length from the bridge (the spectrum cannot tell which); notch list; "evidence" = how much the comb improves the fit in dB (< 1.5 dB = no clear comb); excitation tilt in dB/oct |
| `jawari: BI10` | **pitch-relative buzz index** = energy of partials ≥ 10 over partials < 10 (also BI5, BI20 in `features.json`). start = 20–60 ms; **bloom** = rise after the attack; onset = 10 % of the rise; peak time; plateau = time within 3 dB of the max; **persists to** = last time within 10 dB of the max |
| `formant centre / peak / upper edge / e95` | the buzz band of partials ≥ n_lo (n_lo = 300 Hz / f0; 4 for the kharaj): loudness-weighted (P^0.3) mean partial; the most prominent hump of the 1/3-octave-smoothed profile ("nan" when there is no hump, e.g. a flat buzz plateau); highest partial within 30 dB of the smoothed maximum; partial below which 95 % of the loudness lies. Also in Hz in `features.json` |
| `partial-octave levels` | level of partials 1, 2–3, 4–7, 8–15, 16–31, 32–63, 64–127 as dB of the whole tone at each time. The most robust compact picture of how the spectrum's shape evolves |
| `attack` | 10–90 % rise time of the broadband envelope (smoothed over a quarter period, so ~T0/4 = 3.6 ms is the resolution for the kharaj); **HF burst** = energy above max(4 kHz, 40 f0) in the first 20 ms vs 20–200 ms (a click or finger noise shows as positive dB); **aperiodicity** = residual of a one-period-delay prediction below 12 f0 (first 30 ms vs 0.3–0.6 s); spectral centroid in f0 units, 0–50 ms vs 50–300 ms |
| `stereo` (stereo files) | L/R correlation, side/mid ratio, coherence per octave |
| `abrupt end` (`room`) | if the file contains an abrupt end (string damped by hand, or a fade): T60 of what follows (room plus body ringing; Schroeder curve −5 to −25 dB) |

**Partial envelopes.** The core of the analyser is a heterodyne filter bank in the frequency domain:
- One band per partial. Band edges sit midway between neighbouring measured partials, with power-complementary cross-fades, so the bands tile the spectrum.
- Each band is shifted to baseband and decimated, so |z_n(t)| is the amplitude of *everything* in partial n's band. That makes it robust to the jawari's spreading of energy around the partials.
- The time resolution is about 1/f0 (15 ms for the kharaj), with 20 ms smoothing.
- The instantaneous frequency comes from the band phase.

### Outputs

| file | content |
|---|---|
| `report.txt` | ranked mismatches (compare) and the summaries above |
| `features.json` | every scalar above, plus the 1/3-octave EQ curve (compare) |
| `spectrograms.png` | side-by-side spectrograms, same normalisation |
| `partial_envelopes.png` | partial number × time images (reference, model + EQ), the loudness centre and 95 % line, and the **difference image** model − reference (orange = model louder), sampled at 0.05…11 s |
| `decay.png` | late and early T60 against partial number, with group medians |
| `pitch.png` | f0 tracks in cents; partial frequency deviations from n·f0, with the fitted stretch |
| `pluck.png` | early partial levels with the fitted comb (dotted) |
| `jawari.png` | BI10, buzz-band centre and 95 % edge, and the six partial-octave level tracks |
| `attack_colour.png` | attack envelopes; long-term partial-peak envelopes; the colour (EQ) curve reference/model |

## 3. Comparing with the model

`compare(ref, model)` analyses both signals identically; the model's analysis is cut to the reference's duration. It returns the mismatches ranked by **severity = |difference| / tolerance** (`TOL` in `reference.py`). About 1 is "just noticeable", above 3 is large. The tolerances are engineering choices, not measured JNDs: 5 cents f0, 3 cents glide, 30 % in T60 (|log2 ratio| / log2 1.3), 3 dB on BI and partial-octave tracks, 25 % on formant position, and so on.

**Colour vs string.** First, the long-term spectral balance is matched with a smooth EQ, ref/model:
- It is computed from the time-mean partial levels over the first 4 s, Gaussian-smoothed over ¼ octave.
- It is applied to the model's partial envelopes.
- The level-shape items (jawari, comb) are computed after this EQ. A body or microphone colour difference therefore shows up once, as the `colour` item, and is not blamed on the jawari.
- The EQ curve (`attack_colour.png`, `features.json → eq_third_octave`) is the correction a body filter would need. It also contains any *constant* difference in jawari strength, which the item text says.

Mismatch sentences look like:
- "model partials 11-16 decay 2.0x faster (late T60 1.4 s vs 2.7 s in the reference)"
- "jawari peaks 1.16 s earlier in the model (0.57 s vs 1.73 s; BI10)"
- "reference shows a pluck comb (notches at [3, 6, 9] -> pluck at 0.333 or 0.667 of the length from the bridge) that the model's jawari fills in"
- "attack HF burst …: +12 dB in the reference vs −2 dB in the model -> the reference has a click / finger-noise transient"

## 4. Fitting

```python
from reference import fit
r = fit("references/mine_kharaj_1.wav", string="kharaj",
        params=["target_gap", "delta_ve", "Yb", "xp"], budget=40, jobs=3, dur=5.0, body="tanpura_male")
r["best"], r["best_loss"], r["dir"]     # best_model.wav, history.csv, fit.json, best_vs_ref/ report
```

**Search space.** Each candidate is `presets.preset(string, **params)` → `tanpura1d.simulate` (`dur` seconds) → `common.to_audio` → `body.apply_body`, analysed and scored like `compare`. The parameters are interpretable (`FIT_SPACE`; the box is log-scaled for losses and times):

| name | meaning | range |
|---|---|---|
| `target_gap` | crest gap of the thread-only rest shape (+ = open, − = pressed); statics re-solved | −2 … +3 µm |
| `thread_shift` | thread moved from its grazing position (+ = towards the nut) | ±0.15 mm |
| `delta_ve`, `Qte_inv`, `Yb` | visco-elastic and frequency-independent string loss; body-drain mobility | 1e-3–2e-2; 3e-5–2e-3; 1e-3–2.5e-2 s/kg |
| `xp` (= `xp_frac`) | pluck position, fraction of the speaking length from the bridge | 0.1–0.9 |
| `wp`, `pluck_release`, `pluck_pull`, `pluck_disp` | pad width, release time, pull time, pluck strength (static pad displacement) | 4–30 mm; 0.3–8 ms; 10–150 ms; 0.5–5 mm |

- **Body colour is not a search parameter.** It is matched in closed form, the smooth EQ above, for every candidate.
- **Loss.** The sum of capped (≤ 5) squared severities of the string-related items: pitch glide and stretch, late T60 per group (early T60 at half weight), BI10 track, partial-octave tracks, bloom time, gain and persistence, buzz-band centre and 95 % edge, comb, and attack. The `upper` edge (−30 dB) is reported but not fitted, because it jumps on a flat buzz plateau. The absolute f0 is matched by tuning, not fitted.
- **Search:**
  1. Evaluate the preset default plus a Latin-hypercube design of about 2d + 1 points.
  2. Run a small CMA-ES from the best point, with populations a multiple of `jobs`.
  3. Report the best point evaluated. `target_gap` and `thread_shift` are quantised to 0.05 µm / 5 µm, so the statics, which take ~20 s per geometry, are reused from a disk cache (`out/reference/cache/`).
  4. With `xp` in the fit, finish with a **mirror check**: evaluate the best point plucked at 1 − xp and report its loss.
- **Runtime.** One kharaj candidate costs ~10 s of string CPU for 5 s of sound, plus ~20 s of statics if the geometry is new. With 3 processes, 36 renders took **6.4 min** wall-clock (17 min of worker time) on this 4-core container, so budget 40 is about 7–8 min (see [Validation](#validation)).
- **Noise floor of the loss.** The contact dynamics are chaotic: two renders whose parameters differ by 1e-12 give losses of 0.2–4.4 between them (5 s, through the body). Almost all of this comes from the early decay of high partials (severity ≤ 1.7). Differences of a few loss units are not meaningful.

## 5. Limitations (read before trusting a number)

- **The pluck comb is mostly invisible with a jawari.** The bridge contact refills the notches within the first periods. Model renders at L/3 keep only the partial-3 notch (evidence < 1 dB); linear renders show the full comb. A reference that shows a clear comb suggests a weaker or more open jawari than the model's.
- **L/3 and 2L/3 have identical combs.** The magnitude spectrum cannot separate β from 1 − β. The jawari behaviour can, because the waveform reaching the bridge differs: in the model, 2L/3 reaches a 7 dB higher BI10 and peaks at 0.55 s instead of 1.7 s. `fit` uses that, and the mirror check shows how decisive it is for a given reference.
- **Pad width, release time and body colour all shape the spectral tilt.** The analyser only reports the combined tilt. `fit` can separate them only through their different effects on the time evolution.
- **Phone microphones roll off below ~100 Hz.** The kharaj fundamental is then too weak. f0 does not use partial 1, glide and drift do not use partials 1–3, and the colour EQ absorbs the roll-off. In `compare`/`fit` the colour EQ removes a constant roll-off before the level-shape items are computed. The reference's own absolute BI and octave values from `analyse` still include it.
- **Room reverberation lengthens the late decay of weak, fast-decaying partials.** Record close to the instrument in a dry room.
- **Overlapping strings** (drone cycles) give short windows and contaminated partials (see §1).
- **Chaos.** Fine details of the jawari (individual BI wiggles, the exact upper edge) are not reproducible even model-to-model. Only the smoothed, binned descriptors are compared.

## Validation

`python validate_reference.py synthetic | roundtrip | fit` writes its results to `out/reference/validation/{synthetic,roundtrip,fit}.txt` together with figures and compare reports.

**A. Synthetic plucks (21/21 checks pass, ~20 s).** The test signal is a sum of damped partials:
- f0 = 69.4 Hz + 3 c, with B = 1.6e-4 and T60(n) = 30 s / (1 + (n/8)^1.5);
- levels |sin(nπβ)|/n with β = 0.27;
- a pitch glide of +8 c with τ = 0.15 s, a drift of −0.5 c/s, and noise at −75 dB.

| quantity | result |
|---|---|
| onset start / anchor | +0.16 ms / +0.44 ms re the true onset (1 ms ramp); the anchor follows integer shifts to 0.000 samples |
| f0 | −0.13 c (vs the window mean including glide and drift) |
| B | +0.1 % (all partials and partials ≤ 12) |
| glide / drift | +5.0 c vs +4.9 c (30–150 ms minus 0.5–1.5 s; partials 4–12 since the verify3 fix, +4.0 c with the old partials 2–8) / −0.500 c/s exact |
| late T60, 85 partials (SNR > 30 dB) | median error 0.5 %, 90th percentile 1.7 %, worst 6.7 % (partial 82) |
| pluck comb β = 0.27 / 0.40 / 0.25 | 0.2700 / 0.4000 / 0.2518 |
| moving formant + bloom (harmonic partials) | BI10 track rms error 0.01 dB; partial-octave levels ≤ 0.6 dB; bloom peak 0.26 vs 0.25 s; formant hump tracked to a median of 1.1 partials, found in 99 % of frames |
| kharaj + Pa 1.6 s later | next pluck found at 1.974 s (true 2.0 s, coarse); truncated analysis gives f0 / B / T60 (partials ≤ 20) unchanged (T60 median error 0.0 %) |
| Pa/Sa/Sa/kharaj cycle, 1.2 s apart, analysing the kharaj | all 4 onsets found; kharaj start +0.7 ms (refined in its own band, since it barely rises above the ringing strings); f0 −1.1 c (coinciding Sa/Pa partials), B +1.8 %; 88 partials flagged as contaminated |
| stereo, decorrelated room (T60 0.6 s), damped by hand | correlation 0.88, side/mid −11 dB; room T60 0.63 s |

**B. Model round trip (15/15 checks pass, ~3 min).** These are kharaj renders (6 s) through `common.to_audio`, analysed dry and through `body.apply_body('tanpura_male')`.

*Linear model (contact off)*, checked against the exact damped eigenvalues of the modal string with its thread spring-damper:

| quantity | dry | body |
|---|---|---|
| partial frequencies | 0.03 c median error, 0.10 c 90th percentile | same (0.08 c) |
| B | +0.2 % vs π²EI/(TL²) = 1.621e-4 | same |
| late T60 | 0.1 % median error | same |
| change of late T60 with delta_ve × 2, per group | identical to the exact ratio to 3 decimals (0.999 … 0.732) | same |
| comb at L/3, L/4, 2L/3 | β 0.333, 0.243, 0.336 | β 0.338, 0.245, 0.337 |

The thread damper r_c adds 11 % loss compared with the uncoupled 6.91/σ. The analyser measures the coupled system, as it should.

*Jawari model*: partial 1's instantaneous frequency agrees with the phase slope of the raw 176.4 kHz bridge force to 0.19 c, i.e. +5.8 c of contact sharpening. Further findings, which are model facts the analyser reveals:
- **The jawari locks the upper partials harmonic.** B over partials > 12 is ≈ 0, against the string's 1.6e-4. B over partials ≤ 12 falls to 28 % of the string's value.
- **The pluck comb is filled within the first 40 ms** (evidence ≤ 0.8 dB for L/3, L/4 and 2L/3 alike).
- **L/3 and 2L/3 differ strongly through the jawari**: BI10 maximum +2.3 vs −4.8 dB, bloom peak 0.55 vs 1.73 s, compare loss 66.
- **Grazing vs thread +0.1 mm (gap) vs −0.1 mm (pressed)**, through the body:
  - pressed: the jawari dies at 4.0 s instead of lasting past the 6 s window, mean BI10 over 3–5 s is −17.4 vs −12.6 dB, and the fundamental's late T60 is 24 vs 16 s;
  - gap: the bloom peaks later (2.0 vs 1.7 s) and the fundamental's late T60 is 11 vs 16 s;
  - compare loss vs grazing: 17 (pressed) and 10 (gap), against a twin noise floor of 0.2–4.4.
- **delta_ve × 2 is nearly invisible with the jawari on** (loss 3.6, within twin noise), although it changes the linear string's T60s by up to 27 %. While the contact keeps transferring energy, it sets the decay of the upper partials, not the string's own loss. **Fitting string losses to a jawari tone is therefore weakly identifiable.** Use the late, post-jawari part of a long recording.

**C. Fit recovery.** The pseudo-reference is the kharaj model with non-default parameters: target_gap +1 µm, delta_ve 6.75e-3 (×1.5), pluck at 2L/3, `tanpura_male` body, 5 s. `fit(params=[target_gap, delta_ve, xp])` starts from the preset defaults (0 µm, 4.5e-3, L/3).

| budget (renders) | target_gap | delta_ve | xp | loss: default → best | loss at the true parameters (twin) | wall time (3 processes) |
|---|---|---|---|---|---|---|
| 36 | +0.85 µm ✓ | 1.20e-2 (+77 %) ✗ | 0.641 ✓ | 137 → 14.0 | 2.7 | 6.4 min |
| 54 | +0.70 µm ✓ | 7.84e-3 (+16 %) ✓ | 0.585 (✗: 0.08 L off, but the right side) | 112 → 9.7 | 2.6 | 5.6 min (statics cache warm) |

The two runs scored the default differently, 137 vs 112, because the `upper` edge was taken out of the loss between them. Logs are in `out/reference/validation/fit_budget36.txt` and `fit_budget54.txt`, with the histories in `fit_pseudo_reference*/`.

What the runs show:
- **Side of the string: always found.** The mirror check (same parameters, pluck at 1 − xp) scores 83–84 against 10–14 for the best point.
- **Gap: recovered to 0.15–0.3 µm**, from a 0 µm start and against a true +1 µm.
- **delta_ve: weakly constrained** (+77 % at budget 36, +16 % at 54), as the round trip predicted: the jawari masks the string loss.
- **Not converged.** Neither run reached the loss of the true parameters (≈ 2.6, the chaos floor). The landscape is rugged and the budget small, so a fitted value is "a good point", not "the optimum". For a real reference, use budget ≥ 54, check the history (`history.csv`), and re-run from the best point (`init=`) with fewer parameters.

**D. Independent verification (verify3) and the fixes it made.** A separate reviewer tested the analyser with its own signals, none of them from `validate_reference.py`:
- plucked-string spectra (|sin(nπβ)| · sinc(pad) / n², 12 mm pad) at β = 0.2, 0.37, 0.4 and 0.12;
- two-stage decays, and two polarisations detuned by 1.2 c with double decay;
- room reverb (T60 0.8 s, D/R +6 dB), and a second string 0.8 s later;
- a moving formant on inharmonic partials;
- real AAC (64k, 128k) and MP3 (128k) round trips through ffmpeg, and low-passes at 8, 11 and 16 kHz;
- white noise, and pink noise with 50 Hz hum, at 20–70 dB SNR;
- tuning ±1 semitone from the hint, a phone-like 120 Hz high-pass, and 48 kHz stereo with no pre-roll.

Scripts and logs are in the session scratchpad (`verify3/`).

It found and fixed six problems (marked `verify3` in `reference.py`):

1. **A file that starts with room noise got a fake pluck at t = 0.** This is the 1 s pre-roll that §1 asks for. The STFT's zero padding made the first frame look like a pluck at any SNR ≤ 60 dB. At 40–60 dB SNR, noise flux also passed the 0.7 dB threshold. `analyse` then picked the fake pluck and failed ("only 0.47 s until the next pluck").
   - Now a start candidate is kept only if the file starts within 15 dB of its peak level.
   - A candidate that rises out of near-silence must lift the level ≥ 15 dB above the quiet floor.
   - Detection in the model drone is unchanged (30 / 28 / 30 of 32, no false plucks), and all false onsets at 20–70 dB SNR are gone.
2. **No pre-roll (a file trimmed to the pluck) plus noise.** The late decay fits ran into the noise: median T60 errors of +1000 % to +12 900 %.
   - Now a floor is estimated per band from the spectrum between the partials over the last 2 s. It is used where that floor is stationary, and interpolated over frequency when most bands show one.
   - Median error is now 0.5–1.4 % and p90 4–19 % at 45–70 dB SNR. Clean model renders are unchanged: one late descriptor moved by 0.1 % (dry kharaj, `formant_centre` at 8 s), and the rest of the summaries match to about 1e-7.
3. **The late decay window ended at the last sample above floor + 10 dB.** One late noise excursion near the floor stretched the window over seconds of noise, giving T60s of 60–440 s for partials whose true T60 is about 5 s (30–40 dB SNR).
   - The window now ends where the level stays below floor + 10 dB for 0.5 s.
   - Late fits whose standard error exceeds 30 % of the T60 are dropped; the raw value is kept in `T60_late_raw`. None of the 12 s model renders' late fits is affected.
   - The worst error at 20–50 dB SNR is now 7–83 %, against up to 69 810 % before.
4. **Onset of a pluck over ringing strings.** The broadband refinement was off by −66 to +23 ms whenever the attack rose less than about 10 dB above the background's beating.
   - Both refinements are now scored by that margin, and the band refinement is used when it is better.
   - Below a 12 dB margin the onset falls back to the flux time + 17 ms. The report shows `ONSET approximate` (12–18 dB) or `ONSET coarse`.
   - Synthetic Pa-over-kharaj and 4-string cycle cases: +0.8 to +11.5 ms, against −67 to +23 ms before. Model drone (body): worst anchor error 15 ms, against 65 ms before.
5. **Glide and drift now use partials 4–12, not 2–8.** Body modes near partials 2–3 pull their frequency during the attack. With 2–8, the same kharaj render read −3.3 / +6.3 / +0.2 c of glide through the three body presets and −0.9 c dry. With 4–12 the spread is 1.6 c, and the synthetic +4.9 c glide reads +4.8 to +5.0 c.
6. **Small fixes.** `fit` now passes `--no-truncate` / `--max-dur` to its analysis; before, they were ignored. `.m4a`, `.mp4` and `.mov` files are decoded with ffmpeg.

After the fixes, A is still 21/21 and B 15/15. The B reports match the earlier ones except for the glide items and loss changes of ≤ 3.5.

Results of the independent tests (after the fixes):

| test | result |
|---|---|
| f0, B (clean, β 0.12–0.4, codecs, ±1 semitone off the hint, no hint, 120 Hz high-pass) | f0 within ±0.3 c; B within +3 % (two polarisations: +14 %) |
| pluck comb β = 0.2 / 0.37 / 0.4 / 0.12 | 0.200 / 0.369 / 0.400 / 0.1195 |
| two-stage decay (90 % fast + 10 % slow) | late T60 = slow stage within −1 to −5 % for partials 1–10; early T60 ≈ fast stage +10–20 %; `curv` 1.05–1.2 (mild flag only) |
| two polarisations (+1.2 c, 0.7×, T60 × 2) | late T60 = the long polarisation (ratio 2.0); onset anchor +9 ms (the beating attack peak) |
| room reverb T60 0.8 s, D/R +6 dB | partials 1–16 within 9 %; 26–40 read 19 % long; not flagged |
| AAC 128k / MP3 128k / low-pass 8–16 kHz vs clean | late T60 median 0.0–0.4 %, p90 ≤ 3.5 %; BI10 ≤ 0.02 dB; f0 ≤ 0.015 c. AAC 64k: T60 p90 8 %, onset −2.2 ms |
| moving formant on inharmonic partials | BI10 rms error 0.01 dB; hump (peak) at 0.5 / 1 / 2 / 4 s: 35 / 30 / 21 / 12 vs true 35.3 / 30.7 / 21.3 / 12.0 |
| second string 0.8 s later | analysis cut at 0.77 s; late T60s correctly absent (window too short); f0/B unaffected |
| a Pa analysed over a still-ringing kharaj at equal level | f0 +0.15 c, but **B +55 %** and only 1 of 40 late T60s: the kharaj's partials share every Pa band. A string plucked over ringing strings gives only f0, the early descriptors and the attack |

**E. Fit on an unseen, realistic pseudo-reference (verify3).**
- *The pseudo-reference:* kharaj with target_gap −0.5 µm (pressed), Yb 1e-2 (× 2) and pluck_release 4 ms.
  - It is rendered through the *generic* `tanpura` body, in stereo with room 0.3.
  - It gets a 0.8 s noise pre-roll at 55 dB SNR and an AAC 128k round trip.
- *The fit:* `fit(params=[target_gap, Yb, pluck_release], body='tanpura_male', budget=24, jobs=2, dur=5)`, i.e. with the wrong body, as with a real instrument.
- *Loss at the true string parameters:* 28.8 before fix 5 and 24.5 after. The twin floor is about 2.6, so a body/room/codec mismatch alone costs about 22 loss units. Most of it comes from the early decay of partials 17–60 and the `upper` edge, which is reported but not fitted.

| run | target_gap | Yb | pluck_release | loss: default → best (truth) | wall |
|---|---|---|---|---|---|
| before fix 5 (glide on partials 2–8) | −0.35 µm ✓ | 6.9e-3 (−31 %) | 6.5 ms (+62 %) | 73.1 → 29.2 (28.8) | 5.5 min |
| all fixes | −0.35 µm ✓ | 6.9e-3 (−31 %) | 6.5 ms (+62 %) | 68.6 → 22.6 (24.5) | 2.8 min (statics cache warm) |

The best point now scores *below* the true parameters, and the difference (1.9) is inside the chaos twin noise. Under a body/room mismatch, the gap is recovered (to 0.15 µm) but Yb and the release time are only determined to within about ±50 %: that is a property of the loss, not of the optimiser. Treat fitted loss and pluck-shape values from a real recording as a direction, not a measurement.
