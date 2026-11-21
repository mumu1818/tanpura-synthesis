# Match report: how the model plucks evolve against the real plucks (`match_report.py`)

> Run from inside `jawari/`. The script only **reads** the recordings in `$TANPURA_REFS` (default `/home/user/tanpura-app/assets`) and the round-3 renders in `out/fit3/wav/`. It writes `out/report/match_report.html` (one self-contained page, no network) and `out/report/match_report.json`, both git-ignored. They hold derived numbers only, no audio. It runs no simulation and takes about 20 s on one core.

```bash
cd jawari
python match_report.py        # -> out/report/match_report.html (open it in any browser)
```

## Why

After round 3 you said: **Sa sounds close, Pa and kharaj do not.** The fit's distance D3 says the opposite: Sa is the worst string (269 vs 187 and 167). So this report looks at the plucks over time, partial by partial, to find what the ear follows. It then checks which way of measuring the distance (the loss) agrees with you.

## Method

| what | how |
|---|---|
| partial tracks | level and frequency of each harmonic n·f0, up to 6 kHz. Pitch-adaptive Hann window of 6 periods, 10 ms steps, interpolated peak within ±0.3 f0. Its own code, so it also checks fit2d / fit3. |
| over time | total level; share of partials 1–3 / 4–7 / 8–15 / 16–31 / 32–60; spectral centroid; jawari formant (centre of partials ≥ 4); pitch |
| per partial | time of peak, peak level, decay rate after the peak |
| attack | ERB bands 80 Hz–12 kHz, 11.6 ms window, 2 ms steps, first 300 ms; first 40 ms vs 0.5–1.5 s |
| pitch after the pluck | YIN: window of 3 periods (at least 25 ms), 2 ms steps, 5-frame median, slips over 60 cents removed |
| mismatch split | model − recording per partial and frame, D = c + s_n + e_n(t). s_n is the fixed colour per partial: a smooth part (cubic over log f, EQ-like) plus a rough part. e_n(t) is what changes over time. The parts add up in power. |
| what the ear gets | 40 ERB bands, 50 Hz–15 kHz; Schroeder spreading; Zwicker-type specific loudness (exponent 0.23); loudest frame of the recording at 75 dB SPL; model gain matched for equal mean loudness. Also a small-speaker version with a 4th-order 250 Hz high-pass. |
| candidate losses | D3; harmonic level RMS (all / EQ-invariant / evolution only); perceptual loudness difference (all / shape only / small speaker); loudness-curve RMS; multi-resolution log-STFT (DDSP-style); mel-cepstral distortion |
| yardsticks | the same distances for real pairs: iTanpura CS3 vs your file (largely the same pluck, processed differently) and CS3Alt vs iTanpura CS3 (another real pluck) |
| same-pluck test | phase coherence of harmonics 4–30 over 0.05–2.5 s, with a constant pitch offset allowed: 1 means the same vibration, independent plucks score about 0.4. Plus the level residual after a fixed EQ per harmonic and a common gain over time. Controls are made from your file. |

## Results (round 3, 2026-11-21)

### 1. What follows your ear

| difference (model round 3 minus your recording) | Kharaj (you: not close) | Pa (you: not close) | Sa (you: close) | smallest on Sa? |
|---|---|---|---|---|
| Swell peak time (s) | +0.40 (1.43 vs 1.03) | −0.29 (0.83 vs 1.12) | +0.13 (0.77 vs 0.64) | **yes** |
| Swell size (dB) | +2.0 (17.4 vs 15.5) | +1.0 (19.6 vs 18.7) | −0.6 (18.4 vs 19.0) | **yes** |
| Pitch slide, first 0.75 s (cents) | +3.5 (1.8 vs −1.7) | +9.0 (3.9 vs −5.1) | −3.3 (−5.8 vs −2.5) | **yes** |
| Level at the end (dB) | +1.0 (−9.3 vs −10.3) | +0.2 (−11.9 vs −12.1) | +5.0 (−10.2 vs −15.2) | no |
| Partials 1-3 at 1 s (dB) | +9.0 (−6.7 vs −15.7) | +2.9 (−10.2 vs −13.1) | +11.6 (−3.4 vs −15.0) | no |
| Partials 16-31 at 1 s (dB) | −10.0 (−25.1 vs −15.1) | −12.8 (−30.8 vs −18.0) | −17.9 (−34.5 vs −16.6) | no |
| Jawari formant at 1 s (partial) | −2.3 (6.1 vs 8.3) | −1.5 (8.0 vs 9.5) | −2.5 (7.6 vs 10.0) | no |
| Attack brightness 2-5 kHz (dB) | −2.4 (2.3 vs 4.7) | −11.7 (0.8 vs 12.5) | −9.3 (−5.3 vs 4.0) | no |

* **The swell's timing follows your verdict.**
  * The kharaj model peaks 0.40 s late and the Pa 0.29 s early.
  * Sa is the closest: 0.13 s late.
* **So does the pitch slide after the pluck.**
  * Every real pluck starts a few cents sharp and slides down, as a plucked string does when its amplitude, and with it its tension, falls.
  * The kharaj and Pa models slide **up**. The likely cause is the swell mechanism: the sideways pluck's energy moves into the motion towards the bridge, the string rests more on the curved jawari, and that shortens it and raises the pitch.
  * The Sa model slides down, like the real string.
  * D3 has no pitch term at all.
* **The tone balance does not follow your verdict.** The model has too much in partials 1–3 and too little in 16–31, and it is worst on Sa. These differences are real, but they are not what you hear as wrong.
* **Caveat.** With three strings, a row lines up with your verdict by chance one time in three. These are leads to test by ear, not proof.

### 2. What kind of mismatch it is (RMS, dB)

| pair | Kharaj: total (smooth / per partial / over time) | Pa | Sa |
|---|---|---|---|
| Model round 3 vs your recording | 12.7 (6.6 / 8.8 / 6.5) | 12.9 (6.1 / 9.0 / 7.0) | 13.2 (7.5 / 7.8 / 7.6) |
| Model round 2 vs your recording | 11.9 (2.0 / 9.2 / 7.3) | 13.7 (3.4 / 11.2 / 7.2) | 12.2 (4.4 / 8.7 / 7.4) |
| iTanpura vs your file (largely the same pluck) | 9.5 (2.2 / 7.2 / 5.8) | 9.6 (2.2 / 6.6 / 6.5) | 8.3 (3.4 / 5.0 / 5.7) |
| CS3Alt vs iTanpura (another pluck) | 14.9 (3.2 / 11.3 / 9.2) | 18.2 (4.9 / 14.4 / 10.0) | 16.5 (1.5 / 12.0 / 11.3) |

* About a quarter to a third of the model's mismatch is a smooth fixed tilt: too much bass and too little treble. An EQ, a microphone or a different body response could do the same.
* Your file and the iTanpura file are largely the same pluck, yet they differ by 8–10 dB RMS. Of that, only 2–3 dB is a smooth tilt; the rest is per partial and over time. So that processing was more than an EQ.

### 3. The loss

| distance | Kharaj: model / same pluck / other pluck | Pa | Sa | Sa closest (raw / relative) |
|---|---|---|---|---|
| D3 (the fit's current loss) | 186.7 / 221.9 / 417.8 | 166.9 / 163.4 / 219.2 | 269.1 / 48.6 / 459.6 | no / no |
| Harmonic level difference (dB RMS) | 12.7 / 9.5 / 14.9 | 12.9 / 9.6 / 18.2 | 13.2 / 8.3 / 16.5 | no / no |
| Same, EQ-invariant (dB RMS) | 10.9 / 9.2 / 14.6 | 11.4 / 9.3 / 17.5 | 10.8 / 7.6 / 16.5 | yes / no |
| Time-evolution only (dB RMS) | 6.5 / 5.8 / 9.2 | 7.0 / 6.5 / 10.0 | 7.6 / 5.7 / 11.3 | no / yes |
| Perceptual loudness difference (%) | 37.8 / 18.6 / 33.0 | 38.0 / 20.5 / 39.1 | 43.7 / 16.1 / 39.4 | no / no |
| Perceptual timbre difference (%) | 18.2 / 7.9 / 11.2 | 18.9 / 9.4 / 18.9 | 21.6 / 7.6 / 17.3 | no / no |
| Perceptual, small speaker (%) | 36.6 / 15.3 / 32.2 | 37.4 / 20.7 / 36.1 | 36.8 / 15.3 / 39.6 | no / yes |
| Loudness-curve difference (phon RMS) | 1.2 / 1.6 / 3.8 | 1.3 / 1.7 / 2.4 | 1.0 / 1.1 / 4.7 | yes / yes |
| Multi-resolution spectrogram (dB) | 13.1 / 7.5 / 11.8 | 13.9 / 8.6 / 13.5 | 13.3 / 7.4 / 13.8 | no / yes |
| Mel-cepstral distortion (dB) | 35.3 / 19.6 / 27.1 | 43.1 / 21.3 / 48.5 | 46.2 / 20.2 / 41.1 | no / no |

* **D3 measures processing as much as physics.**
  * It puts the kharaj model (187) closer to your file than the iTanpura file (222), although that is largely the same pluck.
  * It ranks Sa as the worst string.
* **The perceptual distance orders the pairs sensibly.**
  * The same pluck processed twice is about half as far apart (16–21 %) as the model is from your file (38–44 %).
  * The model sits about as far away as another real pluck (33–39 %).
* **Which measures agree with your ear.**
  * The loudness curve agrees with your ranking both raw and relative to the yardstick.
  * The evolution-only, small-speaker and multi-resolution measures agree only relative to it.

### 4. Your recordings

| string | file | coherence vs iTanpura CS3 | level residual (dB) | reading |
|---|---|---|---|---|
| Kharaj | Mumu_Tanpura/Male/tt_CS3_kharaj.mp3 (your reference) | 0.77 | 3.6 | largely the same pluck (heavier processing or mixed) |
| Kharaj | iTanpura/Tanpura_CS3Alt_Kharaj.mp3 | 0.34 | 6.1 | different pluck |
| Kharaj | Mumu_Tanpura/Archive/Kharaj_v2.mp3 | 0.94 | 1.5 | same pluck, processed copy |
| Kharaj | Mumu_Tanpura/Archive/kharaj_v3.mp3 | 1.00 | 0.2 | same pluck, processed copy |
| Kharaj | Mumu_Tanpura/Archive/mumu_kharaj_cs3_230426_20_27.mp3 | 0.53 | 4.8 | different pluck |
| Kharaj | alt-jawari/mumu_EQ_Tanpura_CS3_Kharaj.mp3 | 1.00 | 0.1 | same pluck, processed copy |
| Kharaj | control: your file, EQ + 4:1 compressor + 1.5 cent pitch shift | 0.67 | 4.7 | control made from your file; vs your file: same pluck, processed copy (0.98) |
| Kharaj | control: your file + 0.8 s reverb at -6 dB | 0.73 | 4.7 | control made from your file; vs your file: same pluck, processed copy (1.00) |
| Pa | Mumu_Tanpura/Male/tt_CS3_pa.mp3 (your reference) | 0.61 | 4.3 | unclear |
| Pa | iTanpura/Tanpura_CS3Alt_Pa.mp3 | 0.49 | 5.8 | different pluck |
| Pa | Mumu_Tanpura/Archive/Pa-mix.mp3 | 0.92 | 2.5 | same pluck, processed copy |
| Pa | Mumu_Tanpura/Archive/pa-mix2.mp3 | 0.80 | 4.0 | largely the same pluck (heavier processing or mixed) |
| Pa | alt-jawari/mumu_Pa_Final230426_20_25pm.mp3 | 0.80 | 4.2 | largely the same pluck (heavier processing or mixed) |
| Pa | control: your file, EQ + 4:1 compressor + 1.5 cent pitch shift | 0.60 | 4.4 | control made from your file; vs your file: same pluck, processed copy (1.00) |
| Pa | control: your file + 0.8 s reverb at -6 dB | 0.61 | 4.5 | control made from your file; vs your file: same pluck, processed copy (0.93) |
| Sa | Mumu_Tanpura/Male/tt_CS3_sa.mp3 (your reference) | 0.85 | 3.3 | largely the same pluck (heavier processing or mixed) |
| Sa | iTanpura/Tanpura_CS3Alt_Sa.mp3 | 0.43 | 6.7 | different pluck |
| Sa | Mumu_Tanpura/Archive/mumu_sa_v2.mp3 | 0.13 | 5.6 | different pluck |
| Sa | Mumu_Tanpura/Archive/mumu_sa_v3.mp3 | 0.13 | 5.6 | different pluck |
| Sa | Mumu_Tanpura/Archive/sa-mix.mp3 | 0.94 | 2.9 | same pluck, processed copy |
| Sa | alt-jawari/mumu_sa_Final_230426_20_26.mp3 | 0.41 | 5.0 | different pluck |
| Sa | control: your file, EQ + 4:1 compressor + 1.5 cent pitch shift | 0.58 | 5.9 | control made from your file; vs your file: different pluck (0.52) |
| Sa | control: your file + 0.8 s reverb at -6 dB | 0.79 | 3.6 | control made from your file; vs your file: largely the same pluck (heavier processing or mixed) (0.85) |

* **Your files are not separate recordings.** `Male/tt_CS3` kharaj and Sa share most of their vibration with the iTanpura CS3 files: coherence 0.77 and 0.85.
* **Pa is unclear.** It scores 0.61 and is re-tuned by +1.5 cents.
* **Several files are processed copies** (0.92–1.00) of the iTanpura plucks: `mumu_EQ_Tanpura_CS3_Kharaj`, `Archive/kharaj_v3`, `Kharaj_v2`, `Pa-mix` and `sa-mix`.
* **Some files are independent plucks**, possibly of your own tanpura: `Archive/mumu_kharaj_cs3_230426_20_27`, `mumu_sa_v2/v3` and `alt-jawari/mumu_sa_Final_230426_20_26`.
* **A limit of the test.** Fast, heavy compression lowers the coherence as well: the Sa compressor control scores 0.52 against your file. A "different pluck" reading is therefore less certain than a "same pluck" one. No pair of independent plucks scored above 0.53.

## What it means

**For round 4 (physics).**
* Fix the **pitch slide** and the **timing of the swell** before the tone balance.
* Candidates:
  * a pluck with some motion towards the bridge from the start, so contact and pitch are already set when the swell begins;
  * the contact at rest (gap and thread);
  * stronger tension modulation through the pluck amplitude.
* Check each change against the pitch and swell curves of this report.

**For the loss.**
1. Score what the ear follows: the swell (peak time and size), the pitch slide after the pluck, and how the partials take turns over time.
2. Compare in a hearing-based domain (loudness per ERB band over time), not raw dB per partial.
3. Make the loss blind to a fixed EQ: fit a smooth EQ per reference inside the loss.
4. Divide by real-vs-real distances, so that 1.0 means "as different as another real pluck".
5. Fit several independent plucks at once.
6. Validate the weights on your own ratings: about 10–20 renders rated against the recording, MUSHRA-style.

## Limitations

* **Evidence.** Three strings and one verdict.
* **The auditory model is simple.**
  * No temporal masking.
  * One assumed playback level.
  * No binaural hearing.
* **Pitch at the onset.**
  * The first 50 ms is often unreliable on recordings: there is no clear period yet.
  * The glide is measured from 0.05–0.15 s to 0.5–0.75 s.
* **Time zero differs between signals.**
  * The model renders start 5 ms before the finger's release.
  * The recordings start 5 ms before they first cross −30 dB of their peak.
* **The yardstick.** "Another real pluck" is CS3Alt, which may be a different instrument or jawari setting.
