# `jawari/`: working, validated tanpura string models

This directory replaces the non-functional `modal/`, `fdtd/` and `waveguide/` synthesizers. Those are kept for reference; [`../docs/EVALUATION.md`](../docs/EVALUATION.md) explains, with measurements, why they could not produce jawari.

## The two models

**`tanpura1d.py`: 1-D modal string with jawari**
- One polarisation.
- Exact-pole modal update with all modes below 20 kHz.
- Thread (jiva) that the string rests on.
- Distributed bridge contact using a discrete-gradient (energy-conserving) penalty solved by Newton.
- Output is the bridge force.
- Speed: 0.55–0.8 s of CPU per second of audio.

**`tanpura2d.py`: the same with two polarisations**
- Adds a vertical and a horizontal polarisation.
- Adds bridge friction, nut coupling of the polarisations, and Kirchhoff–Carrier tension modulation.
- Energy-consistent (energy only decreases).
- Speed: about 1.2 s of CPU per second of audio.

Both reproduce the characteristic jawari "formant", a band of boosted partials that sweeps down over seconds. They are checked against `fdstring.py`, an independent finite-difference model that shares no code with them.

## Quick start

```bash
# from the repo root
uv sync
cd jawari
uv run python -c "from tanpura1d import simulate; from common import to_audio; import soundfile as sf; \
p,s,r = simulate(dict(dur=10.0), energy=False); y = to_audio(r['Fbridge'], p['fs']); \
sf.write('out/jawari_demo.wav', 0.9*y/abs(y).max(), 44100); print('wrote out/jawari_demo.wav')"
```

## Files

| File | Purpose |
|---|---|
| `tanpura1d.py` | 1-D model: `simulate(overrides) -> (params, setup, results)`; `DEFAULT` holds all parameters |
| `tanpura2d.py` | 2-polarisation model: same interface |
| `validate1d.py`, `validate2d.py` | physics tests: modal frequencies, energy balance, convergence, penetration, sanity checks |
| `render1d.py`, `render2d.py`, `sens2d.py` | 10 s renders, spectrograms, the thread-placement sweep and the sensitivity analysis |
| `fdstring.py`, `jlib.py` | independent finite-difference cross-check model plus helpers |
| `study_thread_geometry.py` | fixed-size thread slid along the bridge: shows µm-scale sensitivity (EVALUATION §4a) |
| `study_crest_gap.py` | crest-gap sweep, modal vs FD (EVALUATION §4b) |
| `study_thread_geometry_2d.py` | the same fixed-thread sweep for 1-D vs 2-D: geometry dominates polarisation (EVALUATION §4c) |
| `diag_bands.py` | shows why 4× oversampling is used (contact chatter near fs/4) |
| `metrics.py` | jawari metrics: buzz dB, hf fraction, centroid, smoothed spectral peak, spectrogram PNG |
| `common.py`, `common2d.py` | DC block plus band-limited decimation to 44.1 kHz; output directories |
| `MODEL_1D.md`, `MODEL_2D.md` | detailed model notes, validation tables, results, limitations |
| `statics.py`, `STATICS.md` | rest state of the string over a parametrised or **measured** bridge (`BridgeProfile.from_csv`), finite-diameter thread, break angle, bending stiffness; `profile_for_dynamics()` feeds it to `tanpura1d` |
| `validate_statics.py`, `crosscheck_statics.py` | statics tests (8/8 pass) and an independent analytic solver (agrees to < 0.001 µm) |
| `study_statics.py` | crest gap vs thread position, thread diameter, break angle, EI on/off; C3 dynamic sweep from statics |
| `presets.py` | `preset('kharaj')` etc.: the user's string, finger pluck, statics geometry, tuning, losses (see Presets) |
| `study_kharaj.py` | kharaj renders: nominal, old pulse, thread ±0.1/±0.3 mm, steel, linear, loss and convergence checks |
| `body.py`, `BODY.md`, `demo_body.py` | body/radiation filter (tumba + tabli modes, dense HF modes, stereo, optional room) and `apply_ir` for a measured body response |
| `render_demo.py` | full chain: kharaj → statics → string → body; single pluck + re-plucked drone → `out/demo/` |
| `reference.py`, `REFERENCE.md` | analyser for a **recorded pluck** (own instrument or iTanpura): onset, f0/glide, B, partial envelopes, per-partial T60, pluck comb, pitch-relative jawari descriptors, attack, colour, stereo; `compare()` with a ranked list of mismatches and figures; `fit()` of gap, losses, pluck position. REFERENCE.md says how to record references |
| `validate_reference.py` | analyser checks: synthetic plucks, model round trips (exact eigenvalues), fit recovery → `out/reference/validation/` |

All outputs go to `jawari/out/` (git-ignored); set `TANPURA_OUT` to change this.

## Presets

`presets.py`: `preset(name, **overrides)` returns a parameter dict for `tanpura1d.simulate`.

```python
from presets import preset; import tanpura1d as T1
p, s, r = T1.simulate(preset("kharaj", dur=12.0))            # the user's kharaj, finger pluck at L/3
p = preset("kharaj", thread_shift=+0.1e-3)                    # thread 0.1 mm towards the nut from grazing
```

| name | string |
|---|---|
| `dafx16_c3` | exactly `tanpura1d.DEFAULT` (DAFx-16 C3, the reference) |
| `kharaj` (alias `kharaj_bronze`) | the user's kharaj: 0.7 mm **phosphor bronze**, 1.05 m, C#2 = 69.4 Hz (material as the user stated) |
| `kharaj_steel` | the same string in steel (the `shared/params.py` values), on a bridge filed for steel |
| `sa` | the two Sa (jodi) strings: C#3 = 138.8 Hz (2 × 69.4), 0.50 mm **steel** (assumed gauge), 115 modes, its own ψ = 4.51° and thread at 4.00 mm |
| `pa` | the pancham: G#2 = 104.1 Hz (3/2 × 69.4), 0.55 mm **phosphor bronze** (assumed gauge and material), 140 modes, ψ = 4.54° |
| `pa_steel` | the same pancham in steel (alternative material), ψ = 5.12° |

`sa`, `pa` and `pa_steel` are built like the kharaj (statics geometry, thread auto-placed to graze, finger pluck at L/3, `valette` losses) on the same bridge, thread and body. The gauges come from two male-tanpura string-set listings; replace them with the user's measurements. `python render_strings.py` renders 12 s single plucks of Pa, Sa and kharaj and a 30 s Pa–Sa–Sa–kharaj drone (uncoupled strings, one body) to `out/strings/`. See [`STRINGS.md`](STRINGS.md) for the parameters, results and limitations, including the fast decay of Sa and Pa with the thread exactly grazing.

The kharaj presets:

- **Rest geometry from `statics.py`.**
  - The thread is auto-placed so the crest just grazes (`target_gap=0`), and `thread_shift` moves it from there.
  - The run starts from static equilibrium (`static_init`).
  - The statics are re-solved at the tuned tension.
- **Tuning.** `tune='speaking'` gives 69.4000 Hz with contact off. Bridge contact adds about +4 to +6 cents, and up to +25 cents late in the note when a pressed crest stays in permanent contact. Neither is compensated.
- **Discretisation.** All modes below 20 kHz: 140 (bronze) or 120 (steel). That means 124 or 138 bridge points and about 2 s of CPU per second of audio.
- **Excitation.** New `pluck='finger'`: 12 mm pad at L/3, 40 ms pull, 2 ms release, 2 mm deep. Unlike the old pulse, it also excites even modes.
- **Losses.** New `loss='valette'`: air, visco-elastic and internal string losses plus a smooth body drain. T60 is 41 / 21 / 8.5 / 4.5 / 1.5 s at 69 / 139 / 500 / 1000 / 2000 Hz, against 11 / 11 / 8.6 / 4.0 / 0.9 s with the old defaults.

**Assumptions to replace with measurements.** Values and sources are in the `presets.py` docstring.
- Bridge: parabola c = 3/m, crest 2 mm from the back face, 40 mm long.
- Thread 0.2 mm; break angle 10°.
- Speaking-length angle ψ, designed per string: 5.79° (bronze) or 6.94° (steel). A bridge filed for one material does not graze with the other.
- Body-drain mobility (the least certain loss term).

`python study_kharaj.py` renders every case to `out/kharaj/`: nominal, old pulse, thread ±0.1 / ±0.3 mm, steel, linear reference, loss sensitivity and convergence.

## Instrument facts from the user

- The kharaj is **phosphor bronze** (ρ ≈ 8800 kg/m³, E ≈ 1.1e11 Pa), 0.7 mm, speaking length 1.05 m, C#2 = 69.4 Hz.
- It is plucked at about **1/3 (or 2/3) of the length from the bridge** toward the nut.
- The 1-D renders sound close but lack **body depth**.

## Body

`body.apply_body(force, 44100, preset)` turns the bridge force into radiated sound. It is a one-way filter.

| preset | use |
|---|---|
| `tanpura_male` | **use for the kharaj / a male tanpura in C#**: air mode A0 at 65 Hz (just below the kharaj fundamental), soundboard T1 at 120 Hz (between Pa and Sa) |
| `tanpura` | generic; A0 85 Hz and T1 145 Hz were placed for a 131 Hz test string. On the kharaj this thins the fundamental and makes the octave boom (+13 dB). |
| `tanpura_small` | female or smaller instrument |

Both male presets are unmeasured estimates. A measured response through `apply_ir` replaces them; `BODY.md` explains how to record one.

## Known gaps

These are the next steps; see [`../docs/EVALUATION.md`](../docs/EVALUATION.md) §6–7.
- **Bridge geometry is assumed.** It is a parabola, with a 0.2 mm thread and a 10° break angle. Load the measured profile with `statics.BridgeProfile.from_csv`.
- **The body is assumed.** It is a one-way filter with no string–body coupling. Replace it with a measured response via `body.apply_ir`.
- **Losses are not fitted to a recording.** The Valette coefficients and the body-drain mobility are unverified.
- **Pitch.** Bridge contact makes the sounding pitch +4 to +7 cents sharp; there is no jawari-on tuning yet.
- **Other strings.** There is only one string: no Pa/Sa/Sa and no sympathetic coupling.
- **The drone is simplified.** Every pluck is identical, and the finger does not damp the string on re-pluck.
- **Matching real plucks (2-D, sideways pluck; see `FIT2D.md`, `render_match.py`).** The bloom now matches the user's recordings: level curve within 2–4 dB over 0.1–3.5 s. But:
  - the string keeps too much energy in partials 1–3, and the fit hides this with a strong 250–400 Hz high-pass;
  - the attack lacks brightness and the finger click;
  - the formant sits on partials 8–15 instead of 4–7;
  - the polarisation exchange makes a second bloom at about 3.5 s;
  - the fitted coupling settings are specific to one recording.
