# Tanpura Physical Modelling Synthesis

> **Status (Nov 2026).** The original `modal/`, `fdtd/` and `waveguide/` synthesizers below do **not** produce jawari: the string never rests on the jiva, the contact is ~10⁷× too soft, and each method has a fatal bug. The measured root causes are in [`docs/EVALUATION.md`](docs/EVALUATION.md). **Working, validated models are in [`jawari/`](jawari/):** an energy-stable 1-D modal model and a two-polarisation model, both producing the characteristic sweeping jawari formant, cross-checked against an independent finite-difference model. Start there.

An experimental attempt at physically modelling the sound of the Indian Tanpura (Tambura) from first principles -- something that, to our knowledge, has never been fully achieved in computational acoustics.

## Why This Is Hard

The Tanpura produces a shimmering, buzzing drone unlike any other stringed instrument. This sound comes from two physical features that are extraordinarily difficult to model:

### The Curved Bridge (Jawari)

Unlike Western instruments with sharp saddle points, the Tanpura bridge has a **wide, gently curved surface** (~4 cm). The vibrating string doesn't leave the bridge at a fixed point -- it *grazes* along the curve, with the contact point constantly shifting as the string vibrates. This creates a continuously varying effective string length that pumps energy into higher harmonics over time, producing the characteristic "bloom" of overtones.

This grazing contact is governed by nonlinear collision dynamics that are sensitive to sub-millimetre changes in bridge geometry. Most physical modelling literature on strings assumes a point boundary -- that assumption breaks down entirely here.

### The Jiva Thread

A small cotton thread (the *jiva*, literally "life") is placed between the string and the bridge surface. This ~0.3mm bump radically alters the grazing angle and creates the chaotic, buzzing quality that defines the Tanpura's voice. Master players spend considerable time adjusting the jiva position by fractions of a millimetre to achieve the desired tonal quality.

Without the jiva, the simulation produces a flat, lifeless tone. The jiva is bounded by the bridge start and end -- it physically rests on the bridge surface -- and its position within that range is the single most critical tuning parameter.

## Three Synthesis Methods

This project implements three independent physical modelling approaches for comparison:

### `modal/` -- Modal Synthesis (Recommended)

Decomposes the string into 80+ vibrating modes and tracks each mode's amplitude over time. Includes Stormer-Verlet energy-conserving integration, tension modulation (string stretching during bridge contact), and jiva thread modelling. Based on Van Walstijn & Chatziioannou's work.

### `fdtd/` -- Finite Difference Time Domain

Discretises the stiff string PDE on a spatial grid and steps the full equation forward in time. Bridge contact is handled naturally at grid points. Most physically faithful but most computationally expensive.

### `waveguide/` -- Digital Waveguide

Models the string as two travelling waves in bidirectional delay lines with allpass dispersion filters and penalty contact at the bridge. Fastest method, but distributed grazing contact is hardest to model this way.

## Shared Infrastructure (`shared/`)

All three methods share:
- **params.py** -- `TanpuraParams` dataclass with string, bridge, and jiva parameters
- **bridge.py** -- Bridge profile generator (parabolic curve + Gaussian jiva bump)
- **loss.py** -- Multi-objective loss function (spectral + perceptual)
- **optimize.py** -- Method-agnostic Optuna optimization harness
- **audio_io.py** -- Reference loading, WAV output

## Usage

Requires [uv](https://docs.astral.sh/uv/).

```bash
# Render with each method
uv run modal/run.py render --duration 10.0 --out output_modal.wav
uv run fdtd/run.py render --duration 10.0 --out output_fdtd.wav
uv run waveguide/run.py render --duration 10.0 --out output_waveguide.wav

# Optimize modal method against a reference recording
uv run modal/run.py optimize --reference references/kharaj_pluck.mp3 --trials 100 --jobs 4

# Render with optimized parameters
uv run modal/run.py render --params out_modal/best_params.json --duration 10.0
```

## Status

Early experimental trial. The physics are simplified (no torsional modes, no soundboard coupling, no longitudinal string vibration). The bridge contact model is a first approximation.

## References

- Van Walstijn & Chatziioannou, "Numerical Simulation of Tanpura String Vibrations" (ISMA 2014)
- Bridges & Van Walstijn, "Modal Based Tanpura Simulation: Combining Tension Modulation and Distributed Bridge Interaction" (DAFx 2017)
- Van Walstijn, Bridges & Mehes, "A Real-Time Synthesis Oriented Tanpura Model" (DAFx 2016)
- Van Walstijn, Chatziioannou & Bhanuprakash, "Implicit and explicit schemes for energy-stable simulation of string vibrations with collisions" (JSV 2024)
- Bilbao, "Numerical Sound Synthesis" (2009)
- C. V. Raman, "On some Indian stringed instruments" (1921)
- Valette & Cuesta, "Mecanique de la corde vibrante" (1993)
