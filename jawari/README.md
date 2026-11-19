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

All outputs go to `jawari/out/` (git-ignored); set `TANPURA_OUT` to change this.

## Known gaps

These are the next steps; see [`../docs/EVALUATION.md`](../docs/EVALUATION.md) §6–7.
- The rest geometry is assumed (crest exactly grazing), not solved from statics on a measured bridge profile.
- The losses are untuned, so the decay is too fast.
- The pluck is idealised.
- It uses the C3 string from DAFx-16, not C#2 kharaj.
- There is no body and no other strings.
- The nut-coupling strength of the 2-D model is not measured.
