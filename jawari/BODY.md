# Body / radiation model (`body.py`)

The 1-D model outputs the **force on the bridge**. Listening feedback on those renders was "close, just missing some body depth". `body.py` adds the missing acoustic path:

- the **tumba** (gourd resonator) and **tabli** (thin wooden soundboard);
- **radiation** to one or two microphones;
- optionally a light **room**.

It is a **one-way output filter**: bridge force goes in, pressure comes out. Body-to-string coupling (bridge admittance feeding back on the string) is **not** modelled; see [Limitations](#limitations-and-next-steps).

```bash
cd jawari
python body.py        # 11 self-tests, ~3 s
python demo_body.py   # 12 s nominal render + all body versions + metrics -> out/body/, ~40 s
```

```python
from tanpura1d import simulate; from common import to_audio; import body
p, s, r = simulate(dict(dur=12.0)); y = to_audio(r["Fbridge"], p["fs"])        # bridge force, 44.1 kHz
a = body.apply_body(y, 44100)                                  # mono, male tanpura ('tanpura'), dry
a = body.apply_body(y, 44100, stereo=True, room=0.3)           # 2 mic positions + light room -> (N, 2)
a = body.apply_body(y, 44100, t1_hz=135, a0_hz=78, seed=3)     # any preset parameter can be overridden
a = body.apply_ir(y, measured_ir, 44100, ir_sr=48000)          # convolve with a MEASURED body IR
body.plot_body("resp.png", "tanpura")                          # magnitude response + IR + body T60
```

**Parameters of `apply_body(force, sr, preset='tanpura', stereo=False, room=0.0, seed=0, tail=0.0, normalize=True, return_ir=False, **overrides)`:**

| Parameter | Meaning |
|---|---|
| `preset` | `'tanpura'` (male, large tumba, generic), `'tanpura_male'` (male tanpura in C#: use for the kharaj), `'tanpura_small'` (female), `'none'` (pass-through), or a dict |
| `seed` | Draws a different random high-frequency mode set: "another instrument of the same make" |
| `room` | 0..1 wet amount; 1 means wet RMS = dry RMS |
| `tail` | Seconds of silence appended so the body and room tails ring out |
| `normalize` | If true, the output RMS over the first 2 s equals the input's, so A/B comparisons are loudness-fair (RMS, not LUFS) |

## Model

### Transfer function

The force→pressure transfer function is a sum of real normal modes, computed exactly (impulse-invariant two-pole recursions in numba, about 0.1 s per IR). Each mode stops once it has decayed by 120 dB.

- **Explicit low modes:** **acceleration form**, h_k = −(A_k/ω_k²)·d²/dt²[e^(−σt) sin ω_d t].
  - This is the far-field pressure of a monopole-like mode, p ∝ ρ₀ × (volume acceleration).
  - It rises 12 dB/oct below each mode and is mass-like (flat) above it. So the body radiates nothing below A0 (a closed gourd has no bass), and no extra high-pass is needed.
  - The flat high-frequency asymptote of these modes is low-passed at 2 kHz (2nd order) because the dense region takes over there.
- **Dense modes:** damped sines with Gaussian (random-sign) residues, high-passed at 250 Hz.
- **Whole IR:** a 20 Hz DC guard.
- **Causality:** everything is strictly causal, so nothing precedes the force that causes it. There is no pre-ringing.

### 1. Explicit low modes (`explicit_modes(p)`, 15 modes, 85 Hz – 1.05 kHz)

| Mode | Physics | Default f, Q, peak gain* |
|---|---|---|
| A0 | Air/neck mode (gourd volume + hollow dand + small tabli holes), Helmholtz-like | 85 Hz, 15, +9 dB |
| T(0,1) = T1 | Tabli breathing mode | **145 Hz**, 20, +15 dB |
| T(m,n) | Tabli modes, from clamped-circular-plate eigenvalues λ²ₘₙ (Leissa): f = T1·(λ²ₘₙ/λ²₀₁)^γ, with γ = 0.8 for the convexity of the tabli | T(1,1) 261, T(2,1) 387, T(0,2) 430, T(3,1) 525, T(1,2) 604, T(4,1) 674, T(2,2) 787, T(0,3) 820, T(5,1) 832, T(3,2) 978, T(6,1) 997, T(1,3) 1041 Hz |
| A1 | First gourd-cavity mode (rigid sphere, ka = 2.08, D = 43.6 cm) | 521 Hz, 45, +3 dB |

\* Peak |H| at mic 1, in dB re the dense-region mean level at 1 kHz.

**Q and gain rules for the T(m,n) modes:**
- Q = 20·(f/T1)^0.3.
- Axisymmetric modes (m = 0) are monopoles driven at the near-central bridge: +15 / +9 / +6 dB for n = 1, 2, 3.
- m ≥ 1 modes are driven only by the bridge's offset/rocking and radiate as multipoles: 6 − (m−1) − 1.5(n−1) dB, but at least 1 dB.

`'tanpura_small'` scales the modes:
- T1 × 1.18, because plate frequencies go as h/a²;
- A0 × 1.19, because a Helmholtz-like mode goes as V^(−1/2);
- A1 follows the smaller gourd.

### 2. Dense region (from 0.5 kHz, fully in by 1.5 kHz, up to 18 kHz)

There are three mode families. Each has a Weyl modal density computed from the geometry:

| Family | Weyl density | Default value | Power share |
|---|---|---|---|
| Tabli (thin orthotropic plate) | n = S/2·√(ρh/D) | constant **0.022 /Hz** | 60 % |
| Gourd shell | n_∞ = 0.043 /Hz above the ring frequency (2.1 kHz); ∝ √f below it | as given | 25 % |
| Gourd air cavity | n = 4πVf²/c³ + πSf/(2c²) | V = 27 L: 0.015 / 0.16 / 0.91 /Hz at 1 / 4 / 10 kHz; capped at 0.25 /Hz | 15 % |

The total density therefore **grows with frequency**. For seed 0 there are 4819 modes: 15 explicit, 383 plate, 741 shell and 3680 cavity.

**Mode frequencies.** They come from inverting the cumulative count N(f), with ±0.4-spacing jitter. Real mode spacings are neither regular nor Poisson.

**Damping:**
- Plate: η = 0.012·(f/1 kHz)^0.15 + η_rad.
- Shell: 0.015·(f/1 kHz)^0.15.
- Cavity: 1/60.
- Each mode also gets a log-normal scatter of 25 %.
- Resulting body T60: 314 / 232 / 144 / 154 / 95 / 41 / 25 ms in the 125 Hz – 8 kHz octave bands.

**Amplitudes.** The per-mode amplitude is A = G(f)·√(8σ·share/n), chosen so that **E|H(f)|² = G²(f)** whatever the density or damping. The self-test checks this to within 1 dB over 2–12 kHz. G² is a statistical-energy-analysis (SEA) estimate:

  G²(f) ∝ σ_rad(f)·Re{Y_bridge(f)} / (f·η(f))

- **σ_rad:** the Maidanik radiation efficiency of the tabli, averaged over coincidence frequencies between the along-grain (3.7 kHz) and cross-grain (11.6 kHz) values. That smears coincidence, as orthotropic wood does.
- **Y_bridge:** 1/(Z_p + jωm_b), with Z_p = 8√(Dρh) = 44 N·s/m and an **effective** bridge mass of 2.5 g. The mass corner is therefore at 2.8 kHz.
- **Net effect:** the envelope is within about ±2 dB of flat from 0.5 to 6 kHz. It is −4 dB at 8 kHz and −15 dB at 18 kHz.

### 3. Stereo

The two microphones see **the same poles**, because it is the same body, but different residues:
- Explicit mode (m, n) gets the factor cos(m·60°) at mic 2: mic 2 is rotated 60° about the tabli normal.
- Dense residues are drawn with inter-mic correlation 0.2.
- Mic 2 is delayed by 0.25 ms.

**Measured L/R correlation:** 0.90 below 300 Hz (mono-safe), −0.14 above 2 kHz, 0.63 broadband.

An earlier version flipped the sign of every odd-m mode at mic 2. T(1,1) lies on 2f0, so that made the 230–300 Hz band perfectly anti-correlated and the 2nd partial cancelled in mono. That rule was replaced by the cos(m·60°) factor.

### 4. Room (optional)

- **Early reflections:** 8 ms pre-delay, then 6 reflections per channel at 5–24 ms, low-passed at 6 kHz.
- **Late reverberation:** an 8-line Hadamard feedback delay network (FDN) with delays of 10–29 ms. It uses Jot one-pole absorption filters, giving T60 = 0.55 s (0.30 s at high frequencies).
- **Balance:** late energy is 2× the early energy, and the wet signal is normalised to the dry RMS.

## Sources (research log)

**No published, readable measurement of tanpura body modes or bridge admittance was found.** All numbers below are excerpts from search results.

Hosts this session could not reach (WebFetch could not resolve them):
- `www.lemez.net` (Carterette, Vaughn & Jairazbhoy, *Perceptual, acoustical and musical aspects of the tambura drone*, Music Perception 1989);
- `katalog.lib.cas.cz` (the tambura holography record);
- `www.tosslevy.nl` (a tanpura maker's site).

Allow those hosts to read them.

| What | Source | Used for |
|---|---|---|
| Male tanpura: length about 145 cm, tabli width 37–45 cm, tumba circumference about 137 cm (D ≈ 43.6 cm). Female: about 130 cm, 33–36 cm, about 122 cm (D ≈ 38.8 cm) | [Toss Levy, types of tanpura](https://tosslevy.nl/tanpura/types-of-tanpura) (search excerpt); [Wikipedia: Tanpura](https://en.wikipedia.org/wiki/Tanpura) | `tabli_diam`, `gourd_diam` (**sourced**) |
| Tambura resonator: "semi-globular, slightly convex top plank with small holes in two circles beside the bridge"; modes studied by holographic interferometry (no frequencies in the abstract) | [CAS library record](https://katalog.lib.cas.cz/KNAV/EdsRecord/edb,74358300) (abstract via search) | Convex tabli (γ < 1), small holes so a weak, uncertain A0 |
| Sarasvati veena: FEM of the top plate and air cavity; resonator natural frequency **351 Hz** (hammer test); modes measured up to 1.4 kHz; the top plate dominates radiation | Chauhan, Singru & Vathsan, JASA 149(1):540–555 (2021), [doi:10.1121/10.0003356](https://doi.org/10.1121/10.0003356); [JME 2024](https://doi.org/10.21595/jme.2023.23505) | Analogue: the plate dominates, explicit modes below 1.5 kHz |
| Tamburica (bisernica): air mode about 340 Hz, wood mode about 740 Hz | [UBC PHYS341 wiki](https://wiki.ubc.ca/Course:PHYS341/Archive/2016wTerm2/Tamburica_organology) | Order of magnitude (small instrument) |
| Guitar: 1–2 body modes per 100 Hz, **Q 20–60**; Helmholtz–plate coupling gives peaks near 100 and 230 Hz | Richardson & Brooke, [IOA](https://www.ioa.org.uk/system/files/publications/BE%20RICHARDSON%20M%20BROOKE%20MODES%20OF%20VIBRATION%20AND%20RADIATION%20FIELDS%20OF%20GUITARS.pdf); Richardson et al., [mode placement revisited](https://www.ioa.org.uk/system/files/proceedings/be_richardson_hr_johnson_ad_joslin_ia_perry_mode_placement_in_guitars_revisited.pdf) | Q values, A0/T1 structure |
| Oud body modes: 120, 162, 256, 366, 606, 672 Hz | ViennaTalk 2010 proceedings (search excerpt) | Analogue for mode spacing |
| Guitar: modal overlap regimes; roughly constant (plate-like) modal density at mid and high frequencies | Élie et al., [JASA 2012](https://members.loria.fr/BElie/papers/Elieetal_jasa2012.pdf) | Weyl plate density; statistical high-frequency region |
| Soundboard mobility synthesised from modal density and mass; radiation regimes, coincidence | Ege, Boutillon et al., [arXiv 1210.5688](https://arxiv.org/pdf/1210.5688), [1305.3057](https://ar5iv.arxiv.org/html/1305.3057) | SEA envelope: Z_p = 8√(Dρh), coincidence |
| Body modelling: commuted synthesis; parallel second-order resonator banks fitted to measured admittance | J.O. Smith, [PASP: Body Modeling](https://ccrma.stanford.edu/~jos/pasp/Body_Modeling.html); Rau & Smith, [ISMA 2019](https://ccrma.stanford.edu/~mrau/papers/ISMA2019_Rau_Smith.pdf) | Architecture (modal bank / IR) |
| Tanpura signal analysis (no body-resonance numbers in the excerpts) | Datta, Sengupta et al., *Acoustical Analysis of the Tanpura*, Springer 2019 | None |

Some textbook items were used from memory and not re-checked online:
- Maidanik's radiation-efficiency formula;
- Leissa's clamped-circular-plate eigenvalues (λ² = 10.216, 21.26, 34.88, 39.77, …);
- Weyl's law for plates and cavities;
- Jot's FDN absorption filters;
- Farina's exponential sweep;
- the wire-break method.

## Assumptions (not measured: change them with keyword overrides)

| Parameter | Default | Basis |
|---|---|---|
| `tabli_h`, `tabli_rho`, `tabli_EL`/`ER` | 4 mm, 480 kg/m³, 8.5/0.85 GPa | Assumed light hardwood (toon); no thickness found |
| `shell_h`, `shell_rho`, `shell_E`, `shell_area_frac` | 5 mm, 400 kg/m³, 3 GPa, 0.6 | Assumed dried bottle gourd |
| `gourd_fill`, `neck_volume` | 0.6 of the sphere, 1 L → V = 27 L | Assumed |
| `bridge_mass` (effective) | 2.5 g | Assumed. **This is the main high-frequency brightness control.** A real bone bridge is heavier, but rocking and load-spreading lower its effective mass. |
| `eta0`, `eta_exp`, `eta_shell`, `Q_cav`, `q_scatter` | 0.012, 0.15, 0.015, 60, 0.25 | Wood-typical, assumed |
| `a0_hz`, `t1_hz` | **85, 145 Hz** | Estimated ranges: A0 ≈ 60–110 Hz (Helmholtz-like with small holes plus the hollow neck); T1 ≈ 75–200 Hz (a 40 cm, 4 mm plate is 57–116 Hz flat between simply-supported and clamped edges, then ×1.3–1.6 for convexity plus air stiffness) |
| `gamma`, Q/gain rules, `t1_gain_db`, `a0_gain_db`, `a1_gain_db` | 0.8, see above, +15, +9, +3 dB | Assumed, after guitar analogues (the low peaks are 10–20 dB above the mid-frequency mean) |
| `share_*`, `max_density`, `jitter`, `rho_mics`, `mic_spread_deg` | 0.6/0.25/0.15, 0.25 /Hz, 0.8, 0.2, 60° | Modelling choices |
| Room | T60 0.55/0.30 s, small room | Generic |

**Why T1 is 145 Hz.** T1 was placed inside its estimated range so that the C3 test string's fundamental (130.8 Hz) does **not** fall into the A0–T1 antiresonance. The sensitivity table below shows how much this matters. It is a deliberate choice, not a measurement.

## Results (nominal 1-D jawari, C3, 12 s; `out/body/body_results.txt`)

**Sanity checks:**
- All files are RMS-matched over 0–2 s.
- They share one write gain, so the loudest file peaks at −1 dBFS and none clip.
- DC is below 1e-19.
- All values are finite, with no denormals.
- Nothing precedes the onset beyond the dry signal's own −80 dB decimation-filter lead-in.
- `apply_ir` with the saved `body_ir_tanpura.wav` reproduces `apply_body(stereo)` to −158 dB.

| File | LF < 500 Hz (0–12 s) | Centroid (0–12 s) | Centroid (0.3–3 s) | Mean buzz (0.3–3 s) | Mean hf_frac (0.3–3 s) | T30 broadband | T30 at 125 / 250 / 500 / 1k / 2k / 4k Hz (s) |
|---|---|---|---|---|---|---|---|
| dry | 73.2 % | 605 Hz | 924 Hz | −11.3 dB | 0.172 | 7.61 s | 6.92 / 6.70 / 9.77 / 7.56 / 7.04 / 5.89 |
| **body** (mono) | 70.2 % | 491 Hz | 632 Hz | −18.8 dB | 0.109 | 8.32 s | 6.88 / 6.70 / 10.36 / 8.14 / 7.79 / 5.89 |
| body_stereo | 77.9 % | 421 Hz | 572 Hz | −17.8 dB† | 0.067† | 8.14 s | 6.88 / 6.69 / 10.31 / 7.76 / 7.03 / 5.81 |
| body_stereo_room | 77.9 % | 428 Hz | 581 Hz | −16.9 dB† | 0.094† | 7.89 s | 6.90 / 6.69 / 10.08 / 7.76 / 7.02 / 5.75 |
| body_deep (T1 135, A0 78) | 94.3 % | 202 Hz | 317 Hz | −23.5 dB | 0.067 | 7.39 s | |
| body_light (T1 160, A0 92) | 13.5 % | 1096 Hz | 1214 Hz | −16.5 dB | 0.196 | 8.16 s | |
| body_small (female body on a C3 string) | 5.9 % | 1268 Hz | 1323 Hz | −11.2 dB | 0.139 | 7.77 s | |

† Taken from the mono mean of L and R (as in `metrics.py`). That mean partly cancels the decorrelated high-frequency modes. The other columns use channel-averaged spectra and energies.

**Partial levels vs dry**, RMS-matched over the first 2 s, partials 1–12, default body (dB):

−2.2, +3.4, −0.2, +6.6, −2.6, +0.3, +0.9, +7.8, +2.3, +0.2, −2.4, +5.2

**What the body does (objectively):**
1. **Low-mid coloration.** A0 rings below the fundamental at about 85 Hz; this is the thump at the pluck. T1 at 145 Hz rings just above f0. The body boosts and cuts individual partials by up to ±8 dB (some hit modes, some land in dips).
2. **Short body ringing.** Body T60 is 0.3 s at 125 Hz and 25 ms at 8 kHz.
3. **A dense, irregular high-frequency modal texture** in place of the bare force spectrum. The jawari formant sweep survives: the smoothed peak is 1.78–1.79 kHz at 1 s in both dry and body.
4. **Decorrelated stereo** above about 1 kHz, mono-safe below 300 Hz.

**Effect on the metrics:**
- The default body lowers the spectral centroid from 924 to 632 Hz (0.3–3 s) and the buzz metric by about 7 dB. Partials below 1 kHz sit on body peaks that are on average higher than the 2–8 kHz mean.
- The LF < 500 Hz share hardly changes (73 → 70 %; 78 % per channel in stereo). That share is dominated by the fundamental, which already dominates the dry force. So "depth" here comes from resonance and coloration, not from more fundamental.
- T30 rises by up to 0.7 s, but only through spectral re-weighting. A one-way filter cannot lengthen the string's decay, and a coupled body would shorten it.

### Sensitivity: the dominant parameter is where T1 sits relative to f0

Male body, A0 = 85 Hz, mono, mean of seeds 0–3. Dry: LF 73.2 %, centroid 924 Hz, buzz −11.3 dB.

| t1_hz | 120 | 125 | 130 | 135 | 140 | **145** | 150 | 155 | 160 | 170 | 180 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LF < 500 Hz (%) | 77 | 88 | 97 | 94 | 78 | **67** | 54 | 23 | 12 | 7 | 7 | 16 |
| Centroid, 0.3–3 s (Hz) | 764 | 515 | 261 | 354 | 694 | **727** | 1021 | 1334 | 1369 | 1363 | 1320 | 1380 |
| Buzz (dB) | −14.3 | −17.3 | −21.6 | −19.7 | −15.6 | **−14.9** | −11.9 | −10.7 | −12.0 | −10.9 | −10.1 | −9.2 |
| P1 (dB) | +1.2 | +1.9 | +3.2 | +2.6 | +0.9 | **−2.3** | −3.2 | −4.7 | −7.3 | −13.2 | −21.6 | −19.7 |

**Why it is so sensitive.** A0 and T1 are both monopole modes with residues of the same sign. In acceleration form there is an **antiresonance** between them, at about 110 Hz for the defaults (`body_response.png`). Two-oscillator (Helmholtz + top plate) guitar models predict the same kind of dip in radiated pressure.

- If T1 is ≥ 155 Hz, the 131 Hz fundamental falls into it: thin sound, LF share around 10 %.
- If T1 is within about 10 Hz of f0, the fundamental booms: LF share above 90 %.

A0 matters much less: near the default (T1 = 140–150 Hz), moving A0 from 70 to 105 Hz changes P1 by at most 2.2 dB. It matters more once f0 is already in the dip (5 dB at T1 = 160 Hz). The A0/T1 placement of a real tanpura relative to its Sa is unknown. This is the strongest argument for measuring one instrument (see below), and for treating `t1_hz` as a by-ear parameter until then.

**Ear test suggested.** Listen to `dry.wav` vs `body.wav` vs `body_deep.wav` vs `body_light.wav`, then `body_stereo_room.wav`.

**About `tanpura_small`.** It is a female instrument, whose Sa is about F3–A3. On the C3 string its T1 (171 Hz) puts f0 in the dip, so it sounds thin. That is a pitch mismatch, not a bug.

**Kharaj preset `tanpura_male`.** It was added after integration (A0 65 Hz, T1 120 Hz). The `tanpura` defaults were placed for the 131 Hz C3 test string. On the user's 69.4 Hz kharaj, the kharaj fundamental sits below A0, and H2 (139 Hz) sits next to T1. Loudness-matched single pluck, 0.3–3 s (`render_demo.py`):

| body | energy < 100 Hz | H2 re H1 |
|---|---|---|
| dry | 0.234 | −2.2 dB |
| `tanpura_male` | 0.237 | −2.3 dB |
| `tanpura` | 0.033 | **+13.0 dB** |

So `tanpura_male` keeps the fundamental and adds 300 Hz–1 kHz body colour instead of an octave boom. Both values are still estimates inside the ranges above: replace them with a measured response.

**Other by-ear controls:**
- `mode_gain_db` (e.g. −6) reduces the low-mid coloration and keeps more buzz.
- `bridge_mass` (e.g. 1.5e-3) gives a brighter top.
- `seed` gives a different instrument.

## Measure your own body (beats any parametric model)

A measured bridge-to-microphone IR replaces everything above. Use `apply_ir(force, ir, sr, ir_sr=...)`, which takes a mono `(N,)` or stereo `(N, 2)` IR.

1. **Damp all strings.** Weave felt or cloth between them near the nut and over the long string length, without touching the bridge top. Otherwise the IR contains ringing strings and jawari buzz.
2. **Place the microphone.** Use one omni or small-diaphragm microphone (or a pair, for a stereo IR) at 0.5–1 m in front of the tabli, in a quiet, preferably damped room. If you want the room in the IR, record in the room you like instead. Record at 48 kHz / 24-bit.
3. **Excite the bridge normal to the tabli**, which is the direction of the 1-D model's force. Use one of these, best first:
   - **(a) Impact hammer with a force sensor**, recorded in sync with the mic. Use `ir_from_tap(mic, sr, force=f)` (regularised deconvolution) and average 5–10 taps. This gives the true F→p transfer.
   - **(b) Wire break.** Loop a thin copper strand (about 0.1 mm) over the bridge, pull towards the tabli until it snaps, and use `ir_from_wire_break(mic, sr)`. The snap is a force step, so the method differentiates the response. It is very repeatable and needs no sensor.
   - **(c) Sine sweep through a small exciter or shaker** lightly coupled to the bridge. Make the sweep and its inverse with `sweep, inv = exp_sweep(20, 20000, 10, 48000)`, play `sweep`, record, then use `ir_from_sweep(rec, inv, sr)`. This gives the best SNR and discards distortion products. The exciter's own response and mass are included.
   - **(d) Pencil or small steel-ball tap without a sensor.** Use `ir_from_tap(mic, sr)`. It is valid only up to about 1/(contact time), roughly 2–5 kHz.
4. **Prepare the IR.** `prepare_ir` high-passes at 20 Hz, trims the leading silence, keeps up to 2 s and fades the tail. `apply_ir` calls it by default.

The self-tests check (a)–(c) on a synthetic "instrument" (the model's own body IR, plus noise):
- sweep: −47 dB IR error;
- hammer: −36 dB below 8 kHz;
- wire break: exact.

These checks were run on synthetic data only, never on a real recording.

## Verification (`python body.py`, 11/11 pass)

- **Single mode:** peak frequency, Q and peak gain are as specified (300.12 Hz, Q 25.04, −0.001 dB).
- **Dense-region mean power vs the SEA target:** within 1 dB at 2/4/8/12 kHz (8 seeds).
- **Causality:** `apply_body` (mono, stereo, stereo + room) puts nothing before the input onset (−313 dB, i.e. FFT round-off).
- **Normalisation:** output RMS equals input RMS to 1e-9.
- **Identities:** `'none'` and `apply_ir(delta)` return the input.
- **Measurement helpers:** the sweep, hammer and wire-break deconvolutions recover a known IR.

## Limitations and next steps

- **One-way only.** The body does not load the string, so these effects are missing:
  - no energy drain into the body (the string's T60s and the jawari dynamics are unchanged);
  - no frequency pulling near body modes;
  - no bridge-mediated coupling between the four strings (sympathetic resonance);
  - no polarisation coupling.

  **Next step:** use the same explicit modes, given effective masses at the bridge, as a modal bridge admittance in `tanpura1d.py`. Replace the rigid x=0 termination with bridge/body modal oscillators in the energy-consistent scheme. That would also give the physically derived radiated output (EVALUATION §6, steps 4–5). It needs a bridge-admittance measurement to be credible.
- **The explicit-mode frequencies and gains are estimates.** No published tanpura body data were found. Results depend strongly on T1 relative to f0, as the sensitivity table shows.
- **The high-frequency region is statistical.** Its fine structure is random (seeded); only its envelope, density and damping are modelled.
- **What the radiation model leaves out:**
  - two point microphones in the far field, with no directivity beyond the cos(mΔψ) factor;
  - no near field;
  - no dependence on amplitude.
- **Loudness matching is by RMS, not perceptual (LUFS)**, so bass-heavy versions may sound slightly different in loudness.
- **The buzz-metric drop is a model prediction, not a measurement.** The body lowers the buzz metric by about 7 dB at default. Whether a real tanpura's radiated sound is "less buzzy" than its bridge force is unverified. A recording plus a measured IR would settle it.
- **No extra packages were installed.**
