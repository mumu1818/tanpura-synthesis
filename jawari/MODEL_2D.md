# 2-D model (tanpura2d.py): two-polarisation modal tanpura string, and a test of thread-placement sensitivity

> Detailed notes written while building this model. Run all scripts from inside `jawari/`; outputs (WAV, PNG, result files) go to `jawari/out/`. The evaluation that motivates this work is in [`../docs/EVALUATION.md`](../docs/EVALUATION.md).


This directory extends the validated 1-D modal tanpura prototype (`../the 1-D model (tanpura1d.py)`) to two transverse
polarisations with bridge friction, a nut coupling and Kirchhoff-Carrier tension modulation. It then asks
whether the second polarisation makes the jvari depend more strongly on where the thread sits.

## 0. Bottom line

* **A working 2-D model exists and still produces a jvari.** It conserves energy to 3e-11 with damping and friction
  off. With friction on, energy only ever decreases.
* **2-D does not, by itself, add strong thread-placement sensitivity in this model.** Whether the second
  polarisation adds sensitivity depends almost entirely on how strongly the polarisations are coupled at the nut.
  No source I could read gives a value for that coupling.
  * **No nut coupling** (polarisations linked only by bridge friction and tension modulation): 2-D behaves like 1-D.
    S_track is 3.70 dB against 3.65 dB for 1-D+TM.
  * **Nominal coupling** (30 deg, 1 cent): the buzz-trajectory spread S_track rises from 3.76 dB (1-D) to 4.44 dB
    (+18%; +34% at 352.8 kHz, +40% with modes to 40 kHz). The sustain becomes *less* thread-dependent: the spread of
    t_end(-15 dB) drops from 1.48 to 0.52 s, because in 2-D the vertical buzz dies at about 4-5 s at every position.
  * **Strong coupling** (5 cent split): thread placement matters much more.
    * The jvari onset t_on spreads 0.35 -> 3.39 s across 3-8 mm (std 1.35 s), against 0.21 -> 0.79 s in 1-D.
    * The mean buzz spreads by 12.0 dB (std 4.6 dB), against 7.2 dB (std 2.8 dB).
    * The buzz stays above -15 dB for 9.7 s at 3 mm but never at 8 mm.
    * This result is converged against 352.8 kHz (S_track 4.83 vs 4.81 dB) and lies far above its noise floor (0.02 dB).
* **Bridge friction** at the published coefficient has no measurable effect on the sensitivity (4.44 vs 4.38 dB with
  friction off). It removes only about 3% of the pluck energy at 5 mm (0.2% at 8 mm), because slip speeds are 15-140 mm/s,
  which gives mu_f <= 0.15. Near-Coulomb friction (c2 = 2840 s/m) slows the transfer of energy into z and *reduces* the spread
  slightly (3.99 dB).
* **The 1-D model already has a large thread-placement dependence**, about 30 times its own noise floor.
  * Across 3-8 mm the mean buzz changes by 7.2 dB, the jvari onset by 0.6 s, and the time above -20 dB from 9.2 s to 4.8 s.
  * So the premise "1-D fails to show the dependence" is not reproduced in this geometry (thread lift
    3(x_b - x_c)^2 = 12-147 um).
  * I could not read VT-2015 (viennatalk2015.mdw.ac.at and core.ac.uk are blocked from this container), so I cannot
    say how its 1-D comparison was defined.

---------------------------------------------------------------------------------------------------------
## 1. Model (tanpura2d.py)

Coordinates: x along the string (0 = bridge-side termination, L = 1 m = nut); **y** vertical = normal to the
bridge face (and to the soundboard); **z** horizontal = parallel to the bridge face, across the string.
String, thread, bridge, loss, excitation shape and output definition are identical to the 1-D prototype
(C3 steel string, mu = 4.836e-4 kg/m, EI = 6.04e-5 N m^2, T0 = 33.10 N, f1 = 130.81 Hz, 133 modes < 20 kHz,
Woodhouse losses etaF 5e-5 / etaA 1.2 / etaB 1e-2, thread k_c 1.2e5 N/m, r_c 0.5 kg/s on a 1 mm sin^2 window,
bridge h_b = -3 (x - 0.010)^2 m grazing the string at rest, k_b = 1e11 N/m^2, alpha = 1, sampled every 0.2 mm).

Continuous equations (f_e = excitation, psi = sin^2 windows):

    mu y_tt = T(t) y_xx - EI y_xxxx - loss_y + f_b(x,t)          - psi_c (k_c u_cy  + r_c  d/dt u_cy ) + cos(phi) psi_e F_e(t)
    mu z_tt = T(t) z_xx - EI z_xxxx - loss_z + f_fr(x,t)         - psi_c (k_cz u_cz + r_cz d/dt u_cz) + sin(phi) psi_e F_e(t)
    f_b   = k_b max(h_b(x) - y, 0)^alpha               (contact normal = +y: the jawari face is ruled across the string)
    f_fr  = -mu_f(z_t) f_b,   mu_f(v) = c1 atan(c2 v),  c1 = 0.39, c2 = 2.84 s/m   (VT-2015 kinetic law; v mu_f(v) >= 0)
    T(t)  = T0 + (EA / 2L) P,   P = int (y_x^2 + z_x^2) dx     (Kirchhoff-Carrier, couples both polarisations)
    nut coupling: each mode pair (q_i, p_i) has principal axes rotated by theta = 30 deg from y, with frequencies
                  f_i and f_i 2^(1/1200) (1 cent) along them, common damping sigma_i (anisotropic nut support;
                  "coupling of controllable level at the termination opposite the barrier", VT-2015).

Modal form: y = sum q_i sin(beta_i x), z = sum p_i sin(beta_i x), modal mass m = mu L / 2.

Discretisation (fs = 176.4 kHz = 4 x 44.1 kHz, output decimated to 44.1 kHz, as in 1-D):
* linear part: exact-pole two-step recursion per principal axis, written in (y, z) as
  `x_i^{n+1} = A_i x_i^n - b_i x_i^{n-1} + c_i F_i^n` with A_i = R diag(a_1i, a_2i) R^T, which is exactly the
  energy-consistent form m [delta_tt x + 2 sigma* delta_t. x + K*_i x^n] = F (modal frequencies and decay exact);
* tension modulation: linearly implicit, F_TM,i = -(EA/8) P^n beta_i^2 (x_i^{n+1} + x_i^{n-1}) -> a diagonal,
  mode-dependent modification of the force compliance every step (no lag, no iteration);
* bridge normal force: discrete gradient [V(eta^{n+1}) - V(eta^{n-1})] / (eta^{n+1} - eta^{n-1}) at every bridge
  point, solved by Newton on the candidate contact set (thread forces eliminated analytically, as in 1-D; if a
  point outside the candidate set ends up penetrating it is added and the step is re-solved -- never needed);
* friction: f_fr,j = -mu_f(w_j) f_b,j with the centred slip velocity w_j = (z_j^{n+1} - z_j^{n-1}) / 2 dt.  Within a
  step the y-problem does not depend on z (friction acts in z only, the TM tension is known at step n, the nut
  coupling is explicit), so the solve is block-triangular: Newton for y, then a damped Newton for w (unique root:
  the w-system is the gradient of a strictly convex function);
* everything inside one numba `@njit` loop (`_run`).

Discrete energy (x_i = (q_i, p_i)):

    H^{n+1/2} = sum_i (m/2) [ |(x_i^{n+1} - x_i^n)/dt|^2 + x_i^{n+1 T} K*_i x_i^n ]
              + sum_j dx (V(eta_j^{n+1}) + V(eta_j^n)) / 2
              + (k_c/4)(u_cy^{n+1 2} + u_cy^{n 2}) + (k_cz/4)(u_cz^{n+1 2} + u_cz^{n 2}) + (EA/8L) P^{n+1} P^n

    (H^{n+1/2} - H^{n-1/2}) / dt = - sum_i 2 m sigma*_i |delta_t. x_i|^2 - r_c (delta_t. u_cy)^2 - r_cz (delta_t. u_cz)^2
                                   - dx sum_j mu_f(w_j) w_j f_b,j   +   F_e (cos phi delta_t. u_ey + sin phi delta_t. u_ez)

Every loss term is >= 0 (f_b >= 0 because V is convex and non-decreasing, and w mu_f(w) >= 0), and H >= 0 because
omega* dt < 2 on both axes and P >= 0.  So the scheme is stable for any coupling angle, detuning, friction law or
TM strength, and with damping and friction off it conserves H exactly (to round-off).

Excitation: the 1-D whole-string sin^2 force pulse (0.2 N, 10 ms rise, 1 ms release, ~1.15 mm at mid-string),
applied at **phi = 20 deg from the vertical** (see 3).  Outputs: force of the string on the bridge assembly,
vertical F_y = F_0y - F_cy - sum dx f_b and horizontal F_z = F_0z - F_cz - sum dx f_fr (termination + thread
+ barrier), nut forces (y, z), polarisation energies, slip speed, tension rise, solver statistics.

---------------------------------------------------------------------------------------------------------
## 2. Choices that are mine, not the papers'

| item | value | why / status |
|---|---|---|
| pluck direction | 20 deg from vertical (variants 5, 45 deg) | The task asks for "mostly vertical". With a purely vertical pluck and no nut coupling, z is never excited (test A2 gives exactly 0), so 2-D collapses to 1-D. I found no measurement of tanpura pluck direction; a finger stroke across the strings may well have a large horizontal part, hence the 45 deg variant. sin^2(20 deg) puts 12% of the initial energy in z. |
| nut coupling | principal axes 30 deg, 1 cent split, common damping (variants: none, 0.2, 5 cents) | VT-2015 gives only "coupling of controllable level". A constant-cents split is what a stiff anisotropic end support gives (an end correction dL = T/K, the same relative shift for all modes): 1 cent ~ 0.6 mm end correction, 5 cents ~ 2.9 mm. Not measured. |
| friction | mu_f = 0.39 atan(2.84 v) (VT-2015 fit; units of c2 assumed s/m), variant c2 = 2840 s/m | at the nominal pluck the slip is 15-140 mm/s, so mu_f stays small (<= 0.15); the variant is close to Coulomb (mu up to 0.61) |
| thread in z | same k_c, r_c as in y (variant: k_cz = 1.2e3 N/m, r_cz = 0.05) | the thread is assumed to grip the string laterally; max \|F_cz\| = 0.14 N is far below the ~1.4 N grip of a 5 deg break angle with thread friction 0.5 |
| tension modulation | Kirchhoff-Carrier, EA = 1.23e4 N | at the nominal pluck the tension rises by at most 39 mN (0.12%), so it is negligible here (1dtm = 1d within the noise) |
| mono mix | F_y + 0.3 F_z | assumes the soundboard admittance normal to its plane is about 10 dB higher than in-plane. Not measured. All sensitivity numbers are given for the vertical force (directly comparable with the 1-D model (tanpura1d.py)) and for this mix. |
| damping of z | same Woodhouse sigma_i as y | needed so that the rotated two-step update stays symmetric. Different damping per polarisation is not implemented. |

---------------------------------------------------------------------------------------------------------
## 3. Validation (validate2d.py -> validation2d_results.txt / .json; about 2 min)

| test | result | pass |
|---|---|---|
| A. Modal frequencies with contact, thread and TM off, 8 s at 44.1 kHz. Nut force projected on the principal axes (theta 30 deg, 5 cent split) | Max error 0.00001 cents for partials 1-100 on both axes, and the split is exactly 5.0000 cents. With theta = 0 and a vertical pluck the horizontal force is exactly 0. TM mode-1 pitch shift / Duffing prediction 3 g a^2 / (8 w0) = 1.002 at 1.04 mm and 1.000 at 4.05 mm (a 9.75 cent glide). | yes |
| B. Energy. (1) Lossless with contact + thread + TM + nut coupling, friction off, 1 s at 176.4 kHz. (2) Friction on, c2 = 2.84 and 2840. (3) Lossy full model, 3 s | (1) H drift 3.2e-11 relative. (2) The balance H + D_damp + D_fric - W closes to 2.2e-11 / 1.8e-11; H increased in 0 steps; friction removed 27% / 41% of the energy in 1 s (lossless string, so friction is the only loss). (3) Residual 7.4e-12; H never increased. 0 Newton failures. | yes |
| C. Convergence of nominal 2-D, 5 s: fs 352.8k, modes to 40 kHz (222), bridge dx 0.1 mm, k_b 1e12, tolerances x1e-3 / x1e-2; noise floor from +/-1 um thread shifts | Worst deviations: mean buzz 1.12 dB (0.3-3 s) and 1.32 dB (3-5 s); hf_frac 0.0295; bands 0-15 kHz 1.99 dB; 15-20 kHz 3.0 dB; z/y level 1.64 dB. All are set by the 40 kHz mode extension. Criteria were < 1.5 / 2.5 dB, 0.03 and 2 dB, so the mode extension passes **only by a hair**. Noise floor 0.19 / 0.64 dB. The thread-sensitivity measure itself was also re-run at fs x2 and with modes to 40 kHz (section 5). Without x4 oversampling (88.2 kHz) the 15-20 kHz band rises by 6.3 dB. | yes (marginal) |
| D. 1-D limit: pol2 off, TM off vs tanpura1d.py, 10 s | Identical to 2.1e-9 for 0-0.3 s. The waveforms separate (difference > 1e-3) after 0.58 s, because round-off differences are amplified by the **chaotic contact dynamics**. Averaged metrics agree: buzz -11.25 / -11.24 dB (0.3-3 s) and -17.89 / -18.03 dB (3-8 s). | yes |
| E. Contact and thread sanity, x_c 3-8 mm, nominal and 4x pluck | Max penetration 329 nm = 0.36% of the thread lift. No contact at the edge points. 0 Newton or friction failures in E and in all 105 renders; 0 candidate-set re-solves. Max \|F_c\| is 0.71 N (y) and 0.25 N (z), within the bilateral-thread validity set by a 5 deg break angle. Max slip 0.40 m/s, max mu_f 0.33, max tension rise 0.6 N (4x pluck). | yes |

The first run of A3 failed (ratio 0.935 at 4 mm). It took the amplitude from the TM-off run, but the pluck response
of the stiffer TM string is 3.3% smaller. With the amplitude taken from the TM-on run's own energy the ratio is
1.000; both runs are in validation2d_results.txt.

**Chaos and the noise floor.** Because the contact dynamics amplify round-off exponentially, any position effect is
judged against "twins": runs at x_c = 4.999 / 5.000 / 5.001 / 5.002 mm, a 1 um shift that changes the thread lift by
only 0.03 um. The 0.5 s-smoothed metrics of the twins agree to 0.01-0.9 dB (more in 2-D than in 1-D). Every
sensitivity number below is reported next to this floor.

---------------------------------------------------------------------------------------------------------
## 4. Experiments (render2d.py, sens2d.py)

* All runs use 10 s at 176.4 kHz, decimated to 44.1 kHz, and the shared `metrics.py` module.
* Thread positions are x_c = 3, 4, 5, 6.5 and 8 mm (thread lift 147 / 108 / 75 / 37 / 12 um).
* Twins as above.
* Variants:
  * `1d`: horizontal polarisation and TM off, i.e. the colleague's model;
  * `1dtm`: 1d + TM;
  * `2d`: nominal 2-D;
  * `2dnf`: 2d with friction off;
  * `2dcf`: 2d with near-Coulomb friction;
  * `2dnc`: no nut coupling;
  * `2dd02` / `2dd5`: 0.2 / 5 cent nut split;
  * `2dp5` / `2dp45`: pluck at 5 / 45 deg;
  * `2dsz`: laterally soft thread;
  * `@2x` (fs 352.8 kHz) and `@m40` (modes to 40 kHz) repeat the sweep for 1d and 2d, and @2x also for 2dd5.

**Sensitivity measures** (computed on 0.5 s moving-average tracks):
* **S_track (primary)**: the across-position standard deviation of the buzz_db track, averaged over 0.5-6 s.
* **S_late**: the same over 3-8 s.
* Across-position std of these scalars:
  * mean buzz (0.5-6 s);
  * onset t_on: first time hf_frac >= 0.1;
  * persistence t_end15 / t_end20: last time buzz >= -15 / -20 dB;
  * jvari length jv_len: time from the buzz maximum until it has fallen 10 dB.

## 5. Results (vertical bridge force; sensitivity_results.txt has everything, including the mono mix)

Each cell shows the spread across thread positions / the spread across the 1 um twins:

| variant | S_track dB | S_late dB | std mean buzz dB | std t_on s | std t_end15 s | std t_end20 s | std jv_len s |
|---|---|---|---|---|---|---|---|
| 1d | **3.76** / 0.13 | 2.84 / 0.21 | 2.82 / 0.05 | 0.23 / 0.00 | 1.48 / 0.03 | 1.97 / 0.00 | 2.00 / 0.00 |
| 1dtm | 3.65 / 0.15 | 2.69 / 0.22 | 2.66 / 0.02 | 0.23 / 0.00 | 1.66 / 0.02 | 1.90 / 0.00 | 1.86 / 0.01 |
| 2d (nominal) | **4.44** / 0.57 | 3.07 / 0.93 | 2.33 / 0.29 | 0.32 / 0.00 | 0.52 / 0.08 | 0.80 / 0.11 | 0.74 / 0.02 |
| 2dnf (friction off) | 4.38 / 0.55 | 2.47 / 1.08 | 2.51 / 0.08 | 0.33 / 0.00 | 0.51 / 0.09 | 0.63 / 0.07 | 1.56 / 0.09 |
| 2dcf (near-Coulomb) | 3.99 / 0.24 | 2.47 / 0.81 | 2.63 / 0.14 | 0.25 / 0.00 | 0.62 / 0.06 | 0.97 / 1.02 | 1.40 / 0.00 |
| 2dnc (no nut coupling) | 3.70 / 0.17 | 2.79 / 0.28 | 2.68 / 0.04 | 0.23 / 0.00 | 1.67 / 0.03 | 1.90 / 0.01 | 1.88 / 0.00 |
| 2dd02 (0.2 cent) | 3.73 / 0.28 | 2.58 / 0.46 | 2.55 / 0.25 | 0.24 / 0.00 | 1.47 / 0.07 | 1.64 / 0.05 | 1.91 / 0.00 |
| 2dd5 (5 cent) | **4.83** / 0.02 | 3.39 / 0.05 | **4.61** / 0.01 | **1.35** / 0.00 | **4.18** / 0.04 | 1.51 / 0.00 | 1.29 / 0.00 |
| 2dp5 (pluck 5 deg) | 4.15 / 0.91 | 2.70 / 1.91 | 2.78 / 0.55 | 0.32 / 0.00 | 0.34 / 0.69 | 1.84 / 2.89 | 1.54 / 0.02 |
| 2dp45 (pluck 45 deg) | 4.53 / 0.13 | 2.63 / 0.16 | 2.31 / 0.07 | 0.35 / 0.00 | 1.56 / 0.02 | 0.37 / 0.02 | 0.94 / 0.02 |
| 2dsz (soft lateral thread) | 3.68 / 0.18 | 2.69 / 0.26 | 2.67 / 0.10 | 0.23 / 0.00 | 2.73 / 0.03 | 2.00 / 0.02 | 2.08 / 0.00 |
| 1d @ 352.8 kHz | 3.47 | 2.36 | 2.56 | 0.23 | 0.87 | 1.38 | 1.97 |
| 2d @ 352.8 kHz | 4.65 | 3.42 | 1.98 | 0.32 | 0.56 | 0.80 | 0.85 |
| 2dd5 @ 352.8 kHz | 4.81 | 3.45 | 4.60 | 1.34 | 4.26 | 1.51 | 1.46 |
| 1d, modes to 40 kHz | 3.02 | 1.78 | 2.01 | 0.22 | 0.66 | 1.48 | 2.27 |
| 2d, modes to 40 kHz | 4.22 | 2.92 | 2.19 | 0.31 | 0.50 | 0.55 | 1.40 |

The 2-D/1-D ratio of S_track is 1.18 at base resolution, 1.34 at fs x2 and 1.40 with modes to 40 kHz. The
persistence measures (t_end15/20) are poorly converged in 1-D (1.48 -> 0.87 -> 0.66 s): they are threshold
crossings of a fluctuating track. Treat them as qualitative.

Per-position values, vertical force (t in s, buzz in dB; Ez = mean share of string energy in z over 0.5-6 s):

| x_c | 1d: mean buzz / t_on / t_end20 / t_bmax | 2d: mean buzz / t_on / t_end20 / t_bmax / Ez | 2dd5: mean buzz / t_on / t_end15 / Ez |
|---|---|---|---|
| 3 mm | -13.1 / 0.21 / 9.24 / 0.67 | -14.1 / 0.23 / 4.67 / 0.67 / 0.86 | -11.3 / 0.35 / 9.66 / 0.66 |
| 4 mm | -14.1 / 0.26 / 8.59 / 0.72 | -13.1 / 0.28 / 6.50 / 0.72 / 0.85 | -13.1 / 0.49 / 9.31 / 0.63 |
| 5 mm | -14.8 / 0.33 / 9.66 / 0.77 | -13.2 / 0.35 / 4.69 / 0.77 / 0.83 | -15.3 / 2.25 / 7.20 / 0.59 |
| 6.5 mm | -16.1 / 0.46 / 7.36 / 0.74 | -13.0 / 0.53 / 5.85 / 3.39 / 0.79 | -16.7 / 2.62 / 3.16 / 0.52 |
| 8 mm | -20.3 / 0.79 / 4.76 / 1.95 | -18.4 / 1.02 / 5.80 / 4.57 / 0.71 | -23.3 / 3.39 / never / 0.45 |

**What changes in 2-D, and why.**
* **Energy moves into z.** The share of the string energy in z rises from 12% at the pluck to about 95% by 2.5 s
  for x_c = 3-5 mm, and to about 85% by 4 s at 8 mm (fig_1d_vs_2d_by_position.png, right).
  * The vertical polarisation loses energy to the jawari, which pumps it into fast-decaying high partials.
  * The horizontal polarisation never touches the bridge except through weak friction, so it keeps its energy.
* **That shift depends on the thread position.** It is weaker for a low thread (8 mm: 0.71) than for a high one
  (3 mm: 0.86).
* **In the late tone the vertical force is fed mainly from z through the nut coupling.** So after about 4-5 s the
  nominal 2-D buzz is mostly 5-10 dB below the 1-D buzz, and the positions converge. That is why the sustain-time
  spread shrinks.
* **For the low threads (6.5 and 8 mm) the buzz blooms a second time at 3.4-4.6 s.** Energy returning from z drives
  this second bloom, and it is the main source of the larger S_track.
* **With a 5 cent split the y-z exchange is five times faster.** The horizontal polarisation then acts as a
  reservoir that keeps re-supplying the vertical one.
  * The jvari then lasts the whole 10 s for x_c <= 5 mm, and its onset is delayed by seconds for low threads.
  * Thread position then strongly shapes the whole envelope (2dd5 panel of fig_buzz_tracks_extra.png).
* **Friction and TM do not couple the polarisations noticeably** at this pluck amplitude (2dnc ~ 1dtm ~ 1d).

---------------------------------------------------------------------------------------------------------
## 6. Limitations (please read)

* **The answer depends on an unmeasured parameter.** Whether 2-D "adds" thread sensitivity depends on the nut
  coupling strength (detuning). I could not obtain the VT-2015 values or results.
* **The 1-D baseline is already strongly thread-sensitive** in this DAFx-17-style parameterisation, so I cannot
  reproduce the published claim that 1-D "fails". Moving the thread here changes the thread lift 3(x_b - x_c)^2
  over 12-147 um, with the crest fixed.
  * Note that for a real thread of fixed diameter d on an exactly parabolic bridge, the string is tangent to the
    parabola at sqrt(d/c) beyond the thread whatever x_c is. Moving the thread then only *translates* the geometry.
  * So realistic thread-placement studies need a measured, non-parabolic jawari profile, or a model of the thread
    as a contact with finite diameter. Neither is in this model.
* **Contact normal is fixed and vertical.** The bridge face is ruled and flat across the string. There is no groove
  and no transverse tilt, which would couple y and z geometrically at the contact.
* **Unverified friction law.** The VT-2015 friction law, its c2 units and the thread's lateral stiffness are
  unverified. Friction has no static (stick) branch.
* **Same physical simplifications as the 1-D model:**
  * idealised whole-string force pulse instead of a finger pluck;
  * untuned losses;
  * no body, radiation or other strings;
  * 133 modes, so the string cannot wrap the bridge at mm scale.
* **Convergence is partial.**
  * The mode-extension test passes by a hair, and the 15-20 kHz band moves by 3 dB.
  * The instantaneous trajectories are chaotic. Only smoothed or averaged quantities are meaningful.
  * The noise floor of 2-D (0.6 dB on S_track) is about four times that of 1-D.
* **Small samples.** The twins are 4 samples (2-4 for the extra variants) and the sweep has 5 positions. The stds
  are descriptive, not statistically tested.
* **Assumed mono mix.** The 0.3 horizontal weight in the mono mix is an assumption, not a measured value.

---------------------------------------------------------------------------------------------------------
## 7. How to run, files, speed

    cd jawari
    python validate2d.py              # tests A-E (~2 min) -> validation2d_results.txt/.json
    python render2d.py --jobs 2       # the 76 sweep cases incl. twins (~10 min wall with 2 jobs; skips existing runs/<case>.npz)
    python render2d.py "1d_xc3@2x" ...  # convergence cases: {1d,2d,2dd5}_xc{3,4,5,6p5,8}@2x, {1d,2d}_xc*@m40 (rendered: 25)
    python render2d.py 2d_xc5 "2d_xc8@2x"   # selected cases; suffix @2x = fs 352.8 kHz, @m40 = modes to 40 kHz
    python sens2d.py                  # sensitivity tables + figures -> sensitivity_results.txt/.json, png/fig_*.png
    from tanpura2d import simulate; p, s, r = simulate(dict(xc=0.006, dur=5, detune_cents=5), energy=False)
      # r: Fy, Fz (bridge forces), Fny, Fnz (nut), Fcont, Ffric, H/D/Dfric/W (energy), Ey/Ez/dT/slip (decimated), stats

**Files**
* Code:
  * `tanpura2d.py`: the model; the earlier `quick_*.py` scratch files left by the interrupted attempt are in `old_attempt/` (quick_energy.py there mislabels its energy terms; ignore it).
  * `common.py`, `validate2d.py`, `render2d.py`, `sens2d.py`.
  * `tanpura1d.py`: an untouched copy of the 1-D model.
* Data:
  * `runs/*.npz`: 44.1 kHz y/z bridge and nut forces, metric tracks and stats for every case.
  * `render2d_results.txt`: a metrics.py report for every case (vertical, mono and horizontal).
  * `core_reports.txt`: the vertical reports of the core 30 cases.
* Audio, in `wav/`:
  * `<case>_vertical.wav` (F_y), `<case>_mono.wav` (F_y + 0.3 F_z) and `<case>_stereo_y_z.wav`;
  * each normalised to a peak of 0.9;
  * for the 55 main (non-twin) cases.
* Figures, in `png/`:
  * `<case>_vertical.png` and `_mono.png` spectrograms;
  * `fig_buzz_tracks.png` (core variants);
  * `fig_buzz_tracks_extra.png` (coupling, pluck and thread variants);
  * `fig_buzz_tracks_convergence.png`;
  * `fig_1d_vs_2d_by_position.png`;
  * `fig_sensitivity.png`;
  * `_mono` versions of the above.

**Speed**: everything runs inside one numba `@njit` loop, single-threaded. CPU time per second of audio at 176.4 kHz:

| case | s per s of audio |
|---|---|
| 2-D (133 modes x 2 polarisations, TM, friction) | 1.2 |
| 1-D, same code | 1.14 |
| 352.8 kHz | 2.3-2.4 |
| modes to 40 kHz | 2.2-2.3 |

The 2dp45/2dsz runs show 1.9-2.3 s per s, partly because 4 processes shared 4 CPUs.
