from __future__ import annotations
import math
import sys
import numpy as np

sys.path.insert(0, ".")

from shared.params import TanpuraParams, derived
from shared.bridge import create_bridge_profile


def synthesize(params: TanpuraParams, duration_s: float = 1.5, verbose: bool = False) -> tuple[np.ndarray, int, float]:
    from waveguide.kernels import waveguide_step

    d = derived(params)
    c = d["c"]
    EI = d["EI"]
    T = d["T"]

    sr = params.sr
    dt = 1.0 / sr

    # Delay line length = L / (c * dt) = L * sr / c
    delay_samples = int(round(params.L * sr / c))
    delay_samples = max(delay_samples, 10)

    # Dispersion coefficient from string stiffness
    # First-order allpass approximation for inharmonicity
    B = (math.pi ** 2 * EI) / (T * params.L ** 2)
    disp_coeff = min(0.9, B * 100.0)  # Scaled for audible effect

    # Loss coefficient for frequency-dependent damping
    loss_coeff = max(0.0, min(0.999, 1.0 - params.sigma0 * dt * 10.0))

    # Allocate delay lines (with padding)
    buf_len = delay_samples + 64
    upper_line = np.zeros(buf_len)
    lower_line = np.zeros(buf_len)

    # Initial pluck: inject into upper delay line as a pulse shape
    pluck_sample = int(params.pluck_pos * delay_samples)
    for i in range(delay_samples):
        dist = abs(i - pluck_sample)
        width_samples = max(1, int(params.pluck_width * delay_samples / params.L))
        upper_line[i] = params.pluck_amp * math.exp(-(dist / max(width_samples, 1)) ** 2)
        lower_line[i] = upper_line[i]

    write_pos = 0

    # Filter states
    loss_state = np.zeros(1)
    disp_state_upper = np.zeros(2)
    disp_state_lower = np.zeros(2)

    # Bridge geometry mapped to waveguide taps
    n_bridge_pts = 20
    x_bridge = np.linspace(params.bridge_start, params.bridge_end, n_bridge_pts)
    bridge_profile = create_bridge_profile(params, x_bridge)
    bridge_spacing = max(1.0, (params.bridge_end / params.L) * delay_samples / n_bridge_pts)

    # Output tap position
    output_tap = min(delay_samples - 1, max(1, int(params.output_pos * delay_samples)))

    n_audio = int(duration_s * sr)
    audio_out = np.zeros(n_audio, dtype=np.float32)

    for i in range(n_audio):
        write_pos, out_val = waveguide_step(
            upper_line, lower_line, write_pos, delay_samples,
            loss_state, disp_state_upper, disp_state_lower,
            loss_coeff, disp_coeff,
            bridge_profile, n_bridge_pts, bridge_spacing,
            params.K_contact, params.alpha, dt, output_tap
        )

        if abs(out_val) > 10.0:
            if verbose:
                print(f"Waveguide instability at sample {i}")
            return np.zeros(n_audio, dtype=np.float32), params.sr, float(i) / n_audio

        audio_out[i] = out_val

    if not np.all(np.isfinite(audio_out)):
        return np.zeros(n_audio, dtype=np.float32), params.sr, 0.5

    peak = np.max(np.abs(audio_out))
    if peak > 1e-9:
        audio_out /= peak
    return audio_out, params.sr, 1.0
