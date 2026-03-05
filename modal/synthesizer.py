from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, ".")

from shared.params import TanpuraParams, compute_modal_frequencies, compute_mode_shapes, create_pluck_excitation
from shared.bridge import create_bridge_profile
from modal.kernels import modal_step_verlet


def synthesize(params: TanpuraParams, duration_s: float = 1.5, verbose: bool = False) -> tuple[np.ndarray, int, float]:
    OVERSAMPLE = 4
    sr_physics = params.sr * OVERSAMPLE
    dt = 1.0 / sr_physics

    n_audio = int(duration_s * params.sr)

    freqs, omega = compute_modal_frequencies(params)
    sigma = params.sigma0 + (params.sigma1 * omega)

    x_bridge = np.linspace(params.bridge_start, params.bridge_end, 40)
    phi_bridge = compute_mode_shapes(params, x_bridge)
    y_bridge_geo = create_bridge_profile(params, x_bridge)

    phi_output = compute_mode_shapes(params, np.array([params.output_pos * params.L]))[:, 0]
    q, q_dot = create_pluck_excitation(params)

    # Tension modulation coefficient: controls how much bridge contact
    # shifts mode frequencies. Derived from string properties.
    from shared.params import derived
    d = derived(params)
    tension_mod_coeff = params.E * d["A"] / (2.0 * d["T"] * params.L) * 0.01

    audio_out = np.zeros(n_audio, dtype=np.float32)

    for i in range(n_audio):
        for _ in range(OVERSAMPLE):
            q, q_dot = modal_step_verlet(
                q, q_dot, omega, sigma, phi_bridge, y_bridge_geo,
                params.K_contact, params.alpha, dt, tension_mod_coeff
            )

            if abs(q[0]) > 10.0:
                if verbose:
                    print(f"Instability at sample {i}")
                return np.zeros(n_audio, dtype=np.float32), params.sr, float(i) / n_audio

        val = 0.0
        for m in range(params.n_modes):
            val += q[m] * phi_output[m]
        audio_out[i] = val

    if not np.all(np.isfinite(audio_out)):
        return np.zeros(n_audio, dtype=np.float32), params.sr, 0.5

    peak = np.max(np.abs(audio_out))
    if peak > 1e-9:
        audio_out /= peak
    return audio_out, params.sr, 1.0
