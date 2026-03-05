from __future__ import annotations
import math
import sys
import numpy as np

sys.path.insert(0, ".")

from shared.params import TanpuraParams, derived
from shared.bridge import create_bridge_profile
from fdtd.kernels import fdtd_step


def synthesize(params: TanpuraParams, duration_s: float = 1.5, verbose: bool = False) -> tuple[np.ndarray, int, float]:
    d = derived(params)
    T = d["T"]
    mu = d["mu"]
    EI = d["EI"]
    c = d["c"]
    kappa = math.sqrt(EI / mu)

    # Spatial grid
    # CFL condition for stiff string: dt <= dx^2 / (c*dx + sqrt(c^2*dx^2 + 4*kappa^2))
    # We choose N to give a stable dt at the desired sample rate
    OVERSAMPLE = 2
    sr_physics = params.sr * OVERSAMPLE
    dt = 1.0 / sr_physics

    # Stability: dx must satisfy dx >= c*dt + sqrt((c*dt)^2 + 4*kappa*dt)
    # (simplified CFL for stiff string)
    dx_min = c * dt + math.sqrt((c * dt) ** 2 + 4.0 * kappa * dt)
    N = int(math.ceil(params.L / dx_min))
    N = max(N, 50)
    dx = params.L / N

    if verbose:
        print(f"FDTD: N={N} grid points, dx={dx*1000:.2f}mm, dt={dt*1e6:.1f}us, sr_phys={sr_physics}")

    # Precompute coefficients
    c2_dt2_dx2 = (c * dt / dx) ** 2
    kappa2_dt2_dx4 = (kappa * dt / (dx * dx)) ** 2
    sigma0_dt = params.sigma0 * dt
    sigma1_dt_dx2 = params.sigma1 * dt / (dx * dx)

    # Bridge geometry mapped to grid indices
    bridge_start_idx = max(0, int(params.bridge_start / dx))
    bridge_end_idx = min(N - 1, int(params.bridge_end / dx))
    n_bridge_pts = bridge_end_idx - bridge_start_idx + 1
    x_bridge = np.linspace(params.bridge_start, params.bridge_end, n_bridge_pts)
    bridge_geo = create_bridge_profile(params, x_bridge)

    # Initial conditions: pluck shape
    x = np.linspace(0, params.L, N)
    pluck_x = params.pluck_pos * params.L
    y = np.zeros(N)
    for i in range(1, N - 1):
        dist = abs(x[i] - pluck_x)
        y[i] = params.pluck_amp * math.exp(-(dist / params.pluck_width) ** 2)
    y_prev = y.copy()

    # Output position
    out_idx = min(N - 1, max(0, int(params.output_pos * params.L / dx)))

    n_audio = int(duration_s * params.sr)
    audio_out = np.zeros(n_audio, dtype=np.float32)

    step = 0
    for i in range(n_audio):
        for _ in range(OVERSAMPLE):
            y_next = fdtd_step(
                y, y_prev, bridge_geo, bridge_start_idx, bridge_end_idx,
                c2_dt2_dx2, kappa2_dt2_dx4, sigma0_dt, sigma1_dt_dx2,
                params.K_contact, params.alpha, dt
            )

            # Stability check
            max_val = 0.0
            for k in range(len(y_next)):
                if abs(y_next[k]) > max_val:
                    max_val = abs(y_next[k])
            if max_val > 10.0:
                if verbose:
                    print(f"FDTD instability at sample {i}")
                return np.zeros(n_audio, dtype=np.float32), params.sr, float(i) / n_audio

            y_prev = y
            y = y_next
            step += 1

        audio_out[i] = y[out_idx]

    if not np.all(np.isfinite(audio_out)):
        return np.zeros(n_audio, dtype=np.float32), params.sr, 0.5

    peak = np.max(np.abs(audio_out))
    if peak > 1e-9:
        audio_out /= peak
    return audio_out, params.sr, 1.0
