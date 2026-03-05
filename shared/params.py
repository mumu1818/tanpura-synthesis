from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class TanpuraParams:
    # --- REAL WORLD MEASUREMENTS (FIXED) ---
    target_f0: float = 69.4   # C#2
    L: float = 1.05           # 105 cm Length
    d: float = 0.0007         # 0.7 mm Gauge (Kharaj)
    rho: float = 7800.0       # Steel Density
    E: float = 2e11           # Steel Young's Modulus

    # Bridge Dimensions
    bridge_start: float = 0.0
    bridge_end: float = 0.04  # 4 cm Wide

    # --- SIMULATION PARAMETERS (TUNABLE) ---
    sr: int = 44100
    n_modes: int = 80

    # Damping
    sigma0: float = 0.1
    sigma1: float = 5e-6

    # --- THE JIVA (THREAD) ---
    jiva_pos: float = 0.3
    jiva_height: float = 0.0003

    # Bridge Geometry
    bridge_height: float = -0.0005
    bridge_curve_p: float = 2.0

    # Physics
    K_contact: float = 1e6
    alpha: float = 2.0

    # Pluck
    pluck_pos: float = 0.82
    pluck_width: float = 0.02
    pluck_amp: float = 0.003
    output_pos: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TanpuraParams":
        valid = cls.__annotations__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid})


def derived(params: TanpuraParams) -> dict[str, float]:
    A = math.pi * (params.d / 2) ** 2
    mu = params.rho * A
    EI = params.E * (math.pi * (params.d / 2) ** 4 / 4)
    T = mu * (2 * params.L * params.target_f0) ** 2
    c = math.sqrt(T / mu)
    return {"A": A, "mu": mu, "EI": EI, "T": T, "c": c}


def compute_modal_frequencies(params: TanpuraParams) -> tuple[np.ndarray, np.ndarray]:
    d = derived(params)
    n = np.arange(1, params.n_modes + 1)
    B = (np.pi**2 * d["EI"]) / (d["T"] * params.L**2)
    f0_ideal = d["c"] / (2 * params.L)
    freqs = n * f0_ideal * np.sqrt(1 + B * n**2)
    return freqs, 2 * np.pi * freqs


def compute_mode_shapes(params: TanpuraParams, x_positions: np.ndarray) -> np.ndarray:
    n = np.arange(1, params.n_modes + 1)
    return np.sqrt(2 / params.L) * np.sin(np.outer(n * np.pi / params.L, x_positions))


def create_pluck_excitation(params: TanpuraParams) -> tuple[np.ndarray, np.ndarray]:
    n = np.arange(1, params.n_modes + 1)
    q0 = np.zeros(params.n_modes)
    for i, mode_n in enumerate(n):
        phi = np.sin(mode_n * np.pi * params.pluck_pos)
        decay = 1.0 / (mode_n ** 2)
        filt = np.exp(-(mode_n * params.pluck_width * 5) ** 2)
        q0[i] = params.pluck_amp * decay * phi * filt * (2 / params.L)
    return q0, np.zeros(params.n_modes)
