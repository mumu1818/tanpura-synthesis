import numpy as np
from shared.params import TanpuraParams


def create_bridge_profile(params: TanpuraParams, x_bridge: np.ndarray) -> np.ndarray:
    x_norm = (x_bridge - params.bridge_start) / (params.bridge_end - params.bridge_start)
    base_curve = params.bridge_height * (1.0 + x_norm ** float(params.bridge_curve_p))

    bridge_len = params.bridge_end - params.bridge_start
    thread_loc_x = params.bridge_start + (params.jiva_pos * bridge_len)
    thread_width = 0.001

    thread_bump = params.jiva_height * np.exp(
        -((x_bridge - thread_loc_x) ** 2) / (2 * thread_width ** 2)
    )

    return base_curve + thread_bump
