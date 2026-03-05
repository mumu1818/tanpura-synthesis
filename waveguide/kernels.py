from __future__ import annotations
import math
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator


@njit(cache=True, fastmath=True, nogil=True)
def delay_line_read(buffer: np.ndarray, write_pos: int, delay: int) -> float:
    """Read from circular delay line at given delay."""
    N = len(buffer)
    read_pos = (write_pos - delay) % N
    return buffer[read_pos]


@njit(cache=True, fastmath=True, nogil=True)
def delay_line_write(buffer: np.ndarray, write_pos: int, value: float) -> int:
    """Write to circular delay line, return new write position."""
    buffer[write_pos] = value
    return (write_pos + 1) % len(buffer)


@njit(cache=True, fastmath=True, nogil=True)
def allpass_step(x: float, state: np.ndarray, coeff: float) -> float:
    """First-order allpass filter for modelling dispersion.

    H(z) = (coeff + z^-1) / (1 + coeff * z^-1)
    """
    y = coeff * x + state[0] - coeff * state[1]
    state[0] = x
    state[1] = y
    return y


@njit(cache=True, fastmath=True, nogil=True)
def one_pole_lowpass(x: float, state: np.ndarray, g: float) -> float:
    """One-pole lowpass for frequency-dependent damping.

    H(z) = (1-g) / (1 - g*z^-1)
    """
    y = (1.0 - g) * x + g * state[0]
    state[0] = y
    return y


@njit(cache=True, fastmath=True, nogil=True)
def bridge_contact_force(y_string: float, y_bridge: float, K: float, alpha: float) -> float:
    """Penalty-based contact force at a single point."""
    pen = y_bridge - y_string
    if pen > 0.0:
        return K * (pen ** alpha)
    return 0.0


@njit(cache=True, fastmath=True, nogil=True)
def waveguide_step(
    upper_line, lower_line, write_pos, delay_samples,
    loss_state, disp_state_upper, disp_state_lower,
    loss_coeff, disp_coeff,
    bridge_profile, n_bridge_pts, bridge_spacing,
    K_contact, alpha, dt, output_tap
):
    """One time step of the digital waveguide with bridge contact.

    The string is modelled as two travelling waves (upper = right-going,
    lower = left-going). At the bridge end, contact forces inject energy
    back into the waveguide.
    """
    N = delay_samples

    # Read the arriving waves at both ends
    right_arriving = delay_line_read(upper_line, write_pos, N)
    left_arriving = delay_line_read(lower_line, write_pos, N)

    # Bridge end reflection (with loss and dispersion)
    # The string reflects with sign flip at bridge
    reflected_right = -one_pole_lowpass(right_arriving, loss_state, loss_coeff)
    reflected_right = allpass_step(reflected_right, disp_state_upper, disp_coeff)

    # Bridge contact: check displacement at bridge points
    # Displacement at bridge end = right_arriving + left_arriving
    y_at_bridge = right_arriving + left_arriving

    # Compute contact force contribution
    # Sample the bridge profile at a few points near the bridge end
    contact_inject = 0.0
    for j in range(n_bridge_pts):
        # Position along the waveguide for this bridge point
        tap = j * bridge_spacing
        if tap >= N:
            break
        y_j = delay_line_read(upper_line, write_pos, int(tap)) + delay_line_read(lower_line, write_pos, int(tap))
        f = bridge_contact_force(y_j, bridge_profile[j], K_contact, alpha)
        contact_inject += f * dt * dt * 0.5

    # Nut end reflection (simple inversion, rigid termination)
    reflected_left = -left_arriving
    reflected_left = allpass_step(reflected_left, disp_state_lower, disp_coeff)

    # Inject contact forces into both travelling waves
    new_upper = reflected_left + contact_inject * 0.5
    new_lower = reflected_right + contact_inject * 0.5

    # Write new values
    write_pos = delay_line_write(upper_line, write_pos, new_upper)
    delay_line_write(lower_line, write_pos - 1 if write_pos > 0 else len(lower_line) - 1, new_lower)

    # Read output
    out_val = delay_line_read(upper_line, write_pos, output_tap) + delay_line_read(lower_line, write_pos, output_tap)

    return write_pos, out_val
