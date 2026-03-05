from __future__ import annotations
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator


@njit(cache=True, fastmath=True, nogil=True)
def fdtd_step(y, y_prev, bridge_geo, bridge_start_idx, bridge_end_idx,
              c2_dt2_dx2, kappa2_dt2_dx4, sigma0_dt, sigma1_dt_dx2,
              K_contact, alpha, dt):
    """One time step of the stiff string FDTD scheme with bridge contact.

    PDE: y_tt = c^2 y_xx - kappa^2 y_xxxx - 2*sigma0*y_t + 2*sigma1*y_txx + f_contact

    Discretised using centered differences in space and time.
    Energy-conserving contact via penalty potential.
    """
    N = len(y)
    y_next = np.zeros(N)

    # Interior points (indices 2..N-3 for 4th-order spatial derivative)
    for i in range(2, N - 2):
        # Second spatial derivative: y_xx
        d2y = y[i + 1] - 2.0 * y[i] + y[i - 1]

        # Fourth spatial derivative: y_xxxx
        d4y = y[i + 2] - 4.0 * y[i + 1] + 6.0 * y[i] - 4.0 * y[i - 1] + y[i - 2]

        # Velocity approximation for damping: (y - y_prev) / dt
        vel = y[i] - y_prev[i]

        # Frequency-dependent damping: sigma1 * y_txx ~ sigma1 * (d2y - d2y_prev) / dt
        d2y_prev = y_prev[i + 1] - 2.0 * y_prev[i] + y_prev[i - 1]
        vel_xx = d2y - d2y_prev

        # Contact force at bridge points
        f_contact = 0.0
        if bridge_start_idx <= i <= bridge_end_idx:
            # Bridge index mapped to bridge_geo array
            geo_idx = i - bridge_start_idx
            if geo_idx < len(bridge_geo):
                pen = bridge_geo[geo_idx] - y[i]
                if pen > 0.0:
                    f_contact = K_contact * (pen ** alpha) * dt * dt

        # Update equation (centered time difference, rearranged for y_next)
        denom = 1.0 + sigma0_dt
        y_next[i] = (
            2.0 * y[i] - y_prev[i]
            + c2_dt2_dx2 * d2y
            - kappa2_dt2_dx4 * d4y
            - sigma0_dt * y_prev[i]
            + sigma1_dt_dx2 * vel_xx
            + f_contact
            + sigma0_dt * y[i]
        ) / denom

    # Fixed boundary conditions (y=0 at both ends)
    y_next[0] = 0.0
    y_next[1] = 0.0
    y_next[N - 1] = 0.0
    y_next[N - 2] = 0.0

    return y_next
