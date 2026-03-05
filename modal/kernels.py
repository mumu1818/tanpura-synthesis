from __future__ import annotations
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator


@njit(cache=True, fastmath=True, nogil=True)
def modal_step_verlet(
    q, q_dot, omega, sigma, phi_bridge, y_bridge_geo,
    K_contact, alpha, dt, tension_mod_coeff
):
    """Energy-conserving Stormer-Verlet integration with tension modulation.

    Unlike explicit Euler (q += q_dot*dt), Stormer-Verlet is symplectic:
    it conserves a shadow Hamiltonian, preventing energy drift that causes
    the instability bailouts in the original code.
    """
    n_modes = len(q)
    n_bridge = phi_bridge.shape[1]

    # --- Half-step velocity ---
    # First compute forces at current position

    # Modal -> Physical displacement at bridge points
    y_bridge = np.zeros(n_bridge)
    for j in range(n_bridge):
        val = 0.0
        for i in range(n_modes):
            val += q[i] * phi_bridge[i, j]
        y_bridge[j] = val

    # Contact force (penalty method)
    f_space = np.zeros(n_bridge)
    total_pen = 0.0
    for j in range(n_bridge):
        pen = y_bridge_geo[j] - y_bridge[j]
        if pen > 0:
            f_space[j] = K_contact * (pen ** alpha)
            total_pen += pen

    # Physical -> Modal force projection
    f_modal = np.zeros(n_modes)
    for i in range(n_modes):
        val = 0.0
        for j in range(n_bridge):
            val += phi_bridge[i, j] * f_space[j]
        f_modal[i] = val

    # Tension modulation: when string wraps on bridge, it stretches,
    # increasing tension and shifting all mode frequencies upward.
    # Delta_omega ~ tension_mod_coeff * total_penetration
    omega_eff = np.empty(n_modes)
    for i in range(n_modes):
        omega_eff[i] = omega[i] * (1.0 + tension_mod_coeff * total_pen)

    # --- Stormer-Verlet ---
    # Half-step velocity
    for i in range(n_modes):
        acc = -(omega_eff[i] ** 2 * q[i]) - (2.0 * sigma[i] * q_dot[i]) + f_modal[i]
        q_dot[i] += 0.5 * acc * dt

    # Full-step position
    for i in range(n_modes):
        q[i] += q_dot[i] * dt

    # Recompute forces at new position
    for j in range(n_bridge):
        val = 0.0
        for i in range(n_modes):
            val += q[i] * phi_bridge[i, j]
        y_bridge[j] = val

    total_pen2 = 0.0
    for j in range(n_bridge):
        pen = y_bridge_geo[j] - y_bridge[j]
        if pen > 0:
            f_space[j] = K_contact * (pen ** alpha)
            total_pen2 += pen
        else:
            f_space[j] = 0.0

    for i in range(n_modes):
        val = 0.0
        for j in range(n_bridge):
            val += phi_bridge[i, j] * f_space[j]
        f_modal[i] = val

    for i in range(n_modes):
        omega_eff[i] = omega[i] * (1.0 + tension_mod_coeff * total_pen2)

    # Second half-step velocity
    for i in range(n_modes):
        acc = -(omega_eff[i] ** 2 * q[i]) - (2.0 * sigma[i] * q_dot[i]) + f_modal[i]
        q_dot[i] += 0.5 * acc * dt

    return q, q_dot
