"""
Smoothed Particle Hydrodynamics (SPH) - Dam Break Simulation
Naive CPU implementation with N particles (2D).

Physics:
  - Cubic spline smoothing kernel (2D)
  - Tait equation of state (weakly compressible): P = B * ((rho/rho0)^gamma - 1)
  - Gravity: g = (0, -9.81)
  - Monaghan artificial viscosity
  - Reflecting solid-wall boundaries
  - Leapfrog (kick-drift-kick) time integration

Initial condition: rectangular column of water in the bottom-left corner
collapses under gravity and splashes against the opposite wall.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ---------------------------------------------------------------------------
# Kernel: 2D Cubic Spline
# ---------------------------------------------------------------------------

def W(dx, dy, h):
    """Cubic spline smoothing kernel (2D).

    W(r, h) = (10 / 7 pi h^2) * { 1 - 1.5 q^2 + 0.75 q^3   if 0 <= q < 1
                                 { 0.25 (2 - q)^3              if 1 <= q < 2
                                 { 0                            if q >= 2
    where q = r / h.
    """
    r = np.sqrt(dx**2 + dy**2)
    q = r / h
    sigma = 10.0 / (7.0 * np.pi * h**2)

    w = np.zeros_like(q)
    mask1 = q < 1.0
    mask2 = (q >= 1.0) & (q < 2.0)

    w[mask1] = 1.0 - 1.5 * q[mask1]**2 + 0.75 * q[mask1]**3
    w[mask2] = 0.25 * (2.0 - q[mask2])**3

    return sigma * w


def gradW(dx, dy, h):
    """Gradient of the cubic spline kernel (2D).

    Returns (dW/dx, dW/dy) for each pairwise interaction.
    """
    r = np.sqrt(dx**2 + dy**2)
    q = r / h
    sigma = 10.0 / (7.0 * np.pi * h**2)

    # dW/dq
    dwdq = np.zeros_like(q)
    mask1 = q < 1.0
    mask2 = (q >= 1.0) & (q < 2.0)

    dwdq[mask1] = -3.0 * q[mask1] + 2.25 * q[mask1]**2
    dwdq[mask2] = -0.75 * (2.0 - q[mask2])**2

    # dW/dr = sigma * dwdq / h,  then project: dW/dx = (dW/dr) * (dx/r)
    # avoid division by zero
    r_safe = np.where(r > 1e-12, r, 1.0)
    factor = sigma * dwdq / (h * r_safe)
    factor = np.where(r > 1e-12, factor, 0.0)

    return factor * dx, factor * dy


# ---------------------------------------------------------------------------
# Pairwise separations  (naive O(N^2))
# ---------------------------------------------------------------------------

def get_pairwise_separations(ri, rj):
    """Pairwise separation vectors ri[a] - rj[b].

    Parameters
    ----------
    ri : (M, 2)   positions set 1
    rj : (N, 2)   positions set 2

    Returns
    -------
    dx, dy : (M, N)
    """
    dx = ri[:, 0:1] - rj[:, 0:1].T
    dy = ri[:, 1:2] - rj[:, 1:2].T
    return dx, dy


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------

def get_density(r, pos, m, h):
    """SPH density at sampling locations.

    rho_i = sum_j m_j W(r_i - r_j, h)
    """
    dx, dy = get_pairwise_separations(r, pos)
    return np.sum(m * W(dx, dy, h), axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Equation of state: Tait
# ---------------------------------------------------------------------------

def get_pressure(rho, rho0, c0, gamma):
    """Tait equation of state.

    P = B * ((rho / rho0)^gamma - 1)
    B = c0^2 * rho0 / gamma
    """
    B = c0**2 * rho0 / gamma
    return B * ((rho / rho0)**gamma - 1.0)


# ---------------------------------------------------------------------------
# Acceleration
# ---------------------------------------------------------------------------

def get_acceleration(pos, vel, m, h, rho0, c0, gamma_eos, alpha_visc, g_vec):
    """Compute acceleration on every SPH particle.

    Contributions:
      1. Pressure gradient (SPH symmetric form)
      2. Monaghan artificial viscosity
      3. Gravity
    """
    N = pos.shape[0]

    # density & pressure
    rho = get_density(pos, pos, m, h)                   # (N, 1)
    P = get_pressure(rho, rho0, c0, gamma_eos)          # (N, 1)

    # pairwise separations & kernel gradient
    dx, dy = get_pairwise_separations(pos, pos)          # (N, N)
    dWx, dWy = gradW(dx, dy, h)                          # (N, N)

    # --- pressure term ---
    factor = P / rho**2                                   # (N, 1)
    sym = factor + factor.T                               # (N, N)

    # --- artificial viscosity (Monaghan) ---
    # v_ij = v_i - v_j
    dvx = vel[:, 0:1] - vel[:, 0:1].T                    # (N, N)
    dvy = vel[:, 1:2] - vel[:, 1:2].T                    # (N, N)

    # v_ij . r_ij
    vr = dvx * dx + dvy * dy                              # (N, N)

    # |r_ij|^2
    r2 = dx**2 + dy**2
    eta2 = 0.01 * h**2

    # mu_ij = h * (v_ij . r_ij) / (|r_ij|^2 + eta^2)
    mu = h * vr / (r2 + eta2)                             # (N, N)

    # average density and sound speed
    rho_avg = 0.5 * (rho + rho.T)                         # (N, N)
    c_avg = c0  # constant sound speed approximation

    # Pi_ij (only when particles approach: vr < 0)
    Pi = np.where(vr < 0.0, -alpha_visc * c_avg * mu / rho_avg, 0.0)

    # total pairwise factor
    total = sym + Pi                                       # (N, N)

    # acceleration from pressure + viscosity
    ax = -np.sum(m * total * dWx, axis=1, keepdims=True)
    ay = -np.sum(m * total * dWy, axis=1, keepdims=True)

    a = np.hstack((ax, ay))                               # (N, 2)

    # gravity
    a[:, 0] += g_vec[0]
    a[:, 1] += g_vec[1]

    return a, rho


# ---------------------------------------------------------------------------
# Boundary enforcement
# ---------------------------------------------------------------------------

def enforce_boundaries(pos, vel, x_min, x_max, y_min, y_max, damp=0.5):
    """Reflecting boundary conditions.

    When a particle leaves the domain, clamp it back to the wall
    and flip + damp the normal velocity component.
    """
    # left wall
    mask = pos[:, 0] < x_min
    pos[mask, 0] = x_min
    vel[mask, 0] *= -damp

    # right wall
    mask = pos[:, 0] > x_max
    pos[mask, 0] = x_max
    vel[mask, 0] *= -damp

    # bottom wall
    mask = pos[:, 1] < y_min
    pos[mask, 1] = y_min
    vel[mask, 1] *= -damp

    # top wall
    mask = pos[:, 1] > y_max
    pos[mask, 1] = y_max
    vel[mask, 1] *= -damp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- domain ----
    Lx, Ly = 4.0, 3.0          # tank dimensions (m)

    # ---- fluid parameters ----
    rho0      = 1000.0          # reference density (kg/m^3)
    c0        = 20.0            # artificial speed of sound (m/s)
    gamma_eos = 7               # Tait exponent
    alpha_visc = 1.0            # artificial viscosity coefficient
    g_vec     = np.array([0.0, -9.81])  # gravity

    # ---- particle setup ----
    dx_spacing = 0.12           # initial particle spacing (m)  (~200 particles)

    # water column: x in [dx/2, 1], y in [dx/2, 2]
    xs = np.arange(dx_spacing / 2, 1.0, dx_spacing)
    ys = np.arange(dx_spacing / 2, 2.0, dx_spacing)
    xx, yy = np.meshgrid(xs, ys)
    pos = np.column_stack((xx.ravel(), yy.ravel()))  # (N, 2)
    N = pos.shape[0]

    vel = np.zeros_like(pos)
    m = rho0 * dx_spacing**2    # particle mass
    h = 1.3 * dx_spacing        # smoothing length

    print(f"N = {N} particles")
    print(f"h = {h:.4f}, m = {m:.4f}, dx = {dx_spacing}")

    # ---- time stepping ----
    dt   = 1e-3                 # larger dt for fewer particles
    tEnd = 1.5                  # simulation end time (s)
    Nt   = int(np.ceil(tEnd / dt))
    plot_every = max(1, Nt // 100)  # ~100 plot frames

    # initial acceleration
    acc, rho = get_acceleration(pos, vel, m, h, rho0, c0, gamma_eos,
                                alpha_visc, g_vec)

    # ---- run simulation, store frames ----
    frames = []   # list of (pos_copy, vel_copy, t) snapshots
    t = 0.0

    # save initial frame
    frames.append((pos.copy(), vel.copy(), t))

    for i in range(Nt):
        # half-kick
        vel += acc * dt / 2.0

        # drift
        pos += vel * dt

        # boundaries
        enforce_boundaries(pos, vel, 0.0, Lx, 0.0, Ly)

        # acceleration
        acc, rho = get_acceleration(pos, vel, m, h, rho0, c0, gamma_eos,
                                    alpha_visc, g_vec)

        # half-kick
        vel += acc * dt / 2.0

        t += dt

        # store frame
        if i % plot_every == 0 or i == Nt - 1:
            frames.append((pos.copy(), vel.copy(), t))
            print(f"\r  step {i+1}/{Nt}  t={t:.3f}s", end="", flush=True)

    print(f"\nSimulation done. {len(frames)} frames captured.")

    # ---- build GIF ----
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

    def animate(frame_idx):
        p, v, ti = frames[frame_idx]
        speed = np.sqrt(v[:, 0]**2 + v[:, 1]**2)
        ax1.cla()
        ax1.scatter(p[:, 0], p[:, 1], c=speed, cmap='coolwarm',
                    s=10, vmin=0, vmax=5)
        ax1.set_xlim(-0.1, Lx + 0.1)
        ax1.set_ylim(-0.1, Ly + 0.1)
        ax1.set_aspect('equal')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_title(f'Dam Break  t = {ti:.3f} s   (N = {N})')
        ax1.set_facecolor((0.1, 0.1, 0.1))

    anim = FuncAnimation(fig, animate, frames=len(frames), interval=50)
    anim.save('dam_break.gif', writer=PillowWriter(fps=20))
    print("Saved dam_break.gif")
    plt.show()


if __name__ == '__main__':
    main()
