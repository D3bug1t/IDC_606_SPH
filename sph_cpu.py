"""
Smoothed Particle Hydrodynamics (SPH) - Toy Star Simulation
Naive CPU implementation with N particles.

Based on: Philip Mocz (2020), "Smoothed Particle Hydrodynamics: Theory,
Implementation, and Application to Toy Stars"

Physics:
  - Gaussian smoothing kernel in 3D
  - Polytropic equation of state: P = k * rho^(1 + 1/n)
  - External damped harmonic potential: a_ext = -lambda * pos - nu * vel
  - Leapfrog (kick-drift-kick) time integration

The simulation starts N particles from random positions, and they relax
(thanks to the viscous damping) into the equilibrium density profile of
a polytropic star:  rho(r) = (lambda / 4k) * (R^2 - r^2)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------

def W(x, y, z, h):
    """Gaussian smoothing kernel (3D).

    W(r, h) = [1 / (h * sqrt(pi))]^3  *  exp(-r^2 / h^2)
    """
    r2 = x**2 + y**2 + z**2
    return (1.0 / (h * np.sqrt(np.pi)))**3 * np.exp(-r2 / h**2)


def gradW(x, y, z, h):
    """Gradient of the Gaussian smoothing kernel (3D).

    dW/dr_alpha = -2 / (h^5 * pi^(3/2)) * exp(-r^2/h^2) * r_alpha
    where r_alpha is x, y, or z component.
    """
    r2 = x**2 + y**2 + z**2
    coeff = -2.0 * np.exp(-r2 / h**2) / (h**5 * np.pi**1.5)
    return coeff * x, coeff * y, coeff * z


# ---------------------------------------------------------------------------
# Pairwise separations  (naive O(N^2))
# ---------------------------------------------------------------------------

def get_pairwise_separations(ri, rj):
    """Compute all pairwise separation vectors  ri[a] - rj[b].

    Parameters
    ----------
    ri : (M, 3) array   – first set of positions
    rj : (N, 3) array   – second set of positions

    Returns
    -------
    dx, dy, dz : (M, N) arrays of separations
    """
    # broadcasting: (M,1) - (1,N) -> (M,N)
    dx = ri[:, 0:1] - rj[:, 0:1].T
    dy = ri[:, 1:2] - rj[:, 1:2].T
    dz = ri[:, 2:3] - rj[:, 2:3].T
    return dx, dy, dz


# ---------------------------------------------------------------------------
# Density estimation
# ---------------------------------------------------------------------------

def get_density(r, pos, m, h):
    """SPH density at sampling locations r using particles at pos.

    rho_i = sum_j  m_j * W(r_i - r_j, h)

    Parameters
    ----------
    r   : (M, 3) sampling positions
    pos : (N, 3) particle positions
    m   : float, particle mass (uniform)
    h   : float, smoothing length

    Returns
    -------
    rho : (M, 1) density at each sampling point
    """
    dx, dy, dz = get_pairwise_separations(r, pos)
    rho = np.sum(m * W(dx, dy, dz, h), axis=1, keepdims=True)
    return rho


# ---------------------------------------------------------------------------
# Equation of state
# ---------------------------------------------------------------------------

def get_pressure(rho, k, n):
    """Polytropic equation of state.

    P = k * rho^(1 + 1/n)
    """
    return k * rho ** (1.0 + 1.0 / n)


# ---------------------------------------------------------------------------
# Acceleration
# ---------------------------------------------------------------------------

def get_acceleration(pos, vel, m, h, k, n, lmbda, nu):
    """Compute the acceleration on every SPH particle.

    Three contributions:
      1. Pressure gradient (SPH momentum equation):
         a_i^press = - sum_j m_j (P_i/rho_i^2 + P_j/rho_j^2) grad W(r_i - r_j)
      2. External confining potential:  a_ext = -lambda * pos
      3. Viscous damping:               a_visc = -nu * vel

    Parameters
    ----------
    pos    : (N, 3)  positions
    vel    : (N, 3)  velocities
    m      : float   particle mass
    h      : float   smoothing length
    k      : float   EOS constant
    n      : float   polytropic index
    lmbda  : float   external potential strength
    nu     : float   viscosity / damping coefficient

    Returns
    -------
    a : (N, 3) accelerations
    """
    N = pos.shape[0]

    # --- density & pressure at each particle ---
    rho = get_density(pos, pos, m, h)          # (N, 1)
    P   = get_pressure(rho, k, n)              # (N, 1)

    # --- pairwise kernel gradients ---
    dx, dy, dz = get_pairwise_separations(pos, pos)
    dWx, dWy, dWz = gradW(dx, dy, dz, h)      # each (N, N)

    # --- pressure acceleration (SPH symmetric form) ---
    # factor_ij = P_i / rho_i^2  +  P_j / rho_j^2
    # using broadcasting: (N,1)/(N,1)^2 -> (N,1) broadcast with (1,N)
    factor = P / rho**2                         # (N, 1)
    sym = factor + factor.T                     # (N, N)

    ax = -np.sum(m * sym * dWx, axis=1, keepdims=True)
    ay = -np.sum(m * sym * dWy, axis=1, keepdims=True)
    az = -np.sum(m * sym * dWz, axis=1, keepdims=True)

    a = np.hstack((ax, ay, az))                # (N, 3)

    # --- external confining potential ---
    a -= lmbda * pos

    # --- viscous damping ---
    a -= nu * vel

    return a


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def main():
    # ---- simulation parameters ----
    N      = 400       # number of SPH particles
    t      = 0.0       # current simulation time
    tEnd   = 12.0      # final time
    dt     = 0.04      # time-step
    M_star = 2.0       # total star mass
    R      = 0.75      # star radius
    h      = 0.1       # smoothing length
    k      = 0.1       # EOS constant
    n      = 1.0       # polytropic index
    nu     = 1.0       # damping coefficient

    # external potential constant (derived from Lane-Emden solution)
    lmbda = (2.0 * k * (1.0 + n) * np.pi**(-3.0 / (2.0 * n))
             * (M_star * gamma(5.0/2.0 + n) / (R**3 * gamma(1.0 + n)))**(1.0/n)
             / R**2)
    print(f"lambda = {lmbda:.4f}")

    # particle mass (uniform)
    m = M_star / N

    # ---- initial conditions ----
    np.random.seed(42)
    pos = np.random.randn(N, 3)   # random Gaussian blob
    vel = np.zeros_like(pos)

    # initial acceleration
    acc = get_acceleration(pos, vel, m, h, k, n, lmbda, nu)

    Nt = int(np.ceil(tEnd / dt))

    # ---- set up figure ----
    fig = plt.figure(figsize=(4, 5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[2, 0])

    # analytic density profile for comparison
    rlin = np.linspace(0, 1, 100)
    rho_analytic = lmbda / (4.0 * k) * (R**2 - rlin**2)

    # radial sampling points (along x-axis)
    rr = np.zeros((100, 3))
    rr[:, 0] = rlin

    # ---- main time loop (leapfrog kick-drift-kick) ----
    for i in range(Nt):
        # half-kick
        vel += acc * dt / 2.0

        # drift
        pos += vel * dt

        # recompute acceleration
        acc = get_acceleration(pos, vel, m, h, k, n, lmbda, nu)

        # half-kick
        vel += acc * dt / 2.0

        t += dt

        # ---- live plotting ----
        rho = get_density(pos, pos, m, h)

        plt.sca(ax1)
        plt.cla()
        cval = np.minimum((rho - 3) / 3, 1).flatten()
        plt.scatter(pos[:, 0], pos[:, 1], c=cval,
                    cmap=plt.cm.autumn, s=10, alpha=0.5)
        ax1.set(xlim=(-1.4, 1.4), ylim=(-1.2, 1.2))
        ax1.set_aspect('equal', 'box')
        ax1.set_xticks([-1, 0, 1])
        ax1.set_yticks([-1, 0, 1])
        ax1.set_facecolor((0.1, 0.1, 0.1))

        plt.sca(ax2)
        plt.cla()
        ax2.set(xlim=(0, 1), ylim=(0, 3))
        ax2.set_aspect(0.1)
        plt.plot(rlin, rho_analytic, color='gray', linewidth=2,
                 label='Analytic')
        rho_radial = get_density(rr, pos, m, h)
        plt.plot(rlin, rho_radial, color='blue', label='SPH')
        plt.xlabel('radius')
        plt.ylabel('density')
        plt.legend(loc='upper right', fontsize=6)
        plt.pause(0.001)

    plt.savefig('sph.png', dpi=240)
    plt.show()
    print("Done. Saved sph.png")


if __name__ == '__main__':
    main()
