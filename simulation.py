import time
import numpy as np
from density import compute_density
from pressure import compute_pressure
from forces import compute_acceleration
from boundaries import enforce_boundaries
from integrator import leapfrog_step

def run_simulation(config, pos, vel, m, h):

    Nt = int(np.ceil(config.tEnd / config.dt))
    frames = []
    t = 0.0

    print("\n--- SPH Dam Break Simulation ---")
    print(f"Particles: {pos.shape[0]}")
    print(f"dx = {config.dx}")
    print(f"h = {h}")
    print(f"dt = {config.dt}")
    print(f"Total steps = {Nt}")
    print("---------------------------------\n")

    start_time = time.time()

    for i in range(Nt):

        rho = compute_density(pos, m, h)
        P = compute_pressure(rho,
                             config.rho0,
                             config.c0,
                             config.gamma_eos)

        acc = compute_acceleration(pos, vel, m,
                                   rho, P, h,
                                   config.alpha_visc,
                                   config.c0,
                                   config.g_vec)

        pos, vel = leapfrog_step(pos, vel, acc, config.dt)
        enforce_boundaries(pos, vel,
                           config.Lx, config.Ly)

        t += config.dt

        # Store ~100 frames
        # if i % max(1, Nt // 50) == 0:
        if i%10 == 0:
            frames.append((pos.copy(), vel.copy(), t))

        # ---- VERBOSE PRINT ----
        if i % 50 == 0:

            max_rho = float(np.max(rho))
            max_speed = float(np.max(np.linalg.norm(vel, axis=1)))
            CFL = config.c0 * config.dt / h

            elapsed = time.time() - start_time

            print(
                f"Step {i:5d}/{Nt} | "
                f"t = {t:6.3f}s | "
                f"max rho = {max_rho:8.2f} | "
                f"max |v| = {max_speed:6.3f} | "
                f"elapsed = {elapsed:6.1f}s"
            )

    total_time = time.time() - start_time
    print(f"\nSimulation complete in {total_time:.2f} seconds.\n")

    return frames

