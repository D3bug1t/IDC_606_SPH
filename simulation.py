import time
import numpy as np
import warp as wp
from density import compute_density
from pressure import compute_pressure
from forces import compute_acceleration
from boundaries import enforce_boundaries
from integrator import leapfrog_step

def run_simulation(config, pos, vel, m, h):
    wp.init()

    Nt = int(np.ceil(config.tEnd / config.dt))
    frames = []
    t = 0.0
    N = pos.shape[0]

    rho = wp.zeros(N, dtype=float)
    P = wp.zeros(N, dtype=float)
    acc = wp.zeros(N, dtype=wp.vec2)
    g_vec = wp.vec2(float(config.g_vec[0]), float(config.g_vec[1]))

    print("\n--- SPH Dam Break Simulation ---")
    print(f"Particles: {N}")
    print(f"dx = {config.dx}")
    print(f"h = {h}")
    print(f"dt = {config.dt}")
    print(f"Total steps = {Nt}")
    print("---------------------------------\n")

    start_time = time.time()

    for i in range(Nt):
        compute_density(pos, m, h, rho)
        compute_pressure(rho, config.rho0, config.c0, config.gamma_eos, P)
        compute_acceleration(
            pos,
            vel,
            m,
            rho,
            P,
            h,
            config.alpha_visc,
            config.c0,
            g_vec,
            acc,
        )

        pos, vel = leapfrog_step(pos, vel, acc, config.dt)
        enforce_boundaries(pos, vel, config.Lx, config.Ly)

        t += config.dt

        if i % 10 == 0:
            frames.append((pos.numpy().copy(), vel.numpy().copy(), t))

        if i % 50 == 0:
            rho_np = rho.numpy()
            vel_np = vel.numpy()
            max_rho = float(np.max(rho_np))
            max_speed = float(np.max(np.linalg.norm(vel_np, axis=1)))

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
