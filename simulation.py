import time
import numpy as np
import warp as wp
from density import compute_density
from pressure import compute_pressure
from forces import compute_acceleration
from boundaries import enforce_boundaries
from integrator import create_rk4_workspace, rk4_step
from neighbors import create_neighbor_grid, build_neighbor_grid


def run_simulation(config, pos, vel, m, h):
    wp.init()

    Nt = int(np.ceil(config.tEnd / config.dt))
    frames = []
    t = 0.0
    N = pos.shape[0]

    rho = wp.zeros(N, dtype=float)
    P = wp.zeros(N, dtype=float)
    g_vec = wp.vec2(float(config.g_vec[0]), float(config.g_vec[1]))
    use_neighbor_search = bool(getattr(config, "use_neighbor_search", True))
    support_radius = 2.0 * h
    rk4_workspace = create_rk4_workspace(N)
    if use_neighbor_search:
        neighbor_grid, neighbor_points = create_neighbor_grid(config, h, N)
    else:
        neighbor_grid, neighbor_points = None, None

    print("\n--- SPH Dam Break Simulation ---")
    print(f"Particles: {N}")
    print(f"dx = {config.dx}")
    print(f"h = {h}")
    print(f"dt = {config.dt}")
    print("Integrator = RK4")
    print(f"Total steps = {Nt}")
    print(f"Neighbor search = {'ON' if use_neighbor_search else 'OFF (all-pairs)'}")
    print("---------------------------------\n")

    start_time = time.time()

    def compute_state_acceleration(pos_state, vel_state, acc_out):
        if use_neighbor_search:
            build_neighbor_grid(neighbor_grid, pos_state, neighbor_points, support_radius)

        compute_density(
            pos_state,
            m,
            h,
            rho,
            use_neighbor_search=use_neighbor_search,
            grid=neighbor_grid,
            support_radius=support_radius,
        )
        compute_pressure(rho, config.rho0, config.c0, config.gamma_eos, P)
        compute_acceleration(
            pos_state,
            vel_state,
            m,
            rho,
            P,
            h,
            config.alpha_visc,
            config.c0,
            g_vec,
            acc_out,
            use_neighbor_search=use_neighbor_search,
            grid=neighbor_grid,
            support_radius=support_radius,
        )

    for i in range(Nt):
        pos, vel = rk4_step(pos, vel, config.dt, compute_state_acceleration, rk4_workspace)
        enforce_boundaries(pos, vel, config.Lx, config.Ly)

        t += config.dt

        if i % 500 == 0:
            print(i)
            vel_np = vel.numpy()
            max_speed = float(np.max(np.linalg.norm(vel_np, axis=1)))
            print('Max V = ', max_speed)
            frames.append((pos.numpy().copy(), vel.numpy().copy(), t))
#
#        if i % 50 == 0:
#            compute_state_acceleration(pos, vel, rk4_workspace["k1_vel"])
#            rho_np = rho.numpy()
#            vel_np = vel.numpy()
#            max_rho = float(np.max(rho_np))
#            max_speed = float(np.max(np.linalg.norm(vel_np, axis=1)))
#
#            elapsed = time.time() - start_time
#
       # print(
        #        f"Step {i:5d}/{Nt} | "
         #       f"t = {t:6.3f}s | ")
#                f"max rho = {max_rho:8.2f} | "
#                f"max |v| = {max_speed:6.3f} | "
#                f"elapsed = {elapsed:6.1f}s"
#            )

    total_time = time.time() - start_time
    print(f"\nSimulation complete in {total_time:.2f} seconds.\n")

    return frames
