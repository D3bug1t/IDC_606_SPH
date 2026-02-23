import numpy as np
import warp as wp


def initialize_particles(config):
    xs = np.arange(config.dx / 2, 1.0, config.dx)
    ys = np.arange(config.dx / 2, 2.0, config.dx)

    xx, yy = np.meshgrid(xs, ys)  ## Each(xx[i,j],yy[i,j]) represents a particle
    pos_np = np.column_stack((xx.ravel(), yy.ravel())).astype(
        np.float32
    )  ## Same as above but in a cleaner way

    vel_np = np.zeros_like(pos_np, dtype=np.float32)  ## v(0)=0 for all the particles

    m = config.rho0 * config.dx**2  ## rho*(dx)^2 (density*area)
    h = (
        config.h_factor * config.dx
    )  ## Generally 1.3*dx (dx is the initial particle spacing)

    pos = wp.array(pos_np, dtype=wp.vec2)
    vel = wp.array(vel_np, dtype=wp.vec2)

    return pos, vel, float(m), float(h)
