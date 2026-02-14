import numpy as np

def initialize_particles(config):

    xs = np.arange(config.dx/2, 1.0, config.dx)
    ys = np.arange(config.dx/2, 2.0, config.dx)

    xx, yy = np.meshgrid(xs, ys) ## Each(xx[i,j],yy[i,j]) represents a particle
    pos = np.column_stack((xx.ravel(), yy.ravel())) ## Same as above but in a cleaner way

    vel = np.zeros_like(pos) ## v(0)=0 for all the particles

    m = config.rho0 * config.dx**2 ## rho*(dx)^2 (density*area)
    h = config.h_factor * config.dx ## Generally 1.3*dx (dx is the initial particle spacing)

    return pos, vel, m, h
