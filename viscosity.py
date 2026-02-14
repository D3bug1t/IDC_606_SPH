import numpy as np

def artificial_viscosity(dx, dy, vel, rho, h, alpha, c0):

    dvx = vel[:, 0:1] - vel[:, 0:1].T
    dvy = vel[:, 1:2] - vel[:, 1:2].T

    vr = dvx*dx + dvy*dy
    r2 = dx**2 + dy**2
    eta2 = 0.01 * h**2

    mu = h * vr / (r2 + eta2)

    rho_avg = 0.5 * (rho + rho.T)
    Pi = np.where(vr < 0.0,
                  -alpha * c0 * mu / rho_avg,
                  0.0)

    return Pi
