import numpy as np

def W(dx, dy, h):
    r = np.sqrt(dx**2 + dy**2)
    q = r / h
    sigma = 10.0 / (7.0 * np.pi * h**2)

    w = np.zeros_like(q)
    mask1 = q < 1.0
    mask2 = (q >= 1.0) & (q < 2.0)

    w[mask1] = 1.0 - 1.5*q[mask1]**2 + 0.75*q[mask1]**3
    w[mask2] = 0.25*(2.0 - q[mask2])**3

    return sigma * w


def gradW(dx, dy, h):
    r = np.sqrt(dx**2 + dy**2)
    q = r / h
    sigma = 10.0 / (7.0 * np.pi * h**2)

    dwdq = np.zeros_like(q)
    mask1 = q < 1.0
    mask2 = (q >= 1.0) & (q < 2.0)

    dwdq[mask1] = -3.0*q[mask1] + 2.25*q[mask1]**2
    dwdq[mask2] = -0.75*(2.0 - q[mask2])**2

    r_safe = np.where(r > 1e-12, r, 1.0)
    factor = sigma * dwdq / (h * r_safe)
    factor = np.where(r > 1e-12, factor, 0.0)

    return factor * dx, factor * dy
