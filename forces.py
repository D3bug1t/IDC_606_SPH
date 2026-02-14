import numpy as np
from kernels import gradW
from density import pairwise
from viscosity import artificial_viscosity

def compute_acceleration(pos, vel, m, rho, P, h,
                         alpha_visc, c0, g_vec):

    dx, dy = pairwise(pos)
    dWx, dWy = gradW(dx, dy, h)

    factor = P / rho**2
    sym = factor + factor.T

    Pi = artificial_viscosity(dx, dy, vel, rho,
                              h, alpha_visc, c0)

    total = sym + Pi

    ax = -np.sum(m * total * dWx, axis=1, keepdims=True)
    ay = -np.sum(m * total * dWy, axis=1, keepdims=True)

    acc = np.hstack((ax, ay))
    acc[:, 0] += g_vec[0]
    acc[:, 1] += g_vec[1]

    return acc
