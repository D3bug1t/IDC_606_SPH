import warp as wp
from kernels import gradW_xy
from viscosity import artificial_viscosity_ij


@wp.kernel
def _acceleration_kernel(
    pos: wp.array(dtype=wp.vec2),
    vel: wp.array(dtype=wp.vec2),
    m: float,
    rho: wp.array(dtype=float),
    P: wp.array(dtype=float),
    h: float,
    alpha_visc: float,
    c0: float,
    g: wp.vec2,
    acc: wp.array(dtype=wp.vec2),
):
    i = wp.tid()
    N = pos.shape[0]

    p_i = pos[i]
    v_i = vel[i]
    rho_i = rho[i]
    P_i = P[i]

    a_i = wp.vec2(0.0, 0.0)

    for j in range(N):
        dp = p_i - pos[j]
        gradW_ij = gradW_xy(dp[0], dp[1], h)

        rho_j = rho[j]
        P_j = P[j]

        sym = P_i / (rho_i * rho_i) + P_j / (rho_j * rho_j)
        pi_ij = artificial_viscosity_ij(
            dp[0],
            dp[1],
            v_i,
            vel[j],
            rho_i,
            rho_j,
            h,
            alpha_visc,
            c0,
        )

        a_i -= m * (sym + pi_ij) * gradW_ij

    acc[i] = a_i + g


def compute_acceleration(
    pos,
    vel,
    m,
    rho,
    P,
    h,
    alpha_visc,
    c0,
    g_vec,
    acc_out,
):
    wp.launch(
        kernel=_acceleration_kernel,
        dim=pos.shape[0],
        inputs=[pos, vel, m, rho, P, h, alpha_visc, c0, g_vec, acc_out],
    )
    return acc_out
