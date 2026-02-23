import warp as wp
from kernels import W_xy


@wp.kernel
def _density_kernel(
    pos: wp.array(dtype=wp.vec2),
    m: float,
    h: float,
    rho: wp.array(dtype=float),
):
    i = wp.tid()
    N = pos.shape[0]

    rho_i = 0.0
    p_i = pos[i]

    for j in range(N):
        dp = p_i - pos[j]
        rho_i += m * W_xy(dp[0], dp[1], h)

    rho[i] = rho_i


def compute_density(pos, m, h, rho_out):
    wp.launch(
        kernel=_density_kernel,
        dim=pos.shape[0],
        inputs=[pos, m, h, rho_out],
    )
    return rho_out
