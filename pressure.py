import warp as wp


@wp.kernel
def _pressure_kernel(
    rho: wp.array(dtype=float),
    rho0: float,
    c0: float,
    gamma: float,
    P: wp.array(dtype=float),
):
    i = wp.tid()
    B = c0 * c0 * rho0 / gamma
    P[i] = B * (wp.pow(rho[i] / rho0, gamma) - 1.0)


def compute_pressure(rho, rho0, c0, gamma, P_out):
    wp.launch(
        kernel=_pressure_kernel,
        dim=rho.shape[0],
        inputs=[rho, rho0, c0, float(gamma), P_out],
    )
    return P_out
