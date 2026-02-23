import warp as wp


@wp.func
def artificial_viscosity_ij(
    dx: float,
    dy: float,
    vi: wp.vec2,
    vj: wp.vec2,
    rhoi: float,
    rhoj: float,
    h: float,
    alpha: float,
    c0: float,
):
    dv = vi - vj
    vr = dv[0] * dx + dv[1] * dy
    r2 = dx * dx + dy * dy
    eta2 = 0.01 * h * h
    mu = h * vr / (r2 + eta2)

    rho_avg = 0.5 * (rhoi + rhoj)
    if vr < 0.0:
        return -alpha * c0 * mu / rho_avg

    return 0.0
