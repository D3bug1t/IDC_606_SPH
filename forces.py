import warp as wp
from kernels import gradW_xy
from viscosity import artificial_viscosity_ij


@wp.kernel
def _acceleration_neighbors_kernel(
    pos: wp.array(dtype=wp.vec2),
    vel: wp.array(dtype=wp.vec2),
    m: float,
    rho: wp.array(dtype=float),
    P: wp.array(dtype=float),
    h: float,
    grid: wp.uint64,
    support_radius: float,
    alpha_visc: float,
    c0: float,
    g: wp.vec2,
    acc: wp.array(dtype=wp.vec2),
):
    i = wp.tid()

    p_i = pos[i]
    v_i = vel[i]
    rho_i = rho[i]
    P_i = P[i]

    ax = float(0.0)
    ay = float(0.0)
    p_i3 = wp.vec3(p_i[0], p_i[1], 0.0)

    query = wp.hash_grid_query(grid, p_i3, support_radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
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

        coeff = -m * (sym + pi_ij)
        ax += coeff * gradW_ij[0]
        ay += coeff * gradW_ij[1]

    acc[i] = wp.vec2(ax, ay) + g


@wp.kernel
def _acceleration_all_pairs_kernel(
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

    ax = float(0.0)
    ay = float(0.0)

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

        coeff = -m * (sym + pi_ij)
        ax += coeff * gradW_ij[0]
        ay += coeff * gradW_ij[1]

    acc[i] = wp.vec2(ax, ay) + g


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
    use_neighbor_search=True,
    grid=None,
    support_radius=0.0,
):
    if use_neighbor_search:
        if grid is None:
            raise ValueError("Neighbor search enabled but no hash grid was provided.")
        wp.launch(
            kernel=_acceleration_neighbors_kernel,
            dim=pos.shape[0],
            inputs=[
                pos,
                vel,
                m,
                rho,
                P,
                h,
                grid.id,
                support_radius,
                alpha_visc,
                c0,
                g_vec,
                acc_out,
            ],
        )
        return acc_out

    wp.launch(
        kernel=_acceleration_all_pairs_kernel,
        dim=pos.shape[0],
        inputs=[
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
        ],
    )
    return acc_out
