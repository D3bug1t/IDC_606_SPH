import warp as wp
from kernels import W_xy


@wp.kernel
def _density_neighbors_kernel(
    pos: wp.array(dtype=wp.vec2),
    m: float,
    h: float,
    grid: wp.uint64,
    support_radius: float,
    rho: wp.array(dtype=float),
):
    i = wp.tid()

    rho_i = float(0.0)
    p_i = pos[i]
    p_i3 = wp.vec3(p_i[0], p_i[1], 0.0)

    query = wp.hash_grid_query(grid, p_i3, support_radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        dp = p_i - pos[j]
        rho_i += m * W_xy(dp[0], dp[1], h)

    rho[i] = rho_i


@wp.kernel
def _density_all_pairs_kernel(
    pos: wp.array(dtype=wp.vec2),
    m: float,
    h: float,
    rho: wp.array(dtype=float),
):
    i = wp.tid()
    N = pos.shape[0]

    rho_i = float(0.0)
    p_i = pos[i]

    for j in range(N):
        dp = p_i - pos[j]
        rho_i += m * W_xy(dp[0], dp[1], h)

    rho[i] = rho_i


def compute_density(
    pos, m, h, rho_out, use_neighbor_search=True, grid=None, support_radius=0.0
):
    if use_neighbor_search:
        if grid is None:
            raise ValueError("Neighbor search enabled but no hash grid was provided.")
        wp.launch(
            kernel=_density_neighbors_kernel,
            dim=pos.shape[0],
            inputs=[pos, m, h, grid.id, support_radius, rho_out],
        )
        return rho_out

    wp.launch(
        kernel=_density_all_pairs_kernel,
        dim=pos.shape[0],
        inputs=[pos, m, h, rho_out],
    )
    return rho_out
