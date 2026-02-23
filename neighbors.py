import math
import warp as wp


@wp.kernel
def _pack_pos_vec3(
    pos2: wp.array(dtype=wp.vec2),
    pos3: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    p = pos2[i]
    pos3[i] = wp.vec3(p[0], p[1], 0.0)


def create_neighbor_grid(config, h, num_particles):
    cell_width = 2.0 * float(h)
    nx = max(1, int(math.ceil(config.Lx / cell_width)) + 1)
    ny = max(1, int(math.ceil(config.Ly / cell_width)) + 1)
    grid = wp.HashGrid(nx, ny, 1)
    pos3 = wp.zeros(num_particles, dtype=wp.vec3)
    return grid, pos3


def build_neighbor_grid(grid, pos2, pos3, radius):
    wp.launch(kernel=_pack_pos_vec3, dim=pos2.shape[0], inputs=[pos2, pos3])
    grid.build(pos3, radius)
