import math
import warp as wp


def create_neighbor_grid(config, h):
    cell_width = 2.0 * float(h)
    nx = max(1, int(math.ceil(config.Lx / cell_width)) + 1)
    ny = max(1, int(math.ceil(config.Ly / cell_width)) + 1)
    return wp.HashGrid(nx, ny, 1)


def build_neighbor_grid(grid, pos, radius):
    grid.build(pos, radius)
