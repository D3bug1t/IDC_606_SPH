import warp as wp


@wp.kernel
def _boundary_kernel(
    pos: wp.array(dtype=wp.vec2),
    vel: wp.array(dtype=wp.vec2),
    Lx: float,
    Ly: float,
    damp: float,
):
    i = wp.tid()
    p = pos[i]
    v = vel[i]

    if p[0] < 0.0:
        p = wp.vec2(0.0, p[1])
        v = wp.vec2(-damp * v[0], v[1])
    elif p[0] > Lx:
        p = wp.vec2(Lx, p[1])
        v = wp.vec2(-damp * v[0], v[1])

    if p[1] < 0.0:
        p = wp.vec2(p[0], 0.0)
        v = wp.vec2(v[0], -damp * v[1])
    elif p[1] > Ly:
        p = wp.vec2(p[0], Ly)
        v = wp.vec2(v[0], -damp * v[1])

    pos[i] = p
    vel[i] = v


def enforce_boundaries(pos, vel, Lx, Ly, damp=0.5):
    wp.launch(kernel=_boundary_kernel, dim=pos.shape[0], inputs=[pos, vel, Lx, Ly, damp])
