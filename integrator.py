import warp as wp


@wp.kernel
def _leapfrog_kernel(
    pos: wp.array(dtype=wp.vec2),
    vel: wp.array(dtype=wp.vec2),
    acc: wp.array(dtype=wp.vec2),
    dt: float,
):
    i = wp.tid()
    v = vel[i]
    a = acc[i]

    v = v + a * (0.5 * dt)
    pos[i] = pos[i] + v * dt
    vel[i] = v + a * (0.5 * dt)


def leapfrog_step(pos, vel, acc, dt):
    wp.launch(kernel=_leapfrog_kernel, dim=pos.shape[0], inputs=[pos, vel, acc, dt])
    return pos, vel
