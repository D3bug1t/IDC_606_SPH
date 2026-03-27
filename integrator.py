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


@wp.kernel
def _copy_vec2_kernel(
    src: wp.array(dtype=wp.vec2),
    dst: wp.array(dtype=wp.vec2),
):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def _combine_state_kernel(
    base_pos: wp.array(dtype=wp.vec2),
    base_vel: wp.array(dtype=wp.vec2),
    delta_pos: wp.array(dtype=wp.vec2),
    delta_vel: wp.array(dtype=wp.vec2),
    scale: float,
    out_pos: wp.array(dtype=wp.vec2),
    out_vel: wp.array(dtype=wp.vec2),
):
    i = wp.tid()
    out_pos[i] = base_pos[i] + delta_pos[i] * scale
    out_vel[i] = base_vel[i] + delta_vel[i] * scale


@wp.kernel
def _rk4_finalize_kernel(
    pos: wp.array(dtype=wp.vec2),
    vel: wp.array(dtype=wp.vec2),
    k1_pos: wp.array(dtype=wp.vec2),
    k1_vel: wp.array(dtype=wp.vec2),
    k2_pos: wp.array(dtype=wp.vec2),
    k2_vel: wp.array(dtype=wp.vec2),
    k3_pos: wp.array(dtype=wp.vec2),
    k3_vel: wp.array(dtype=wp.vec2),
    k4_pos: wp.array(dtype=wp.vec2),
    k4_vel: wp.array(dtype=wp.vec2),
    dt: float,
):
    i = wp.tid()
    pos[i] = pos[i] + (dt / 6.0) * (
        k1_pos[i] + 2.0 * k2_pos[i] + 2.0 * k3_pos[i] + k4_pos[i]
    )
    vel[i] = vel[i] + (dt / 6.0) * (
        k1_vel[i] + 2.0 * k2_vel[i] + 2.0 * k3_vel[i] + k4_vel[i]
    )


def create_rk4_workspace(num_particles):
    def zeros():
        return wp.zeros(num_particles, dtype=wp.vec2)

    return {
        "stage_pos": zeros(),
        "stage_vel": zeros(),
        "k1_pos": zeros(),
        "k1_vel": zeros(),
        "k2_pos": zeros(),
        "k2_vel": zeros(),
        "k3_pos": zeros(),
        "k3_vel": zeros(),
        "k4_pos": zeros(),
        "k4_vel": zeros(),
    }


def rk4_step(pos, vel, dt, acceleration_fn, workspace):
    n = pos.shape[0]

    wp.launch(kernel=_copy_vec2_kernel, dim=n, inputs=[vel, workspace["k1_pos"]])
    acceleration_fn(pos, vel, workspace["k1_vel"])

    wp.launch(
        kernel=_combine_state_kernel,
        dim=n,
        inputs=[
            pos,
            vel,
            workspace["k1_pos"],
            workspace["k1_vel"],
            0.5 * dt,
            workspace["stage_pos"],
            workspace["stage_vel"],
        ],
    )
    wp.launch(
        kernel=_copy_vec2_kernel,
        dim=n,
        inputs=[workspace["stage_vel"], workspace["k2_pos"]],
    )
    acceleration_fn(workspace["stage_pos"], workspace["stage_vel"], workspace["k2_vel"])

    wp.launch(
        kernel=_combine_state_kernel,
        dim=n,
        inputs=[
            pos,
            vel,
            workspace["k2_pos"],
            workspace["k2_vel"],
            0.5 * dt,
            workspace["stage_pos"],
            workspace["stage_vel"],
        ],
    )
    wp.launch(
        kernel=_copy_vec2_kernel,
        dim=n,
        inputs=[workspace["stage_vel"], workspace["k3_pos"]],
    )
    acceleration_fn(workspace["stage_pos"], workspace["stage_vel"], workspace["k3_vel"])

    wp.launch(
        kernel=_combine_state_kernel,
        dim=n,
        inputs=[
            pos,
            vel,
            workspace["k3_pos"],
            workspace["k3_vel"],
            dt,
            workspace["stage_pos"],
            workspace["stage_vel"],
        ],
    )
    wp.launch(
        kernel=_copy_vec2_kernel,
        dim=n,
        inputs=[workspace["stage_vel"], workspace["k4_pos"]],
    )
    acceleration_fn(workspace["stage_pos"], workspace["stage_vel"], workspace["k4_vel"])

    wp.launch(
        kernel=_rk4_finalize_kernel,
        dim=n,
        inputs=[
            pos,
            vel,
            workspace["k1_pos"],
            workspace["k1_vel"],
            workspace["k2_pos"],
            workspace["k2_vel"],
            workspace["k3_pos"],
            workspace["k3_vel"],
            workspace["k4_pos"],
            workspace["k4_vel"],
            dt,
        ],
    )

    return pos, vel
