import warp as wp

PI = 3.141592653589793


@wp.func
def W_xy(dx: float, dy: float, h: float):
    # monaghan cubic spline kernel
    r = wp.sqrt(dx * dx + dy * dy)
    q = r / h
    sigma = 10.0 / (7.0 * PI * h * h)

    w = 0.0
    if q < 1.0:
        w = 1.0 - 1.5 * q * q + 0.75 * q * q * q
    elif q < 2.0:
        t = 2.0 - q
        w = 0.25 * t * t * t

    return sigma * w


@wp.func
def gradW_xy(dx: float, dy: float, h: float):
    r = wp.sqrt(dx * dx + dy * dy)
    if r <= 1.0e-12:
        return wp.vec2(0.0, 0.0)

    q = r / h
    sigma = 10.0 / (7.0 * PI * h * h)

    dwdq = 0.0
    if q < 1.0:
        dwdq = -3.0 * q + 2.25 * q * q
    elif q < 2.0:
        t = 2.0 - q
        dwdq = -0.75 * t * t

    factor = sigma * dwdq / (h * r)
    return wp.vec2(factor * dx, factor * dy)
