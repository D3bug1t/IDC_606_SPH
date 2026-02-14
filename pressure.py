def compute_pressure(rho, rho0, c0, gamma):
    B = c0**2 * rho0 / gamma
    return B * ((rho / rho0)**gamma - 1.0)
