import numpy as np
from dataclasses import dataclass, field


@dataclass
class SPHConfig:
    # Domain
    Lx: float = 4.0  ## Dimensions of the box
    Ly: float = 3.0

    # Fluid
    rho0: float = 1000.0
    c0: float = 20.0
    gamma_eos: int = 7
    alpha_visc: float = 0.6
    g_vec: np.ndarray = field(default_factory=lambda: np.array([0.0, -9.81]))

    # Discretization
    dx: float = 0.02  ## Particle Spacing
    h_factor: float = 1.3  ## Smoothing factor (h = h_factor*dx)

    # Time
    dt: float = 1e-3  ## Time steps
    tEnd: float = 5

    # Performance/validation
    use_neighbor_search: bool = True
