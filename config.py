import numpy as np
from dataclasses import dataclass, field


@dataclass
class SPHConfig:
    # Domain
    Lx: float = 20.0  ## Dimensions of the box
    Ly: float = 15.0

    # Fluid
    rho0: float = 1000.0
    c0: float = 20.0
    gamma_eos: int = 7
    alpha_visc: float = 1
    g_vec: np.ndarray = field(default_factory=lambda: np.array([0.0, -9.81]))

    # Discretization
    dx: float = 0.05  ## Particle Spacing
    h_factor: float = 1.6  ## Smoothing factor (h = h_factor*dx)

    # Time
    dt: float = 1e-3  ## Time steps
    tEnd: float = 6.0  ## Long enough for the dam to fully collapse (~sqrt(H/g)) plus settling time

    # Output
    frame_stride: int = 20  ## Save every Nth step; with dt=1e-3 this is 50 frames/s of sim time

    # Performance/validation
    use_neighbor_search: bool = True
