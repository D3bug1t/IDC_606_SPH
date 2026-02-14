import numpy as np
from kernels import W

def pairwise(pos):
    dx = pos[:, 0:1] - pos[:, 0:1].T ## Basically creading a ditance matrix in x direction
    dy = pos[:, 1:2] - pos[:, 1:2].T ## Same as above but in y direction (column - row vector creates a matrix)
    #(I prefer dist(x) in R tho)
    return dx, dy

def compute_density(pos, m, h):
    dx, dy = pairwise(pos)
    return np.sum(m * W(dx, dy, h), axis=1, keepdims=True)
