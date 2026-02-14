def enforce_boundaries(pos, vel, Lx, Ly, damp=0.5):

    mask = pos[:,0] < 0
    pos[mask,0] = 0
    vel[mask,0] *= -damp ## Particle loses energy on hitting the wall

    mask = pos[:,0] > Lx
    pos[mask,0] = Lx
    vel[mask,0] *= -damp

    mask = pos[:,1] < 0
    pos[mask,1] = 0
    vel[mask,1] *= -damp

    mask = pos[:,1] > Ly
    pos[mask,1] = Ly
    vel[mask,1] *= -damp
