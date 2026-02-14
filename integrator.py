def leapfrog_step(pos, vel, acc, dt):
    vel += acc * dt/2 ## Basically a half step integration for velocity (Philip Mocz is the god here!{Maybe :P})
    pos += vel * dt  ## Update the position of the particle (again refer Mocz for the equation)
    vel += acc * dt/2 ## Half step to get v_{n+1}
    return pos, vel ## We're done
