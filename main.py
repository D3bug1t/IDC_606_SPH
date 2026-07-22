import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.animation import FFMpegWriter
from config import SPHConfig
from initialization import initialize_particles
from simulation import run_simulation
import numpy as np


def main():
    config = SPHConfig()
    pos, vel, m, h = initialize_particles(config)

    frames = run_simulation(config, pos, vel, m, h)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    def animate(i):
        p, v, t = frames[i]
        speed = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)

        ax.cla()
        ax.scatter(p[:, 0], p[:, 1], c=speed, cmap="coolwarm", s=10, vmin=0, vmax=5)

        ax.set_xlim(0, config.Lx)
        ax.set_ylim(0, config.Ly)
        ax.set_aspect("equal")
        ax.set_title(f"Dam Break  t={t:.3f}s")

    anim = FuncAnimation(fig, animate, frames=len(frames), interval=50)

    writer = FFMpegWriter(fps=40)

    anim.save("dam_break_0.05dx.mp4", writer=writer)

    print("Saved dam_break.mp4")


#

if __name__ == "__main__":
    main()
