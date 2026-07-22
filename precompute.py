"""
Pre-computes SPH dam break simulations across a dx grid and saves them
as gzip-compressed uint16 binary files for web playback at 24 fps.

Run once on your GPU machine:
    python precompute.py

Then upload sph_cache/*.bin.gz to a GitHub Release (tag: sph-cache).

Binary layout per file (after gzip decompression):
  Header — 16 bytes:
    N         uint32   particle count
    Lx        float32  domain width
    Ly        float32  domain height
    max_speed float32  global peak speed across all frames (for colormap)
  Then 144 frames, each 6*N bytes:
    x[N]  uint16   position x, mapped [0, Lx]  → [0, 65535]
    y[N]  uint16   position y, mapped [0, Ly]  → [0, 65535]
    s[N]  uint16   speed,      mapped [0, max] → [0, 65535]
"""

import argparse
import gzip
import os
import struct
import sys
import time

import numpy as np
import warp as wp

from boundaries import enforce_boundaries
from config import SPHConfig
from density import compute_density
from forces import compute_acceleration
from initialization import initialize_particles
from integrator import create_rk4_workspace, rk4_step
from neighbors import build_neighbor_grid, create_neighbor_grid
from pressure import compute_pressure

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true",
                    help="Quick local test: 1s sim, dx=0.2 only, no GPU check")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TARGET_FPS   = 24
SIM_DURATION = 1.0 if args.test else 6.0
N_FRAMES     = int(TARGET_FPS * SIM_DURATION)
DX_GRID      = [0.2] if args.test else [0.05, 0.1, 0.15, 0.2]
OUT_DIR      = "sph_cache"

os.makedirs(OUT_DIR, exist_ok=True)
wp.init()

# ---------------------------------------------------------------------------
# GPU check — bail out early if running on CPU (e.g. Mac)
# ---------------------------------------------------------------------------
if not args.test:
    try:
        device = wp.get_device("cuda:0")
    except Exception:
        device = None

    if device is None or not wp.is_device_available("cuda:0"):
        print("ERROR: No CUDA GPU detected. Warp would fall back to CPU.")
        print("This script needs to run on your Linux/GPU machine, not your Mac.")
        print("")
        print("To test locally (CPU, coarse, fast):")
        print("  python precompute.py --test")
        sys.exit(1)

    print(f"GPU detected: {wp.get_device('cuda:0')}")

# ---------------------------------------------------------------------------
# Simulation loop — collects exactly N_FRAMES evenly spaced in sim time
# ---------------------------------------------------------------------------

def run_and_collect(dx: float):
    config = SPHConfig(dx=dx, tEnd=SIM_DURATION, dt=1e-4)
    pos, vel, m, h = initialize_particles(config)
    N = pos.shape[0]

    Nt           = int(np.ceil(config.tEnd / config.dt))   # 60 000
    frame_stride = max(1, Nt // N_FRAMES)                  # ~416

    rho    = wp.zeros(N, dtype=float)
    P      = wp.zeros(N, dtype=float)
    g_vec  = wp.vec2(float(config.g_vec[0]), float(config.g_vec[1]))
    sr     = 2.0 * h
    rk4_ws = create_rk4_workspace(N)
    ngrid, npoints = create_neighbor_grid(config, h, N)

    def acc_fn(p, v, a):
        build_neighbor_grid(ngrid, p, npoints, sr)
        compute_density(p, m, h, rho, use_neighbor_search=True, grid=ngrid, support_radius=sr)
        compute_pressure(rho, config.rho0, config.c0, config.gamma_eos, P)
        compute_acceleration(p, v, m, rho, P, h, config.alpha_visc, config.c0,
                             g_vec, a, use_neighbor_search=True, grid=ngrid, support_radius=sr)

    frames_pos   = []
    frames_speed = []

    t0 = time.perf_counter()
    for i in range(Nt):
        pos, vel = rk4_step(pos, vel, config.dt, acc_fn, rk4_ws)
        enforce_boundaries(pos, vel, config.Lx, config.Ly)

        if i % frame_stride == 0 and len(frames_pos) < N_FRAMES:
            p_np = pos.numpy().copy()
            v_np = vel.numpy()
            frames_pos.append(p_np)
            frames_speed.append(np.linalg.norm(v_np, axis=1).astype(np.float32))

        if i % 5000 == 0:
            pct = i / Nt * 100
            print(f"  step {i:6d}/{Nt}  {pct:4.0f}%  ({time.perf_counter()-t0:.1f}s)")

    elapsed = time.perf_counter() - t0
    print(f"  done — {N} particles, {len(frames_pos)} frames in {elapsed:.1f}s")
    return frames_pos, frames_speed, config.Lx, config.Ly


# ---------------------------------------------------------------------------
# Export to gzipped uint16 binary
# ---------------------------------------------------------------------------

def export(dx: float, frames_pos, frames_speed, Lx: float, Ly: float):
    N         = frames_pos[0].shape[0]
    max_speed = float(np.max(np.concatenate(frames_speed))) or 1.0

    tag  = f"{int(dx * 100):03d}"          # 0.05 → "005", 0.1 → "010"
    path = os.path.join(OUT_DIR, f"sph_dx{tag}.bin.gz")

    with gzip.open(path, "wb", compresslevel=6) as f:
        # Header
        f.write(struct.pack("<I", N))
        f.write(struct.pack("<f", Lx))
        f.write(struct.pack("<f", Ly))
        f.write(struct.pack("<f", max_speed))

        # Frames
        for pos_np, spd_np in zip(frames_pos, frames_speed):
            x_q = (pos_np[:, 0] / Lx * 65535).clip(0, 65535).astype(np.uint16)
            y_q = (pos_np[:, 1] / Ly * 65535).clip(0, 65535).astype(np.uint16)
            s_q = (spd_np / max_speed * 65535).clip(0, 65535).astype(np.uint16)
            f.write(x_q.tobytes())
            f.write(y_q.tobytes())
            f.write(s_q.tobytes())

    kb = os.path.getsize(path) / 1024
    print(f"  → {path}  ({kb:.0f} KB)\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Rough particle counts: (5/dx) × (10/dx)
PARTICLE_ESTIMATES = {0.05: "~19 700", 0.1: "~5 000", 0.15: "~2 178", 0.2: "~1 250"}

for dx in DX_GRID:
    est = PARTICLE_ESTIMATES.get(dx, "")
    print(f"\n{'='*52}")
    print(f"  dx = {dx}   {est} particles   {N_FRAMES} frames @ {TARGET_FPS} fps")
    print(f"{'='*52}")
    frames_pos, frames_speed, Lx, Ly = run_and_collect(dx)
    export(dx, frames_pos, frames_speed, Lx, Ly)

print("All done.")
print(f"Upload everything in {OUT_DIR}/ to a GitHub Release with tag: sph-cache")
print("Example URL the site will fetch:")
print("  https://github.com/apranav22/apranav22.github.io/releases/download/sph-cache/sph_dx010.bin.gz")
