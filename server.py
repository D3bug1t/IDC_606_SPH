"""
FastAPI WebSocket server for the SPH dam break simulation.

Usage:
    pip install fastapi uvicorn
    python server.py

Then expose via ngrok:
    ngrok http 8000
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import uvicorn
import warp as wp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import SPHConfig
from initialization import initialize_particles
from simulation import stream_simulation

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI()

# Allow requests from your GitHub Pages domain and localhost for local dev.
# Update ALLOWED_ORIGINS with your actual domain.
ALLOWED_ORIGINS = [
    "https://apranav22.github.io",
    "http://localhost:4000",   # jekyll serve
    "http://127.0.0.1:4000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Only one simulation runs at a time
_sim_lock = asyncio.Lock()
_executor = ThreadPoolExecutor(max_workers=1)

# Initialise Warp once at startup
wp.init()

# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    # --- Receive config from browser ---
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        params = json.loads(raw)
    except Exception as e:
        await websocket.close(code=1003, reason=f"Bad config: {e}")
        return

    if _sim_lock.locked():
        await websocket.send_text(json.dumps({"error": "Simulation already running. Try again shortly."}))
        await websocket.close()
        return

    async with _sim_lock:
        # Build config from browser params
        gravity = float(params.get("gravity", -9.81))
        config = SPHConfig(
            dx=float(params.get("dx", 0.05)),
            alpha_visc=float(params.get("alpha_visc", 1.0)),
            g_vec=np.array([0.0, gravity]),
            dt=float(params.get("dt", 1e-4)),
            tEnd=float(params.get("tEnd", 0.5)),
        )

        pos, vel, m, h = initialize_particles(config)
        N = pos.shape[0]

        # Send metadata so the browser knows domain size and particle count
        await websocket.send_text(json.dumps({
            "type": "meta",
            "N": N,
            "Lx": config.Lx,
            "Ly": config.Ly,
        }))

        stop_flag = [False]
        frame_queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        loop = asyncio.get_event_loop()

        # --- Run simulation in a background thread ---
        def run_sim():
            try:
                for frame_bytes in stream_simulation(config, pos, vel, m, h,
                                                     stop_flag=stop_flag,
                                                     target_fps=24):
                    # blocks when queue is full → natural backpressure
                    asyncio.run_coroutine_threadsafe(
                        frame_queue.put(frame_bytes), loop
                    ).result()
            finally:
                asyncio.run_coroutine_threadsafe(
                    frame_queue.put(None), loop  # sentinel = done
                ).result()

        sim_future = loop.run_in_executor(_executor, run_sim)

        # Listen for "stop" message from browser
        async def listen_for_stop():
            try:
                while True:
                    msg = await websocket.receive_text()
                    if msg == "stop":
                        stop_flag[0] = True
                        break
            except Exception:
                stop_flag[0] = True

        stop_task = asyncio.create_task(listen_for_stop())

        # --- Stream frames to browser ---
        try:
            while True:
                frame = await asyncio.wait_for(frame_queue.get(), timeout=30.0)
                if frame is None:
                    break
                await websocket.send_bytes(frame)
        except (WebSocketDisconnect, asyncio.TimeoutError):
            stop_flag[0] = True
        finally:
            stop_task.cancel()
            await sim_future

        await websocket.close()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok", "sim_running": _sim_lock.locked()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
