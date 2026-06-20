"""The Newton sim in its own process - so it can own the main thread for the viewer.

A GL viewer must own the process main thread, but an env's asyncio server pins it
forever, so the env spawns the sim as a separate process (`SimHost`): the chosen
interface - the FastMCP `mcp` server or the `robot` bridge - is served on a worker
thread, and the main thread runs the viewer when `WORLDSIM_VIEWER=1` (else it just serves).

`SimHost` is the env-side handle (spawn / wait-until-up / stop); everything from
`main()` down runs in the spawned process. It stays a module separate from
`sim/server.py` so `sim.server` is imported once under its canonical name - the bridge
and the viewer must share one `_sim`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"


def _free_port() -> int:
    """Grab an ephemeral port so concurrent rollouts don't collide."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


class SimHost:
    """Spawns the sim in its own process and exposes where it serves.

    One per env process. `serve="mcp"` hosts the FastMCP tool server (LLM tasks);
    `serve="robot"` hosts the Franka bridge + its control RPC (VLA tasks). The env
    picks the port; `start()` blocks until the child listens, `stop()` kills it.
    """

    host = HOST

    def __init__(self, serve: str) -> None:  # serve: "mcp" | "robot"
        self.serve = serve
        self.port = _free_port()  # mcp: HTTP port; robot: control-RPC port
        self._proc: subprocess.Popen | None = None

    @property
    def mcp_url(self) -> str:
        return f"http://{HOST}:{self.port}/mcp"

    async def start(self, timeout: float = 120.0) -> None:
        # Hand off mode + port through the child's environment (it's only ever spawned here).
        child_env = {**os.environ, "WORLDSIM_SIM_SERVE": self.serve, "WORLDSIM_SIM_PORT": str(self.port)}
        self._proc = subprocess.Popen([sys.executable, "-m", "sim.host"], cwd=str(ROOT), env=child_env)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"sim host exited early (code {self._proc.returncode})")
            try:  # keep trying to connect until the sim is up
                socket.create_connection((HOST, self.port), timeout=0.5).close()
                return
            except OSError:
                await asyncio.sleep(0.2)
        raise RuntimeError(f"sim host never listened on {HOST}:{self.port}")

    async def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(self._proc.wait, 10)
        self._proc = None


# ── the spawned process: serve the sim, viewer on the main thread when asked ──


async def _serve_robot(port: int) -> None:
    """Serve the Franka bridge (openpi/0 WS) + its control RPC on the loop thread."""
    from hud.environment.robot import RobotEndpoint

    from sim.franka_bridge import WorldsimFrankaBridge

    bridge = WorldsimFrankaBridge(port=0)  # ephemeral WS; the env reads it back via url()
    await bridge.start()
    server = await RobotEndpoint(bridge).serve(HOST, port)
    async with server:
        await server.serve_forever()


def main() -> None:
    serve = os.environ["WORLDSIM_SIM_SERVE"]  # "mcp" | "robot", handed off by SimHost
    port = int(os.environ["WORLDSIM_SIM_PORT"])

    import sim.server as sim_server

    if serve == "mcp":
        def _run() -> None:
            asyncio.run(sim_server.server.run_async(
                transport="http", host=HOST, port=port, show_banner=False))
    else:
        def _run() -> None:
            asyncio.run(_serve_robot(port))

    # Serve on a worker thread so the main thread is free for the (thread-affine) viewer.
    driver = threading.Thread(target=_run, daemon=True)
    driver.start()

    if os.environ.get("WORLDSIM_VIEWER") == "1":
        sim_server.run_viewer(driver)  # owns the main thread; headless-safe fallback
    else:
        driver.join()


if __name__ == "__main__":
    main()
