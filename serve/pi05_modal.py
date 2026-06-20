"""Serve the pi0.5 policy on a Modal GPU. `modal run serve/pi05_modal.py` -> ws://HOST:PORT.

The zero-infrastructure way to run the policy on a remote GPU box: no machine to
rent or SSH into. This runs the same `serve/policy_server.py` server on a Modal
A100, forwards a public TCP tunnel, and prints the `ws://HOST:PORT` to pass to the
eval machine:

    pip install modal && modal token new          # one-time Modal setup
    modal run serve/pi05_modal.py                  # prints ws://HOST:PORT, stays up

    # then on the (CPU-only) eval machine:
    python run_vla.py --remote HOST:PORT --group 10

The checkpoint downloads once into a Modal Volume and is cached across runs. Stop
the server with Ctrl-C (or let `--timeout` expire).
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

CHECKPOINT = "lerobot/pi05_libero_finetuned_v044"
PORT = 8000
CACHE = "/cache"  # HF cache (checkpoint + processors), Volume-backed so it persists

# lerobot is pinned to a git commit (0.5.2 isn't on PyPI; PyPI's 0.5.1 lacks pi05).
_LEROBOT = "lerobot @ git+https://github.com/huggingface/lerobot.git@b8ad81bf397d59dda69ccfc7e74e847f0a9d4fbf"

# Mount this package's serve/ dir so the container imports the SAME server code the
# GPU box would run; only meaningful locally (the container hydrates the image).
_SERVE_DIR = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "hud-python[robot]",  # openpi-client (the openpi/0 wire codec) + numpy
        _LEROBOT,             # PI05Policy + pre/post processors
        "torch", "transformers", "accelerate", "safetensors", "huggingface_hub",
        "websockets", "msgpack", "pillow", "scipy", "einops",
    )
    .add_local_dir(str(_SERVE_DIR), "/root/serve", copy=True)
    .env({"HF_HOME": CACHE, "PYTHONPATH": "/root"})
)

app = modal.App("worldsim-pi05-serve")
cache_vol = modal.Volume.from_name("worldsim-pi05-cache", create_if_missing=True)


@app.function(image=image, gpu="A100", timeout=24 * 3600, volumes={CACHE: cache_vol})
def serve() -> None:
    import asyncio

    sys.path.insert(0, "/root/serve")
    from policy_server import build_pi05_infer, serve_openpi

    infer = build_pi05_infer(CHECKPOINT, device="cuda")
    with modal.forward(PORT, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"[serve] pi0.5 ready - run: python run_vla.py --remote {host}:{port}", flush=True)
        asyncio.run(serve_openpi("0.0.0.0", PORT, infer, metadata={"checkpoint": CHECKPOINT}))


@app.local_entrypoint()
def main() -> None:
    serve.remote()
