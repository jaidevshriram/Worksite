"""openpi/0 websocket policy server - run this on the GPU box.

A standalone server that holds the policy weights and answers stateless
observation→action-chunk requests over the `openpi/0` (msgpack-numpy) websocket.
The eval machine runs the sim + loop and connects with `run_vla.py --remote
HOST:PORT` (a `RemoteAgent`), so the box that has the GPU is the *only* thing that
needs torch/lerobot/CUDA.

    # real pi0.5 baseline (GPU box, needs `.[robot,vla]`):
    python serve/policy_server.py --port 8000

    # weightless wire check (no GPU / no model) - pairs with run_vla.py --remote:
    python serve/policy_server.py --port 8000 --noop

For a managed GPU (no box to rent/SSH), `serve/pi05_modal.py` runs this same
server on Modal and prints the public `ws://HOST:PORT` to use with `--remote`.

Protocol (matches openpi-client `WebsocketClientPolicy`): on connect the server
sends a metadata dict, then for each received observation dict replies with
`{"actions": <[T, A] chunk>}`. The observation arrives with the env's openpi wire
keys (`observation/image`, `observation/wrist_image`, `observation/state`) plus
`prompt` - exactly what `WorldsimFrankaBridge.get_observation` emits, shipped as-is by
the agent's `OpenPIAdapter`.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, Callable

import numpy as np
import websockets.asyncio.server as wss
import websockets.exceptions
from openpi_client import msgpack_numpy

# The env's two cameras in contract order; mapped onto the policy's image slots.
ENV_IMAGE_KEYS = ["observation/image", "observation/wrist_image"]
DEFAULT_CHECKPOINT = "lerobot/pi05_libero_finetuned_v044"

InferFn = Callable[[dict[str, Any]], dict[str, Any]]


async def serve_openpi(host: str, port: int, infer: InferFn, *, metadata: dict | None = None) -> None:
    """Serve `infer` over the openpi/0 websocket forever (one inference per request)."""
    packer = msgpack_numpy.Packer()

    async def handler(ws: Any) -> None:
        await ws.send(packer.pack(metadata or {}))  # openpi handshake: metadata first
        try:
            while True:
                obs = msgpack_numpy.unpackb(await ws.recv())
                result = await asyncio.to_thread(infer, obs)  # keep the event loop free
                await ws.send(packer.pack(result))
        except websockets.exceptions.ConnectionClosed:
            pass

    async with wss.serve(handler, host, port, compression=None, max_size=None) as server:
        print(f"[serve] openpi/0 policy server: ws://{host}:{port}", flush=True)
        await server.serve_forever()


def build_pi05_infer(
    checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None, horizon: int = 10
) -> InferFn:
    """Load a pi0.5 LIBERO checkpoint and return its openpi `infer` (needs a GPU + lerobot).

    Mirrors the in-process `PI05Agent`: same checkpoint, same LeRobot pre/post, same
    replan horizon - only the transport differs. The chunk is truncated to `horizon`
    here so the remote client replans every `horizon` steps (pi0.5 run whole open-loop
    drifts in sim-to-sim; the published LIBERO eval replans ~10).
    """
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    from hud.agents.robot.model import LeRobotModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[serve] loading policy: {checkpoint} (device={device})", flush=True)
    policy = PI05Policy.from_pretrained(checkpoint).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    model = LeRobotModel(policy, preprocess, postprocess)
    image_keys = list(policy.config.image_features)  # model slots in contract order

    def infer(obs: dict[str, Any]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(np.asarray(obs["observation/state"], dtype=np.float32)),
            "task": obs.get("prompt", ""),
        }
        for model_key, env_key in zip(image_keys, ENV_IMAGE_KEYS, strict=False):
            batch[model_key] = torch.from_numpy(np.asarray(obs[env_key])).permute(2, 0, 1).float() / 255.0
        chunk = model.infer(batch)[0, :horizon]  # [N, T, A] -> [horizon, A]
        return {"actions": chunk}

    return infer


def build_noop_infer(horizon: int = 10) -> InferFn:
    """Weightless infer: hold position, gripper open. The remote analogue of NoopAgent."""
    chunk = np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], (horizon, 1)).astype(np.float32)

    def infer(obs: dict[str, Any]) -> dict[str, Any]:
        return {"actions": chunk}

    return infer


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a VLA policy over the openpi/0 websocket.")
    ap.add_argument("--host", default="0.0.0.0", help="bind address")
    ap.add_argument("--port", type=int, default=8000, help="bind port")
    ap.add_argument("--noop", action="store_true", help="serve a weightless no-op policy (no GPU/model)")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="LeRobot checkpoint for pi0.5")
    ap.add_argument("--horizon", type=int, default=10, help="actions per returned chunk (replan period)")
    args = ap.parse_args()

    infer = build_noop_infer(args.horizon) if args.noop else build_pi05_infer(args.checkpoint, horizon=args.horizon)
    asyncio.run(serve_openpi(args.host, args.port, infer))


if __name__ == "__main__":
    main()
