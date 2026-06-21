"""Record an offscreen MP4 of a scene running in the Newton sim.

Boots the sim in-process (no LLM, no HUD key), resets the given scene, steps it
forward, and renders frames from a framed free camera into an MP4 via ffmpeg.
Handy for scenes with no authored cameras (e.g. Gizmo room exports).

Usage:
    python scripts/record_scene.py --scene living-room-v1
    python scripts/record_scene.py --scene living-room-v1 --frames 240 --fps 30 \
        --azimuth 120 --elevation -15 --distance 7 --out media/living-room.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="living-room-v1")
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--steps-per-frame", type=int, default=2)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--settle", type=int, default=0,
                    help="physics steps to run before recording starts")
    ap.add_argument("--distance", type=float, default=8.0)
    ap.add_argument("--azimuth", type=float, default=120.0)
    ap.add_argument("--elevation", type=float, default=-18.0)
    ap.add_argument("--lookat", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH; cannot encode MP4.", file=sys.stderr)
        return 1

    import sim.server as S

    r = S.reset(scene_id=args.scene, settle_steps=args.settle)
    if not isinstance(r, dict) or r.get("status") != "ready":
        print(f"reset failed: {r}", file=sys.stderr)
        return 1
    print(f"reset ok: {args.scene} | solver={r.get('solver')} "
          f"| bodies={r.get('n_bodies')} joints={r.get('n_joints')}")

    sim = S._require_sim()
    model = sim.mj_model or sim.solver.mj_model
    nu = sim.solver.mj_model.nu
    zero = [0.0] * nu

    # Enlarge the offscreen framebuffer to fit the requested resolution
    # (default is 640x480; avoids editing the scene XML's <visual> block).
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = args.lookat
    cam.distance = args.distance
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    out = Path(args.out) if args.out else (ROOT / "media" / f"{args.scene}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for f in range(args.frames):
            with sim.lock:
                sim._sync_render_data()
                renderer.update_scene(sim.mj_data, cam)
                pixels = renderer.render()
            Image.fromarray(pixels).save(tmp / f"frame_{f:05d}.png")
            for _ in range(args.steps_per_frame):
                S.step(zero)
            if f % 30 == 0:
                print(f"  frame {f}/{args.frames}")
        renderer.close()

        cmd = [
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(tmp / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(out),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # also drop a single preview PNG next to the video
    preview = out.with_suffix(".png")
    Image.fromarray(pixels).save(preview)
    print(f"wrote {out}  ({args.frames} frames @ {args.fps}fps)")
    print(f"wrote {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
