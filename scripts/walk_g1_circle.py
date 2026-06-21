"""Kinematic 'walk in a circle' animation for the Unitree G1 in a scene.

IMPORTANT: this is a *scripted kinematic animation*, NOT learned locomotion.
The G1 has a floating base and no balance controller, so true physical walking
requires an RL/whole-body controller (see chat notes). Here we drive the base
along a circle and play a simple leg/arm gait by writing joint angles directly,
using mj_forward (kinematics only - no dynamics, so it never falls). Good for a
demo/preview video of the robot moving through the room.

Usage:
    python scripts/walk_g1_circle.py --scene living-room-v1 \
        --radius 1.6 --center 0 0.2 --laps 1 --frames 300 --fps 30
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

# G1 joints we animate (others held at their standing default).
DEF = {  # standing defaults (rad), arms slightly forward like the 'stand' keyframe
    "left_shoulder_pitch_joint": 0.2, "left_elbow_joint": 1.28,
    "right_shoulder_pitch_joint": 0.2, "right_elbow_joint": 1.28,
}


def _adr(model: mujoco.MjModel, joint: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if jid < 0:
        raise SystemExit(f"joint {joint!r} not found in model")
    return int(model.jnt_qposadr[jid])


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="living-room-v1")
    ap.add_argument("--radius", type=float, default=1.6)
    ap.add_argument("--center", type=float, nargs=2, default=(0.0, 0.2))
    ap.add_argument("--base-height", type=float, default=0.80)
    ap.add_argument("--laps", type=float, default=1.0)
    ap.add_argument("--cadence", type=float, default=1.6, help="steps per second")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--cam-distance", type=float, default=6.5)
    ap.add_argument("--cam-elevation", type=float, default=-55.0)
    ap.add_argument("--cam-azimuth", type=float, default=135.0)
    ap.add_argument("--cam-lookat", type=float, nargs=3, default=(0.0, 0.3, 0.3))
    ap.add_argument("--track", action="store_true",
                    help="track the robot with the camera (default: fixed room view)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH; cannot encode MP4.", file=sys.stderr)
        return 1

    scene = ROOT / "scenes" / args.scene / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)  # qpos = qpos0 (free bodies at initial poses)

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)

    base = _adr(model, "floating_base_joint")
    j = {n: _adr(model, n) for n in [
        "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
        "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
        *DEF.keys(),
    ]}
    for n, v in DEF.items():
        data.qpos[j[n]] = v

    cx, cy = args.center
    hip_amp, knee_amp, ankle_amp, arm_amp = 0.45, 0.65, 0.25, 0.35

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = args.cam_distance
    cam.elevation = args.cam_elevation
    cam.azimuth = args.cam_azimuth
    cam.lookat[:] = args.cam_lookat

    out = Path(args.out) if args.out else (ROOT / "media" / f"{args.scene}_g1_walk.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    def gait(phase: float, lead: bool) -> None:
        # lead=True -> left leg leads; mirror for right with phase+pi
        for side, ph in (("left", phase), ("right", phase + math.pi)):
            s = math.sin(ph)
            data.qpos[j[f"{side}_hip_pitch_joint"]] = hip_amp * s
            data.qpos[j[f"{side}_knee_joint"]] = knee_amp * max(0.0, math.sin(ph + 0.4))
            data.qpos[j[f"{side}_ankle_pitch_joint"]] = -ankle_amp * s
        # arms swing opposite to same-side leg
        data.qpos[j["left_shoulder_pitch_joint"]] = DEF["left_shoulder_pitch_joint"] - arm_amp * math.sin(phase)
        data.qpos[j["right_shoulder_pitch_joint"]] = DEF["right_shoulder_pitch_joint"] + arm_amp * math.sin(phase)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pixels = None
        for f in range(args.frames):
            t = f / args.fps
            theta = 2.0 * math.pi * args.laps * (f / args.frames)
            # base position on the circle, heading tangent (CCW)
            x = cx + args.radius * math.cos(theta)
            y = cy + args.radius * math.sin(theta)
            yaw = theta + math.pi / 2.0
            bob = 0.012 * math.sin(2.0 * 2.0 * math.pi * args.cadence * t)
            data.qpos[base + 0] = x
            data.qpos[base + 1] = y
            data.qpos[base + 2] = args.base_height + bob
            data.qpos[base + 3], data.qpos[base + 4], data.qpos[base + 5], data.qpos[base + 6] = _yaw_quat(yaw)
            gait(2.0 * math.pi * args.cadence * t, lead=True)

            mujoco.mj_forward(model, data)  # kinematics only; no integration
            if args.track:
                cam.lookat[:] = (x, y, 0.5)
            renderer.update_scene(data, cam)
            pixels = renderer.render()
            Image.fromarray(pixels).save(tmp / f"frame_{f:05d}.png")
            if f % 60 == 0:
                print(f"  frame {f}/{args.frames}")
        renderer.close()

        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(tmp / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if pixels is not None:
            Image.fromarray(pixels).save(out.with_suffix(".png"))

    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
