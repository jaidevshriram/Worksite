"""Skill-based loco-manipulation demo for the Unitree G1 (kinematic).

Demonstrates that fundamental abilities - walk to A, pick an object, carry it to
B, place it - are demoable and, crucially, *composable from high-level skills*.
An LLM/Foreman would emit a plan like the PLAN list below; the executor turns
each step into deterministic motion. This is a *kinematic animation* (mj_forward,
no dynamics), so it is reliable by construction - it cannot fall or fail. Real
physical loco-manipulation on a humanoid needs a learned controller; for the RL
env + grading, the abstract floating manipulators do the real physics.

    python scripts/loco_manip_g1.py            # runs the default pick-and-place plan
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

# A plan an LLM/Foreman could emit: "carry the side table to the SW corner".
PLAN = [
    {"skill": "walk_to", "x": 1.0, "y": 1.35},        # approach the side table
    {"skill": "turn_to", "x": 1.52, "y": 1.96},       # face it
    {"skill": "pick", "object": "asset_2_2"},          # lift it
    {"skill": "walk_to", "x": -1.35, "y": -1.35},      # carry to the SW corner
    {"skill": "turn_to", "x": -2.4, "y": -2.4},        # face the corner
    {"skill": "place"},                                 # set it down
    {"skill": "idle", "frames": 20},
]

ARM_DEF = {"left_shoulder_pitch_joint": 0.2, "left_elbow_joint": 1.28,
           "right_shoulder_pitch_joint": 0.2, "right_elbow_joint": 1.28}


def yaw_quat(yaw: float):
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def ang_lerp(a: float, b: float, t: float) -> float:
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + d * t


class Animator:
    def __init__(self, scene: str, fps: int, w: int, h: int, cam):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "scenes" / scene / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, w)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, h)
        self.fps = fps
        self.renderer = mujoco.Renderer(self.model, height=h, width=w)
        self.cam = cam
        self.frames_dir = Path(tempfile.mkdtemp())
        self.n = 0
        self.last = None

        self.adr = {n: int(self.model.jnt_qposadr[self._jid(n)]) for n in [
            "floating_base_joint", "asset_2_2_joint",
            "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
            "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
            *ARM_DEF.keys(),
        ]}
        for n, v in ARM_DEF.items():
            self.data.qpos[self.adr[n]] = v

        # robot planar pose + gait clock + carried object
        self.x, self.y, self.yaw = 0.2, -0.4, math.pi / 2
        self.base_h = 0.80
        self.cadence = 1.7
        self.gait_t = 0.0
        self.held = False
        self.obj_rest = self.data.qpos[self.adr["asset_2_2_joint"]: self.adr["asset_2_2_joint"] + 7].copy()
        self.carry_z = 0.52

    def _jid(self, name):
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise SystemExit(f"joint {name!r} missing")
        return j

    # --- low level: pose the robot + (optionally) the carried object, render ---
    def _gait(self, crouch=0.0, reach=0.0):
        gp = 2 * math.pi * self.cadence * self.gait_t
        hip_amp, knee_amp, ank_amp, arm_amp = 0.42, 0.62, 0.22, 0.32
        for side, ph in (("left", gp), ("right", gp + math.pi)):
            s = math.sin(ph)
            self.data.qpos[self.adr[f"{side}_hip_pitch_joint"]] = hip_amp * s + crouch
            self.data.qpos[self.adr[f"{side}_knee_joint"]] = knee_amp * max(0.0, math.sin(ph + 0.4)) + 1.6 * crouch
            self.data.qpos[self.adr[f"{side}_ankle_pitch_joint"]] = -ank_amp * s - 0.7 * crouch
        self.data.qpos[self.adr["left_shoulder_pitch_joint"]] = ARM_DEF["left_shoulder_pitch_joint"] - arm_amp * math.sin(gp) + reach
        self.data.qpos[self.adr["right_shoulder_pitch_joint"]] = ARM_DEF["right_shoulder_pitch_joint"] + arm_amp * math.sin(gp) + reach
        self.data.qpos[self.adr["left_elbow_joint"]] = ARM_DEF["left_elbow_joint"] - reach
        self.data.qpos[self.adr["right_elbow_joint"]] = ARM_DEF["right_elbow_joint"] - reach

    def _carry_pose(self):
        cx = self.x + 0.42 * math.cos(self.yaw)
        cy = self.y + 0.42 * math.sin(self.yaw)
        return np.array([cx, cy, self.carry_z, *yaw_quat(self.yaw)])

    def _set_object(self, pose):
        a = self.adr["asset_2_2_joint"]
        self.data.qpos[a:a + 7] = pose

    def _frame(self, crouch=0.0, reach=0.0, moving=True):
        b = self.adr["floating_base_joint"]
        bob = 0.012 * math.sin(2 * 2 * math.pi * self.cadence * self.gait_t) if moving else 0.0
        self.data.qpos[b + 0], self.data.qpos[b + 1], self.data.qpos[b + 2] = self.x, self.y, self.base_h - crouch + bob
        self.data.qpos[b + 3], self.data.qpos[b + 4], self.data.qpos[b + 5], self.data.qpos[b + 6] = yaw_quat(self.yaw)
        self._gait(crouch=crouch, reach=reach)
        if self.held:
            self._set_object(self._carry_pose())
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.cam)
        px = self.renderer.render()
        self.last = px
        Image.fromarray(px).save(self.frames_dir / f"frame_{self.n:05d}.png")
        self.n += 1

    # --- skills ---
    def walk_to(self, tx, ty, speed=0.85):
        heading = math.atan2(ty - self.y, tx - self.x)
        for k in range(10):  # turn toward travel direction
            self.yaw = ang_lerp(self.yaw, heading, (k + 1) / 10)
            self.gait_t += 1 / self.fps * 0.4
            self._frame()
        self.yaw = heading
        d = math.hypot(tx - self.x, ty - self.y)
        steps = max(1, int(d / speed * self.fps))
        x0, y0 = self.x, self.y
        for k in range(steps):
            f = (k + 1) / steps
            self.x, self.y = x0 + (tx - x0) * f, y0 + (ty - y0) * f
            self.gait_t += 1 / self.fps
            self._frame()

    def turn_to(self, tx, ty):
        heading = math.atan2(ty - self.y, tx - self.x)
        y0 = self.yaw
        for k in range(14):
            self.yaw = ang_lerp(y0, heading, (k + 1) / 14)
            self.gait_t += 1 / self.fps * 0.3
            self._frame()
        self.yaw = heading

    def pick(self, obj, frames=26):
        target = self._carry_pose()
        start = self.obj_rest.copy()
        for k in range(frames):  # crouch + reach down, object rises to hands
            f = (k + 1) / frames
            crouch = 0.18 * math.sin(math.pi * f)  # dip then back up
            reach = 0.5 * math.sin(math.pi * min(1.0, f * 1.2))
            self._set_object(start + (target - start) * f)
            self._frame(crouch=crouch, reach=reach, moving=False)
        self.held = True

    def place(self, frames=26):
        floor = self.obj_rest.copy()
        floor[0] = self.x + 0.5 * math.cos(self.yaw)
        floor[1] = self.y + 0.5 * math.sin(self.yaw)
        floor[2] = self.obj_rest[2]
        floor[3:7] = yaw_quat(self.yaw)
        start = self._carry_pose()
        for k in range(frames):  # crouch + lower object to floor, then release
            f = (k + 1) / frames
            crouch = 0.18 * math.sin(math.pi * f)
            reach = 0.5 * math.sin(math.pi * min(1.0, f * 1.2))
            self._set_object(start + (floor - start) * f)
            self._frame(crouch=crouch, reach=reach, moving=False)
        self.held = False
        self.obj_rest = floor  # object stays where placed

    def idle(self, frames=20):
        for _ in range(frames):
            self.gait_t += 1 / self.fps * 0.2
            self._frame(moving=False)

    def run(self, plan):
        for step in plan:
            sk = step["skill"]
            print(f"  -> {sk} {({k: v for k, v in step.items() if k != 'skill'})}")
            if sk == "walk_to":
                self.walk_to(step["x"], step["y"])
            elif sk == "turn_to":
                self.turn_to(step["x"], step["y"])
            elif sk == "pick":
                self.pick(step.get("object"))
            elif sk == "place":
                self.place()
            elif sk == "idle":
                self.idle(step.get("frames", 20))
            else:
                raise SystemExit(f"unknown skill {sk!r}")
        self.renderer.close()

    def encode(self, out: Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(self.fps),
            "-i", str(self.frames_dir / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if self.last is not None:
            Image.fromarray(self.last).save(out.with_suffix(".png"))
        shutil.rmtree(self.frames_dir, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="living-room-v1")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH", file=sys.stderr)
        return 1

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 7.0, -55.0, 135.0
    cam.lookat[:] = (0.0, 0.2, 0.3)

    a = Animator(args.scene, args.fps, args.width, args.height, cam)
    a.run(PLAN)
    out = Path(args.out) if args.out else (ROOT / "media" / f"{args.scene}_g1_pickplace.mp4")
    a.encode(out)
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
