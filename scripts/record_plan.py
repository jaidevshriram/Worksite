"""Record a video of the planning agents executing the tidy-room / move-table plan.

The symbolic env only emits per-step render() snapshots; this replays the scripted
skill plan with smooth interpolation + a walking gait and encodes an MP4, using the
same 3D scene + humanoids as the live render() tool.

    python scripts/record_plan.py                 # cooperative two-agent tidy-room
    python scripts/record_plan.py --mode single    # single-agent move-side-table

This is a *visualization* of the plan (kinematic), not a physics rollout.
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
sys.path.insert(0, str(ROOT))
from sim.scene_render import get_renderer  # noqa: E402
from sim.agent_spacing import formation_offsets, MIN_AGENT_SEP  # noqa: E402

LEGS = ["left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
        "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint"]
ARMS = {"left_shoulder_pitch_joint": 0.2, "left_elbow_joint": 1.28,
        "right_shoulder_pitch_joint": 0.2, "right_elbow_joint": 1.28}
BASE_H = 1.01  # pelvis height for living-room-v1 (floor top at z=0.2, measured from foot contact)
FPS = 30


class Recorder:
    def __init__(self, scene_id: str, agent_prefixes: list[str]):
        n = len(agent_prefixes)
        self.sr = get_renderer(scene_id, n_agents=n)
        self.model, self.data = self.sr.model, self.sr.data
        self.prefixes = agent_prefixes  # e.g. ["", "B_"]; index aligns with base_adr
        self.base_adr = self.sr.base_adr
        self.joint_adr = {p: {j: self._jadr(f"{p}{j}") for j in (*LEGS, *ARMS)} for p in agent_prefixes}
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, 1280)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, 960)
        self.renderer = mujoco.Renderer(self.model, height=640, width=960)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 7.2, -55.0, 135.0
        self.cam.lookat[:] = (0.0, 0.2, 0.4)
        # per-agent kinematic state
        self.pos = {p: [0.0, 0.0] for p in agent_prefixes}
        self.yaw = {p: math.pi / 2 for p in agent_prefixes}
        self.phase = {p: 0.0 for p in agent_prefixes}
        self.frames_dir = Path(tempfile.mkdtemp())
        self.n = 0
        self.last = None

    def _jadr(self, name):
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_qposadr[j]) if j >= 0 else -1

    def set_agent_start(self, prefix, x, y, yaw=math.pi / 2):
        self.pos[prefix] = [x, y]
        self.yaw[prefix] = yaw

    def set_object(self, body, x, y, z, yaw=0.0):
        a = self.sr._free_adr(body)
        if a >= 0:
            self.data.qpos[a:a + 7] = [x, y, z, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    def _pose_agent(self, prefix, walking):
        a = self.base_adr[self.prefixes.index(prefix)]
        if a < 0:
            return
        gp = self.phase[prefix]
        bob = 0.012 * math.sin(2 * gp) if walking else 0.0
        x, y = self.pos[prefix]
        yaw = self.yaw[prefix]
        self.data.qpos[a:a + 7] = [x, y, BASE_H + bob, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        adr = self.joint_adr[prefix]
        for jn, v in ARMS.items():
            if adr[jn] >= 0:
                self.data.qpos[adr[jn]] = v
        if not walking:
            for jn in LEGS:
                if adr[jn] >= 0:
                    self.data.qpos[adr[jn]] = 0.0
            return
        hip, knee, ank = 0.42, 0.62, 0.22
        for side, ph in (("left", gp), ("right", gp + math.pi)):
            s = math.sin(ph)
            if adr[f"{side}_hip_pitch_joint"] >= 0:
                self.data.qpos[adr[f"{side}_hip_pitch_joint"]] = hip * s
            if adr[f"{side}_knee_joint"] >= 0:
                self.data.qpos[adr[f"{side}_knee_joint"]] = knee * max(0.0, math.sin(ph + 0.4))
            if adr[f"{side}_ankle_pitch_joint"] >= 0:
                self.data.qpos[adr[f"{side}_ankle_pitch_joint"]] = -ank * s
        arm = 0.32
        if adr["left_shoulder_pitch_joint"] >= 0:
            self.data.qpos[adr["left_shoulder_pitch_joint"]] = ARMS["left_shoulder_pitch_joint"] - arm * math.sin(gp)
        if adr["right_shoulder_pitch_joint"] >= 0:
            self.data.qpos[adr["right_shoulder_pitch_joint"]] = ARMS["right_shoulder_pitch_joint"] + arm * math.sin(gp)

    def _frame(self, walking: set[str]):
        for p in self.prefixes:
            self._pose_agent(p, p in walking)
        prev = self.model.opt.disableflags
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        mujoco.mj_forward(self.model, self.data)
        self.model.opt.disableflags = prev
        self.renderer.update_scene(self.data, self.cam)
        px = self.renderer.render()
        self.last = px
        Image.fromarray(px).save(self.frames_dir / f"f_{self.n:05d}.png")
        self.n += 1

    def walk(self, targets: dict[str, tuple[float, float]], speed=1.1):
        """Move the given agents to their targets simultaneously (others stand)."""
        dist = max((math.hypot(targets[p][0] - self.pos[p][0], targets[p][1] - self.pos[p][1])
                    for p in targets), default=0.0)
        frames = max(6, int(dist / speed * FPS))
        starts = {p: tuple(self.pos[p]) for p in targets}
        for p in targets:
            self.yaw[p] = math.atan2(targets[p][1] - self.pos[p][1], targets[p][0] - self.pos[p][0])
        for k in range(frames):
            f = (k + 1) / frames
            for p in targets:
                self.pos[p] = [starts[p][0] + (targets[p][0] - starts[p][0]) * f,
                               starts[p][1] + (targets[p][1] - starts[p][1]) * f]
                self.phase[p] += 2 * math.pi * 1.7 / FPS
            self._frame(set(targets))

    def carry(self, agents: list[str], body: str, target: tuple[float, float], z: float,
              speed=0.9, offsets: dict[str, tuple[float, float]] | None = None):
        """Carriers + the object move together to the target (offsets keep co-carriers apart)."""
        offsets = offsets or {p: (0.0, 0.0) for p in agents}
        start = [self.pos[agents[0]][0] - offsets[agents[0]][0],
                 self.pos[agents[0]][1] - offsets[agents[0]][1]]
        dist = math.hypot(target[0] - start[0], target[1] - start[1])
        frames = max(6, int(dist / speed * FPS))
        for p in agents:
            self.yaw[p] = math.atan2(target[1] - self.pos[p][1], target[0] - self.pos[p][0])
        for k in range(frames):
            f = (k + 1) / frames
            x = start[0] + (target[0] - start[0]) * f
            y = start[1] + (target[1] - start[1]) * f
            for p in agents:
                self.pos[p] = [x + offsets[p][0], y + offsets[p][1]]
                self.phase[p] += 2 * math.pi * 1.7 / FPS
            self.set_object(body, x, y, z)
            self._frame(set(agents))

    def pause(self, frames=10):
        for _ in range(frames):
            self._frame(set())

    def encode(self, out: Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(self.frames_dir / "f_%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if self.last is not None:
            Image.fromarray(self.last).save(out.with_suffix(".png"))
        shutil.rmtree(self.frames_dir, ignore_errors=True)


def record_coop() -> Path:
    from environment.coop_env import AGENTS, PLACEMENTS
    from sim import coop_world as cw
    cw.reset_world("living-room-v1", placements=PLACEMENTS, agents=AGENTS)
    objs = {n: (o["x"], o["y"], o["z"]) for n, o in cw._WORLD.objects.items()}

    rec = Recorder("living-room-v1", agent_prefixes=["", "B_"])
    rec.set_agent_start("", *AGENTS["A"])
    rec.set_agent_start("B_", *AGENTS["B"])
    for name, (x, y, z) in objs.items():
        rec.set_object(name, x, y, z)
    rec.pause(8)

    light = [p for p in PLACEMENTS if p["carriers"] == 1]
    heavy = next(p for p in PLACEMENTS if p["carriers"] >= 2)
    a_item, b_item = light[0], light[1]

    # Phase 1: A and B each fetch a light item in parallel.
    rec.walk({"": (objs[a_item["object"]][0], objs[a_item["object"]][1]),
              "B_": (objs[b_item["object"]][0], objs[b_item["object"]][1])})
    rec.pause(6)
    rec.carry([""], a_item["object"], tuple(a_item["goal"]), objs[a_item["object"]][2])
    # B continues while A is done
    rec.carry(["B_"], b_item["object"], tuple(b_item["goal"]), objs[b_item["object"]][2])
    rec.pause(6)

    # Phase 2: both converge on the heavy TV (formation spacing), then co-carry it.
    hx, hy, hz = objs[heavy["object"]]
    pair = formation_offsets(2, MIN_AGENT_SEP)
    rec.walk({"": (hx + pair[0][0], hy + pair[0][1]),
              "B_": (hx + pair[1][0], hy + pair[1][1])})
    rec.pause(8)
    rec.carry(["", "B_"], heavy["object"], tuple(heavy["goal"]), hz,
              offsets={"": pair[0], "B_": pair[1]})
    rec.pause(20)

    out = ROOT / "media" / "coop_tidy_room.mp4"
    rec.encode(out)
    return out


def record_single() -> Path:
    from sim import skill_world as sw
    sw.reset_world("living-room-v1")
    objs = {n: (o["x"], o["y"], o["z"]) for n, o in sw._WORLD.objects.items()}
    t = objs["asset_2_2"]

    rec = Recorder("living-room-v1", agent_prefixes=[""])
    rec.set_agent_start("", sw._WORLD.agent["x"], sw._WORLD.agent["y"])
    for name, (x, y, z) in objs.items():
        rec.set_object(name, x, y, z)
    rec.pause(8)
    rec.walk({"": (t[0], t[1])})
    rec.pause(6)
    rec.carry([""], "asset_2_2", (-2.0, -2.0), t[2])
    rec.pause(20)

    out = ROOT / "media" / "living-room-v1_move_table.mp4"
    rec.encode(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["coop", "single"], default="coop")
    args = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH", file=sys.stderr)
        return 1
    out = record_coop() if args.mode == "coop" else record_single()
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
