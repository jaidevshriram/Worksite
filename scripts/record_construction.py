"""Record a walkabout video for the construction-v1 scene.

4 Unitree G1 humanoids + 2 Boston Dynamics Spot robots walk around the site
with full-body animated walking cycles (hip/knee/ankle/arm for G1; trot gait
for Spot).

Output:
    media/construction_walkabout.mp4
    media/construction_walkabout.png   (thumbnail)
"""
from __future__ import annotations

import math
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sim.scene_render import get_renderer, WALK_ANGULAR_VEL  # noqa: E402

AGENT_TYPES = ["g1", "g1", "g1", "g1", "spot", "spot"]

# Spawn positions: 4 G1 in centre cluster, 2 Spot flanking them
SPAWN = [
    (-1.376,  2.249, 0.0),    # G1-0
    ( 2.810,  2.102, 3.14),   # G1-1
    (-1.667, -2.276, 1.57),   # G1-2
    ( 2.658, -2.473, -1.57),  # G1-3
    (-6.0,    4.0,   0.0),    # Spot-0 — west side of foundation
    ( 7.0,   -4.0,  -1.57),   # Spot-1 — east side of foundation
]

# Waypoints each agent visits in order (loops after the last)
WAYPOINTS = [
    [(-7.0, 6.0), (8.0, -3.0), (-1.4,  2.2)],    # G1-0
    [( 8.0, 5.0), (9.0, -5.0), ( 2.8,  2.1)],    # G1-1
    [( 3.0, 6.5), (-7.0, -4.0), (-1.7, -2.3)],   # G1-2
    [( 7.0, -4.5), (-5.0, -3.0), ( 2.7, -2.5)],  # G1-3
    [( 0.0,  0.0), (8.0,  4.0),  (-6.0,  4.0)],  # Spot-0
    [( 0.0, -1.0), (-5.0, 5.0),  ( 7.0, -4.0)],  # Spot-1
]

FPS          = 30
TOTAL_FRAMES = 240   # 8 seconds
WALK_SPEED   = {"g1": 1.8, "spot": 1.5}   # m/s
PAUSE_FRAMES = 20    # frames to linger at each waypoint
W, H         = 960, 540

# Phase offsets so agents are not in identical pose (spread ⅙ cycle apart)
PHASE_OFFSET = [i * (2 * math.pi / len(AGENT_TYPES)) for i in range(len(AGENT_TYPES))]

# Camera keyframes: (frame, distance, elevation, azimuth, lookat_x, lookat_y)
CAM_KF = [
    (  0, 25, -40, 135,  0.0,  0.0),
    ( 80, 18, -50, 180,  2.0,  0.5),
    (140, 14, -45,  90,  9.5, -4.5),
    (180, 22, -38, 220, -6.0,  4.0),
    (220, 20, -42, 160,  0.5,  1.0),
]


def _lerp(a, b, t):
    return a + (b - a) * t


def _cam(frame: int) -> tuple:
    kf = CAM_KF
    if frame <= kf[0][0]:   return kf[0][1:]
    if frame >= kf[-1][0]:  return kf[-1][1:]
    for i in range(len(kf) - 1):
        f0, f1 = kf[i][0], kf[i + 1][0]
        if f0 <= frame < f1:
            t = (frame - f0) / (f1 - f0)
            t = t * t * (3 - 2 * t)  # smooth-step easing
            return tuple(_lerp(kf[i][j], kf[i + 1][j], t) for j in range(1, 6))
    return kf[-1][1:]


def main():
    print("Building model (4×G1 + 2×Spot)…")
    sr = get_renderer("construction-v1", AGENT_TYPES)
    m, d = sr.model, sr.data
    print(f"  nbody={m.nbody}  ngeom={m.ngeom}  agents={len(AGENT_TYPES)}")

    n    = len(AGENT_TYPES)
    dt   = 1.0 / FPS

    # Per-agent mutable state
    cx        = [s[0] for s in SPAWN]
    cy        = [s[1] for s in SPAWN]
    cyaw      = [s[2] for s in SPAWN]
    wpi       = [0] * n          # current waypoint index
    pause     = [0] * n          # frames still paused at waypoint
    walk_phase = list(PHASE_OFFSET)  # gait phase, advances when moving

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    vid_renderer = mujoco.Renderer(m, H, W)

    frames_dir = tempfile.mkdtemp(prefix="constr_frames_")
    print(f"Rendering {TOTAL_FRAMES} frames → {frames_dir}")

    thumbnail_px = None

    for frame in range(TOTAL_FRAMES):
        # ── advance agents ────────────────────────────────────────────
        for i in range(n):
            rtype = AGENT_TYPES[i]
            if pause[i] > 0:
                pause[i] -= 1
                continue
            if wpi[i] >= len(WAYPOINTS[i]):
                continue
            tx, ty = WAYPOINTS[i][wpi[i]]
            dx, dy = tx - cx[i], ty - cy[i]
            dist = math.hypot(dx, dy)
            if dist < 0.15:
                pause[i] = PAUSE_FRAMES
                wpi[i] += 1
            else:
                step = min(WALK_SPEED[rtype] * dt, dist)
                cyaw[i] = math.atan2(dy, dx)
                cx[i] += step * math.cos(cyaw[i])
                cy[i] += step * math.sin(cyaw[i])
                # Advance walk phase at a speed proportional to step size
                # (faster cadence = more phase per frame; clamp to WALK_ANGULAR_VEL)
                walk_phase[i] += WALK_ANGULAR_VEL
            # keep phase in [0, 2π)
            walk_phase[i] %= (2 * math.pi)

        # ── pose & forward kinematics ─────────────────────────────────
        agent_poses  = [(cx[i], cy[i], cyaw[i]) for i in range(n)]
        # Only animate agents that are moving (not paused at waypoint)
        phases = [
            walk_phase[i] if (wpi[i] < len(WAYPOINTS[i]) and pause[i] == 0) else None
            for i in range(n)
        ]
        # pose() accepts None entries — use standing joints for paused agents
        mujoco.mj_resetData(m, d)
        for i, ((x, y, yaw), adr) in enumerate(zip(agent_poses, sr._base_adr)):
            if adr < 0:
                continue
            bh  = sr._base_heights[i]
            rtype = AGENT_TYPES[i]
            φ   = phases[i]

            from sim.scene_render import _g1_walk_joints, _spot_walk_joints
            if φ is not None:
                if rtype == "g1":
                    sj = _g1_walk_joints(φ)
                elif rtype == "spot":
                    sj = _spot_walk_joints(φ)
                else:
                    sj = sr._standing_joints[i]
            else:
                sj = sr._standing_joints[i]

            d.qpos[adr:adr + 7] = [x, y, bh,
                                    math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]
            if sj:
                d.qpos[adr + 7:adr + 7 + len(sj)] = sj

        prev = m.opt.disableflags
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        mujoco.mj_forward(m, d)
        m.opt.disableflags = prev

        # ── camera ────────────────────────────────────────────────────
        cam_dist, elev, azim, lx, ly = _cam(frame)
        cam.distance, cam.elevation, cam.azimuth = cam_dist, elev, azim
        cam.lookat[:] = [lx, ly, 1.2]

        vid_renderer.update_scene(d, cam)
        px = vid_renderer.render()
        Image.fromarray(px).save(f"{frames_dir}/f{frame:05d}.png")

        if frame == 60:
            thumbnail_px = px
        if frame % 30 == 0:
            print(f"  frame {frame}/{TOTAL_FRAMES}")

    vid_renderer.close()

    # ── thumbnail ─────────────────────────────────────────────────────
    out_png = str(REPO_ROOT / "media" / "construction_walkabout.png")
    (Image.fromarray(thumbnail_px) if thumbnail_px is not None
     else Image.open(f"{frames_dir}/f00060.png")).save(out_png)
    print(f"Thumbnail → {out_png}")

    # ── encode video ──────────────────────────────────────────────────
    out_mp4 = str(REPO_ROOT / "media" / "construction_walkabout.mp4")
    print("Encoding MP4…")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", f"{frames_dir}/f%05d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        out_mp4,
    ], check=True, capture_output=True)
    print(f"Video  → {out_mp4}")

    import shutil
    shutil.rmtree(frames_dir)


if __name__ == "__main__":
    main()
