"""Record a video of the construction block_line scripted agent task.

The construction scene XML is a static photogrammetry mesh — it has NO movable
block bodies (only fixed `pallet_of_bricks_*` meshes).  So the old recorder's
`objs` were always empty and the bricks never appeared to be carried.  This
recorder injects free-body box markers (`block_1..4`) into the scene via
`SceneRenderer(movable_blocks=...)` and animates them so each brick visibly:

    walk to brick → crouch & pick up → carry (brick lifted, moves WITH agent)
    → walk to scaffold line → crouch & set down

mirroring the living-room recorder (`scripts/record_plan.py`).

Motion is intentionally slow and deliberate:
  MOVE_FRAMES   = 60   walk_to            (~2.0 s at 30 fps)
  CARRY_FRAMES  = 70   carry_to           (~2.3 s, heavier/slower)
  GRAB_FRAMES   = 22   crouch + pick up   (dip the base, lift the brick)
  RELEASE_FRAMES= 22   crouch + set down
  PAUSE_FRAMES  = 14   checkpoint / beat
  SETUP_FRAMES  = 24   opening hold
  FINAL_FRAMES  = 40   closing hold

Camera: gentle orbit (0.03°/frame) so it doesn't add to the sense of speed.

Output
------
  media/construction_task.mp4
  media/construction_task.png   (thumbnail — first carry frame)
"""
from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sim import construction_world as cw  # noqa: E402
from sim.scene_render import (  # noqa: E402
    SceneRenderer, WALK_ANGULAR_VEL,
    _g1_walk_joints, _spot_walk_joints,
)
from environment.construction_env import (  # noqa: E402
    AGENTS, AGENT_SPAWN, SCENE,
    _SE, _SCAFFOLD_LINE, _GUARD_CPS,
)

# ── config ─────────────────────────────────────────────────────────────────────
FPS  = 30
W, H = 960, 540

AGENT_ORDER  = ["A", "B", "C", "D", "E", "F"]
AGENT_TYPES  = ["g1", "g1", "g1", "g1", "spot", "spot"]

# Spread phase offsets so robots aren't all in lockstep (⅙ cycle apart)
PHASE_OFFSETS = [i * (2 * math.pi / 6) for i in range(6)]

# Frame budget — deliberately slow, deliberate movement.
MOVE_FRAMES    = 60
CARRY_FRAMES   = 70
GRAB_FRAMES    = 22
RELEASE_FRAMES = 22
PAUSE_FRAMES   = 14
SETUP_FRAMES   = 24
FINAL_FRAMES   = 40

CAM_AZIMUTH_START = 135.0
CAM_DEG_PER_FRAME = 0.03      # gentle orbit right

CAMERA_BASE = {
    "distance":  18.0,
    "elevation": -45.0,
    "lookat":    [0.0, 0.0, 1.2],
}

# ── block markers ───────────────────────────────────────────────────────────────
BLOCK_HALF = 0.33
GROUND_Z   = 1.52          # brick centre resting on the site floor (floor top ≈ 1.19)
CARRY_Z    = 1.78          # brick centre while lifted/carried
CROUCH_DZ  = 0.14          # how far an agent's base dips when bending to grab/release

# Which G1 carries which brick.
CARRIER = {"A": "block_1", "B": "block_2", "C": "block_3", "D": "block_4"}

MOVABLE_BLOCKS = [
    {"name": "block_1", "half": BLOCK_HALF, "rgba": [0.90, 0.22, 0.16, 1.0]},
    {"name": "block_2", "half": BLOCK_HALF, "rgba": [0.96, 0.56, 0.12, 1.0]},
    {"name": "block_3", "half": BLOCK_HALF, "rgba": [0.95, 0.83, 0.18, 1.0]},
    {"name": "block_4", "half": BLOCK_HALF, "rgba": [0.30, 0.66, 0.92, 1.0]},
]


# ── math helpers ───────────────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _ease(t: float) -> float:
    """Smooth-step (cubic ease-in-out)."""
    return t * t * (3.0 - 2.0 * t)


def _poses_from_snap(snap: dict) -> list[tuple[float, float, float]]:
    """Return (x, y, yaw) for each agent in AGENT_ORDER."""
    agents = snap["agents"]
    return [(agents[aid]["x"], agents[aid]["y"], agents[aid].get("yaw", 0.0))
            for aid in AGENT_ORDER]


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> str:
    print("Building renderer (4×G1 + 2×Spot + 4 movable bricks)…")
    sr = SceneRenderer(SCENE, AGENT_TYPES, movable_blocks=MOVABLE_BLOCKS)
    m, d = sr.model, sr.data
    print(f"  nbody={m.nbody}  ngeom={m.ngeom}")

    block_adr = {b["name"]: sr._free_adr(b["name"]) for b in MOVABLE_BLOCKS}
    print(f"  block free-joint addrs: {block_adr}")

    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance  = CAMERA_BASE["distance"]
    cam.elevation = CAMERA_BASE["elevation"]
    cam.lookat[:] = CAMERA_BASE["lookat"]

    vid_renderer = mujoco.Renderer(m, H, W)
    frames_dir   = tempfile.mkdtemp(prefix="task_frames_")
    frame_offset = 0
    phases       = list(PHASE_OFFSETS)
    thumbnail_px = None

    # Persistent block render state {name: [x, y, z]} — seeded at spawn positions.
    block_pos: dict[str, list[float]] = {
        b["name"]: [next(s["x"] for s in _SE if s["name"] == b["name"]),
                    next(s["y"] for s in _SE if s["name"] == b["name"]),
                    GROUND_Z]
        for b in MOVABLE_BLOCKS
    }

    # ── low-level pose helpers ─────────────────────────────────────────────────

    def _pose_agents(agent_poses, walking: set[str], dz: dict[str, float]) -> None:
        for i, (adr, (x, y, yaw)) in enumerate(zip(sr._base_adr, agent_poses)):
            if adr < 0:
                continue
            aid   = AGENT_ORDER[i]
            bh    = sr._base_heights[i] - dz.get(aid, 0.0)
            rtype = AGENT_TYPES[i]
            if aid in walking:
                phi = phases[i]
                sj  = _g1_walk_joints(phi) if rtype == "g1" else _spot_walk_joints(phi)
            else:
                sj  = sr._standing_joints[i]
            d.qpos[adr:adr + 7] = [
                x, y, bh, math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]
            if sj:
                d.qpos[adr + 7:adr + 7 + len(sj)] = sj

    def _pose_blocks() -> None:
        for name, adr in block_adr.items():
            if adr < 0:
                continue
            x, y, z = block_pos[name]
            d.qpos[adr:adr + 7] = [x, y, z, 1., 0., 0., 0.]

    def _advance_phases(walking: set[str]) -> None:
        for i, aid in enumerate(AGENT_ORDER):
            if aid in walking:
                phases[i] = (phases[i] + WALK_ANGULAR_VEL) % (2 * math.pi)

    def _render_frame(agent_poses, walking: set[str],
                      dz: dict[str, float] | None = None) -> "np.ndarray":
        nonlocal frame_offset
        mujoco.mj_resetData(m, d)
        _pose_agents(agent_poses, walking, dz or {})
        _pose_blocks()

        prev = m.opt.disableflags
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        mujoco.mj_forward(m, d)
        m.opt.disableflags = prev

        cam.azimuth = CAM_AZIMUTH_START + CAM_DEG_PER_FRAME * frame_offset
        vid_renderer.update_scene(d, cam)
        px = vid_renderer.render()
        Image.fromarray(px).save(f"{frames_dir}/f{frame_offset:05d}.png")
        frame_offset += 1
        return px

    # ── high-level animation primitives ────────────────────────────────────────

    def hold(snap, n_frames: int, animate: bool = True) -> None:
        poses   = _poses_from_snap(snap)
        walking = set(AGENT_ORDER) if animate else set()
        for _ in range(n_frames):
            _render_frame(poses, walking)
            _advance_phases(walking)

    def walk(label: str, fn, n: int = MOVE_FRAMES) -> None:
        snap_before = cw.world_snapshot()
        fn()
        snap_after = cw.world_snapshot()
        poses_before = _poses_from_snap(snap_before)
        poses_after  = _poses_from_snap(snap_after)

        moved = {AGENT_ORDER[j] for j in range(len(AGENT_ORDER))
                 if math.hypot(poses_after[j][0] - poses_before[j][0],
                               poses_after[j][1] - poses_before[j][1]) > 0.05}
        a_after = snap_after["agents"]
        print(f"  [{label}]  moved={sorted(moved)}  "
              + "  ".join(f"{aid}=({a_after[aid]['x']:.1f},{a_after[aid]['y']:.1f})"
                          for aid in AGENT_ORDER))

        # Any held bricks ride along (they sit at CARRY_Z above their carrier).
        held = {CARRIER[aid]: aid for aid in moved
                if aid in CARRIER and a_after[aid].get("holding")}

        for fi in range(n):
            t  = _ease(fi / max(n - 1, 1))
            interp = []
            for j in range(len(AGENT_ORDER)):
                bx, by, _      = poses_before[j]
                ax, ay, ay_yaw = poses_after[j]
                ix, iy = _lerp(bx, ax, t), _lerp(by, ay, t)
                dx, dy = ax - bx, ay - by
                iyaw = math.atan2(dy, dx) if (abs(dx) > 0.01 or abs(dy) > 0.01) else ay_yaw
                interp.append((ix, iy, iyaw))
            for bname, aid in held.items():
                ax, ay, _ = interp[AGENT_ORDER.index(aid)]
                block_pos[bname][0], block_pos[bname][1] = ax, ay
                block_pos[bname][2] = CARRY_Z
            _render_frame(interp, moved)
            _advance_phases(moved)

    def crouch(label: str, aid: str, fn, picking_up: bool, n: int = GRAB_FRAMES):
        """Bend down to grab (lift brick) or release (lower brick)."""
        nonlocal thumbnail_px
        fn()
        snap = cw.world_snapshot()
        poses = _poses_from_snap(snap)
        bname = CARRIER.get(aid)
        print(f"  [{label}]  {aid} {'grab' if picking_up else 'release'} {bname}")

        z_from = GROUND_Z if picking_up else CARRY_Z
        z_to   = CARRY_Z if picking_up else GROUND_Z
        for fi in range(n):
            t   = fi / max(n - 1, 1)
            dip = math.sin(math.pi * t) * CROUCH_DZ
            if bname is not None:
                block_pos[bname][2] = _lerp(z_from, z_to, _ease(t))
            px = _render_frame(poses, set(), dz={aid: dip})
            if thumbnail_px is None and picking_up:
                thumbnail_px = Image.fromarray(px)

    def beat(label: str, fn, n: int = PAUSE_FRAMES) -> None:
        fn()
        snap = cw.world_snapshot()
        poses = _poses_from_snap(snap)
        print(f"  [{label}]")
        for _ in range(n):
            _render_frame(poses, set())

    # ── reset world ────────────────────────────────────────────────────────────
    print("Resetting world…")
    snap = cw.reset_world(
        scene_id=SCENE, agents=AGENTS, agent_positions=AGENT_SPAWN,
        blocks=_SE, targets=_SCAFFOLD_LINE, checkpoints=_GUARD_CPS,
    )
    placed = sum(1 for b in snap["blocks"].values() if b.get("at_goal"))
    print(f"  Initial: placed={placed}/{len(snap['blocks'])}")

    # ── opening hold ───────────────────────────────────────────────────────────
    print(f"Opening hold ({SETUP_FRAMES} frames)…")
    hold(snap, SETUP_FRAMES, animate=False)

    cw.say("A", "I'll take block_1 to (-4, 7)")
    cw.say("B", "I'll take block_2 to (-2, 7)")
    cw.say("C", "I'll take block_3 to (0, 7)")
    cw.say("D", "I'll take block_4 to (2, 7)")
    cw.say("E", "Heading to guard_west")
    cw.say("F", "Heading to guard_east")

    # ── Step 1: walk to bricks / guard posts ───────────────────────────────────
    print("\nStep 1 — Walk to bricks / guard posts…")
    walk("walk A→block_1", lambda: cw.walk_to("A", 10.0, -4.7))
    walk("walk B→block_2", lambda: cw.walk_to("B",  8.5, -5.0))
    walk("walk C→block_3", lambda: cw.walk_to("C",  9.9, -6.5))
    walk("walk D→block_4", lambda: cw.walk_to("D", 11.4, -5.0))
    walk("walk E→guard_W", lambda: cw.walk_to("E", -6.0,  2.0))
    walk("walk F→guard_E", lambda: cw.walk_to("F",  7.0, -2.0))

    # ── Step 2: checkpoints + pick up (crouch) ─────────────────────────────────
    print("\nStep 2 — Checkpoint + Grab…")
    beat("cp E guard_west", lambda: cw.checkpoint("E", "guard_west"))
    beat("cp F guard_east", lambda: cw.checkpoint("F", "guard_east"))
    crouch("grab A block_1", "A", lambda: cw.grab("A", "block_1"), picking_up=True)
    crouch("grab B block_2", "B", lambda: cw.grab("B", "block_2"), picking_up=True)
    crouch("grab C block_3", "C", lambda: cw.grab("C", "block_3"), picking_up=True)
    crouch("grab D block_4", "D", lambda: cw.grab("D", "block_4"), picking_up=True)

    # ── Step 3: carry bricks to scaffold line (slow) ───────────────────────────
    print("\nStep 3 — Carry to scaffold line…")
    walk("carry A → (-4,7)", lambda: cw.carry_to("A", -4.0, 7.0), n=CARRY_FRAMES)
    walk("carry B → (-2,7)", lambda: cw.carry_to("B", -2.0, 7.0), n=CARRY_FRAMES)
    walk("carry C →  (0,7)", lambda: cw.carry_to("C",  0.0, 7.0), n=CARRY_FRAMES)
    walk("carry D →  (2,7)", lambda: cw.carry_to("D",  2.0, 7.0), n=CARRY_FRAMES)

    # ── Step 4: set down (crouch) ──────────────────────────────────────────────
    print("\nStep 4 — Release…")
    crouch("release A", "A", lambda: cw.release("A"), picking_up=False, n=RELEASE_FRAMES)
    crouch("release B", "B", lambda: cw.release("B"), picking_up=False, n=RELEASE_FRAMES)
    crouch("release C", "C", lambda: cw.release("C"), picking_up=False, n=RELEASE_FRAMES)
    crouch("release D", "D", lambda: cw.release("D"), picking_up=False, n=RELEASE_FRAMES)

    # ── closing hold ───────────────────────────────────────────────────────────
    snap   = cw.world_snapshot()
    placed = sum(1 for b in snap["blocks"].values() if b.get("at_goal"))
    m_str  = (f"placed={placed}/{len(snap['blocks'])}  "
              f"fleet_util={snap['metrics']['fleet_utilization']:.2f}  "
              f"collisions={snap['metrics']['collision_events']}")
    print(f"\nFinal state: {m_str}")
    hold(snap, FINAL_FRAMES, animate=False)

    vid_renderer.close()
    total_frames = frame_offset
    duration = total_frames / FPS
    print(f"\nTotal frames rendered: {total_frames}  ({duration:.1f} s @ {FPS} fps)")

    # ── thumbnail ──────────────────────────────────────────────────────────────
    out_png = REPO_ROOT / "media" / "construction_task.png"
    if thumbnail_px is None:
        thumbnail_px = Image.open(f"{frames_dir}/f{max(frame_offset - 1, 0):05d}.png")
    thumbnail_px.save(str(out_png))
    print(f"Thumbnail → {out_png}")

    # ── encode MP4 ─────────────────────────────────────────────────────────────
    out_mp4 = REPO_ROOT / "media" / "construction_task.mp4"
    print("Encoding MP4…")
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", f"{frames_dir}/f%05d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(out_mp4),
    ], check=True, capture_output=True)

    size = out_mp4.stat().st_size
    print(f"Video  → {out_mp4}  ({size:,} bytes, {size / 1024:.1f} KB)")

    shutil.rmtree(frames_dir)
    return str(out_mp4)


if __name__ == "__main__":
    main()
