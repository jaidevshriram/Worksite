"""Symbolic skill world - a physics-free planning layer served as an `mcp` capability.

For grading *planning* (not low-level control), the agent drives high-level skills
with explicit preconditions and effects over a symbolic world state:

    walk_to(x, y)        move the agent to a floor point
    pick(object_name)    grab an object  (pre: near it + hands empty)
    place(x, y)          set the held object down  (pre: holding + near the point)
    get_world_state()    observe agent pose, held object, object + goal positions
    render()             top-down schematic of the room (for the trace)

The world state lives in this module's process; both the agent (over MCP) and the
env-side grader (in-process, via `world_snapshot()`) read the same `_WORLD`. No
contact dynamics - skills deterministically transform world state, so a *valid*
plan always succeeds and an invalid one fails on a precondition. Object/goal data
is loaded from the scene's metadata.json, so this generalizes to any scene with a
`tasks.*.goal_region`.
"""

from __future__ import annotations

import io
import json
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.utilities.types import Image as MCPImage
from PIL import Image, ImageDraw

SCENES_DIR = Path(os.environ.get("SCENES_DIR", Path(__file__).resolve().parents[1] / "scenes"))

REACH = 0.75          # planar distance (m) within which pick/place are allowed
ROOM_HALF = 2.3       # agents/objects are clamped to the interior (walls at +-2.5)

server = FastMCP(name="worldsim-skills")


@dataclass
class _World:
    scene_id: str = ""
    agent: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": -1.0, "yaw": math.pi / 2})
    holding: str | None = None
    objects: dict[str, dict[str, Any]] = field(default_factory=dict)  # name -> {x,y,z,movable}
    target: str = ""
    goal: dict[str, float] = field(default_factory=dict)              # {x,y,radius}
    target_initial_distance: float = 1.0
    target_was_picked: bool = False
    steps: int = 0
    log: list[str] = field(default_factory=list)


_WORLD = _World()
_LOCK = threading.Lock()


def _clamp(v: float) -> float:
    return max(-ROOM_HALF, min(ROOM_HALF, v))


def reset_world(scene_id: str = "living-room-v1", task: str | None = None) -> dict[str, Any]:
    """(Re)load the symbolic world from a scene's metadata.json. Returns a summary."""
    meta_path = SCENES_DIR / scene_id / "metadata.json"
    meta = json.loads(meta_path.read_text())

    objects: dict[str, dict[str, Any]] = {}
    for name, spec in meta.get("objects", {}).items():
        pos = spec.get("initial_position")
        if not pos:
            continue
        objects[name] = {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                         "movable": spec.get("type") == "free"}

    # Pick the goal task: explicit arg, else the first task with a goal_region.
    tasks = meta.get("tasks", {})
    task_name = task
    if task_name is None:
        task_name = next((k for k, v in tasks.items() if "goal_region" in v), None)
    tspec = tasks.get(task_name, {}) if task_name else {}
    target = tspec.get("target_object", "")
    gr = tspec.get("goal_region", {"center": [-2.0, -2.0], "radius": 0.6})
    goal = {"x": float(gr["center"][0]), "y": float(gr["center"][1]), "radius": float(gr["radius"])}

    spawn = (meta.get("robot") or {}).get("spawn_position") or [0.0, -1.0, 0.0]
    start = objects.get(target, {"x": 0.0, "y": 0.0})

    global _WORLD
    with _LOCK:
        _WORLD = _World(
            scene_id=scene_id,
            agent={"x": _clamp(float(spawn[0])), "y": _clamp(float(spawn[1])), "yaw": -math.pi / 2},
            holding=None,
            objects=objects,
            target=target,
            goal=goal,
            target_initial_distance=max(1e-3, math.hypot(start["x"] - goal["x"], start["y"] - goal["y"])),
            target_was_picked=False,
            steps=0,
            log=[f"reset {scene_id}: target={target!r} goal={goal}"],
        )
    return _summary()


def _dist_agent_to(x: float, y: float) -> float:
    return math.hypot(_WORLD.agent["x"] - x, _WORLD.agent["y"] - y)


def _summary() -> dict[str, Any]:
    w = _WORLD
    tx = w.objects.get(w.target, {}).get("x", 0.0)
    ty = w.objects.get(w.target, {}).get("y", 0.0)
    tdist = math.hypot(tx - w.goal.get("x", 0.0), ty - w.goal.get("y", 0.0)) if w.goal else None
    return {
        "scene_id": w.scene_id,
        "agent": {k: round(v, 4) for k, v in w.agent.items()},
        "holding": w.holding,
        "objects": {n: {"x": round(o["x"], 4), "y": round(o["y"], 4), "z": round(o["z"], 4),
                        "movable": o["movable"]} for n, o in w.objects.items()},
        "target_object": w.target,
        "goal": w.goal,
        "target_distance_to_goal": round(tdist, 4) if tdist is not None else None,
        "target_in_goal": (tdist is not None and tdist <= w.goal["radius"]),
        "target_was_picked": w.target_was_picked,
        "reach_radius": REACH,
        "steps": w.steps,
    }


def world_snapshot() -> dict[str, Any]:
    """In-process read for the env-side grader (same state the MCP tools mutate)."""
    with _LOCK:
        snap = _summary()
        snap["target_initial_distance"] = round(_WORLD.target_initial_distance, 4)
        return snap


# ── skills (MCP tools) ────────────────────────────────────────────────────────


@server.tool()
def reset(scene_id: str = "living-room-v1") -> dict[str, Any]:
    """Reset the symbolic world for a scene. Returns agent pose, objects, and the goal."""
    return reset_world(scene_id)


@server.tool()
def get_world_state() -> dict[str, Any]:
    """Observe the world: agent pose, held object, every object's position, and the goal region.

    Call this first, and after each skill, to plan and verify.
    """
    with _LOCK:
        return _summary()


@server.tool()
def walk_to(x: float, y: float) -> dict[str, Any]:
    """Walk the agent to floor point (x, y). Always succeeds (clamped to the room).

    If holding an object, it is carried (its position tracks the agent).
    """
    with _LOCK:
        w = _WORLD
        tx, ty = _clamp(x), _clamp(y)
        w.agent["yaw"] = math.atan2(ty - w.agent["y"], tx - w.agent["x"])
        w.agent["x"], w.agent["y"] = tx, ty
        if w.holding:
            w.objects[w.holding]["x"], w.objects[w.holding]["y"] = tx, ty
        w.steps += 1
        w.log.append(f"walk_to({tx:.3f},{ty:.3f})")
        return {"ok": True, "agent": {k: round(v, 4) for k, v in w.agent.items()}, "holding": w.holding}


@server.tool()
def pick(object_name: str) -> dict[str, Any]:
    """Pick up an object. Preconditions: hands empty AND agent within reach of the object.

    Effect: the agent is now holding the object.
    """
    with _LOCK:
        w = _WORLD
        if w.holding is not None:
            return {"ok": False, "error": f"already holding '{w.holding}'; place it first"}
        obj = w.objects.get(object_name)
        if obj is None:
            return {"ok": False, "error": f"no object named '{object_name}'"}
        if not obj["movable"]:
            return {"ok": False, "error": f"'{object_name}' is not movable"}
        d = _dist_agent_to(obj["x"], obj["y"])
        if d > REACH:
            return {"ok": False, "error": f"too far to pick '{object_name}' "
                                          f"(distance {d:.2f} m > reach {REACH} m); walk_to it first"}
        w.holding = object_name
        if object_name == w.target:
            w.target_was_picked = True
        w.steps += 1
        w.log.append(f"pick({object_name})")
        return {"ok": True, "holding": object_name}


@server.tool()
def place(x: float, y: float) -> dict[str, Any]:
    """Put the held object down at (x, y). Preconditions: holding something AND agent within reach of (x, y).

    Effect: the object rests at (x, y); hands are empty.
    """
    with _LOCK:
        w = _WORLD
        if w.holding is None:
            return {"ok": False, "error": "not holding anything"}
        tx, ty = _clamp(x), _clamp(y)
        d = _dist_agent_to(tx, ty)
        if d > REACH:
            return {"ok": False, "error": f"too far to place at ({tx:.2f},{ty:.2f}) "
                                          f"(distance {d:.2f} m > reach {REACH} m); walk_to it first"}
        name = w.holding
        w.objects[name]["x"], w.objects[name]["y"] = tx, ty
        w.holding = None
        w.steps += 1
        w.log.append(f"place({tx:.3f},{ty:.3f}) -> {name}")
        return {"ok": True, "placed": name, "at": {"x": round(tx, 4), "y": round(ty, 4)}}


@server.tool()
def render(width: int = 960, height: int = 640) -> MCPImage:
    """Render the room in 3D: the humanoid agent and the movable objects posed at their
    current positions (falls back to a top-down schematic if 3D render fails)."""
    with _LOCK:
        try:
            from sim.scene_render import get_renderer

            sr = get_renderer(_WORLD.scene_id, n_agents=1)
            agents = [(_WORLD.agent["x"], _WORLD.agent["y"], _WORLD.agent["yaw"])]
            objs = {n: (o["x"], o["y"], o["z"], 0.0) for n, o in _WORLD.objects.items()}
            png = sr.render(agents, objs, width=width, height=height)
        except Exception:
            png = _render_topdown(min(width, height), min(width, height))
    return MCPImage(data=png, format="png")


@server.tool()
def render_map(width: int = 600, height: int = 600) -> MCPImage:
    """Top-down schematic map (2D): goal region (green), objects, and the agent (blue)."""
    with _LOCK:
        png = _render_topdown(width, height)
    return MCPImage(data=png, format="png")


def _w2p(x: float, y: float, size: int) -> tuple[int, int]:
    px = (x + 2.5) / 5.0 * size
    py = size - (y + 2.5) / 5.0 * size  # north = up
    return int(px), int(py)


def _render_topdown(width: int, height: int) -> bytes:
    s = min(width, height)
    img = Image.new("RGB", (s, s), (235, 232, 226))
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, s - 3, s - 3], outline=(60, 60, 70), width=4)  # walls

    w = _WORLD
    if w.goal:
        gx, gy = _w2p(w.goal["x"], w.goal["y"], s)
        rr = int(w.goal["radius"] / 5.0 * s)
        d.ellipse([gx - rr, gy - rr, gx + rr, gy + rr], fill=(200, 235, 200), outline=(40, 150, 40), width=3)
        d.text((gx - 14, gy - 6), "GOAL", fill=(20, 110, 20))

    for name, o in w.objects.items():
        ox, oy = _w2p(o["x"], o["y"], s)
        is_target = name == w.target
        color = (150, 90, 40) if is_target else ((120, 120, 200) if o["movable"] else (170, 170, 170))
        r = 12 if is_target else 8
        d.ellipse([ox - r, oy - r, ox + r, oy + r], fill=color, outline=(30, 30, 30), width=2)
        if is_target:
            d.text((ox + r + 2, oy - 6), "table", fill=(90, 50, 20))

    ax, ay = _w2p(w.agent["x"], w.agent["y"], s)
    yaw = w.agent["yaw"]
    tip = (ax + 18 * math.cos(yaw), ay - 18 * math.sin(yaw))
    left = (ax + 11 * math.cos(yaw + 2.5), ay - 11 * math.sin(yaw + 2.5))
    right = (ax + 11 * math.cos(yaw - 2.5), ay - 11 * math.sin(yaw - 2.5))
    d.polygon([tip, left, right], fill=(40, 90, 220), outline=(10, 10, 60))
    if w.holding:
        d.text((ax - 10, ay + 14), f"holds {w.holding[:10]}", fill=(20, 20, 120))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":  # manual: serve standalone for poking with an MCP client
    import asyncio

    reset_world()
    port = int(os.environ.get("WORLDSIM_SKILL_PORT", "9100"))
    asyncio.run(server.run_async(transport="http", host="127.0.0.1", port=port, show_banner=False))
