"""Multi-agent symbolic skill world - a team of embodied agents in one shared room.

A physics-free planning world for benchmarking *multi-agent coordination*. One LLM
acts as the team controller, driving agent-scoped skills over MCP:

    walk_to(agent, x, y)                 move one agent (carries what it holds)
    pick(agent, object_name)             grab a LIGHT object (one agent)
    place(agent, x, y)                   set a singly-held object down
    joint_lift(agent_a, agent_b, obj)    lift a HEAVY object (needs two agents)
    joint_place(agent_a, agent_b, x, y)  set a jointly-held object down together
    get_world_state()                    observe all agents, objects, the goal, blockers
    render()                             top-down schematic of the room

What makes the task genuinely multi-agent (not two independent pick-and-places):
  * the side table is HEAVY - a single `pick` is refused; it needs `joint_lift` by
    two adjacent agents, who then move in lockstep.
  * the target corner is BLOCKED by a cushion - placing into the corner is refused
    until some agent clears the cushion out (an ordering dependency between agents).
So the only winning plan requires decomposition + role assignment + ordering +
a synchronized joint action. Skills are deterministic, so a *valid team plan*
always succeeds and an invalid one fails on a precondition.
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

REACH = 0.9           # planar distance (m) within which pick/place/lift are allowed
ROOM_HALF = 2.3       # agents/objects clamped to the interior (walls at +-2.5)
AGENT_IDS = ("alpha", "bravo", "carol")
AGENT_COLORS = {"alpha": (40, 90, 220), "bravo": (210, 70, 60), "carol": (40, 160, 80)}
_START = {"alpha": (-0.6, -0.5), "bravo": (0.6, -0.5), "carol": (0.0, 0.5)}

server = FastMCP(name="worldsim-team-skills")


@dataclass
class _World:
    scene_id: str = ""
    agents: dict[str, dict[str, Any]] = field(default_factory=dict)   # id -> {x,y,yaw,holding}
    objects: dict[str, dict[str, Any]] = field(default_factory=dict)  # name -> {x,y,z,movable,heavy}
    target: str = ""
    blocker: str = ""
    goal: dict[str, float] = field(default_factory=dict)
    target_initial_distance: float = 1.0
    target_was_lifted: bool = False
    steps: int = 0
    log: list[str] = field(default_factory=list)


_WORLD = _World()
_LOCK = threading.Lock()


def _clamp(v: float) -> float:
    return max(-ROOM_HALF, min(ROOM_HALF, v))


def reset_world(scene_id: str = "living-room-v1", task: str | None = None) -> dict[str, Any]:
    """(Re)load the multi-agent world from a scene's metadata.json. Returns a summary."""
    meta = json.loads((SCENES_DIR / scene_id / "metadata.json").read_text())

    objects: dict[str, dict[str, Any]] = {}
    for name, spec in meta.get("objects", {}).items():
        pos = spec.get("initial_position")
        if not pos:
            continue
        objects[name] = {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                         "movable": spec.get("type") == "free", "heavy": False}

    tasks = meta.get("tasks", {})
    task_name = task or next((k for k, v in tasks.items() if "goal_region" in v), None)
    tspec = tasks.get(task_name, {}) if task_name else {}
    target = tspec.get("target_object", "")
    gr = tspec.get("goal_region", {"center": [-2.0, -2.0], "radius": 0.6})
    goal = {"x": float(gr["center"][0]), "y": float(gr["center"][1]), "radius": float(gr["radius"])}

    if target in objects:
        objects[target]["heavy"] = True  # the side table needs two agents

    # A cushion starts INSIDE the goal corner -> must be cleared before the table lands.
    blocker = next((n for n in ("asset_0_0_fb_cushion_mid", "asset_0_0_fb_cushion_left",
                                "asset_0_0_fb_cushion_right") if n in objects), "")
    if blocker:
        objects[blocker]["x"] = goal["x"] + 0.12
        objects[blocker]["y"] = goal["y"] + 0.12

    start = objects.get(target, {"x": 0.0, "y": 0.0})
    agents = {a: {"x": _clamp(_START[a][0]), "y": _clamp(_START[a][1]), "yaw": math.pi / 2, "holding": None}
              for a in AGENT_IDS}

    global _WORLD
    with _LOCK:
        _WORLD = _World(
            scene_id=scene_id, agents=agents, objects=objects, target=target, blocker=blocker, goal=goal,
            target_initial_distance=max(1e-3, math.hypot(start["x"] - goal["x"], start["y"] - goal["y"])),
            target_was_lifted=False, steps=0,
            log=[f"reset {scene_id}: target={target!r} heavy, blocker={blocker!r} in corner, goal={goal}"],
        )
    return _summary()


def _carriers(obj_name: str) -> list[str]:
    return [a for a, s in _WORLD.agents.items() if s["holding"] == obj_name]


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _objects_in_goal(exclude: str | None = None) -> list[str]:
    w = _WORLD
    return [n for n, o in w.objects.items()
            if n != exclude and _dist(o["x"], o["y"], w.goal["x"], w.goal["y"]) <= w.goal["radius"]]


def _summary() -> dict[str, Any]:
    w = _WORLD
    tx = w.objects.get(w.target, {}).get("x", 0.0)
    ty = w.objects.get(w.target, {}).get("y", 0.0)
    tdist = _dist(tx, ty, w.goal["x"], w.goal["y"]) if w.goal else None
    blockers = _objects_in_goal(exclude=w.target)
    return {
        "scene_id": w.scene_id,
        "agents": {a: {"x": round(s["x"], 3), "y": round(s["y"], 3), "holding": s["holding"]}
                   for a, s in w.agents.items()},
        "objects": {n: {"x": round(o["x"], 3), "y": round(o["y"], 3), "movable": o["movable"],
                        "weight": "heavy" if o["heavy"] else "light",
                        "carried_by": _carriers(n)} for n, o in w.objects.items()},
        "target_object": w.target,
        "goal": w.goal,
        "target_distance_to_goal": round(tdist, 3) if tdist is not None else None,
        "target_in_goal": (tdist is not None and tdist <= w.goal["radius"]),
        "blockers_in_goal": blockers,
        "corner_cleared": len(blockers) == 0,
        "reach_radius": REACH,
        "steps": w.steps,
    }


def world_snapshot() -> dict[str, Any]:
    with _LOCK:
        snap = _summary()
        snap["target_initial_distance"] = round(_WORLD.target_initial_distance, 3)
        snap["target_was_lifted"] = _WORLD.target_was_lifted
        return snap


def _move_object_with(name: str, x: float, y: float) -> None:
    _WORLD.objects[name]["x"], _WORLD.objects[name]["y"] = x, y


# ── skills (MCP tools) ────────────────────────────────────────────────────────


@server.tool()
def reset(scene_id: str = "living-room-v1") -> dict[str, Any]:
    """Reset the multi-agent world for a scene. Returns agents, objects, goal, and blockers."""
    return reset_world(scene_id)


@server.tool()
def get_world_state() -> dict[str, Any]:
    """Observe everything: each agent's pose + what it holds, every object (with weight and who
    carries it), the goal region, and which objects still block the corner. Call first and often."""
    with _LOCK:
        return _summary()


@server.tool()
def walk_to(agent: str, x: float, y: float) -> dict[str, Any]:
    """Walk one agent to (x, y). Carries any held object. If the agent co-carries a heavy object,
    its partner and the object move together (they stay in lockstep)."""
    with _LOCK:
        w = _WORLD
        if agent not in w.agents:
            return {"ok": False, "error": f"no agent '{agent}'; agents are {list(w.agents)}"}
        tx, ty = _clamp(x), _clamp(y)
        held = w.agents[agent]["holding"]
        movers = [agent]
        if held is not None:
            crs = _carriers(held)
            movers = crs  # move all carriers of the held object together
            _move_object_with(held, tx, ty)
        for m in movers:
            w.agents[m]["yaw"] = math.atan2(ty - w.agents[m]["y"], tx - w.agents[m]["x"])
            w.agents[m]["x"], w.agents[m]["y"] = tx, ty
        w.steps += 1
        w.log.append(f"walk_to({agent},{tx:.2f},{ty:.2f})")
        return {"ok": True, "moved": movers, "at": {"x": round(tx, 3), "y": round(ty, 3)}, "carrying": held}


@server.tool()
def pick(agent: str, object_name: str) -> dict[str, Any]:
    """One agent picks up a LIGHT object. Pre: agent free + within reach. Heavy objects need joint_lift."""
    with _LOCK:
        w = _WORLD
        if agent not in w.agents:
            return {"ok": False, "error": f"no agent '{agent}'"}
        if w.agents[agent]["holding"] is not None:
            return {"ok": False, "error": f"'{agent}' already holds '{w.agents[agent]['holding']}'"}
        obj = w.objects.get(object_name)
        if obj is None:
            return {"ok": False, "error": f"no object '{object_name}'"}
        if not obj["movable"]:
            return {"ok": False, "error": f"'{object_name}' is not movable"}
        if obj["heavy"]:
            return {"ok": False, "error": f"'{object_name}' is too heavy for one agent; "
                                          f"use joint_lift(agent_a, agent_b, '{object_name}')"}
        if _carriers(object_name):
            return {"ok": False, "error": f"'{object_name}' is already held"}
        if _dist(w.agents[agent]["x"], w.agents[agent]["y"], obj["x"], obj["y"]) > REACH:
            return {"ok": False, "error": f"'{agent}' is too far from '{object_name}'; walk_to it first"}
        w.agents[agent]["holding"] = object_name
        w.steps += 1
        w.log.append(f"pick({agent},{object_name})")
        return {"ok": True, "holding": {agent: object_name}}


@server.tool()
def place(agent: str, x: float, y: float) -> dict[str, Any]:
    """One agent puts its singly-held object down at (x, y). Pre: holding + within reach. Placing
    into the goal corner is refused while another object still blocks it."""
    with _LOCK:
        w = _WORLD
        if agent not in w.agents or w.agents[agent]["holding"] is None:
            return {"ok": False, "error": f"'{agent}' is not holding anything"}
        name = w.agents[agent]["holding"]
        if len(_carriers(name)) > 1:
            return {"ok": False, "error": f"'{name}' is jointly carried; use joint_place"}
        tx, ty = _clamp(x), _clamp(y)
        if _dist(w.agents[agent]["x"], w.agents[agent]["y"], tx, ty) > REACH:
            return {"ok": False, "error": f"'{agent}' is too far from ({tx:.2f},{ty:.2f}); walk_to it first"}
        blockers = _objects_in_goal(exclude=name)
        if _dist(tx, ty, w.goal["x"], w.goal["y"]) <= w.goal["radius"] and blockers:
            return {"ok": False, "error": f"corner is blocked by {blockers}; clear it before placing there"}
        _move_object_with(name, tx, ty)
        w.agents[agent]["holding"] = None
        w.steps += 1
        w.log.append(f"place({agent},{tx:.2f},{ty:.2f})->{name}")
        return {"ok": True, "placed": name, "at": {"x": round(tx, 3), "y": round(ty, 3)}}


@server.tool()
def joint_lift(agent_a: str, agent_b: str, object_name: str) -> dict[str, Any]:
    """Two agents lift a HEAVY object together. Pre: distinct free agents, BOTH within reach of it."""
    with _LOCK:
        w = _WORLD
        if agent_a == agent_b:
            return {"ok": False, "error": "joint_lift needs two different agents"}
        for a in (agent_a, agent_b):
            if a not in w.agents:
                return {"ok": False, "error": f"no agent '{a}'"}
            if w.agents[a]["holding"] is not None:
                return {"ok": False, "error": f"'{a}' already holds '{w.agents[a]['holding']}'"}
        obj = w.objects.get(object_name)
        if obj is None or not obj["movable"]:
            return {"ok": False, "error": f"'{object_name}' is not a movable object"}
        if _carriers(object_name):
            return {"ok": False, "error": f"'{object_name}' is already held"}
        for a in (agent_a, agent_b):
            if _dist(w.agents[a]["x"], w.agents[a]["y"], obj["x"], obj["y"]) > REACH:
                return {"ok": False, "error": f"'{a}' is too far from '{object_name}'; walk_to it first"}
        w.agents[agent_a]["holding"] = object_name
        w.agents[agent_b]["holding"] = object_name
        w.target_was_lifted = w.target_was_lifted or (object_name == w.target)
        w.steps += 1
        w.log.append(f"joint_lift({agent_a},{agent_b},{object_name})")
        return {"ok": True, "carried_by": [agent_a, agent_b]}


@server.tool()
def joint_place(agent_a: str, agent_b: str, x: float, y: float) -> dict[str, Any]:
    """Two agents set their jointly-held object down at (x, y). Pre: both carry the same object and
    BOTH are within reach of (x, y). Refused if another object still blocks the goal corner."""
    with _LOCK:
        w = _WORLD
        for a in (agent_a, agent_b):
            if a not in w.agents:
                return {"ok": False, "error": f"no agent '{a}'"}
        ha, hb = w.agents[agent_a]["holding"], w.agents[agent_b]["holding"]
        if ha is None or ha != hb:
            return {"ok": False, "error": f"'{agent_a}' and '{agent_b}' are not jointly holding one object"}
        name = ha
        tx, ty = _clamp(x), _clamp(y)
        for a in (agent_a, agent_b):
            if _dist(w.agents[a]["x"], w.agents[a]["y"], tx, ty) > REACH:
                return {"ok": False, "error": f"'{a}' is too far from ({tx:.2f},{ty:.2f}); walk both there first"}
        blockers = _objects_in_goal(exclude=name)
        if _dist(tx, ty, w.goal["x"], w.goal["y"]) <= w.goal["radius"] and blockers:
            return {"ok": False, "error": f"corner is blocked by {blockers}; clear it before placing there"}
        _move_object_with(name, tx, ty)
        w.agents[agent_a]["holding"] = None
        w.agents[agent_b]["holding"] = None
        w.steps += 1
        w.log.append(f"joint_place({agent_a},{agent_b},{tx:.2f},{ty:.2f})->{name}")
        return {"ok": True, "placed": name, "at": {"x": round(tx, 3), "y": round(ty, 3)}}


@server.tool()
def render(width: int = 600, height: int = 600) -> MCPImage:
    """Top-down schematic: goal (green), target table (brown), blocker (red ring), agents (colored)."""
    with _LOCK:
        png = _render_topdown(width, height)
    return MCPImage(data=png, format="png")


def _w2p(x: float, y: float, s: int) -> tuple[int, int]:
    return int((x + 2.5) / 5.0 * s), int(s - (y + 2.5) / 5.0 * s)


def _render_topdown(width: int, height: int) -> bytes:
    s = min(width, height)
    img = Image.new("RGB", (s, s), (235, 232, 226))
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, s - 3, s - 3], outline=(60, 60, 70), width=4)
    w = _WORLD
    if w.goal:
        gx, gy = _w2p(w.goal["x"], w.goal["y"], s)
        rr = int(w.goal["radius"] / 5.0 * s)
        d.ellipse([gx - rr, gy - rr, gx + rr, gy + rr], fill=(200, 235, 200), outline=(40, 150, 40), width=3)
        d.text((gx - 14, gy - 6), "GOAL", fill=(20, 110, 20))
    for name, o in w.objects.items():
        ox, oy = _w2p(o["x"], o["y"], s)
        if name == w.target:
            color, r = (150, 90, 40), 13
        elif name == w.blocker:
            color, r = (200, 80, 80), 9
        else:
            color, r = ((120, 120, 200) if o["movable"] else (170, 170, 170)), 7
        d.ellipse([ox - r, oy - r, ox + r, oy + r], fill=color, outline=(30, 30, 30), width=2)
        if name == w.target:
            d.text((ox + r + 2, oy - 6), "table(heavy)", fill=(90, 50, 20))
        elif name == w.blocker:
            d.text((ox + r + 2, oy - 6), "blocker", fill=(140, 30, 30))
    for a, st in w.agents.items():
        ax, ay = _w2p(st["x"], st["y"], s)
        col = AGENT_COLORS.get(a, (40, 90, 220))
        d.ellipse([ax - 9, ay - 9, ax + 9, ay + 9], fill=col, outline=(10, 10, 30), width=2)
        d.text((ax - 8, ay + 10), a, fill=col)
        if st["holding"]:
            d.text((ax - 8, ay - 22), "holds", fill=col)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    import asyncio

    reset_world()
    port = int(os.environ.get("WORLDSIM_SKILL_PORT", "9101"))
    asyncio.run(server.run_async(transport="http", host="127.0.0.1", port=port, show_banner=False))
