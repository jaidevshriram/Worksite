"""Cooperative multi-agent skill world - the planning layer for >1 embodied agent.

Several agents share one symbolic room and a tool API parameterized by `agent`.
The key coordination hook: every object has a `carriers_required` count, and a
HEAVY object (2+) cannot be moved until that many agents have `grab`bed it - so a
single-agent plan provably cannot finish a task that contains a heavy object. That
makes coordination *necessary*, not just nice-to-have, while light objects let a
good planner divide labor in parallel.

Skills (all take `agent`):
    walk_to(agent, x, y)        move an agent (must have free hands)
    grab(agent, object_name)    take hold of an object   (pre: near it, hands free)
    carry_to(agent, x, y)       move the held object + all its carriers together
                                (pre: enough agents are grabbing it = carriers_required)
    release(agent)              let go
    say(agent, text)            post to a shared message log (cheap coordination)
    get_world_state()           observe agents, objects (+carriers), goals, messages
    render()                    top-down schematic

Physics-free: skills deterministically transform world state. Served as an `mcp`
capability; the env-side grader reads the same state in-process via world_snapshot().
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

from sim.agent_spacing import MIN_AGENT_SEP, resolve_walk, spread_agents, spread_all

REACH = 0.8           # planar distance (m) within which grab/carry are allowed
ROOM_HALF = 2.3
AGENT_COLORS = {"A": (40, 90, 220), "B": (230, 130, 30), "C": (40, 170, 90), "D": (170, 60, 200)}

server = FastMCP(name="worldsim-coop")


@dataclass
class _World:
    scene_id: str = ""
    agents: dict[str, dict[str, float]] = field(default_factory=dict)   # id -> {x,y,yaw}
    objects: dict[str, dict[str, Any]] = field(default_factory=dict)    # name -> {x,y,z,movable,carriers_required,grippers:set,max_grippers}
    targets: dict[str, dict[str, float]] = field(default_factory=dict)  # name -> {x,y,radius}
    initial_distance: dict[str, float] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    steps: int = 0


_WORLD = _World()
_LOCK = threading.Lock()


def _clamp(v: float) -> float:
    return max(-ROOM_HALF, min(ROOM_HALF, v))


def reset_world(scene_id: str = "living-room-v1",
                placements: list[dict[str, Any]] | None = None,
                agents: dict[str, list[float]] | None = None) -> dict[str, Any]:
    """(Re)load the cooperative world from scene metadata + a task spec.

    placements: [{object, goal:[x,y], radius, carriers}], agents: {id:[x,y]}.
    """
    meta = json.loads((SCENES_DIR / scene_id / "metadata.json").read_text())
    placements = placements or []
    agents = agents or {"A": [-0.6, -0.6], "B": [0.6, -0.6]}

    objects: dict[str, dict[str, Any]] = {}
    for name, spec in meta.get("objects", {}).items():
        pos = spec.get("initial_position")
        if not pos:
            continue
        objects[name] = {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                         "movable": spec.get("type") == "free",
                         "carriers_required": 1, "grippers": set(), "max_grippers": 0}

    targets: dict[str, dict[str, float]] = {}
    initial: dict[str, float] = {}
    for p in placements:
        name = p["object"]
        if name not in objects:
            continue
        objects[name]["carriers_required"] = int(p.get("carriers", 1))
        gx, gy = float(p["goal"][0]), float(p["goal"][1])
        targets[name] = {"x": gx, "y": gy, "radius": float(p.get("radius", 0.6))}
        initial[name] = max(1e-3, math.hypot(objects[name]["x"] - gx, objects[name]["y"] - gy))

    global _WORLD
    with _LOCK:
        _WORLD = _World(
            scene_id=scene_id,
            agents={aid: {"x": _clamp(xy[0]), "y": _clamp(xy[1]), "yaw": math.pi / 2}
                    for aid, xy in agents.items()},
            objects=objects,
            targets=targets,
            initial_distance=initial,
            messages=[],
            steps=0,
        )
    return _summary()


def _agent_holding(aid: str) -> str | None:
    for name, o in _WORLD.objects.items():
        if aid in o["grippers"]:
            return name
    return None


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _summary() -> dict[str, Any]:
    w = _WORLD
    objs = {}
    for n, o in w.objects.items():
        entry = {"x": round(o["x"], 4), "y": round(o["y"], 4), "movable": o["movable"],
                 "carriers_required": o["carriers_required"], "grabbed_by": sorted(o["grippers"])}
        if n in w.targets:
            t = w.targets[n]
            d = _dist(o["x"], o["y"], t["x"], t["y"])
            entry["goal"] = {"x": t["x"], "y": t["y"], "radius": t["radius"]}
            entry["distance_to_goal"] = round(d, 4)
            entry["in_goal"] = d <= t["radius"]
        objs[n] = entry
    return {
        "scene_id": w.scene_id,
        "agents": {aid: {"x": round(a["x"], 4), "y": round(a["y"], 4),
                         "holding": _agent_holding(aid)} for aid, a in w.agents.items()},
        "objects": objs,
        "targets": list(w.targets.keys()),
        "reach_radius": REACH,
        "messages": list(w.messages),
        "steps": w.steps,
    }


def world_snapshot() -> dict[str, Any]:
    with _LOCK:
        snap = _summary()
        snap["initial_distance"] = {k: round(v, 4) for k, v in _WORLD.initial_distance.items()}
        snap["coordinated"] = {n: (o["max_grippers"] >= o["carriers_required"] and o["carriers_required"] >= 2)
                               for n, o in _WORLD.objects.items() if n in _WORLD.targets}
        return snap


def _require_agent(aid: str) -> dict[str, float] | None:
    return _WORLD.agents.get(aid)


# ── skills (MCP tools) ────────────────────────────────────────────────────────


@server.tool()
def get_world_state() -> dict[str, Any]:
    """Observe everything: each agent's pose + what it holds, every object (with
    carriers_required and who is grabbing it), the goals, and the shared message log."""
    with _LOCK:
        return _summary()


@server.tool()
def walk_to(agent: str, x: float, y: float) -> dict[str, Any]:
    """Walk `agent` to (x, y). Fails if the agent is currently holding an object
    (release it or use carry_to instead)."""
    with _LOCK:
        a = _require_agent(agent)
        if a is None:
            return {"ok": False, "error": f"no agent '{agent}'"}
        held = _agent_holding(agent)
        if held is not None:
            return {"ok": False, "error": f"agent '{agent}' is holding '{held}'; "
                                          f"use carry_to or release first"}
        tx, ty = _clamp(x), _clamp(y)
        tx, ty = resolve_walk(agent, tx, ty, _WORLD.agents)
        tx, ty = _clamp(tx), _clamp(ty)
        a["yaw"] = math.atan2(ty - a["y"], tx - a["x"])
        a["x"], a["y"] = tx, ty
        spread_all(_WORLD.agents)
        _WORLD.steps += 1
        return {"ok": True, "agent": agent, "x": round(tx, 4), "y": round(ty, 4)}


@server.tool()
def grab(agent: str, object_name: str) -> dict[str, Any]:
    """`agent` grabs an object. Pre: hands free AND within reach of the object.

    For a HEAVY object (carriers_required > 1) every required agent must grab it
    before anyone can carry_to."""
    with _LOCK:
        a = _require_agent(agent)
        if a is None:
            return {"ok": False, "error": f"no agent '{agent}'"}
        if _agent_holding(agent) is not None:
            return {"ok": False, "error": f"agent '{agent}' already holding something"}
        o = _WORLD.objects.get(object_name)
        if o is None:
            return {"ok": False, "error": f"no object '{object_name}'"}
        if not o["movable"]:
            return {"ok": False, "error": f"'{object_name}' is not movable"}
        d = _dist(a["x"], a["y"], o["x"], o["y"])
        if d > REACH:
            return {"ok": False, "error": f"too far to grab '{object_name}' "
                                          f"({d:.2f} m > reach {REACH} m); walk_to it first"}
        o["grippers"].add(agent)
        o["max_grippers"] = max(o["max_grippers"], len(o["grippers"]))
        if len(o["grippers"]) > 1:
            spread_agents(_WORLD.agents, list(o["grippers"]), o["x"], o["y"], yaw=a["yaw"])
        else:
            a["x"], a["y"] = o["x"], o["y"]
        _WORLD.steps += 1
        need = o["carriers_required"]
        have = len(o["grippers"])
        return {"ok": True, "agent": agent, "object": object_name,
                "carriers": f"{have}/{need}",
                "liftable": have >= need,
                "note": (f"need {need - have} more agent(s) grabbing before carry_to"
                         if have < need else "ready to carry")}


@server.tool()
def carry_to(agent: str, x: float, y: float) -> dict[str, Any]:
    """Carry the object `agent` is holding to (x, y). Pre: enough agents are grabbing
    it (>= carriers_required). All grabbing agents move there together with the object."""
    with _LOCK:
        a = _require_agent(agent)
        if a is None:
            return {"ok": False, "error": f"no agent '{agent}'"}
        name = _agent_holding(agent)
        if name is None:
            return {"ok": False, "error": f"agent '{agent}' is not holding anything; grab first"}
        o = _WORLD.objects[name]
        need, have = o["carriers_required"], len(o["grippers"])
        if have < need:
            return {"ok": False, "error": f"'{name}' needs {need} carriers but only {have} "
                                          f"grabbing ({sorted(o['grippers'])}); have the other "
                                          f"agent walk_to it and grab before carrying"}
        tx, ty = _clamp(x), _clamp(y)
        o["x"], o["y"] = tx, ty
        spread_agents(_WORLD.agents, list(o["grippers"]), tx, ty, yaw=a["yaw"])
        _WORLD.steps += 1
        return {"ok": True, "object": name, "carried_by": sorted(o["grippers"]),
                "at": {"x": round(tx, 4), "y": round(ty, 4)}}


@server.tool()
def release(agent: str) -> dict[str, Any]:
    """`agent` lets go of whatever it is holding (the object stays where it is)."""
    with _LOCK:
        name = _agent_holding(agent)
        if name is None:
            return {"ok": False, "error": f"agent '{agent}' is not holding anything"}
        _WORLD.objects[name]["grippers"].discard(agent)
        _WORLD.steps += 1
        return {"ok": True, "agent": agent, "released": name}


@server.tool()
def say(agent: str, text: str) -> dict[str, Any]:
    """Post a short message to the shared log so agents can coordinate."""
    with _LOCK:
        _WORLD.messages.append(f"{agent}: {text}")
        return {"ok": True, "messages": list(_WORLD.messages)}


@server.tool()
def render(width: int = 960, height: int = 640) -> MCPImage:
    """Render the room in 3D: the two humanoid agents and the movable objects posed at
    their current positions (falls back to a top-down schematic if 3D render fails)."""
    with _LOCK:
        try:
            from sim.scene_render import get_renderer

            sr = get_renderer(_WORLD.scene_id, n_agents=len(_WORLD.agents))
            agents = [(a["x"], a["y"], a["yaw"]) for _, a in sorted(_WORLD.agents.items())]
            objs = {n: (o["x"], o["y"], o["z"], 0.0) for n, o in _WORLD.objects.items()}
            png = sr.render(agents, objs, width=width, height=height)
        except Exception:
            png = _render_topdown(min(width, height), min(width, height))
    return MCPImage(data=png, format="png")


@server.tool()
def render_map(width: int = 600, height: int = 600) -> MCPImage:
    """Top-down schematic map (2D): per-object goals, objects (heavy = red ring), agents."""
    with _LOCK:
        png = _render_topdown(width, height)
    return MCPImage(data=png, format="png")


def _w2p(x: float, y: float, size: int) -> tuple[int, int]:
    return int((x + 2.5) / 5.0 * size), int(size - (y + 2.5) / 5.0 * size)


def _render_topdown(width: int, height: int) -> bytes:
    s = min(width, height)
    img = Image.new("RGB", (s, s), (235, 232, 226))
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, s - 3, s - 3], outline=(60, 60, 70), width=4)
    w = _WORLD

    for name, t in w.targets.items():
        gx, gy = _w2p(t["x"], t["y"], s)
        rr = int(t["radius"] / 5.0 * s)
        d.ellipse([gx - rr, gy - rr, gx + rr, gy + rr], outline=(40, 150, 40), width=3)

    for name, o in w.objects.items():
        ox, oy = _w2p(o["x"], o["y"], s)
        heavy = o["carriers_required"] >= 2
        is_target = name in w.targets
        color = (150, 90, 40) if is_target else (170, 170, 170)
        r = 13 if is_target else 7
        d.ellipse([ox - r, oy - r, ox + r, oy + r], fill=color,
                  outline=(200, 30, 30) if heavy else (30, 30, 30), width=4 if heavy else 2)
        if is_target:
            tag = f"{name[:8]}{' x2' if heavy else ''}"
            d.text((ox + r + 2, oy - 6), tag, fill=(60, 40, 20))

    for aid, a in w.agents.items():
        ax, ay = _w2p(a["x"], a["y"], s)
        col = AGENT_COLORS.get(aid, (60, 60, 60))
        yaw = a["yaw"]
        tip = (ax + 18 * math.cos(yaw), ay - 18 * math.sin(yaw))
        l = (ax + 11 * math.cos(yaw + 2.5), ay - 11 * math.sin(yaw + 2.5))
        rr = (ax + 11 * math.cos(yaw - 2.5), ay - 11 * math.sin(yaw - 2.5))
        d.polygon([tip, l, rr], fill=col, outline=(10, 10, 30))
        d.text((ax - 3, ay - 4), aid, fill=(255, 255, 255))
        held = _agent_holding(aid)
        if held:
            hx, hy = _w2p(w.objects[held]["x"], w.objects[held]["y"], s)
            d.line([ax, ay, hx, hy], fill=col, width=2)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
