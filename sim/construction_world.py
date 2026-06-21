"""Multi-agent construction site planning world.

Mixed fleet
-----------
G1 humanoids (A, B, C, D) — can grab blocks and carry them.
Spot quadrupeds (E, F)     — can walk and register patrol checkpoints only.

Skills (MCP tools)
------------------
    walk_to(agent, x, y)          move any agent
    grab(agent, block_name)       G1 only: pick up a block (must be within reach)
    carry_to(agent, x, y)         G1 holding a block: transport it there
    release(agent)                 G1 puts block down at current position
    checkpoint(agent, cp_name)    Spot only: register arrival at a named checkpoint
    say(agent, text)              post to shared message log
    get_world_state()             observe agents, blocks, goals, metrics, messages
    render()                      3D snapshot of the site

Fleet Metrics (in every get_world_state response)
--------------------------------------------------
    total_steps           – tool calls made so far
    fleet_utilization     – fraction of (agent × step) slots spent actively working
    peak_simultaneous     – max agents that took an action in the same step
    agent_uptime          – per-agent action count / total_steps
    collision_events      – times any two agents ended within MIN_SEP of each other

Coordination is rewarded: a plan that dispatches G1s and Spots in parallel will
show higher fleet_utilization and peak_simultaneous than a strictly sequential one.
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
from PIL import Image, ImageDraw, ImageFont

SCENES_DIR = Path(os.environ.get("SCENES_DIR", Path(__file__).resolve().parents[1] / "scenes"))

# ── constants ──────────────────────────────────────────────────────────────────
REACH = 2.0           # metres — grab / checkpoint activation radius
# Must be strictly below agent_spacing.MIN_AGENT_SEP (1.0 m) so that agents
# properly nudged apart by resolve_walk / spread_agents don't count as collisions.
MIN_SEP = 0.8         # metres — true collision threshold
SITE_BOUNDS = (-9.0, 12.0, -9.0, 10.0)   # xmin, xmax, ymin, ymax

AGENT_ROLES: dict[str, str] = {}   # filled by reset_world; id -> "g1" | "spot"

server = FastMCP(name="worldsim-construction")


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class _Metrics:
    total_steps: int = 0
    # how many steps each agent was the caller of an action
    agent_active: dict[str, int] = field(default_factory=dict)
    peak_simultaneous: int = 0
    # agents who acted THIS step (reset each step)
    _this_step: set[str] = field(default_factory=set)
    collision_events: int = 0   # cumulative pair-collision checks


@dataclass
class _World:
    scene_id: str = ""
    agents: dict[str, dict[str, float]] = field(default_factory=dict)
    # block_name -> {x, y, z, movable, carriers_required, grippers, max_grippers}
    blocks: dict[str, dict[str, Any]] = field(default_factory=dict)
    # block_name -> {x, y, radius}
    targets: dict[str, dict[str, float]] = field(default_factory=dict)
    # checkpoint_name -> {x, y, radius, visited_by: set[str]}
    checkpoints: dict[str, dict[str, Any]] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    metrics: _Metrics = field(default_factory=_Metrics)


_WORLD = _World()
_LOCK = threading.Lock()


# ── helpers ───────────────────────────────────────────────────────────────────

def _dist(ax, ay, bx, by) -> float:
    return math.hypot(ax - bx, ay - by)


def _clamp_site(x: float, y: float) -> tuple[float, float]:
    xmin, xmax, ymin, ymax = SITE_BOUNDS
    return max(xmin, min(xmax, x)), max(ymin, min(ymax, y))


def _holding(agent_id: str) -> str | None:
    for name, b in _WORLD.blocks.items():
        if agent_id in b["grippers"]:
            return name
    return None


def _co_carriers() -> set[frozenset[str]]:
    """Return pairs of agents that are co-carrying the same block (intentionally close)."""
    pairs: set[frozenset[str]] = set()
    for b in _WORLD.blocks.values():
        grippers = list(b["grippers"])
        for i in range(len(grippers)):
            for j in range(i + 1, len(grippers)):
                pairs.add(frozenset([grippers[i], grippers[j]]))
    return pairs


def _check_collisions() -> int:
    """Return number of new collision events (pairs within MIN_SEP).
    Co-carriers of the same block are excluded since they are intentionally close."""
    skip = _co_carriers()
    agents = list(_WORLD.agents.items())
    count = 0
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            aid_a, a = agents[i]
            aid_b, b = agents[j]
            if frozenset([aid_a, aid_b]) in skip:
                continue
            if _dist(a["x"], a["y"], b["x"], b["y"]) < MIN_SEP:
                count += 1
    return count


def _advance_step(caller: str) -> None:
    """Call ONCE per tool invocation to update metrics."""
    m = _WORLD.metrics
    m._this_step.add(caller)
    # flush at the start of the NEXT call or on snapshot
    # (we flush immediately so each call is its own step)
    m.agent_active[caller] = m.agent_active.get(caller, 0) + 1
    n_active = len(m._this_step)
    m.peak_simultaneous = max(m.peak_simultaneous, n_active)
    m._this_step.clear()
    m.collision_events += _check_collisions()
    m.total_steps += 1


def _fleet_utilization() -> float:
    m = _WORLD.metrics
    n = len(_WORLD.agents)
    if n == 0 or m.total_steps == 0:
        return 0.0
    return sum(m.agent_active.values()) / (n * m.total_steps)


def _summary() -> dict[str, Any]:
    w = _WORLD
    m = w.metrics
    n = len(w.agents)

    agents_out = {}
    for aid, a in w.agents.items():
        role = AGENT_ROLES.get(aid, "unknown")
        agents_out[aid] = {
            "role": role,
            "x": round(a["x"], 3),
            "y": round(a["y"], 3),
            "yaw": round(a.get("yaw", 0), 3),
            "holding": _holding(aid),
        }

    blocks_out = {}
    for name, b in w.blocks.items():
        entry = {
            "x": round(b["x"], 3),
            "y": round(b["y"], 3),
            "carriers_required": b["carriers_required"],
            "grabbed_by": sorted(b["grippers"]),
        }
        if name in w.targets:
            t = w.targets[name]
            d = _dist(b["x"], b["y"], t["x"], t["y"])
            entry["goal"] = {"x": t["x"], "y": t["y"], "radius": t["radius"]}
            entry["distance_to_goal"] = round(d, 3)
            entry["at_goal"] = d <= t["radius"]
        blocks_out[name] = entry

    cp_out = {}
    for name, cp in w.checkpoints.items():
        cp_out[name] = {
            "x": cp["x"], "y": cp["y"],
            "visited_by": sorted(cp["visited_by"]),
        }

    uptime = {aid: round(m.agent_active.get(aid, 0) / max(m.total_steps, 1), 3)
              for aid in w.agents}

    return {
        "scene_id": w.scene_id,
        "agents": agents_out,
        "blocks": blocks_out,
        "checkpoints": cp_out,
        "messages": list(w.messages),
        "metrics": {
            "total_steps": m.total_steps,
            "fleet_utilization": round(_fleet_utilization(), 3),
            "peak_simultaneous_active": m.peak_simultaneous,
            "agent_uptime": uptime,
            "collision_events": m.collision_events,
            "note": (
                "fleet_utilization = fraction of (agent×step) slots spent acting. "
                "Peak=6 means all agents acted in the same step (maximum parallelism)."
            ),
        },
        "reach_radius": REACH,
        "min_separation": MIN_SEP,
    }


# ── public API (used by the grader) ──────────────────────────────────────────

def world_snapshot() -> dict[str, Any]:
    with _LOCK:
        snap = _summary()
        snap["_internal"] = {
            "initial_distance": {
                name: round(v, 3)
                for name, v in _WORLD._initial_dist.items()
            } if hasattr(_WORLD, "_initial_dist") else {},
        }
        return snap


def reset_world(
    scene_id: str = "construction-v1",
    agents: dict[str, str] | None = None,       # id -> "g1" | "spot"
    agent_positions: dict[str, list[float]] | None = None,   # id -> [x, y]
    blocks: list[dict[str, Any]] | None = None,              # list of block specs
    targets: list[dict[str, Any]] | None = None,             # list of target specs
    checkpoints: list[dict[str, Any]] | None = None,         # list of cp specs
) -> dict[str, Any]:
    """(Re)initialize the world.

    agents: {"A": "g1", "B": "g1", "E": "spot", ...}
    agent_positions: {"A": [x, y], ...}
    blocks: [{"name": "block_1", "x": 10, "y": -5, "carriers": 1}, ...]
    targets: [{"block": "block_1", "x": 0, "y": 7, "radius": 1.2}, ...]
    checkpoints: [{"name": "excavator", "x": -10, "y": -0.12}, ...]
    """
    global AGENT_ROLES, _WORLD

    if agents is None:
        agents = {"A": "g1", "B": "g1", "C": "g1", "D": "g1", "E": "spot", "F": "spot"}
    if agent_positions is None:
        agent_positions = {
            "A": [-1.4, 2.2], "B": [2.8, 2.1],
            "C": [-1.7, -2.3], "D": [2.7, -2.5],
            "E": [-6.0, 4.0],  "F": [7.0, -4.0],
        }
    blocks = blocks or []
    targets = targets or []
    checkpoints = checkpoints or []

    AGENT_ROLES = dict(agents)

    w_agents = {}
    for aid, role in agents.items():
        pos = agent_positions.get(aid, [0.0, 0.0])
        w_agents[aid] = {"x": float(pos[0]), "y": float(pos[1]), "yaw": 0.0}

    w_blocks: dict[str, dict[str, Any]] = {}
    for b in blocks:
        w_blocks[b["name"]] = {
            "x": float(b["x"]), "y": float(b["y"]), "z": float(b.get("z", 0.0)),
            "movable": True,
            "carriers_required": int(b.get("carriers", 1)),
            "grippers": set(),
            "max_grippers": 0,
        }

    w_targets: dict[str, dict[str, float]] = {}
    init_dist: dict[str, float] = {}
    for t in targets:
        bname = t["block"]
        if bname not in w_blocks:
            continue
        gx, gy = float(t["x"]), float(t["y"])
        w_targets[bname] = {"x": gx, "y": gy, "radius": float(t.get("radius", 1.2))}
        bpos = w_blocks[bname]
        init_dist[bname] = max(0.01, _dist(bpos["x"], bpos["y"], gx, gy))

    w_checkpoints: dict[str, dict[str, Any]] = {}
    for cp in checkpoints:
        w_checkpoints[cp["name"]] = {
            "x": float(cp["x"]), "y": float(cp["y"]),
            "radius": float(cp.get("radius", REACH)),
            "visited_by": set(),
        }

    metrics = _Metrics(agent_active={aid: 0 for aid in agents})

    with _LOCK:
        _WORLD = _World(
            scene_id=scene_id,
            agents=w_agents,
            blocks=w_blocks,
            targets=w_targets,
            checkpoints=w_checkpoints,
            messages=[],
            metrics=metrics,
        )
        _WORLD._initial_dist = init_dist  # type: ignore[attr-defined]

    return _summary()


# ── MCP tools ─────────────────────────────────────────────────────────────────

@server.tool()
def get_world_state() -> dict[str, Any]:
    """Return the full world state: agents (role, position, holding), blocks (position,
    goal, distance), checkpoints (visited_by), fleet metrics, and messages.
    Call this first to understand the scene."""
    with _LOCK:
        return _summary()


@server.tool()
def walk_to(agent: str, x: float, y: float) -> dict[str, Any]:
    """Walk `agent` to world position (x, y).
    Fails if the agent is currently carrying a block — use carry_to instead."""
    with _LOCK:
        a = _WORLD.agents.get(agent)
        if a is None:
            return {"ok": False, "error": f"unknown agent '{agent}'"}
        held = _holding(agent)
        if held is not None:
            return {"ok": False, "error": f"'{agent}' is carrying '{held}'; use carry_to('{agent}', x, y)"}
        nx, ny = _clamp_site(float(x), float(y))
        a["yaw"] = math.atan2(ny - a["y"], nx - a["x"])
        a["x"], a["y"] = nx, ny
        _advance_step(agent)
        return {"ok": True, "agent": agent, "x": round(nx, 3), "y": round(ny, 3)}


@server.tool()
def grab(agent: str, block_name: str) -> dict[str, Any]:
    """G1 humanoid only. Pick up `block_name`.
    Pre-conditions: agent is a G1, has free hands, and is within {reach} m of the block.
    Heavy blocks (carriers_required > 1) cannot be carried until enough agents grab them."""
    with _LOCK:
        a = _WORLD.agents.get(agent)
        if a is None:
            return {"ok": False, "error": f"unknown agent '{agent}'"}
        if AGENT_ROLES.get(agent) != "g1":
            return {"ok": False, "error": f"'{agent}' is a Spot — only G1 humanoids can grab blocks"}
        if _holding(agent) is not None:
            return {"ok": False, "error": f"'{agent}' already holding something; release first"}
        b = _WORLD.blocks.get(block_name)
        if b is None:
            return {"ok": False, "error": f"no block named '{block_name}'"}
        d = _dist(a["x"], a["y"], b["x"], b["y"])
        if d > REACH:
            return {"ok": False, "error": f"too far: {d:.1f} m > reach {REACH} m; "
                                          f"walk_to({a['x']:.1f}, {a['y']:.1f}) → ({b['x']:.1f}, {b['y']:.1f})"}
        b["grippers"].add(agent)
        b["max_grippers"] = max(b["max_grippers"], len(b["grippers"]))
        if len(b["grippers"]) > 1:
            # snap co-carriers to ring around block
            from sim.agent_spacing import spread_agents
            spread_agents(_WORLD.agents, list(b["grippers"]), b["x"], b["y"], yaw=a["yaw"])
        else:
            a["x"], a["y"] = b["x"], b["y"]
        need = b["carriers_required"]
        have = len(b["grippers"])
        _advance_step(agent)
        return {
            "ok": True, "agent": agent, "block": block_name,
            "carriers": f"{have}/{need}",
            "liftable": have >= need,
            "note": (f"need {need - have} more G1(s) to also grab before carry_to"
                     if have < need else "ready to carry_to"),
        }


@server.tool()
def carry_to(agent: str, x: float, y: float) -> dict[str, Any]:
    """Carry the block `agent` is holding to (x, y).
    All agents currently grabbing the same block move together.
    Pre-condition: enough agents grabbing (>= carriers_required)."""
    with _LOCK:
        a = _WORLD.agents.get(agent)
        if a is None:
            return {"ok": False, "error": f"unknown agent '{agent}'"}
        bname = _holding(agent)
        if bname is None:
            return {"ok": False, "error": f"'{agent}' is not holding anything; grab first"}
        b = _WORLD.blocks[bname]
        need, have = b["carriers_required"], len(b["grippers"])
        if have < need:
            return {
                "ok": False,
                "error": (f"'{bname}' needs {need} carriers but only {have} grabbing "
                          f"({sorted(b['grippers'])}); have another G1 walk_to and grab it first"),
            }
        nx, ny = _clamp_site(float(x), float(y))
        b["x"], b["y"] = nx, ny
        from sim.agent_spacing import spread_agents
        spread_agents(_WORLD.agents, list(b["grippers"]), nx, ny, yaw=a["yaw"])
        _advance_step(agent)
        return {
            "ok": True, "block": bname,
            "carried_by": sorted(b["grippers"]),
            "now_at": {"x": round(nx, 3), "y": round(ny, 3)},
        }


@server.tool()
def release(agent: str) -> dict[str, Any]:
    """G1 humanoid: release the block you are holding. It stays at the current position."""
    with _LOCK:
        bname = _holding(agent)
        if bname is None:
            return {"ok": False, "error": f"'{agent}' is not holding anything"}
        _WORLD.blocks[bname]["grippers"].discard(agent)
        _advance_step(agent)
        return {"ok": True, "agent": agent, "released": bname,
                "block_at": {"x": round(_WORLD.blocks[bname]["x"], 3),
                             "y": round(_WORLD.blocks[bname]["y"], 3)}}


@server.tool()
def checkpoint(agent: str, cp_name: str) -> dict[str, Any]:
    """Spot robot: register your arrival at checkpoint `cp_name`.
    The Spot must be within the checkpoint radius. Use this to log patrol progress."""
    with _LOCK:
        a = _WORLD.agents.get(agent)
        if a is None:
            return {"ok": False, "error": f"unknown agent '{agent}'"}
        if AGENT_ROLES.get(agent) != "spot":
            return {"ok": False, "error": f"'{agent}' is a G1 — only Spot robots use checkpoint()"}
        cp = _WORLD.checkpoints.get(cp_name)
        if cp is None:
            return {"ok": False, "error": f"no checkpoint '{cp_name}'; "
                                          f"available: {list(_WORLD.checkpoints.keys())}"}
        d = _dist(a["x"], a["y"], cp["x"], cp["y"])
        if d > cp["radius"]:
            return {"ok": False, "error": f"too far: {d:.1f} m > radius {cp['radius']:.1f} m; "
                                          f"walk_to({cp['x']}, {cp['y']}) first"}
        cp["visited_by"].add(agent)
        _advance_step(agent)
        all_cps = list(_WORLD.checkpoints.keys())
        remaining = [n for n, c in _WORLD.checkpoints.items()
                     if agent not in c["visited_by"]]
        return {
            "ok": True, "agent": agent, "checkpoint": cp_name,
            "remaining_for_this_agent": remaining,
        }


@server.tool()
def say(agent: str, text: str) -> dict[str, Any]:
    """Post a coordination message to the shared log (visible to all agents in get_world_state)."""
    with _LOCK:
        _WORLD.messages.append(f"{agent}: {text}")
        return {"ok": True, "messages": list(_WORLD.messages)}


@server.tool()
def render(width: int = 1024, height: int = 640) -> MCPImage:
    """3D render of the construction site with current agent and block positions."""
    with _LOCK:
        try:
            from sim.scene_render import get_renderer
            agent_order = sorted(_WORLD.agents.keys())  # A,B,C,D,E,F
            types = [AGENT_ROLES.get(aid, "g1") for aid in agent_order]
            sr = get_renderer(_WORLD.scene_id, types)
            agent_poses = [(a["x"], a["y"], a.get("yaw", 0.0))
                           for aid in agent_order
                           for a in [_WORLD.agents[aid]]]
            objs = {n: (b["x"], b["y"], b["z"], 0.0) for n, b in _WORLD.blocks.items()}
            png = sr.render(agent_poses, objs, width=width, height=height)
        except Exception as e:
            png = _render_topdown(min(width, 640))
    return MCPImage(data=png, format="png")


@server.tool()
def render_map(width: int = 700, height: int = 500) -> MCPImage:
    """Top-down 2D schematic of the site: agents, blocks, targets, checkpoints."""
    with _LOCK:
        png = _render_topdown(width, height)
    return MCPImage(data=png, format="png")


# ── top-down schematic ────────────────────────────────────────────────────────

_ROLE_COLORS = {
    "g1":   (50, 100, 220),   # blue humanoid
    "spot": (220, 130, 30),   # orange dog
}
_AGENT_LABELS = {
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F",
}


def _world_to_px(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    """Map site coords to pixel coords. Site: x∈[-12,14], y∈[-11,12]."""
    px = int((x + 12) / 26.0 * w)
    py = int(h - (y + 11) / 23.0 * h)
    return px, py


def _render_topdown(width: int, height: int = 0) -> bytes:
    if height == 0:
        height = int(width * 23 / 26)
    img = Image.new("RGB", (width, height), (195, 175, 145))
    d = ImageDraw.Draw(img)

    def wp(x, y):
        return _world_to_px(x, y, width, height)

    # Foundation outline
    corners = [(-9, -9), (12, -9), (12, 10), (-9, 10)]
    pts = [wp(cx, cy) for cx, cy in corners]
    d.polygon(pts, fill=(210, 190, 160), outline=(120, 100, 70), width=3)

    w = _WORLD

    # Checkpoints — grey circles
    for cp_name, cp in w.checkpoints.items():
        cx, cy = wp(cp["x"], cp["y"])
        r = max(8, int(cp["radius"] / 26 * width))
        visited = bool(cp["visited_by"])
        d.ellipse([cx - r, cy - r, cx + r, cy + r],
                  outline=(30, 150, 30) if visited else (120, 120, 120), width=2)
        d.text((cx + r + 2, cy - 6), cp_name[:8], fill=(80, 80, 80))

    # Target zones — green rings
    for bname, t in w.targets.items():
        tx, ty = wp(t["x"], t["y"])
        r = max(10, int(t["radius"] / 26 * width))
        b = w.blocks.get(bname, {})
        placed = b.get("x") is not None and \
                 _dist(b["x"], b["y"], t["x"], t["y"]) <= t["radius"]
        d.ellipse([tx - r, ty - r, tx + r, ty + r],
                  outline=(30, 200, 30) if placed else (50, 180, 50), width=3)
        d.text((tx - r, ty - r - 14), bname[:6], fill=(30, 130, 30))

    # Blocks — brown squares
    for bname, b in w.blocks.items():
        bx, by = wp(b["x"], b["y"])
        heavy = b["carriers_required"] >= 2
        held = bool(b["grippers"])
        c = (140, 90, 40) if not held else (200, 130, 60)
        d.rectangle([bx - 7, by - 7, bx + 7, by + 7],
                    fill=c, outline=(200, 50, 50) if heavy else (80, 60, 30), width=3 if heavy else 1)
        d.text((bx + 8, by - 6), bname[-4:], fill=(80, 50, 20))

    # Agents — arrows
    for aid, a in w.agents.items():
        ax, ay = wp(a["x"], a["y"])
        role = AGENT_ROLES.get(aid, "g1")
        col = _ROLE_COLORS.get(role, (80, 80, 80))
        yaw = a.get("yaw", 0)
        tip = (ax + 16 * math.cos(yaw), ay - 16 * math.sin(yaw))
        l = (ax + 10 * math.cos(yaw + 2.4), ay - 10 * math.sin(yaw + 2.4))
        r = (ax + 10 * math.cos(yaw - 2.4), ay - 10 * math.sin(yaw - 2.4))
        d.polygon([tip, l, r], fill=col, outline=(10, 10, 30))
        d.text((ax - 4, ay - 5), aid, fill=(255, 255, 255))
        held = _holding(aid)
        if held and held in w.blocks:
            hx, hy = wp(w.blocks[held]["x"], w.blocks[held]["y"])
            d.line([ax, ay, hx, hy], fill=col, width=2)

    # Legend
    d.text((4, 4), "■ G1=blue  ●Spot=orange  □block  ○checkpoint  ◯goal", fill=(50, 50, 50))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
