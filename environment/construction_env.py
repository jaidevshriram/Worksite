"""Construction site multi-agent coordination environment.

Four tasks test increasingly complex fleet coordination for a mixed robot team
of 4 Unitree G1 humanoids (A–D) and 2 Boston Dynamics Spot quadrupeds (E–F).

Tasks
-----
1. block_line          – All 4 G1s move one block each to a line; Spots clear the path.
2. parallel_supply     – Two parallel G1 teams race to fill two staging areas; score
                         rewards the peak number of agents active simultaneously.
3. heavy_relay         – One HEAVY block (requires 2 carriers) is ferried in two legs
                         via a relay point; Spots patrol the route.
4. full_coordination   – 4 blocks to a diamond, Spots complete a full site patrol.
                         Scores blocks (50%) + patrol (30%) + fleet utilization (20%).

Run with:
    hud eval environment/construction_env.py --tasks block_line
    hud eval environment/construction_env.py --tasks full_coordination

Metrics reported in every get_world_state() response guide Claude toward
parallel execution: fleet_utilization, peak_simultaneous_active, agent_uptime.
"""

from __future__ import annotations

import asyncio
import math
import socket
import sys
import threading
from pathlib import Path
from typing import Any

from hud import Environment, Taskset
from hud.capabilities import Capability
from hud.graders import EvaluationResult, SubScore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import construction_world as cw  # noqa: E402

HOST = "127.0.0.1"
SCENE = "construction-v1"

# ── agent config ──────────────────────────────────────────────────────────────
AGENTS      = {"A": "g1", "B": "g1", "C": "g1", "D": "g1", "E": "spot", "F": "spot"}
AGENT_SPAWN = {
    "A": [-1.4,  2.2], "B": [2.8,  2.1],
    "C": [-1.7, -2.3], "D": [2.7, -2.5],
    "E": [-6.0,  4.0], "F": [7.0, -4.0],
}

# ── scene landmarks used across tasks ────────────────────────────────────────
# Brick cluster (SE corner)
_SE = [
    {"name": "block_1", "x": 10.0, "y": -4.7,  "carriers": 1},
    {"name": "block_2", "x":  8.5, "y": -5.0,  "carriers": 1},
    {"name": "block_3", "x":  9.9, "y": -6.5,  "carriers": 1},
    {"name": "block_4", "x": 11.4, "y": -5.0,  "carriers": 1},
]
# Scaffolding base targets (line along north edge, y≈7)
_SCAFFOLD_LINE = [
    {"block": "block_1", "x": -4.0, "y": 7.0, "radius": 1.2},
    {"block": "block_2", "x": -2.0, "y": 7.0, "radius": 1.2},
    {"block": "block_3", "x":  0.0, "y": 7.0, "radius": 1.2},
    {"block": "block_4", "x":  2.0, "y": 7.0, "radius": 1.2},
]
# Diamond-pattern targets at centre
_DIAMOND = [
    {"block": "block_1", "x":  0.0, "y":  3.0, "radius": 1.2},
    {"block": "block_2", "x":  3.0, "y":  0.0, "radius": 1.2},
    {"block": "block_3", "x":  0.0, "y": -3.0, "radius": 1.2},
    {"block": "block_4", "x": -3.0, "y":  0.0, "radius": 1.2},
]
# Site patrol checkpoints
_ALL_CPS = [
    {"name": "excavator",   "x": -10.0, "y": -0.12, "radius": 2.0},
    {"name": "trucks_north","x":  11.4, "y":  6.0,  "radius": 2.0},
    {"name": "trucks_south","x":  11.4, "y":  0.0,  "radius": 2.0},
]
# Guard posts used in task 1
_GUARD_CPS = [
    {"name": "guard_west",  "x": -6.0,  "y":  2.0,  "radius": 2.0},
    {"name": "guard_east",  "x":  7.0,  "y": -2.0,  "radius": 2.0},
]

SKILLS = (
    "walk_to(agent, x, y), grab(agent, block), carry_to(agent, x, y), "
    "release(agent), checkpoint(agent, cp_name), say(agent, text), "
    "get_world_state(), render()"
)

env = Environment(name="worldsim-construction")
_server_state: dict[str, object] = {}


# ── server lifecycle ──────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0)); return s.getsockname()[1]


async def _ensure_server(timeout: float = 30.0) -> str:
    if "url" in _server_state:
        return str(_server_state["url"])
    port = _free_port()

    def _run():
        asyncio.run(cw.server.run_async(
            transport="http", host=HOST, port=port, show_banner=False))

    threading.Thread(target=_run, daemon=True).start()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            socket.create_connection((HOST, port), timeout=0.5).close()
            _server_state["url"] = f"http://{HOST}:{port}/mcp"
            return str(_server_state["url"])
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"construction skill server never came up on {HOST}:{port}")


@env.initialize
async def _up() -> None:
    url = await _ensure_server()
    env.add_capability(Capability.mcp(name="skills", url=url))


@env.shutdown
async def _down() -> None:
    pass


# ── shared grading helper ─────────────────────────────────────────────────────

def _block_score(snap: dict, targets: list[dict]) -> tuple[float, list[SubScore], list[str]]:
    """Returns (raw_fraction_placed, subscores, lines)."""
    subs, lines, placed = [], [], 0
    for t in targets:
        bname = t["block"]
        b = snap["blocks"].get(bname, {})
        at_goal = bool(b.get("at_goal", False))
        d = b.get("distance_to_goal", float("inf"))
        init = snap.get("_internal", {}).get("initial_distance", {}).get(bname, 1.0) or 1.0
        progress = max(0.0, min(1.0, 1.0 - d / init)) if d != float("inf") else 0.0
        score = 1.0 if at_goal else 0.4 * progress
        if at_goal:
            placed += 1
        subs.append(SubScore(name=f"placed_{bname}", weight=round(1 / len(targets), 4),
                             value=round(score, 4)))
        lines.append(f"{bname}: {'✓ at goal' if at_goal else f'dist={d:.1f}m'}")
    return placed / len(targets), subs, lines


def _patrol_score(snap: dict, required_visitors: dict[str, list[str]]) -> float:
    """required_visitors: {cp_name: [agent_ids who must visit it]}.
    Returns fraction of required visits completed."""
    total, done = 0, 0
    for cp_name, agents in required_visitors.items():
        cp = snap["checkpoints"].get(cp_name, {})
        visited = set(cp.get("visited_by", []))
        for aid in agents:
            total += 1
            if aid in visited:
                done += 1
    return done / total if total > 0 else 1.0


def _metrics_bonus(snap: dict) -> dict[str, float]:
    m = snap["metrics"]
    return {
        "fleet_utilization": m["fleet_utilization"],
        "peak_simultaneous": m["peak_simultaneous_active"],
        "collision_events": m["collision_events"],
        "steps": m["total_steps"],
    }


# ── Task 1: Block Line ────────────────────────────────────────────────────────

@env.template(
    id="block_line",
    description=(
        "4 G1s carry one brick each from the SE cluster to a line at the scaffolding base. "
        "Spots must first clear the path by visiting their guard posts."
    ),
)
async def block_line(scene_id: str = SCENE):
    """Place 4 bricks in a line; Spots must guard both flanks."""
    cw.reset_world(
        scene_id=scene_id,
        agents=AGENTS,
        agent_positions=AGENT_SPAWN,
        blocks=_SE,
        targets=_SCAFFOLD_LINE,
        checkpoints=_GUARD_CPS,
    )
    snap = cw.world_snapshot()

    goals_txt = "; ".join(
        f"{t['block']} → ({t['x']:.0f},{t['y']:.0f})" for t in _SCAFFOLD_LINE)
    cp_txt = "; ".join(f"{c['name']} ({c['x']:.0f},{c['y']:.0f})" for c in _GUARD_CPS)

    yield (
        f"You command a mixed construction fleet on site '{scene_id}'.\n\n"
        f"Agents\n"
        f"  G1 humanoids (can grab/carry blocks): A {AGENT_SPAWN['A']}, "
        f"B {AGENT_SPAWN['B']}, C {AGENT_SPAWN['C']}, D {AGENT_SPAWN['D']}\n"
        f"  Spot robots (patrol/guard only): E {AGENT_SPAWN['E']}, F {AGENT_SPAWN['F']}\n\n"
        f"Skills: {SKILLS}\n\n"
        f"OBJECTIVE: Move all 4 bricks from the SE brick cluster to the scaffolding baseline.\n"
        f"  Brick targets (place within 1.2 m): {goals_txt}\n\n"
        f"SECONDARY: Spots must visit their guard posts to clear the path.\n"
        f"  Checkpoints: {cp_txt}\n"
        f"  Spot E → guard_west;  Spot F → guard_east\n\n"
        f"SCORING:\n"
        f"  70% — fraction of bricks placed at goal\n"
        f"  15% — guard posts cleared (Spots at checkpoints)\n"
        f"  15% — fleet utilization (reward parallel G1 movement)\n\n"
        f"TIPS: Use say() to coordinate. Check metrics in get_world_state() to see "
        f"how well you are parallelising. Start with get_world_state()."
    )

    snap = cw.world_snapshot()
    mb = _metrics_bonus(snap)

    block_frac, b_subs, b_lines = _block_score(snap, _SCAFFOLD_LINE)
    patrol_frac = _patrol_score(snap, {"guard_west": ["E"], "guard_east": ["F"]})
    fleet = mb["fleet_utilization"]

    reward = 0.70 * block_frac + 0.15 * patrol_frac + 0.15 * fleet
    reward = round(max(0.0, min(1.0, reward)), 4)

    all_placed = block_frac == 1.0
    summary = (
        f"{'ALL BRICKS PLACED' if all_placed else 'INCOMPLETE'} | "
        f"steps={mb['steps']} | patrol={patrol_frac:.2f} | "
        f"fleet_util={fleet:.2f} | collisions={mb['collision_events']} | "
        + " | ".join(b_lines)
    )
    subs = b_subs + [
        SubScore(name="patrol", weight=0.15, value=round(patrol_frac, 4)),
        SubScore(name="fleet_utilization", weight=0.15, value=round(fleet, 4)),
    ]
    yield EvaluationResult(reward=reward, done=True, content=summary, subscores=subs)


# ── Task 2: Parallel Supply ───────────────────────────────────────────────────

@env.template(
    id="parallel_supply",
    description=(
        "Two G1 pairs race in parallel to fill two staging zones from the same brick cache. "
        "Score rewards synchronous high-fleet-utilization execution."
    ),
)
async def parallel_supply(scene_id: str = SCENE):
    """Parallel workstreams — bonus for peak simultaneous agents."""
    # Two staging zones: west cluster and east cluster
    blocks_ps = [
        {"name": "block_1", "x": 10.0, "y": -4.7, "carriers": 1},
        {"name": "block_2", "x":  8.5, "y": -5.0, "carriers": 1},
        {"name": "block_3", "x":  9.9, "y": -6.5, "carriers": 1},
        {"name": "block_4", "x": 11.4, "y": -5.0, "carriers": 1},
    ]
    # Team 1 (A,B) fills west staging; Team 2 (C,D) fills east staging
    targets_ps = [
        {"block": "block_1", "x": -4.0, "y":  0.0, "radius": 1.5},   # west
        {"block": "block_2", "x": -4.0, "y": -2.5, "radius": 1.5},   # west
        {"block": "block_3", "x":  4.5, "y":  0.0, "radius": 1.5},   # east
        {"block": "block_4", "x":  4.5, "y": -2.5, "radius": 1.5},   # east
    ]
    cw.reset_world(
        scene_id=scene_id,
        agents=AGENTS,
        agent_positions=AGENT_SPAWN,
        blocks=blocks_ps,
        targets=targets_ps,
        checkpoints=[],
    )

    team1 = "A and B"; team2 = "C and D"
    goals_txt = (
        "block_1 → (-4, 0), block_2 → (-4, -2.5) [Team 1: A,B]; "
        "block_3 → (4.5, 0), block_4 → (4.5, -2.5) [Team 2: C,D]"
    )
    yield (
        f"Mixed fleet on '{scene_id}'. Two teams work in parallel to stock two staging zones.\n\n"
        f"Agents\n"
        f"  Team 1 — G1s A, B: fill WEST staging zone (target x≈-4)\n"
        f"  Team 2 — G1s C, D: fill EAST staging zone (target x≈+4.5)\n"
        f"  Spots E, F: provide traffic coordination along the route\n\n"
        f"Skills: {SKILLS}\n\n"
        f"OBJECTIVE: {goals_txt}\n\n"
        f"SCORING:\n"
        f"  60% — fraction of blocks placed\n"
        f"  25% — peak simultaneous agents (max 6); higher = more parallel\n"
        f"  15% — fleet utilization\n\n"
        f"KEY INSIGHT: Teams 1 and 2 can work completely in parallel — blocks in the SE "
        f"cluster are independent. Spots can patrol concurrently. The best score comes from "
        f"dispatching ALL 6 agents at the same time. get_world_state() shows metrics.\n\n"
        f"Start with get_world_state()."
    )

    snap = cw.world_snapshot()
    mb = _metrics_bonus(snap)

    block_frac, b_subs, b_lines = _block_score(snap, targets_ps)
    peak = mb["peak_simultaneous"]
    fleet = mb["fleet_utilization"]

    # peak bonus: 6/6 agents = 1.0, 3/6 = 0.5 etc.
    peak_score = min(1.0, peak / 6)

    reward = 0.60 * block_frac + 0.25 * peak_score + 0.15 * fleet
    reward = round(max(0.0, min(1.0, reward)), 4)

    summary = (
        f"blocks={block_frac:.2f} | peak_simultaneous={peak}/6 | "
        f"fleet_util={fleet:.2f} | steps={mb['steps']} | collisions={mb['collision_events']} | "
        + " | ".join(b_lines)
    )
    subs = b_subs + [
        SubScore(name="peak_simultaneous", weight=0.25, value=round(peak_score, 4)),
        SubScore(name="fleet_utilization", weight=0.15, value=round(fleet, 4)),
    ]
    yield EvaluationResult(reward=reward, done=True, content=summary, subscores=subs)


# ── Task 3: Heavy Relay ───────────────────────────────────────────────────────

@env.template(
    id="heavy_relay",
    description=(
        "One HEAVY block (2 carriers required) must travel SE → relay point → scaffolding base "
        "via two carrying teams. Spots patrol the route."
    ),
)
async def heavy_relay(scene_id: str = SCENE):
    """Co-carry a heavy block across two relay legs; Spots patrol."""
    blocks_hr = [
        {"name": "heavy_block", "x": 10.0, "y": -5.5, "carriers": 2},
    ]
    relay_x, relay_y = 4.0, -1.0
    final_x, final_y = -2.0, 7.0
    targets_hr = [
        {"block": "heavy_block", "x": final_x, "y": final_y, "radius": 1.5},
    ]
    # Relay checkpoints: G1s A+B carry leg 1, C+D carry leg 2
    relay_cps = [
        {"name": "relay_midpoint", "x": relay_x, "y": relay_y, "radius": 2.0},
        {"name": "scaffold_base",  "x": final_x, "y": final_y, "radius": 2.0},
    ]
    cw.reset_world(
        scene_id=scene_id,
        agents=AGENTS,
        agent_positions=AGENT_SPAWN,
        blocks=blocks_hr,
        targets=targets_hr,
        checkpoints=relay_cps,
    )

    yield (
        f"Mixed fleet on '{scene_id}'. A HEAVY block at (10, -5.5) requires 2 G1 carriers.\n\n"
        f"Agents\n"
        f"  G1s A, B, C, D (humanoids — can grab/carry)\n"
        f"  Spots E, F (quadrupeds — patrol only, no grabbing)\n\n"
        f"Skills: {SKILLS}\n\n"
        f"OBJECTIVE: Carry heavy_block to the scaffolding base at ({final_x}, {final_y}).\n\n"
        f"STRATEGY (relay):\n"
        f"  Leg 1: A + B walk to the block, both grab it, carry_to relay_midpoint "
        f"({relay_x}, {relay_y}), release.\n"
        f"  Leg 2: C + D walk to the relay, both grab, carry_to scaffold_base "
        f"({final_x}, {final_y}), release.\n"
        f"  Meanwhile: E and F patrol the route (visit relay_midpoint and scaffold_base checkpoints).\n\n"
        f"SCORING:\n"
        f"  60% — heavy_block placed at scaffold_base\n"
        f"  20% — relay checkpoints visited by Spots\n"
        f"  20% — fleet utilization (Spots should work concurrently with G1 relay)\n\n"
        f"NOTE: heavy_block needs carriers=2; carry_to will fail until BOTH A and B "
        f"(or both C and D) have grabbed it.\n\n"
        f"Start with get_world_state()."
    )

    snap = cw.world_snapshot()
    mb = _metrics_bonus(snap)

    block_frac, b_subs, b_lines = _block_score(snap, targets_hr)
    patrol_frac = _patrol_score(snap, {
        "relay_midpoint": ["E", "F"],
        "scaffold_base":  ["E", "F"],
    })
    fleet = mb["fleet_utilization"]

    reward = 0.60 * block_frac + 0.20 * patrol_frac + 0.20 * fleet
    reward = round(max(0.0, min(1.0, reward)), 4)

    summary = (
        f"block_placed={block_frac:.2f} | patrol={patrol_frac:.2f} | "
        f"fleet_util={fleet:.2f} | steps={mb['steps']} | collisions={mb['collision_events']} | "
        + " | ".join(b_lines)
    )
    subs = b_subs + [
        SubScore(name="patrol", weight=0.20, value=round(patrol_frac, 4)),
        SubScore(name="fleet_utilization", weight=0.20, value=round(fleet, 4)),
    ]
    yield EvaluationResult(reward=reward, done=True, content=summary, subscores=subs)


# ── Task 4: Full Coordination ─────────────────────────────────────────────────

@env.template(
    id="full_coordination",
    description=(
        "4 bricks placed in a diamond at centre while both Spots complete a full 3-point "
        "site patrol. Best score requires maximum parallelism with zero collisions."
    ),
)
async def full_coordination(scene_id: str = SCENE):
    """Full-site coordination: blocks + patrol + fleet utilization."""
    cw.reset_world(
        scene_id=scene_id,
        agents=AGENTS,
        agent_positions=AGENT_SPAWN,
        blocks=_SE,
        targets=_DIAMOND,
        checkpoints=_ALL_CPS,
    )

    goals_txt = "; ".join(
        f"{t['block']} → ({t['x']:.0f},{t['y']:.0f})" for t in _DIAMOND)
    cp_txt = "; ".join(f"{c['name']} ({c['x']:.0f},{c['y']:.0f})" for c in _ALL_CPS)

    yield (
        f"FULL SITE COORDINATION — '{scene_id}'\n\n"
        f"You command: G1 humanoids A, B, C, D  +  Spot robots E, F.\n"
        f"Skills: {SKILLS}\n\n"
        f"OBJECTIVE 1 — Brick diamond (50% of score):\n"
        f"  {goals_txt}\n\n"
        f"OBJECTIVE 2 — Site patrol (30% of score):\n"
        f"  Both Spots (E and F) must visit ALL 3 checkpoints:\n"
        f"  {cp_txt}\n"
        f"  Use: walk_to(E, cx, cy) then checkpoint(E, name)\n\n"
        f"OBJECTIVE 3 — Fleet utilization (20% of score):\n"
        f"  fleet_utilization in metrics rewards parallel execution.\n"
        f"  A plan where all 6 agents act concurrently scores near 1.0 here.\n\n"
        f"COLLISION PENALTY: −0.05 per collision event (agents within {cw.MIN_SEP}m).\n\n"
        f"TIPS:\n"
        f"  • G1s A–D each handle one brick independently (no coordination needed for light blocks).\n"
        f"  • E and F can patrol concurrently while G1s carry.\n"
        f"  • Avoid routing G1s through the same narrow gaps simultaneously.\n"
        f"  • say() is free — use it to announce intentions and avoid conflicts.\n\n"
        f"Aim for: all 4 bricks placed + both Spots visit all 3 checkpoints + high fleet_util.\n"
        f"Start with get_world_state()."
    )

    snap = cw.world_snapshot()
    mb = _metrics_bonus(snap)

    block_frac, b_subs, b_lines = _block_score(snap, _DIAMOND)
    # Both Spots must visit all 3 checkpoints
    patrol_frac = _patrol_score(snap, {
        "excavator":    ["E", "F"],
        "trucks_north": ["E", "F"],
        "trucks_south": ["E", "F"],
    })
    fleet = mb["fleet_utilization"]
    collision_penalty = min(0.20, mb["collision_events"] * 0.05)

    reward = (0.50 * block_frac + 0.30 * patrol_frac + 0.20 * fleet
              - collision_penalty)
    reward = round(max(0.0, min(1.0, reward)), 4)

    all_bricks = block_frac == 1.0
    full_patrol = patrol_frac == 1.0
    summary = (
        f"{'FULL SUCCESS' if all_bricks and full_patrol else 'INCOMPLETE'} | "
        f"bricks={block_frac:.2f} | patrol={patrol_frac:.2f} | "
        f"fleet_util={fleet:.2f} | collisions={mb['collision_events']} | "
        f"steps={mb['steps']} | "
        + " | ".join(b_lines)
    )
    subs = b_subs + [
        SubScore(name="patrol_completion", weight=0.30, value=round(patrol_frac, 4)),
        SubScore(name="fleet_utilization", weight=0.20, value=round(fleet, 4)),
        SubScore(name="collision_penalty", weight=0.00, value=round(collision_penalty, 4)),
    ]
    yield EvaluationResult(reward=reward, done=True, content=summary, subscores=subs)


# ── taskset (for `hud eval`) ──────────────────────────────────────────────────

_t1 = block_line(scene_id=SCENE);         _t1.slug = "block_line"
_t2 = parallel_supply(scene_id=SCENE);    _t2.slug = "parallel_supply"
_t3 = heavy_relay(scene_id=SCENE);        _t3.slug = "heavy_relay"
_t4 = full_coordination(scene_id=SCENE);  _t4.slug = "full_coordination"

taskset = Taskset("worldsim-construction", [_t1, _t2, _t3, _t4])
