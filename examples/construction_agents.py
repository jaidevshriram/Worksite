"""Scripted reference agents for the construction coordination tasks.

These implement optimal strategies for each task and serve two purposes:
  1. Smoke-testing that the environment mechanics work end-to-end.
  2. Providing a baseline for comparison against LLM agents.

Usage:
    cd worldsim-template
    # run scripted agent against one task (no HUD key needed):
    .venv/bin/python examples/construction_agents.py block_line
    .venv/bin/python examples/construction_agents.py parallel_supply
    .venv/bin/python examples/construction_agents.py heavy_relay
    .venv/bin/python examples/construction_agents.py full_coordination

    # run all tasks:
    .venv/bin/python examples/construction_agents.py all
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim import construction_world as cw
from environment.construction_env import (
    AGENTS, AGENT_SPAWN, SCENE,
    _SE, _SCAFFOLD_LINE, _DIAMOND, _GUARD_CPS, _ALL_CPS,
)


# ── helper ────────────────────────────────────────────────────────────────────

def _fmt(snap: dict) -> str:
    m = snap["metrics"]
    placed = sum(1 for b in snap["blocks"].values() if b.get("at_goal"))
    total  = len(snap["blocks"])
    return (
        f"  steps={m['total_steps']} | placed={placed}/{total} | "
        f"fleet_util={m['fleet_utilization']:.2f} | "
        f"peak_sim={m['peak_simultaneous_active']} | "
        f"collisions={m['collision_events']}"
    )


# ── Task 1: block_line ────────────────────────────────────────────────────────

def run_block_line():
    """Optimal strategy: dispatch all 4 G1s and both Spots in parallel."""
    print("\n=== TASK 1: block_line ===")
    snap = cw.reset_world(
        scene_id=SCENE, agents=AGENTS, agent_positions=AGENT_SPAWN,
        blocks=_SE, targets=_SCAFFOLD_LINE, checkpoints=_GUARD_CPS,
    )
    print("Initial state:", _fmt(snap))

    # Announce intentions
    cw.say("A", "I'll take block_1 to (-4, 7)")
    cw.say("B", "I'll take block_2 to (-2, 7)")
    cw.say("C", "I'll take block_3 to (0, 7)")
    cw.say("D", "I'll take block_4 to (2, 7)")
    cw.say("E", "Heading to guard_west")
    cw.say("F", "Heading to guard_east")

    # Step 1: All agents walk to their targets simultaneously (representing parallel dispatch)
    # (In reality a single-threaded agent calls them sequentially but we model parallelism
    #  by making all 6 calls in a "batch" — get_world_state will show peak_simultaneous=1
    #  since each call is its own step; in a real async setup these could be concurrent.)
    cw.walk_to("A", 10.0, -4.7)    # walk to block_1
    cw.walk_to("B",  8.5, -5.0)    # walk to block_2
    cw.walk_to("C",  9.9, -6.5)    # walk to block_3
    cw.walk_to("D", 11.4, -5.0)    # walk to block_4
    cw.walk_to("E", -6.0,  2.0)    # walk to guard_west
    cw.walk_to("F",  7.0, -2.0)    # walk to guard_east

    # Step 2: Spots register guard checkpoints; G1s grab blocks
    cw.checkpoint("E", "guard_west")
    cw.checkpoint("F", "guard_east")
    cw.grab("A", "block_1")
    cw.grab("B", "block_2")
    cw.grab("C", "block_3")
    cw.grab("D", "block_4")

    # Step 3: Carry to targets
    cw.carry_to("A", -4.0, 7.0)
    cw.carry_to("B", -2.0, 7.0)
    cw.carry_to("C",  0.0, 7.0)
    cw.carry_to("D",  2.0, 7.0)

    # Step 4: Release
    cw.release("A"); cw.release("B"); cw.release("C"); cw.release("D")

    snap = cw.world_snapshot()
    print("Final state:", _fmt(snap))
    placed = [n for n, b in snap["blocks"].items() if b.get("at_goal")]
    print(f"  Placed: {placed}")
    return snap


# ── Task 2: parallel_supply ───────────────────────────────────────────────────

def run_parallel_supply():
    """Two teams work in parallel to fill two staging zones."""
    print("\n=== TASK 2: parallel_supply ===")
    blocks_ps = [
        {"name": "block_1", "x": 10.0, "y": -4.7, "carriers": 1},
        {"name": "block_2", "x":  8.5, "y": -5.0, "carriers": 1},
        {"name": "block_3", "x":  9.9, "y": -6.5, "carriers": 1},
        {"name": "block_4", "x": 11.4, "y": -5.0, "carriers": 1},
    ]
    targets_ps = [
        {"block": "block_1", "x": -4.0, "y":  0.0, "radius": 1.5},
        {"block": "block_2", "x": -4.0, "y": -2.5, "radius": 1.5},
        {"block": "block_3", "x":  4.5, "y":  0.0, "radius": 1.5},
        {"block": "block_4", "x":  4.5, "y": -2.5, "radius": 1.5},
    ]
    snap = cw.reset_world(
        scene_id=SCENE, agents=AGENTS, agent_positions=AGENT_SPAWN,
        blocks=blocks_ps, targets=targets_ps, checkpoints=[],
    )
    print("Initial:", _fmt(snap))

    # Team 1 (A,B) picks block_1 and block_2
    # Team 2 (C,D) picks block_3 and block_4
    # Spots E,F walk and coordinate

    # All walk to blocks in parallel
    cw.walk_to("A", 10.0, -4.7); cw.walk_to("B",  8.5, -5.0)
    cw.walk_to("C",  9.9, -6.5); cw.walk_to("D", 11.4, -5.0)
    cw.walk_to("E", -4.0, -1.3); cw.walk_to("F",  4.5, -1.3)   # Spots move to staging zones

    cw.grab("A", "block_1"); cw.grab("B", "block_2")
    cw.grab("C", "block_3"); cw.grab("D", "block_4")

    cw.carry_to("A", -4.0, 0.0);  cw.carry_to("B", -4.0, -2.5)
    cw.carry_to("C",  4.5, 0.0);  cw.carry_to("D",  4.5, -2.5)

    cw.release("A"); cw.release("B"); cw.release("C"); cw.release("D")

    snap = cw.world_snapshot()
    print("Final:", _fmt(snap))
    placed = [n for n, b in snap["blocks"].items() if b.get("at_goal")]
    print(f"  Placed: {placed}")
    return snap


# ── Task 3: heavy_relay ───────────────────────────────────────────────────────

def run_heavy_relay():
    """Relay a heavy block in two legs while Spots patrol."""
    print("\n=== TASK 3: heavy_relay ===")
    blocks_hr = [{"name": "heavy_block", "x": 10.0, "y": -5.5, "carriers": 2}]
    relay_x, relay_y = 4.0, -1.0
    final_x, final_y = -2.0, 7.0
    targets_hr = [{"block": "heavy_block", "x": final_x, "y": final_y, "radius": 1.5}]
    relay_cps = [
        {"name": "relay_midpoint", "x": relay_x, "y": relay_y, "radius": 2.0},
        {"name": "scaffold_base",  "x": final_x, "y": final_y, "radius": 2.0},
    ]
    snap = cw.reset_world(
        scene_id=SCENE, agents=AGENTS, agent_positions=AGENT_SPAWN,
        blocks=blocks_hr, targets=targets_hr, checkpoints=relay_cps,
    )
    print("Initial:", _fmt(snap))

    # Leg 1: A+B carry to relay
    cw.say("A", "Walking to heavy_block with B for leg 1")
    cw.walk_to("A", 10.0, -5.5); cw.walk_to("B", 10.0, -5.5)
    # Spots patrol on opposite sides of the relay midpoint to avoid colliding
    cw.walk_to("E", relay_x - 1.5, relay_y)   # E: west of relay
    cw.walk_to("F", relay_x + 1.5, relay_y)   # F: east of relay
    cw.grab("A", "heavy_block"); cw.grab("B", "heavy_block")
    cw.carry_to("A", relay_x, relay_y)
    cw.release("A"); cw.release("B")
    # A and B step aside to clear space for C and D
    cw.walk_to("A", relay_x - 3.0, relay_y + 1.0)
    cw.walk_to("B", relay_x - 3.0, relay_y - 1.0)
    cw.checkpoint("E", "relay_midpoint"); cw.checkpoint("F", "relay_midpoint")
    # Spots clear the relay zone north before C and D arrive
    cw.walk_to("E", relay_x - 1.5, relay_y + 3.5)
    cw.walk_to("F", relay_x + 1.5, relay_y + 3.5)

    # Leg 2: C+D approach relay from opposite sides (within REACH=2m but offset)
    cw.say("C", "Walking to relay with D for leg 2")
    cw.walk_to("C", relay_x + 1.2, relay_y)   # C approaches from east
    cw.walk_to("D", relay_x - 1.2, relay_y)   # D approaches from west
    # Spots head to final zone after leg 2 is underway
    cw.walk_to("E", final_x - 1.5, final_y); cw.walk_to("F", final_x + 1.5, final_y)
    cw.grab("C", "heavy_block"); cw.grab("D", "heavy_block")
    cw.carry_to("C", final_x, final_y)
    cw.release("C"); cw.release("D")
    cw.checkpoint("E", "scaffold_base"); cw.checkpoint("F", "scaffold_base")

    snap = cw.world_snapshot()
    print("Final:", _fmt(snap))
    b = snap["blocks"]["heavy_block"]
    print(f"  heavy_block at_goal={b.get('at_goal')} dist={b.get('distance_to_goal'):.2f}m")
    return snap


# ── Task 4: full_coordination ─────────────────────────────────────────────────

def run_full_coordination():
    """Place 4 blocks in a diamond while Spots complete full patrol."""
    print("\n=== TASK 4: full_coordination ===")
    snap = cw.reset_world(
        scene_id=SCENE, agents=AGENTS, agent_positions=AGENT_SPAWN,
        blocks=_SE, targets=_DIAMOND, checkpoints=_ALL_CPS,
    )
    print("Initial:", _fmt(snap))

    # Announce plan
    for aid, tgt in zip("ABCD", _DIAMOND):
        cw.say(aid, f"Taking {tgt['block']} to ({tgt['x']:.0f},{tgt['y']:.0f})")
    cw.say("E", "Starting patrol: excavator → trucks_north → trucks_south")
    cw.say("F", "Mirroring E's patrol for full coverage")

    # All G1s walk to blocks (parallel)
    block_positions = {b["name"]: (b["x"], b["y"]) for b in _SE}
    for aid, tgt in zip("ABCD", _DIAMOND):
        bx, by = block_positions[tgt["block"]]
        cw.walk_to(aid, bx, by)

    # Spot patrol (interleaved with G1 work) — offset by 1 m to avoid collision
    cw.walk_to("E", -10.0,  0.8)
    cw.walk_to("F", -10.0, -1.0)
    cw.checkpoint("E", "excavator")
    cw.checkpoint("F", "excavator")

    # G1s grab
    for aid, tgt in zip("ABCD", _DIAMOND):
        cw.grab(aid, tgt["block"])

    # Spots continue patrol while G1s carry — offset to avoid stacking
    cw.walk_to("E", 11.4, 6.0)
    cw.walk_to("F", 11.4, 7.5)
    cw.checkpoint("E", "trucks_north")
    cw.checkpoint("F", "trucks_north")

    # G1s carry to diamond targets
    for aid, tgt in zip("ABCD", _DIAMOND):
        cw.carry_to(aid, tgt["x"], tgt["y"])

    cw.walk_to("E", 11.4,  0.0)
    cw.walk_to("F", 11.4, -1.5)
    cw.checkpoint("E", "trucks_south")
    cw.checkpoint("F", "trucks_south")

    # G1s release
    for aid in "ABCD":
        cw.release(aid)

    snap = cw.world_snapshot()
    print("Final:", _fmt(snap))
    placed = [n for n, b in snap["blocks"].items() if b.get("at_goal")]
    print(f"  Placed: {placed}")
    for cp_name, cp in snap["checkpoints"].items():
        print(f"  Checkpoint {cp_name}: visited_by={cp['visited_by']}")
    return snap


# ── scoring helper ────────────────────────────────────────────────────────────

def _compute_score(snap: dict, block_targets: list, patrol_reqs: dict,
                   weights: tuple = (0.50, 0.30, 0.20)) -> float:
    from environment.construction_env import _block_score, _patrol_score
    block_frac, _, _ = _block_score(snap, block_targets)
    patrol_frac = _patrol_score(snap, patrol_reqs) if patrol_reqs else 1.0
    fleet = snap["metrics"]["fleet_utilization"]
    w1, w2, w3 = weights
    collisions = snap["metrics"]["collision_events"]
    return round(
        w1 * block_frac + w2 * patrol_frac + w3 * fleet - 0.05 * min(4, collisions), 4
    )


# ── main ──────────────────────────────────────────────────────────────────────

TASKS = {
    "block_line":       run_block_line,
    "parallel_supply":  run_parallel_supply,
    "heavy_relay":      run_heavy_relay,
    "full_coordination":run_full_coordination,
}

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "all"
    if task == "all":
        for name, fn in TASKS.items():
            fn()
    elif task in TASKS:
        TASKS[task]()
    else:
        print(f"Unknown task '{task}'. Choose: {list(TASKS.keys())} or 'all'")
        sys.exit(1)
