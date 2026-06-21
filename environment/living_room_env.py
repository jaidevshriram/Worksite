"""Living-room planning environment - grades an LLM's *plan*, not low-level control.

The agent drives a symbolic skill layer (`sim/skill_world.py`) served as an `mcp`
capability: walk_to / pick / place with explicit preconditions + effects over a
physics-free world state. The single task, `move-side-table`, scores whether the
agent successfully lands the side table in the target corner.

Unlike the Newton env (`environment/env.py`), the skill server is light and runs on
a worker thread *in this process* (no GL viewer, so no separate process needed); the
grader reads the same world state in-process via `skill_world.world_snapshot()`.

Serve it like any environment: `hud eval environment/living_room_env.py`.
"""

from __future__ import annotations

import asyncio
import socket
import sys
import threading
from pathlib import Path

from hud import Environment, Taskset
from hud.capabilities import Capability
from hud.graders import EvaluationResult, SubScore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import skill_world  # noqa: E402

HOST = "127.0.0.1"
SKILLS = "walk_to(x, y), pick(object_name), place(x, y), get_world_state(), render()"

env = Environment(name="worldsim-living-room")
_server_state: dict[str, object] = {}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


async def _ensure_skill_server(timeout: float = 30.0) -> str:
    """Start the FastMCP skill server on a daemon thread (once) and return its URL."""
    if "url" in _server_state:
        return str(_server_state["url"])
    port = _free_port()

    def _run() -> None:
        asyncio.run(skill_world.server.run_async(
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
    raise RuntimeError(f"skill server never listened on {HOST}:{port}")


@env.initialize
async def _up() -> None:
    url = await _ensure_skill_server()
    env.add_capability(Capability.mcp(name="skills", url=url))


@env.shutdown
async def _down() -> None:
    pass  # daemon thread exits with the process


@env.template(id="move-side-table",
              description="Plan a pick-and-place: carry the side table into the target corner.")
async def move_side_table(scene_id: str = "living-room-v1", target_object: str = "asset_2_2",
                          goal_x: float = -2.0, goal_y: float = -2.0, goal_radius: float = 0.6):
    skill_world.reset_world(scene_id)
    reach = skill_world.REACH
    yield (
        f"You control a humanoid in a living room through HIGH-LEVEL SKILLS only "
        f"(no motor control). Skills: {SKILLS}.\n"
        f"GOAL: move the side table '{target_object}' into the corner at "
        f"({goal_x:.2f}, {goal_y:.2f}) - it must end within {goal_radius:.2f} m of that point.\n"
        f"Rules: pick(obj) needs you within {reach:.2f} m of the object and hands empty; "
        f"place(x,y) needs you holding something and within {reach:.2f} m of (x,y). "
        f"Carrying moves the object with you.\n"
        f"Plan suggestion: get_world_state() -> walk_to the table -> pick it -> "
        f"walk_to the corner -> place it there -> get_world_state() to verify."
    )

    snap = skill_world.world_snapshot()
    obj = snap["objects"].get(target_object, {})
    dist = ((obj.get("x", 0.0) - goal_x) ** 2 + (obj.get("y", 0.0) - goal_y) ** 2) ** 0.5
    initial = snap.get("target_initial_distance", 1.0) or 1.0
    progress = max(0.0, min(1.0, 1.0 - dist / initial))
    landed = dist <= goal_radius
    picked = snap.get("target_was_picked", False)

    # a correct placement is full credit; partial credit only when not yet landed.
    reward = 1.0 if landed else 0.4 * progress
    yield EvaluationResult(
        reward=round(reward, 4),
        done=True,
        content=(f"side table '{target_object}' final distance to corner = {dist:.3f} m "
                 f"(goal <= {goal_radius} m). picked={'YES' if picked else 'NO'}, "
                 f"progress={progress:.2f}, steps={snap.get('steps')}. "
                 f"{'SUCCESS' if landed else 'INCOMPLETE'}"),
        subscores=[
            SubScore(name="placed", weight=1.0, value=round(reward, 4)),
        ],
    )


# ── Taskset ───────────────────────────────────────────────────────────────────
# Defined here (not a separate tasks.py) so `hud eval environment/living_room_env.py`
# spawns *this* env: hud's runtime serves a .py source directly only when it builds an
# `Environment(...)`, else it falls back to a sibling env.py.

_move_table = move_side_table(scene_id="living-room-v1", target_object="asset_2_2",
                              goal_x=-2.0, goal_y=-2.0, goal_radius=0.6)
_move_table.slug = "move-side-table"

taskset = Taskset("worldsim-living-room", [_move_table])
