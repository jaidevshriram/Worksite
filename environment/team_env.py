"""Multi-agent living-room planning environment - grades TEAM coordination.

One LLM acts as the controller for a team of three embodied agents (alpha, bravo,
carol) sharing one room, driving agent-scoped skills (`sim/multi_skill_world.py`)
over an `mcp` capability. The task, `team-stage-corner`, can only be solved by
coordinating: the side table is heavy (needs a two-agent `joint_lift`) and the
target corner is blocked by a cushion that must be cleared first.

Reward = the table landed in the corner (dominant) + the corner cleared + distance
shaping, so the score explains itself.

Serve it like any environment: `hud eval environment/team_env.py claude --max-steps 40`.
The skill server is physics-free and runs on a worker thread in this process.
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
from sim import multi_skill_world as world  # noqa: E402

HOST = "127.0.0.1"
SKILLS = ("walk_to(agent, x, y), pick(agent, object_name), place(agent, x, y), "
          "joint_lift(agent_a, agent_b, object_name), joint_place(agent_a, agent_b, x, y), "
          "get_world_state(), render()")

env = Environment(name="worldsim-team")
_server_state: dict[str, object] = {}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


async def _ensure_skill_server(timeout: float = 30.0) -> str:
    if "url" in _server_state:
        return str(_server_state["url"])
    port = _free_port()

    def _run() -> None:
        asyncio.run(world.server.run_async(transport="http", host=HOST, port=port, show_banner=False))

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
    raise RuntimeError(f"team skill server never listened on {HOST}:{port}")


@env.initialize
async def _up() -> None:
    url = await _ensure_skill_server()
    env.add_capability(Capability.mcp(name="skills", url=url))


@env.shutdown
async def _down() -> None:
    pass


@env.template(id="team-stage-corner",
              description="Coordinate three agents to clear the corner and carry the heavy table into it.")
async def team_stage_corner(scene_id: str = "living-room-v1", target_object: str = "asset_2_2",
                            goal_x: float = -2.0, goal_y: float = -2.0, goal_radius: float = 0.6):
    world.reset_world(scene_id)
    reach = world.REACH
    yield (
        f"You are the controller of a THREE-AGENT team (alpha, bravo, carol) in a living room. "
        f"Drive them with high-level skills (no motor control): {SKILLS}.\n"
        f"GOAL: get the side table '{target_object}' into the corner at ({goal_x:.2f}, {goal_y:.2f}) "
        f"- it must end within {goal_radius:.2f} m of that point.\n"
        f"CONSTRAINTS that force coordination:\n"
        f"  - The table is HEAVY: a single pick() is refused. Two agents must joint_lift() it while "
        f"both stand within {reach:.2f} m of it, then walk it to the corner and joint_place() it.\n"
        f"  - The corner is BLOCKED by a cushion: placing into the corner is refused until some agent "
        f"picks up that cushion and places it OUTSIDE the goal region.\n"
        f"Rules: pick/lift need agents within {reach:.2f} m of the object; place needs agents within "
        f"{reach:.2f} m of the drop point. Start with get_world_state() (it lists blockers_in_goal), "
        f"assign roles, and verify with get_world_state() at the end."
    )

    snap = world.world_snapshot()
    obj = snap["objects"].get(target_object, {})
    dist = ((obj.get("x", 0.0) - goal_x) ** 2 + (obj.get("y", 0.0) - goal_y) ** 2) ** 0.5
    initial = snap.get("target_initial_distance", 1.0) or 1.0
    progress = max(0.0, min(1.0, 1.0 - dist / initial))
    landed = dist <= goal_radius
    cleared = snap.get("corner_cleared", False)
    lifted = snap.get("target_was_lifted", False)

    reward = 0.3 * progress + 0.2 * (1.0 if cleared else 0.0) + 0.5 * (1.0 if landed else 0.0)
    yield EvaluationResult(
        reward=round(reward, 4),
        done=True,
        content=(f"table '{target_object}' dist to corner = {dist:.3f} m (goal <= {goal_radius}). "
                 f"corner_cleared={'YES' if cleared else 'NO'}, joint_lifted={'YES' if lifted else 'NO'}, "
                 f"steps={snap.get('steps')}. {'SUCCESS' if landed and cleared else 'INCOMPLETE'}"),
        subscores=[
            SubScore(name="distance_progress", weight=0.3, value=round(progress, 4)),
            SubScore(name="corner_cleared", weight=0.2, value=1.0 if cleared else 0.0),
            SubScore(name="table_landed", weight=0.5, value=1.0 if landed else 0.0),
        ],
    )


# ── Taskset (defined here so `hud eval environment/team_env.py` spawns this env) ──

_team = team_stage_corner(scene_id="living-room-v1", target_object="asset_2_2",
                          goal_x=-2.0, goal_y=-2.0, goal_radius=0.6)
_team.slug = "team-stage-corner"

taskset = Taskset("worldsim-team", [_team])
