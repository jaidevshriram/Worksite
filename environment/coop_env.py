"""Cooperative multi-agent planning environment.

Two embodied agents (A, B) share a living room and a skill API parameterized by
`agent` (see sim/coop_world.py). The `tidy-room` task asks for three placements:

    side table (asset_2_2)  -> SW corner   (1 carrier)
    lamp       (asset_5_5)  -> NW corner   (1 carrier)
    TV console (asset_6_6)  -> NE corner   (HEAVY: 2 carriers)

The TV cannot be moved until BOTH agents grab it, so a single-agent plan tops out
at 2/3. A good multi-agent plan parallelizes the two light items, then both agents
converge to co-carry the TV. Reward = mean per-object score (distance-shaped +
landed), so the breakdown shows exactly which placements succeeded.

Serve it like any environment: `hud eval environment/coop_env.py`.
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
from sim import coop_world  # noqa: E402

HOST = "127.0.0.1"
SKILLS = ("walk_to(agent, x, y), grab(agent, object_name), carry_to(agent, x, y), "
          "release(agent), say(agent, text), get_world_state(), render()")

# The task spec: object -> goal corner, radius, and how many agents must carry it.
PLACEMENTS = [
    {"object": "asset_2_2", "goal": [-2.0, -2.0], "radius": 0.6, "carriers": 1, "label": "side table"},
    {"object": "asset_5_5", "goal": [2.0, -2.0], "radius": 0.6, "carriers": 1, "label": "lamp"},
    {"object": "asset_6_6", "goal": [2.0, 2.0], "radius": 0.7, "carriers": 2, "label": "TV console (heavy)"},
]
AGENTS = {"A": [-0.8, -0.6], "B": [0.8, -0.6]}

env = Environment(name="worldsim-living-room-coop")
_server_state: dict[str, object] = {}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


async def _ensure_server(timeout: float = 30.0) -> str:
    if "url" in _server_state:
        return str(_server_state["url"])
    port = _free_port()

    def _run() -> None:
        asyncio.run(coop_world.server.run_async(
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
    raise RuntimeError(f"coop skill server never listened on {HOST}:{port}")


@env.initialize
async def _up() -> None:
    url = await _ensure_server()
    env.add_capability(Capability.mcp(name="skills", url=url))


@env.shutdown
async def _down() -> None:
    pass


@env.template(id="tidy-room",
              description="Two agents tidy the room; one item is too heavy for one agent.")
async def tidy_room(scene_id: str = "living-room-v1"):
    coop_world.reset_world(scene_id, placements=PLACEMENTS, agents=AGENTS)
    reach = coop_world.REACH
    goals_txt = "; ".join(
        f"{p['label']} '{p['object']}' -> ({p['goal'][0]:.1f}, {p['goal'][1]:.1f}) "
        f"[{p['carriers']} carrier{'s' if p['carriers'] > 1 else ''}]"
        for p in PLACEMENTS)
    yield (
        f"You direct TWO humanoid agents, 'A' and 'B', tidying a living room through "
        f"high-level skills. Every skill takes the agent id as its first argument.\n"
        f"Skills: {SKILLS}.\n"
        f"GOALS (place each object within its goal radius): {goals_txt}.\n"
        f"Rules: grab(agent, obj) needs that agent within {reach:.1f} m of the object with free "
        f"hands. A HEAVY object (carriers > 1) will NOT move with carry_to until that many agents "
        f"have grabbed it - so coordinate both agents onto the heavy item. walk_to fails while an "
        f"agent is holding something. Work the two light items in parallel, then co-carry the heavy one.\n"
        f"Start with get_world_state(). Verify with get_world_state() / render() before finishing."
    )

    snap = coop_world.world_snapshot()
    subs, lines, total = [], [], 0.0
    for p in PLACEMENTS:
        name = p["object"]
        o = snap["objects"].get(name, {})
        dist = o.get("distance_to_goal", float("inf"))
        initial = snap["initial_distance"].get(name, 1.0) or 1.0
        progress = max(0.0, min(1.0, 1.0 - dist / initial)) if dist != float("inf") else 0.0
        placed = bool(o.get("in_goal", False))
        # a correct placement is full credit; partial credit only for unplaced objects.
        score = 1.0 if placed else 0.4 * progress
        total += score
        subs.append(SubScore(name=f"placed_{name}", weight=round(1.0 / len(PLACEMENTS), 4),
                             value=round(score, 4)))
        lines.append(f"{p['label']}: dist={dist:.2f}m placed={'YES' if placed else 'NO'}"
                     f"{' (needs 2 agents)' if p['carriers'] > 1 else ''}")

    reward = total / len(PLACEMENTS)
    all_placed = all(snap["objects"].get(p["object"], {}).get("in_goal", False) for p in PLACEMENTS)
    yield EvaluationResult(
        reward=round(reward, 4),
        done=True,
        content=f"{'ALL PLACED' if all_placed else 'INCOMPLETE'} | steps={snap.get('steps')} | "
                + " | ".join(lines),
        subscores=subs,
    )


_tidy = tidy_room(scene_id="living-room-v1")
_tidy.slug = "tidy-room"
taskset = Taskset("worldsim-living-room-coop", [_tidy])
