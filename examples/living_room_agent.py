"""Drive the living-room planning task with a scripted or LLM agent.

    # 1) Scripted planner (default): emits the obvious skill plan, scores 1.0, no model.
    python examples/living_room_agent.py

    # 2) LLM planner: the model decides which skills to call (routed via the HUD gateway).
    python examples/living_room_agent.py --llm

Both drive the SAME high-level skill API (walk_to / pick / place) over MCP - the only
difference is who emits the plan. This is the point: reliability comes from the LLM
*composing reliable skills*, not from low-level control.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from hud import LocalRuntime
from hud.agents import create_agent
from hud.agents.base import Agent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from environment.living_room_env import move_side_table  # noqa: E402

GOAL = (-2.0, -2.0)  # must match the task args below


def _as_dict(result: Any) -> dict:
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict):
        return sc.get("result", sc)
    content = getattr(result, "content", None)
    if content:
        text = getattr(content[0], "text", None)
        if text:
            return json.loads(text)
    return {}


class ScriptedPlanner(Agent):
    """The deterministic plan an LLM/Foreman would emit: walk -> pick -> walk -> place."""

    async def __call__(self, run) -> None:
        skills = await run.client.open("skills")

        async def call(name: str, **kw) -> dict:
            return _as_dict(await skills.call_tool(name, kw))

        st = await call("get_world_state")
        t = st["objects"][st["target_object"]]
        gx, gy = st["goal"]["x"], st["goal"]["y"]

        await call("walk_to", x=t["x"], y=t["y"])      # go to the table
        await call("pick", object_name=st["target_object"])
        await call("walk_to", x=gx, y=gy)              # carry to the corner
        await call("place", x=gx, y=gy)                # set it down

        final = await call("get_world_state")
        run.trace.content = (f"placed {final['target_object']} at corner; "
                             f"dist={final['target_distance_to_goal']} m, "
                             f"landed={final['target_in_goal']}")


async def main_async(args: argparse.Namespace) -> None:
    task = move_side_table(scene_id="living-room-v1", target_object="asset_2_2",
                           goal_x=GOAL[0], goal_y=GOAL[1], goal_radius=0.6)
    agent = create_agent(args.model) if args.llm else ScriptedPlanner()
    job = await task.run(agent, runtime=LocalRuntime(str(ROOT / "environment" / "living_room_env.py")))
    print(f"reward: {job.reward}   (1.0 = side table landed within 0.6 m of the corner)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Drive the living-room planning task.")
    ap.add_argument("--llm", action="store_true", help="let an LLM emit the skill plan (via the HUD gateway)")
    ap.add_argument("--model", default="claude-sonnet-4-5", help="model id for --llm")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
