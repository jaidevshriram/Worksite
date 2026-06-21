"""Drive the multi-agent team task with a scripted or LLM controller.

    # 1) Scripted team plan (default): coordinated, scores 1.0, no model.
    python examples/team_agent.py

    # 2) LLM controller: the model decides how to coordinate the three agents.
    python examples/team_agent.py --llm

Both drive the SAME agent-scoped skill API over MCP. The scripted path encodes the
coordinated plan (clear the corner + two-agent joint lift); the LLM path must figure
out that coordination itself - which is exactly what the task measures.
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
from environment.team_env import team_stage_corner  # noqa: E402

GOAL = (-2.0, -2.0)


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


class ScriptedTeam(Agent):
    """The coordinated plan a good controller should find:
    carol clears the corner; alpha+bravo joint-lift the heavy table and carry it in."""

    async def __call__(self, run) -> None:
        skills = await run.client.open("skills")

        async def call(name: str, **kw) -> dict:
            return _as_dict(await skills.call_tool(name, kw))

        st = await call("get_world_state")
        gx, gy = st["goal"]["x"], st["goal"]["y"]
        target = st["target_object"]
        table = st["objects"][target]
        blocker = (st["blockers_in_goal"] or [None])[0]

        # carol clears the corner: pick the blocking cushion, drop it well outside the goal.
        if blocker:
            b = st["objects"][blocker]
            await call("walk_to", agent="carol", x=b["x"], y=b["y"])
            await call("pick", agent="carol", object_name=blocker)
            await call("walk_to", agent="carol", x=0.5, y=0.5)
            await call("place", agent="carol", x=0.5, y=0.5)

        # alpha + bravo co-lift the heavy table and carry it into the corner.
        await call("walk_to", agent="alpha", x=table["x"], y=table["y"])
        await call("walk_to", agent="bravo", x=table["x"], y=table["y"])
        await call("joint_lift", agent_a="alpha", agent_b="bravo", object_name=target)
        await call("walk_to", agent="alpha", x=gx, y=gy)  # moves both carriers + table
        await call("joint_place", agent_a="alpha", agent_b="bravo", x=gx, y=gy)

        final = await call("get_world_state")
        run.trace.content = (f"table dist={final['target_distance_to_goal']} m, "
                             f"landed={final['target_in_goal']}, cleared={final['corner_cleared']}")


async def main_async(args: argparse.Namespace) -> None:
    task = team_stage_corner(scene_id="living-room-v1", target_object="asset_2_2",
                             goal_x=GOAL[0], goal_y=GOAL[1], goal_radius=0.6)
    agent = create_agent(args.model) if args.llm else ScriptedTeam()
    job = await task.run(agent, runtime=LocalRuntime(str(ROOT / "environment" / "team_env.py")))
    print(f"reward: {job.reward}   (1.0 = corner cleared AND heavy table landed in it)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Drive the multi-agent team task.")
    ap.add_argument("--llm", action="store_true", help="let an LLM coordinate the team (via the HUD gateway)")
    ap.add_argument("--model", default="claude-sonnet-4-5", help="model id for --llm")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
