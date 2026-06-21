"""Drive the cooperative two-agent tidy-room task: scripted or LLM orchestrator.

    # 1) Scripted multi-agent planner (default): divides labor + co-carries the heavy
    #    item. Deterministic, scores 1.0, no model.
    python examples/coop_agents.py

    # 2) LLM orchestrator: one model commands both agents A and B via the skill API.
    python examples/coop_agents.py --llm

Both drive the SAME skill API (walk_to / grab / carry_to / release, each taking an
agent id). The point: the heavy TV is unmovable by one agent, so a passing plan MUST
coordinate both - which is what makes this a meaningful multi-agent test.
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
from environment.coop_env import PLACEMENTS, tidy_room  # noqa: E402


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


class ScriptedCoopPlanner(Agent):
    """Centralized planner: A and B each grab a light item and place it in parallel,
    then both converge on the heavy TV and co-carry it."""

    async def __call__(self, run) -> None:
        skills = await run.client.open("skills")

        async def call(name: str, **kw) -> dict:
            return _as_dict(await skills.call_tool(name, kw))

        st = await call("get_world_state")
        objs = st["objects"]
        light = [p for p in PLACEMENTS if p["carriers"] == 1]
        heavy = next(p for p in PLACEMENTS if p["carriers"] >= 2)

        async def solo(agent: str, p: dict) -> None:
            o = objs[p["object"]]
            await call("walk_to", agent=agent, x=o["x"], y=o["y"])
            await call("grab", agent=agent, object_name=p["object"])
            await call("carry_to", agent=agent, x=p["goal"][0], y=p["goal"][1])
            await call("release", agent=agent)

        # divide the two light items between A and B
        await solo("A", light[0])
        await solo("B", light[1])

        # both converge on the heavy item, both grab, then co-carry
        ho = objs[heavy["object"]]
        await call("walk_to", agent="A", x=ho["x"], y=ho["y"])
        await call("walk_to", agent="B", x=ho["x"], y=ho["y"])
        await call("grab", agent="A", object_name=heavy["object"])
        await call("grab", agent="B", object_name=heavy["object"])
        await call("carry_to", agent="A", x=heavy["goal"][0], y=heavy["goal"][1])
        await call("release", agent="A")
        await call("release", agent="B")

        final = await call("get_world_state")
        placed = {n: o.get("in_goal") for n, o in final["objects"].items() if "in_goal" in o}
        run.trace.content = f"placements: {placed}"


async def main_async(args: argparse.Namespace) -> None:
    task = tidy_room(scene_id="living-room-v1")
    agent = create_agent(args.model) if args.llm else ScriptedCoopPlanner()
    job = await task.run(agent, runtime=LocalRuntime(str(ROOT / "environment" / "coop_env.py")))
    print(f"reward: {job.reward}   (1.0 = all three objects placed; TV requires both agents)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Drive the cooperative tidy-room task.")
    ap.add_argument("--llm", action="store_true", help="one LLM orchestrates both agents (via the HUD gateway)")
    ap.add_argument("--model", default="claude-sonnet-4-5", help="model id for --llm")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
