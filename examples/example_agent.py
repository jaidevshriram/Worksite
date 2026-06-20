"""Example agents driving the sim's `mcp` capability directly - no policy server.

Shows that a Worldsim scene is just a live environment with a tool API. Two ways to
drive the `move-object` task (push the mug to a goal spot on the table):

    # 1) Scripted agent (default): deterministic, scores ~1.0, no model.
    python examples/example_agent.py

    # 2) LLM agent: Claude decides which tools to call (routed via the HUD gateway).
    python examples/example_agent.py --llm

Both drive the SAME loop a policy eval uses - reset (env-side), look, act, score  - 
through MCP tool calls on the sim capability. The scripted path is a custom
`Agent`; the LLM path is just `create_agent(...)`, which wires the tools itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

from hud import LocalRuntime
from hud.agents.base import Agent
from hud.agents import create_agent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from environment.env import move_object  # noqa: E402

GOAL = (-0.2, 0.0, 0.75)  # must match the move_object task args below


def _as_dict(result: Any) -> dict:
    """Unwrap an MCPToolResult into a plain dict."""
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict):
        return sc.get("result", sc)
    content = getattr(result, "content", None)
    if content:
        text = getattr(content[0], "text", None)
        if text:
            return json.loads(text)
    return {}


class ScriptedAgent(Agent):
    """Closed-loop control through the generic tool API: read state, command raw
    actuator velocities with step(), repeat. tabletop-v1's actuators are
    [vx, vy, vz, vyaw, finger_left, finger_right]."""

    async def __call__(self, run) -> None:
        sim = await run.client.open("sim")

        async def call(name: str, **kwargs) -> dict:
            return _as_dict(await sim.call_tool(name, kwargs))

        async def gripper() -> list[float]:
            return (await call("get_state"))["gripper_position"]

        async def mug() -> dict:
            return (await call("get_object_state", object_name="mug"))["position"]

        async def servo_to(tx, ty, tz, tol=0.005, rounds=60) -> None:
            for _ in range(rounds):
                x, y, z = await gripper()
                ex, ey, ez = tx - x, ty - y, tz - z
                if max(abs(ex), abs(ey), abs(ez)) < tol:
                    return
                vx, vy, vz = (max(-0.3, min(0.3, 8.0 * e)) for e in (ex, ey, ez))
                for _ in range(25):
                    await call("step", action=[vx, vy, vz, 0, 0.04, 0.04])

        async def push_axis(axis: str, target: float, stop_margin=0.012) -> dict:
            m = await mug()
            sign = 1 if target > m[axis] else -1
            dx, dy = (-0.085 * sign, 0.0) if axis == "x" else (0.0, -0.085 * sign)
            await servo_to(m["x"] + dx, m["y"] + dy, 0.93)   # above, beside the mug
            await servo_to(m["x"] + dx, m["y"] + dy, 0.845)  # down to push height
            for _ in range(220):
                m = await mug()
                if sign * (target - m[axis]) <= stop_margin:
                    break
                v = [0.0, 0.0]
                v[0 if axis == "x" else 1] = 0.10 * sign
                for _ in range(25):
                    await call("step", action=[v[0], v[1], 0, 0, 0.04, 0.04])
            x, y, _ = await gripper()
            await servo_to(x, y, 1.0, tol=0.02)              # retreat
            return await mug()

        await call("close_gripper")                          # fingers together = pusher
        await push_axis("x", GOAL[0])
        m = await push_axis("y", GOAL[1])
        if abs(m["x"] - GOAL[0]) > 0.035:                    # touch-up if x drifted
            m = await push_axis("x", GOAL[0])
        dist = math.dist((m["x"], m["y"], m["z"]), GOAL)
        run.trace.content = f"moved mug to ({m['x']:.3f}, {m['y']:.3f}, {m['z']:.3f}); dist={dist:.4f}m"


async def main_async(args: argparse.Namespace) -> None:
    task = move_object(scene_id="tabletop-v1", target_object="mug",
                       goal_x=GOAL[0], goal_y=GOAL[1], goal_z=GOAL[2])
    agent = create_agent(args.model) if args.llm else ScriptedAgent()
    job = await task.run(agent, runtime=LocalRuntime(str(ROOT / "environment" / "env.py")))
    print(f"reward: {job.reward}   (1.0 = mug within 5 cm of the goal)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Drive the sim with a scripted or LLM agent.")
    ap.add_argument("--llm", action="store_true", help="let an LLM drive the tools (via the HUD gateway)")
    ap.add_argument("--model", default="claude-sonnet-4-5", help="model id for --llm")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
