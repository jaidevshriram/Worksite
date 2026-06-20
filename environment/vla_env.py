"""Worldsim VLA environment - the Franka/LIBERO pick task over the `robot` capability.

The VLA counterpart of `env.py` (which serves the LLM tool tasks over an `mcp`
capability). The Newton sim + Franka bridge run in their own process (`sim/host.py`,
spawned in `@env.initialize`) and are served over the `robot` (openpi/0) protocol: an
agent connects, reads this env's contract, and runs its policy in a closed
`observe -> infer -> act` loop the framework drives. The separate process is what lets
the live viewer own the main thread; the env drives the bridge through a remote
`RobotEndpoint`. Sim, scoring, and recording all live in the bridge (sim/franka_bridge.py).

Serve it like any env: `python run_vla.py` (LocalRuntime), a container CMD, or
`python -m hud.environment.server environment/vla_env.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hud.environment import Environment
from hud.environment.robot import RobotEndpoint
from hud.graders import EvaluationResult, SubScore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.host import SimHost  # noqa: E402

CONTRACT = json.loads((Path(__file__).resolve().parents[1] / "contracts" / "franka_libero.json").read_text())

# The bridge lives in the sim process; the endpoint drives it remotely over its
# control RPC (reset/result/url) while the agent's openpi loop hits the bridge WS.
sim_host = SimHost("robot")
endpoint = RobotEndpoint.remote(SimHost.host, sim_host.port)

env = Environment(name="worldsim-vla")


@env.initialize
async def _up() -> None:
    await sim_host.start()
    await endpoint.connect()
    env.add_capability(await endpoint.capability(contract=CONTRACT))


@env.shutdown
async def _down() -> None:
    await endpoint.close()
    await sim_host.stop()


@env.template(id="vla-pick", description="Pick-and-lift for a VLA policy on a Franka scene.")
async def vla_pick(
    scene_id: str = "franka-libero-v1",
    target_object: str = "block",
    instruction: str = "pick up the red block",
    lift_height: float = 0.55,
    seed: int = 0,
    max_steps: int = 200,
):
    """The franka-libero-v1 / pick-the-red-block task.

    The bridge resets the scene and the prompt is just the instruction (the VLA's
    `task`); the agent drives the robot loop; grading is lift-progress shaping +
    binary success, emitted as self-consistent subscores.
    """
    prompt = await endpoint.reset(
        scene_id=scene_id, target_object=target_object, instruction=instruction,
        lift_height=lift_height, seed=seed, max_steps=max_steps,
    )
    yield {"prompt": prompt}

    res = await endpoint.result()
    progress, success = res["lift_progress"], res["success"]
    # reward = weighted sum of the subscores below (self-consistent).
    yield EvaluationResult(
        reward=round(res["score"], 4),
        done=True,
        content=f"VLA pick '{target_object}': z={res['final_z']:.4f} / {lift_height:.4f} "
                f"({progress * 100:.1f}% progress). {'SUCCESS' if success else 'INCOMPLETE'}",
        subscores=[
            SubScore(name="lift_progress", weight=0.5, value=round(progress, 4)),
            SubScore(name="binary_success", weight=0.5, value=1.0 if success else 0.0),
        ],
    )
