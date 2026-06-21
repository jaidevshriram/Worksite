<h1 align="center">
  <img src="antim.png" alt="Antim Labs" height="30"> &nbsp;&nbsp;×&nbsp;&nbsp; <img src="hud_logo.svg" alt="HUD" height="28"> &nbsp;&nbsp;|&nbsp;&nbsp; Worksite — Multi-Agent Construction RL Environment
</h1>

<p align="center"><b>Worksite</b> — <i>Where AI learns to run the floor.</i><br/>
A simulation-to-training platform that benchmarks how well AI coordinates fleets of real-world robots.<br/>
Powered by <b>HUD</b> · <b>Fireworks</b> · <b>Exa</b> · <b>World Labs</b> · <b>MuJoCo</b></p>

Worksite is a HUD-native RL environment where a fleet of heterogeneous robots
cooperates on a construction site. A physics scene is a live environment with a
tool API; you drive it with an LLM agent (or a VLA policy) and it scores the rollout.

> **reset** the scene  →  **drive** the fleet through the tool API (or a VLA policy)  →  **grade** from sim state

There is also a self-contained **website** at the repo root — open
[`index.html`](index.html) in any browser (an engineering-blueprint landing page with the
demo videos, renders, the pipeline, and the measured results). It needs no build step and
uses relative `media/...` paths, so it also works on GitHub Pages.

## The Worksite construction environment

`environment/construction_env.py` runs the `worldsim-construction` taskset on the
`construction-v1` scene: a fleet of heterogeneous embodiments moves bricks along a
supply line to a build pad while patrolling the site, and HUD grades the **real** task
(blocks placed, patrol coverage, fleet utilization, collision safety) into a reward with
an explainable breakdown. Four tasks ship — `block_line`, `parallel_supply`,
`heavy_relay`, `full_coordination` — each with its own partial-credit reward weights.

```bash
hud eval environment/construction_env.py            # run the construction taskset
python examples/construction_agents.py              # scripted fleet reference
python scripts/construction_rl_train.py             # Fireworks GRPO calibration / training loop
```

> ### ⚠️ Large scene assets are gitignored
> The heavy `construction-v1` mesh and textures — most notably
> `scenes/construction-v1/scene.xml` (~260 MB) and the scene's `*.obj` / `*.stl` /
> texture `*.png` files — are **excluded from this repo** via `.gitignore` to keep it
> under GitHub's 100 MB file limit and lean (no Git LFS). Source code, tasks, the
> website, and the small `media/` clips are all included. **Regenerate or download the
> scene meshes separately** and drop them under `scenes/construction-v1/` (matching the
> `scene.xml` + `metadata.json` layout) before running the construction env locally.

- **LLM tool tasks** - four manipulation tasks (`open-drawer`, `pick-object`,
  `move-object`, `force-grasp`) on `tabletop-v1`, served as an `mcp` capability.
- **VLA policy eval** - `franka-libero-v1` / "pick up the red block", served as a
  `robot` (openpi/0) capability: bring a policy, get a success rate.

The sim runs in its own process (the env spawns it); set `WORLDSIM_VIEWER=1` to watch
any run in a live 3D window.

**Scenes are environments.** Any folder under `scenes/` is a live env you can
reset/step/render/score - bring your own scene + reward to make a new benchmark.
Generate new scenes at **[gizmo.antimlabs.com](https://gizmo.antimlabs.com)** (the same
Gizmo engine this runs on) and drop them under `scenes/` as a new folder, matching the
bundled scenes' layout (`scene.xml` + `metadata.json`).

## Tasks

Four LLM tool-control tasks on `tabletop-v1`, served as an `mcp` capability:

| Task | What the agent does | Scored on |
|------|---------------------|-----------|
| `move-object` | push the mug to a target (x, y, z) on the table | distance to goal + reached |
| `pick-object` | grasp the mug and lift it above a height | lift progress + reached |
| `force-grasp` | grip the mug firmly (>= 0.5 N per finger) and hold | grip quality + lifted |
| `open-drawer` | grasp the under-table handle and pull the drawer out | drawer travel + opened |

Each grades from sim state with partial credit, so the reward breakdown always explains the
score. Plus a VLA task, `vla-pick` on `franka-libero-v1` ("pick up the red block"), over the
`robot` (openpi/0) capability.

### Planning task (symbolic skills, no physics)

`move-side-table` on `living-room-v1` grades an LLM's **plan**, not low-level control. The
agent drives a high-level skill API served as an `mcp` capability - `walk_to(x,y)`,
`pick(object)`, `place(x,y)`, `get_world_state()`, `render()` - with explicit
**preconditions + effects** over a physics-free world state (`sim/skill_world.py`). Reward =
whether the side table is landed in the target corner (with distance-shaped partial credit).
This is the right layer for benchmarking planning / multi-agent orchestration: reliability
comes from the LLM composing reliable skills, not from contact dynamics.

### Cooperative multi-agent task

`tidy-room` on `living-room-v1` (`environment/coop_env.py`) puts **two agents** (`A`, `B`) in
the room with skills parameterized by agent id (`walk_to`/`grab`/`carry_to`/`release`/`say`).
Three objects must be placed in three corners; one (the TV console) is **heavy and needs both
agents to grab it before it can be carried**. So coordination is *necessary* - a single-agent
plan caps at 0.667 (2/3), while a plan that parallelizes the light items and co-carries the
heavy one scores 1.0. Reward is the mean per-object score, so the breakdown shows which
placements (and the cooperation) succeeded.

### Multi-agent task (team coordination)

`team-stage-corner` on `living-room-v1` makes one LLM the controller of a three-agent team
(alpha, bravo, carol) over agent-scoped skills (`sim/multi_skill_world.py`). It can only be
solved by coordinating: the side table is **heavy** (a single `pick` is refused - two agents
must `joint_lift` it together) and the corner is **blocked** by a cushion that must be cleared
first. So the winning plan needs decomposition, role assignment, ordering, and a synchronized
joint action. Reward = table landed in the corner (0.5) + corner cleared (0.2) + distance
shaping (0.3).

## Install

Python 3.12. `uv` installs everything from the lockfile, including the bundled Newton
wheel in `wheels/` (wired up via `[tool.uv.sources]`).

```bash
uv sync                              # all deps incl. the bundled Newton wheel (add --extra viewer for the live 3D viewer)
source .venv/bin/activate            # so the python / hud commands below resolve to this env
hud set HUD_API_KEY=your-key-here    # routes models via the gateway, traces to the platform
```

## Run

```bash
# readiness check - boots the sim + grades one scripted rollout (first reset compiles Warp, ~1 min)
python scripts/check_setup.py

# LLM tasks against the platform
hud eval environment/tasks.py claude --all --group 3

# planning task (symbolic skills, no GPU/physics boot) - trace shows the skill plan
hud eval environment/living_room_env.py claude
python examples/living_room_agent.py        # scripted planner, deterministic (1.0)
python examples/living_room_agent.py --llm  # an LLM emits the skill plan

# cooperative multi-agent task (two agents; the heavy item needs both)
hud eval environment/coop_env.py claude --max-steps 60
python examples/coop_agents.py              # scripted multi-agent planner (1.0)
python examples/coop_agents.py --llm        # one LLM orchestrates agents A and B

# multi-agent team task (needs more steps: clear corner + two-agent joint lift)
hud eval environment/team_env.py claude --max-steps 40
python examples/team_agent.py               # scripted team plan, deterministic (1.0)
python examples/team_agent.py --llm         # an LLM coordinates the three agents

# the example agent on move-object
python examples/example_agent.py            # scripted, deterministic (~1.0)
python examples/example_agent.py --llm      # an LLM drives the tools

# VLA wire check - no GPU, no model
python run_vla.py --noop --group 1

# watch any run live in a 3D window (needs a display)
WORLDSIM_VIEWER=1 hud eval environment/tasks.py claude --group 1
```

## VLA policy eval

The pi0.5 baseline, policy local (needs a GPU + `uv sync --extra vla`):

```bash
python run_vla.py --group 10
```

Or keep the policy on a GPU box and run the sim + loop on a CPU-only machine - only
the observation→action forward crosses the network:

```bash
modal run serve/pi05_modal.py               # managed GPU; prints ws://HOST:PORT
# or your own GPU box (uv sync --extra robot --extra vla):
python serve/policy_server.py --port 8000   # prints ws://HOST:PORT

python run_vla.py --remote HOST:PORT --group 10        # on the eval machine
python run_vla.py --group 5 --record ./datasets/pick   # eval + record a dataset
```

Bring your own policy: copy the `CustomModel`/`CustomAgent` scaffold in
`agents/vla_agent.py`, fill in `infer()`, then
`python run_vla.py --agent agents.vla_agent:CustomAgent --group 10`.

## Layout

```
Worksite/
├── index.html     the static blueprint website (open directly; site.css / site.js)
├── media/         demo videos + render stills used by the website
├── environment/   the envs + tasks: construction_env.py, env.py (LLM), vla_env.py (VLA), tasks.py
├── examples/      example_agent.py - an agent on the tool API (scripted + LLM)
├── agents/        vla_agent.py - VLA policies: pi0.5 baseline + bring-your-own
├── serve/         policy servers for a GPU box (policy_server.py, pi05_modal.py)
├── run_vla.py     the VLA eval runner
├── scripts/       check_setup.py - readiness check
├── contracts/     franka_libero.json - the VLA env↔policy contract
├── scenes/        each folder is a scene (tabletop-v1, franka-libero-v1)
├── sim/           the Newton sim (server.py) + EE control, the bridge, and host.py   # internals
└── wheels/        the Newton engine, pre-built                                        # internals
```
