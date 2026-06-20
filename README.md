<h1 align="center">
  <img src="antim.png" alt="Antim Labs" height="30"> &nbsp;&nbsp;×&nbsp;&nbsp; <img src="hud_logo.svg" alt="HUD" height="28"> &nbsp;&nbsp;|&nbsp;&nbsp; World Sim RL Environment Template
</h1>

Worldsim robotics tasks on the **HUD SDK**. A Newton physics scene is a live
environment with a tool API; you drive it with an LLM agent or a VLA policy and it
scores the rollout.

> **reset** the scene  →  **drive** the gripper through the tool API (or a VLA policy)  →  **grade** from sim state

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
worldsim-template/
├── environment/   the envs + tasks: env.py (LLM), vla_env.py (VLA), tasks.py
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
