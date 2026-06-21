# Worksite

**Where AI learns to run the floor.**

Worksite is a simulation-to-training platform for benchmarking how well AI coordinates fleets of real-world robots. It ships HUD-native environments, construction fleet tasks with explainable metrics, optional Fireworks GRPO training, and a static demo site in the repo root.

## Quick start

```bash
uv sync
source .venv/bin/activate
hud eval environment/construction_env.py
```

Copy `.env.example` to `.env` and fill in keys as needed (never commit `.env`).

## Key features

- **`construction-v1`** — heterogeneous robot fleet on a construction site (`environment/construction_env.py`, `sim/construction_world.py`)
- **Four tasks** — `block_line`, `parallel_supply`, `heavy_relay`, `full_coordination` with partial-credit rewards (blocks placed, patrol coverage, fleet utilization, collision safety)
- **Fleet metrics** — graded rollouts with an explainable reward breakdown via HUD
- **Fireworks RL** — `python scripts/construction_rl_train.py` for GRPO calibration / training
- **Exa search proxy** — `serve/exa_search.py` for web search in agent loops
- **Demo site** — open [`index.html`](index.html) locally (uses `media/`, `site.css`, `site.js`; no build step)

Reference agents and recorders live under `examples/` and `scripts/`. Living-room planning, coop, and team envs are included for multi-agent benchmarks.

## Scene assets

Large scene meshes are **gitignored** (`scenes/**/scene.xml`, scene `*.obj` / `*.stl`, and heavy textures). The repo includes metadata, robot XML, code, and `media/` clips only. Download or regenerate the construction scene assets separately and place them under `scenes/construction-v1/` before running the construction env locally.

## Also in this repo

The upstream WorldSim template pieces remain: tabletop LLM tasks, VLA eval (`run_vla.py`), and additional scenes under `scenes/`. See `environment/` and `scripts/check_setup.py` for broader eval commands.
