# Worksite — Multi-Agent Construction RL Environment

**Where AI learns to run the floor.**

Worksite is a simulation-to-training platform for benchmarking how AI coordinates fleets of robots on a construction site — from HUD eval rollouts and scripted reference agents to optional Fireworks GRPO training and a static demo site.

**Powered by:** HUD · Fireworks · Exa · World Labs · MuJoCo

## Quick start

```bash
uv sync
source .venv/bin/activate
hud set HUD_API_KEY=...
hud eval environment/construction_env.py
python examples/construction_agents.py
python scripts/construction_rl_train.py --calibrate-only --calibration-backend inference
```

Copy `.env.example` to `.env` and fill in keys as needed (never commit `.env`).

## Tasks

Mixed fleet: 4 Unitree G1 humanoids + 2 Boston Dynamics Spots on `construction-v1`.

| Task | Goal |
|------|------|
| `block_line` | Each G1 moves one block into a line; Spots clear the path. |
| `parallel_supply` | Two G1 teams fill two staging areas; rewards peak simultaneous activity. |
| `heavy_relay` | A two-carrier heavy block is relayed in two legs; Spots patrol the route. |
| `full_coordination` | Four blocks to a diamond layout plus a full Spot site patrol. |

## Fleet metrics

Every rollout reports **fleet_utilization** (share of agent×step slots spent acting), **peak_simultaneous** (max agents active in one step), and **collision_events** (proximity violations). Parallel plans score higher than strictly sequential ones.

## Demo site

Open [`index.html`](index.html) locally — uses `media/`, `site.css`, and `site.js` (no build step).

## Exa spec lookup

Run the search proxy for agent spec lookup:

```bash
uvicorn serve.exa_search:app --port 7823
```

Requires `EXA_API_KEY` in `.env`. Query `GET /search?q=...`.

## Scene assets

Large scene meshes are **gitignored** (`scenes/**/scene.xml` is ~260MB and excluded, along with scene `*.obj` / `*.stl` and heavy textures). The repo ships metadata, robot XML, code, and `media/` clips. Download or regenerate construction scene assets under `scenes/construction-v1/` before running locally.

## Layout

```
environment/   # HUD task definitions (construction_env.py)
sim/           # Construction world sim + metrics
scripts/       # RL training, recorders
examples/      # Reference construction agents
media/         # Demo clips and screenshots
serve/         # Exa search proxy (exa_search.py)
index.html     # Static demo site
```

**Repository:** [github.com/jaidevshriram/Worksite](https://github.com/jaidevshriram/Worksite)
