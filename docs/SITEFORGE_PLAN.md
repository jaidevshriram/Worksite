# SiteForge — Hackathon Plan & Discussion Overview

> **HUD × YC — Frontier RL Environments Hackathon**
> An RL environment for **multi-agent autonomous construction**, built on the HUD Worldsim Robotics track (HUD × Antim Labs).
> This doc captures the full thinking: the literature that grounds it, the pitch, the phased build plan, sponsor integration, and the repo-scope findings that constrain what we build.

---

## 0. TL;DR

We are building **SiteForge**: a HUD-native RL environment where a fleet of heterogeneous robots cooperates to build a small structure (a toy block house). HUD measures the *real* task — using fleet resources efficiently to construct the target — and we **hill-climb** an agent toward it (calibration → GRPO via Fireworks).

- **Domain:** physical-world construction, framed around a manufacturing/building shortage.
- **Core thesis:** the future of construction robotics is **multi-agent fleets** of different embodiments, not one humanoid. We need RL environments that measure fleet safety, utilization, and coordination over time.
- **Why it wins:** HUD is the *spine* (env, tasks, rewards, traces, graders, training-ready trajectories), not bolted on. It's an RL environment + agentic eval, not "LLM calls a tool."

---

## 1. Motivation & Pitch

### The problem

- There is a **manufacturing / construction shortage**. Projects take too long and cost too much today.
- Most robotics work today is framed around **the home and a single agent**. But you will not build a house with one humanoid — you'll use **an army of them**.
- The real future is **fleets of heterogeneous embodiments**: autonomous vehicles, robodogs, humanoids, wheeled manipulators, cranes — each with different strengths.
- Operating such a fleet means caring about **fleet safety, fleet utilization, embodiment specialization, and coordination over a long horizon**.

### The thesis

> Before autonomous agents can run real construction sites, we need RL environments where safety, cost, time, physical uncertainty, and multi-robot coordination actually interact. **SiteForge is that environment.** HUD records every decision, grades the physical outcome (did the structure get built, safely, efficiently?), and lets us hill-climb an agent toward competence.

### What we explicitly avoid

Shallow framings like "use a VLM to classify damage" or "detect hazards." SiteForge is the **training/evaluation arena** for autonomous physical-world operation, starting with construction.

---

## 2. Literature Context

Two papers frame where SiteForge sits. They probe **opposite ends** of the multi-agent coordination problem, and SiteForge occupies the missing middle.

### 2.1 RoboFactory (ICCV 2025) — arXiv:2503.16408

*Qin et al. — "Exploring Embodied Agent Collaboration with Compositional Constraints"*

- **Problem:** existing methods can't automatically generate **safe, efficient training data** for multi-robot embodied systems.
- **Key idea — compositional constraints** (three types that must all hold for a valid collaborative trajectory):
  - **Logical** — prevent wrong interaction forms (e.g. grabbing a camera by the lens).
  - **Spatial** — prevent collisions between arms.
  - **Temporal** — prevent inefficient scheduling / unnecessary waiting.
- **System:** `RoboBrain` (LLM planner: decomposes the task, emits textual constraints + trajectories) + `RoboChecker` (turns constraints into executable interfaces, validates trajectories, feeds failures back).
- **Benchmark:** RoboFactory — first benchmark for embodied multi-agent manipulation, 11 tasks in ManiSkill, 2–4 robots.
- **Learning:** **imitation learning only** (ACT, Diffusion Policy), across Independent / Shared / Collaborative architectures. *No RL* — the contribution is the data pipeline + benchmark; IL gives a clean signal. RL is implied future work.
- **Key result:** collaborative (cross-attention) policies dominate as agent count / constraint complexity grows (e.g. ~35% vs ~18% on hard 3-agent tasks; ~3× better on 4-agent tasks).

### 2.2 CREW-Wildfire (TMLR 2025) — arXiv:2507.05178

*Hyun, Waytowich, Chen (Duke / Army Research Lab) — "Benchmarking Agentic Multi-Agent Collaborations at Scale"*

- **Problem:** LLM-based multi-agent benchmarks are small-scale, fully observable, low-complexity — they don't test real scalability/coordination.
- **Environment:** procedurally generated wildfire response — large maps, **heterogeneous agents** (drones, helicopters, bulldozers, firefighters), partial observability, stochastic dynamics, long horizons. Scales to **2000+ agents**.
- **Modular `Perception` + `Execution` modules** bridge raw sim ↔ LLM (ASCII/text summaries in; natural-language commands → action primitives out).
- **7 behavioral competencies** used as fine-grained graders: Task Designation, Agent Capitalization, Spatial Reasoning, Observation Sharing, Realtime Coordination, **Plan Adaptation**, Objective Prioritization.
- **Key findings:** state-of-the-art LLM frameworks coordinate fine on simple tasks but **fail to generalize** — weakest at Plan Adaptation and Observation Sharing. Token cost scales ~O(N²) with global-state sharing, capping realistic agent counts.

### 2.3 How they contextualize SiteForge


|                    | RoboFactory              | CREW-Wildfire        | **SiteForge (ours)**                   |
| ------------------ | ------------------------ | -------------------- | -------------------------------------- |
| Layer              | Low-level sensorimotor   | High-level strategic | **Operational (the middle)**           |
| Agents             | 2–4 robots               | up to 2000+          | 3–4 embodiments                        |
| Coordination       | Cross-attention policies | Natural language     | Foreman + capability-aware scheduling  |
| Learning           | IL only                  | LLM zero/few-shot    | **RL hill-climb (calibration → GRPO)** |
| Core gap addressed | data generation          | reasoning at scale   | **physical fleet ops as an RL env**    |


**Takeaways we steal:**

- From RoboFactory: make **compositional constraints** explicit in the reward and (later) in task generation — logical (correct grasp/placement), spatial (no collision), temporal (scheduling/makespan). Collaborative beats independent → motivates fleet coordination.
- From CREW-Wildfire: borrow the **behavioral-competency grading taxonomy** instead of a single scalar reward (utilization, spatial reasoning, plan adaptation…), and respect the **O(N²) token cost** → keep the active fleet small (3–5).

---

## 3. The Repo We're Building On

Worldsim Robotics template (HUD × Antim Labs). **Mental model:** a scene is a physics world (MJCF/USD/URDF); HUD wraps it as an *environment* exposing *tools* over MCP; an LLM agent drives the tools; a task reads final sim state → *reward + subscores*.

```
reset(scene) → drive robot via tools → grade from sim state → reward + subscores
```

**What we edit:** `environment/env.py` + `environment/tasks.py` (env + `@env.template` tasks) and `scenes/<id>/` (each folder = a world: `scene.xml` + `metadata.json`).
**Plumbing we leave alone:** `sim/host.py`, `sim/server.py` (Newton physics + FastMCP tool server), `sim/control.py` (Franka IK).

### Scene formats

MJCF (`scene.xml`) ← **use this**, USD (`scene.usd`), URDF (`scene.urdf`), or Python `build_scene.py`. Only `.xml`/`.mjcf` also load the MuJoCo names/cameras layer the graders read.

### Two findings that CONSTRAIN scope

1. **Robot control is the hard part, not geometry.** The only motion controller (`sim/control.py`) is **Franka-specific IK**, auto-activated only when exact Franka joint names are present. A dropped-in Unitree/Spot humanoid is **inert geometry with no locomotion controller** — days of work, not hours.
2. **High-level gripper tools are single-robot.** `move_gripper`/`open_gripper`/`close_gripper`/`rotate_gripper` are hardcoded to `gripper_base` and actuator indices `ctrl[0..5]`. With a second robot they silently break → multi-robot must drive raw `step(action)`.

*(Full detail in `docs/scope-guide.html`.)*

---

## 4. Locked Decisions

1. **Embodiments:** NOT real Spot/Unitree locomotion. Use **3 abstract floating manipulators with capability profiles**, themed as:
  - **Crane / Gantry** — heavy lift, large reach, slow (base blocks).
  - **Humanoid** — dexterous, medium (fiddly roof / fine pieces).
  - **Wheeled mobile manipulator** — fast, low payload (fetch/stage).
2. **Target structure:** start at **2 blocks (base + roof)**, stretch to 3 (two walls + roof). Reference: classic wooden toy block house.
3. **Multi-agent approach:** attempt true multi-gripper physics, **time-boxed to 60 min**; if it fights us, drop to the **symbolic-fleet fallback** (one physics arm; scheduler assigns blocks to named robots with capability constraints; reward credits utilization/makespan as if parallel).

---

## 5. The Reward (the "real task")

```
reward = 0.40 * structural_completion   # each block within tolerance of its target pose in the house
       + 0.20 * fleet_utilization        # work spread across robots, not one doing everything
       + 0.15 * makespan_efficiency      # parallel build beats sequential
       + 0.15 * safety_no_collision      # no two manipulators in one cell; nothing knocked over
       + 0.10 * stability                # structure still standing after a settle step
```

Same partial-credit-from-sim-state pattern the template already uses for `move-object`, composed over multiple blocks + robots. Composability is what makes it tractable.

**Hard-fail / anti-reward-hacking guards (from RoboFactory's constraint thinking):**

- Logical: can't "place" a block that isn't grasped; roof can't go before base settles.
- Spatial: two manipulators in the same cell = collision fail; knocking the structure over zeroes stability.
- Temporal: can't stall forever; makespan is bounded by an episode cap.

---

## 6. Phased Plan (≤10 hours, each phase independently demoable)

### Phase 0 — Setup (~45 min) · *Foundation* — **HUD, Antim, Modal, Fireworks**

- `uv sync`, `hud set HUD_API_KEY=...`, `python scripts/check_setup.py`, `hud eval environment/tasks.py claude --group 1`.
- Claim credits: HUD `YC-RL-HACKATHON`, Modal `SQ8-USG-5K2`, Fireworks `HUD-HACK-2026`, Daytona, Exa `HUDHACK`.
- **Milestone:** baseline tabletop eval produces a trace + reward on hud.ai. *(Repo understanding captured in `docs/scope-guide.html`.)*

### Phase 1 — Single-robot block build MVP (~2.5 h) · *Guaranteed deliverable* — **HUD + Antim**

- Copy `tabletop-v1` → `scenes/construction-v1/`: floor + build pad + `base_block`/`roof_block` (free bodies) + the proven floating gripper.
- Write a `build_house` task: reset → prompt → grade `structural_completion` (per-block closeness to target pose) + `stability` after settle.
- **Milestone:** an LLM stacks blocks into a house; HUD shows the reward breakdown. *This is "an RL env that measures the real construction task."*

### Phase 2 — Multi-agent fleet (~2.5 h) · *The ambition* — **Anthropic + Fireworks**

- Add 2 more manipulators to the MJCF with name-prefixed actuators; capability profiles enforced in a thin tool layer; drive via `step(action)`.
- Add a **Foreman** orchestrator (Claude) that assigns blocks to robots; specialists are the embodiments.
- Extend reward with `fleet_utilization`, `makespan_efficiency`, `safety_no_collision`.
- **Fallback (time-box 60 min):** symbolic fleet over single-arm physics.
- **Milestone:** naive single-arm vs. coordinated fleet → side-by-side reward gap (the money slide).

### Phase 3 — Hill-climb with Fireworks (~2 h) · *The RL story* — **Fireworks + Modal**

- **Calibrate first** (cheap): tune the task to **20–50% reward with variance** (`uv run train.py --calibrate-only ...`). Aim `reward_mean ~0.2–0.5`.
- Run a few GRPO steps on Qwen3-8B; plot reward rising = "hill-climbing to the real task." Modal for parallel rollouts.
- ⚠️ Cookbook notes a preview-account trainer blocker → treat **calibration + a couple steps** as the target; the calibration reward distribution is the guaranteed artifact.
- **Milestone:** a reward-vs-step curve, even if short.

### Phase 4 — Polish + remaining sponsors (~2 h) · *Wow factor*

- **Antim/Gizmo** (`gizmo.antimlabs.com`): generate a prettier construction scene → drop under `scenes/` (visual layer; symbolic sim stays source of truth).
- Dashboard: HUD trace + reward-breakdown panel beside the 3D viewer (`WORLDSIM_VIEWER=1`).
- **MiniMax:** narrated incident/build recap video.
- **Exa:** ground a reward term in a real building-code/sequencing rule (e.g. base must settle before roof).
- **Hillclimb / Protege:** framing — failed traces → curriculum data; future real-dataset bridge.

---

## 7. Sponsor → Milestone Map (deep, not bolted-on)


| Phase | Sponsors that become essential                                                                              |
| ----- | ----------------------------------------------------------------------------------------------------------- |
| 0–1   | **HUD** (env/tasks/reward/traces), **Antim** (scene engine)                                                 |
| 2     | **Anthropic** (foreman), **Fireworks** (specialist models)                                                  |
| 3     | **Fireworks** (GRPO), **Modal** (parallel rollouts)                                                         |
| 4     | **Gizmo/Antim** (visual), **MiniMax** (recap), **Exa** (code grounding), **Hillclimb/Protege** (data story) |


**Credits:** HUD `YC-RL-HACKATHON` ($200) · Modal `SQ8-USG-5K2` ($250) · Daytona `DAYTONA_RL_ENVIRONMENTS_HACK_Y6ZDQBG5` ($100) · Exa `HUDHACK` ($50) · Fireworks `HUD-HACK-2026` ($30) · MiniMax (form) · DeepMind ($25 GCP) · SixtyFour (64).

---

## 8. Demo Story (work backwards from one build)

1. **Setup shot:** the construction site — build pad, scattered blocks, 3 embodiments, the target house ghosted in.
2. **Naive run:** one robot tries to do everything → low reward (structure incomplete / unstable / collisions). Show the HUD reward breakdown.
3. **Coordinated run:** Foreman assigns base→crane, roof→humanoid, staging→wheeled → high reward, stable house. Side-by-side breakdown.
4. **Hill-climb:** the reward-vs-step curve from Fireworks rising.
5. **Leaderboard slide:** naive vs. multi-agent vs. trained — the proof the benchmark measures something real.

---

## 9. What Makes This Win

- **HUD is central**, not decorative: environment, taskset, graders, traces, training-ready trajectories all required.
- **Real research gap:** operational multi-agent physical coordination — the layer between RoboFactory and CREW-Wildfire.
- **Genuine RL/hill-climb loop**, not "LLM calls a tool."
- **Tangible physical-world business case:** construction/manufacturing shortage, fleet utilization & safety.
- **Demoable + visual:** a 3D site, a building structure, a reward dashboard, an improvement curve.

---

## Appendix · Quick Commands

```bash
uv sync                                                   # deps incl. bundled Newton wheel
hud set HUD_API_KEY=...                                   # gateway + traces
python scripts/check_setup.py                             # boots sim + grades one scripted rollout
hud eval environment/tasks.py claude --group 1            # LLM drives the tools
WORLDSIM_VIEWER=1 hud eval environment/tasks.py claude --group 1   # watch live
```

## Appendix · References

- RoboFactory — [https://arxiv.org/abs/2503.16408](https://arxiv.org/abs/2503.16408) · project: [https://iranqin.github.io/robofactory/](https://iranqin.github.io/robofactory/)
- CREW-Wildfire — [https://arxiv.org/abs/2507.05178](https://arxiv.org/abs/2507.05178) · project: [https://generalroboticslab.com/CREW-Wildfire](https://generalroboticslab.com/CREW-Wildfire)
- HUD build docs — [https://docs.hud.ai/v6/build/overview](https://docs.hud.ai/v6/build/overview)
- Worldsim template — [https://github.com/hud-evals/worldsim-template](https://github.com/hud-evals/worldsim-template)
- Fireworks RL cookbook — [https://github.com/hud-evals/hud-python/tree/main/cookbooks/fireworks-rl-training](https://github.com/hud-evals/hud-python/tree/main/cookbooks/fireworks-rl-training)
- Hackathon — [https://www.hud.ai/hackathon](https://www.hud.ai/hackathon)

