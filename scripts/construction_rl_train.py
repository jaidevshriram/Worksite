"""Fireworks RL Training loop over the construction-v1 coordination task.

Adapts the hud-evals/hud-python Fireworks RL Training cookbook to use our
symbolic construction planning environment as the task source.

Instead of arithmetic (multiply a × b), the LLM receives a JSON description
of a construction site (agent positions, block positions, goals) and must
output a valid JSON action plan that moves blocks to their target positions.
Reward = fraction of blocks placed at goal (0.0–1.0, continuous → good GRPO spread).

Architecture
------------
The loop follows the original cookbook's three-phase structure:

Phase 1 – CALIBRATION (--calibrate-only --calibration-backend inference)
  • Uses Fireworks' OpenAI-compatible inference endpoint only.
  • No training API / account key needed — just FIREWORKS_API_KEY.
  • Runs N task groups × R rollouts, grades locally, reports reward spread.
  • USE THIS FIRST to verify task difficulty before spinning up training infra.

Phase 2 – CALIBRATION (--calibrate-only --calibration-backend managed)
  • Provisions the same managed deployment sampler that training uses.
  • Requires FIREWORKS_ACCOUNT_ID and the Training API preview access.
  • Samples from the ACTUAL base model (Qwen3-8B by default), not a
    proxy model, so within_group_reward_std is the signal that counts.

Phase 3 – TRAINING (--steps N)
  • Full GRPO loop: rollout → grade → forward_backward_custom → optim_step
    → save_weights_for_sampler → reload → repeat.
  • Same managed deployment sampler as Phase 2.
  • Requires Training API access (currently preview on Fireworks).

Quick start
-----------
1. Add keys to .env (see .env.example):
       FIREWORKS_API_KEY=fw_...
       FIREWORKS_ACCOUNT_ID=...   # only needed for phases 2 & 3

2. Install extras:
       uv add --optional fireworks openai python-dotenv transformers torch

3. Calibration run (inference backend, no training keys required):
       uv run scripts/construction_rl_train.py \\
           --calibrate-only --calibration-backend inference \\
           --groups-per-step 4 --rollouts-per-prompt 4 --debug-samples 2

4. Calibration run (managed backend, actual base model):
       uv run scripts/construction_rl_train.py \\
           --calibrate-only --calibration-backend managed \\
           --groups-per-step 4 --rollouts-per-prompt 6 --parallelism 12

5. Training run (5 steps, production-sized groups):
       uv run scripts/construction_rl_train.py \\
           --steps 5 --groups-per-step 6 --rollouts-per-prompt 6 --parallelism 18

Reward design
-------------
For GRPO to learn, within_group_reward_std must be clearly > 0 (each prompt group
must have at least some rollouts with different rewards). With 4 blocks the reward
can be 0, 0.25, 0.50, 0.75, or 1.0.  Task difficulty is tunable via --n-blocks:

  --n-blocks 1   → very easy: should be mostly 1.0 (too easy for RL)
  --n-blocks 2   → easy: good spread for early training
  --n-blocks 3   → medium: recommended for the managed calibration check
  --n-blocks 4   → hard: use if the model solves 2–3 blocks reliably

References
----------
• Fireworks Training API: https://docs.fireworks.ai/fine-tuning/training-api/introduction
• Original cookbook: https://github.com/hud-evals/hud-python/tree/main/cookbooks/fireworks-rl-training
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BASE_MODEL        = "accounts/fireworks/models/qwen3-8b"
DEFAULT_TOKENIZER_MODEL   = "Qwen/Qwen3-8B"
DEFAULT_TRAINING_SHAPE    = "accounts/fireworks/trainingShapes/qwen3-8b-128k"
# llama-v3p1-70b-instruct was deprecated (no serverless endpoint) Nov 2025.
# GLM 5.2 supports serverless, has 131K context, and emits clean JSON plans.
DEFAULT_INFERENCE_MODEL   = "accounts/fireworks/models/glm-5p2"
DEFAULT_INFERENCE_BASE_URL = "https://api.fireworks.ai/inference/v1"


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------
# Site geometry constants (same as construction_env.py)
_ALL_BLOCKS = [
    {"name": "block_1", "x": 10.0, "y": -4.7},
    {"name": "block_2", "x":  8.5, "y": -5.0},
    {"name": "block_3", "x":  9.9, "y": -6.5},
    {"name": "block_4", "x": 11.4, "y": -5.0},
]
_ALL_TARGETS = [
    {"block": "block_1", "x": -4.0, "y": 7.0, "radius": 1.5},
    {"block": "block_2", "x": -2.0, "y": 7.0, "radius": 1.5},
    {"block": "block_3", "x":  0.0, "y": 7.0, "radius": 1.5},
    {"block": "block_4", "x":  2.0, "y": 7.0, "radius": 1.5},
]
_AGENTS = {
    "A": {"role": "g1", "x": -1.4, "y":  2.2},
    "B": {"role": "g1", "x":  2.8, "y":  2.1},
    "C": {"role": "g1", "x": -1.7, "y": -2.3},
    "D": {"role": "g1", "x":  2.7, "y": -2.5},
    "E": {"role": "spot", "x": -6.0, "y": 4.0},
    "F": {"role": "spot", "x":  7.0, "y": -4.0},
}
_REACH = 2.0
_GOAL_RADIUS = 1.5


@dataclass(frozen=True, slots=True)
class ConstructionTask:
    group_index: int
    n_blocks: int        # how many blocks to place (1–4)
    seed: int

    @property
    def blocks(self) -> list[dict]:
        """Active blocks for this task instance."""
        rng = random.Random(self.seed)
        pool = list(_ALL_BLOCKS)
        rng.shuffle(pool)
        return pool[: self.n_blocks]

    @property
    def targets(self) -> list[dict]:
        """Targets for each active block (same order as blocks)."""
        target_by_name = {t["block"]: t for t in _ALL_TARGETS}
        return [target_by_name[b["name"]] for b in self.blocks
                if b["name"] in target_by_name]

    @property
    def prompt(self) -> str:
        """System + user prompt for the LLM."""
        blocks_desc = "\n".join(
            f'  - {b["name"]} at ({b["x"]:.1f}, {b["y"]:.1f})'
            for b in self.blocks
        )
        targets_desc = "\n".join(
            f'  - {t["block"]} → ({t["x"]:.1f}, {t["y"]:.1f}) within {t["radius"]:.1f}m'
            for t in self.targets
        )
        agents_desc = "\n".join(
            f'  - {aid}: {a["role"]} at ({a["x"]:.1f}, {a["y"]:.1f})'
            for aid, a in _AGENTS.items()
        )
        return f"""You are coordinating a mixed robot fleet on a construction site.
The site uses a 2D coordinate system with the foundation spanning roughly x∈[-9,12], y∈[-9,10].

Agents (reach radius {_REACH}m — must be within this to grab a block):
{agents_desc}

Roles:
  - G1 humanoids (A, B, C, D): can walk_to, grab a block, carry_to, and release.
  - Spot quadrupeds (E, F): can only walk_to (no grabbing).

Blocks to move:
{blocks_desc}

Goals (place each block at its target within the radius):
{targets_desc}

Output a JSON action plan — a list of steps in order. Each step is one of:
  {{"agent": "X", "action": "walk_to",  "x": <float>, "y": <float>}}
  {{"agent": "X", "action": "grab",     "block": "<name>"}}
  {{"agent": "X", "action": "carry_to", "x": <float>, "y": <float>}}
  {{"agent": "X", "action": "release"}}

Rules:
1. An agent must walk_to within {_REACH}m of a block before grab.
2. After grab, use carry_to to move block+agent, then release.
3. Multiple agents can work in parallel (interleave their steps).
4. Only G1 humanoids (A-D) can grab/carry/release.

Respond with ONLY a JSON object: {{"steps": [...]}}"""


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def _dist(ax, ay, bx, by) -> float:
    return math.hypot(ax - bx, ay - by)


def _execute_plan(task: ConstructionTask, steps: list[dict]) -> float:
    """Run the action plan against the construction world and return reward ∈ [0,1]."""
    from sim import construction_world as cw

    agents_cfg  = {aid: a["role"] for aid, a in _AGENTS.items()}
    agent_pos   = {aid: [a["x"], a["y"]] for aid, a in _AGENTS.items()}

    cw.reset_world(
        scene_id="construction-v1",
        agents=agents_cfg,
        agent_positions=agent_pos,
        blocks=[{"name": b["name"], "x": b["x"], "y": b["y"]} for b in task.blocks],
        targets=[{"block": t["block"], "x": t["x"], "y": t["y"],
                  "radius": t["radius"]} for t in task.targets],
        checkpoints=[],
    )

    ACTION_LIMIT = max(60, len(task.blocks) * 20)
    for i, step in enumerate(steps[:ACTION_LIMIT]):
        if not isinstance(step, dict):
            continue
        agent  = step.get("agent", "")
        action = step.get("action", "")
        try:
            if action == "walk_to":
                cw.walk_to(agent, float(step["x"]), float(step["y"]))
            elif action == "grab":
                cw.grab(agent, str(step["block"]))
            elif action == "carry_to":
                cw.carry_to(agent, float(step["x"]), float(step["y"]))
            elif action == "release":
                cw.release(agent)
        except Exception:
            pass  # bad step → skip silently

    snap = cw.world_snapshot()
    placed = sum(1 for b in snap["blocks"].values() if b.get("at_goal"))
    return round(placed / max(1, len(task.targets)), 4)


def grade_plan(text: str, task: ConstructionTask) -> tuple[float, int]:
    """Parse + execute the LLM's JSON plan. Returns (reward, n_steps_executed)."""
    # Extract the first JSON object from the text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return 0.0, 0
    try:
        obj = json.loads(match.group())
        steps = obj.get("steps", [])
        if not isinstance(steps, list):
            return 0.0, 0
    except (json.JSONDecodeError, TypeError):
        return 0.0, 0

    reward = _execute_plan(task, steps)
    return reward, len(steps)


# ---------------------------------------------------------------------------
# Rollout record (mirrors the cookbook's RolloutRecord)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class RolloutRecord:
    task: ConstructionTask
    text: str
    reward: float
    n_steps: int          # number of valid action steps in the plan
    # set only in managed-sampler / training path:
    tokens: list[int] = field(default_factory=list)
    rollout_logprobs: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Inference-backend sampling (OpenAI-compatible Fireworks endpoint)
# ---------------------------------------------------------------------------
async def _sample_one_inference(
    client,
    task: ConstructionTask,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    _max_retries: int = 8,
) -> RolloutRecord:
    from openai import RateLimitError as _RateLimitError
    delay = 2.0
    for attempt in range(_max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": task.prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format={"type": "json_object"},
            )
            break
        except _RateLimitError:
            if attempt == _max_retries - 1:
                raise
            print(f"  [rate-limit] retrying in {delay:.1f}s (attempt {attempt+1}/{_max_retries})")
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, 60.0)
    text = response.choices[0].message.content or ""
    reward, n_steps = grade_plan(text, task)
    return RolloutRecord(task=task, text=text, reward=reward, n_steps=n_steps)


async def sample_rollouts_inference(
    tasks: list[ConstructionTask],
    *,
    api_key: str,
    model: str,
    base_url: str,
    rollouts_per_prompt: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    parallelism: int,
) -> list[RolloutRecord]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(parallelism)

    async def _one(task: ConstructionTask) -> RolloutRecord:
        async with sem:
            return await _sample_one_inference(
                client, task, model=model, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p,
            )

    jobs = [_one(task) for task in tasks for _ in range(rollouts_per_prompt)]
    return list(await asyncio.gather(*jobs))


# ---------------------------------------------------------------------------
# Managed-sampler / training-API sampling  (mirrors cookbook exactly)
# ---------------------------------------------------------------------------
async def sample_rollouts_managed(
    sampler,
    tokenizer,
    tasks: list[ConstructionTask],
    *,
    rollouts_per_prompt: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> list[RolloutRecord]:
    """Sample using the Fireworks deployment sampler (training path)."""
    import torch
    from fireworks.training.sdk import AdaptiveConcurrencyController  # noqa

    async def _one(task: ConstructionTask) -> RolloutRecord:
        prompt_tokens = list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": task.prompt}],
                tokenize=True, add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        completions = await sampler.sample_with_prompt_tokens(
            prompt_tokens, n=1, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p, logprobs=True,
        )
        c = completions[0]
        tokens      = list(c.full_tokens)
        prompt_len  = int(c.prompt_len)
        output_len  = max(0, len(tokens) - prompt_len)
        output_logprobs = list(c.inference_logprobs)
        text = str(c.text)

        model_input_len    = max(0, len(tokens) - 1)
        rollout_logprobs   = ([0.0] * max(0, prompt_len - 1)
                              + output_logprobs[:output_len])
        if len(rollout_logprobs) < model_input_len:
            rollout_logprobs += [0.0] * (model_input_len - len(rollout_logprobs))
        else:
            rollout_logprobs = rollout_logprobs[:model_input_len]

        reward, n_steps = grade_plan(text, task)
        return RolloutRecord(
            task=task, text=text, reward=reward, n_steps=n_steps,
            tokens=tokens, rollout_logprobs=rollout_logprobs,
        )

    jobs = [_one(task) for task in tasks for _ in range(rollouts_per_prompt)]
    return list(await asyncio.gather(*jobs))


# ---------------------------------------------------------------------------
# Reward / advantage helpers
# ---------------------------------------------------------------------------
def reward_stats(records: list[RolloutRecord]) -> dict[str, float]:
    if not records:
        return {"reward_mean": 0.0, "reward_std": 0.0,
                "reward_min": 0.0, "reward_max": 0.0}
    rewards = [r.reward for r in records]
    mean = sum(rewards) / len(rewards)
    var  = sum((r - mean) ** 2 for r in rewards) / max(1, len(rewards) - 1)
    return {
        "reward_mean": round(mean, 4),
        "reward_std":  round(math.sqrt(var), 4),
        "reward_min":  round(min(rewards), 4),
        "reward_max":  round(max(rewards), 4),
    }


def within_group_reward_std(records: list[RolloutRecord]) -> float:
    """Mean per-group reward std — the signal GRPO actually trains on."""
    grouped: dict[int, list[float]] = {}
    for r in records:
        grouped.setdefault(r.task.group_index, []).append(r.reward)
    stds = []
    for rewards in grouped.values():
        if len(rewards) < 2:
            continue
        mean = sum(rewards) / len(rewards)
        var  = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
        stds.append(math.sqrt(var))
    return round(sum(stds) / len(stds), 4) if stds else 0.0


def advantages_by_record(records: list[RolloutRecord]) -> list[float]:
    grouped: dict[int, list[float]] = {}
    for r in records:
        grouped.setdefault(r.task.group_index, []).append(r.reward)
    stats: dict[int, tuple[float, float]] = {}
    for g, rewards in grouped.items():
        mean = sum(rewards) / len(rewards)
        var  = sum((r - mean) ** 2 for r in rewards) / max(1, len(rewards) - 1)
        stats[g] = (mean, math.sqrt(var) if var > 1e-12 else 1.0)
    return [
        (r.reward - stats[r.task.group_index][0]) / stats[r.task.group_index][1]
        for r in records
    ]


# ---------------------------------------------------------------------------
# GRPO loss (mirrors cookbook)
# ---------------------------------------------------------------------------
def make_grpo_loss(records: list[RolloutRecord], advantages: list[float]):
    import tinker
    import torch
    rollout_lps  = [torch.tensor(r.rollout_logprobs, dtype=torch.float32)
                    for r in records]
    adv_tensors  = [torch.tensor(v, dtype=torch.float32) for v in advantages]

    def loss_fn(data, logprobs_list):
        total_loss   = torch.tensor(0.0)
        total_tokens = 0.0
        ratios: list[float] = []
        for i, lps in enumerate(logprobs_list):
            weights = torch.tensor(
                data[i].loss_fn_inputs["weights"].data, dtype=torch.float32)
            ml = min(len(lps), len(weights), len(rollout_lps[i]))
            if ml == 0:
                continue
            pi     = lps[:ml].float()
            old    = rollout_lps[i][:ml]
            mask   = weights[:ml]
            ratio  = torch.exp((pi - old).clamp(-8.0, 8.0))
            clipped = torch.clamp(ratio, 0.8, 1.2)
            surrogate = torch.minimum(
                ratio * adv_tensors[i], clipped * adv_tensors[i])
            total_loss   = total_loss - torch.dot(surrogate, mask)
            total_tokens += float(mask.sum().item())
            if mask.sum().item() > 0:
                ratios.append(
                    float((ratio * mask).sum().item() / mask.sum().item()))
        mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        return total_loss, {
            "policy_loss_sum": float(total_loss.item()),
            "tokens": total_tokens,
            "mean_ratio": mean_ratio,
        }
    return loss_fn


def make_datums(records: list[RolloutRecord]):
    import tinker
    return [
        tinker.Datum(
            model_input=tinker.ModelInput.from_ints(r.tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(
                    data=r.tokens[1:], dtype="int64",
                    shape=[len(r.tokens) - 1]),
                "weights": tinker.TensorData(
                    data=[1.0] * (len(r.tokens) - 1),  # all output tokens
                    dtype="float32", shape=[len(r.tokens) - 1]),
            },
        )
        for r in records
    ]


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def maybe_plot(metrics_path: Path, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if l]
    plottable = [r for r in rows if r.get("phase") in {"calibrate", "train"}]
    if not plottable:
        return
    steps   = [r["step"] for r in plottable]
    rewards = [r["reward_mean"] for r in plottable]
    losses  = [r.get("policy_loss_sum", 0.0) for r in plottable]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(steps, rewards, "o-", label="reward_mean", color="tab:green")
    ax1.set_xlabel("step"); ax1.set_ylabel("reward_mean", color="tab:green")
    ax1.set_ylim(-0.05, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(steps, losses, "x-", label="policy_loss_sum", color="tab:blue")
    ax2.set_ylabel("policy_loss_sum", color="tab:blue")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot → {out_path}")


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------
def make_tasks(*, groups: int, n_blocks: int, seed: int) -> list[ConstructionTask]:
    rng = random.Random(seed)
    return [
        ConstructionTask(
            group_index=i,
            n_blocks=n_blocks,
            seed=rng.randint(0, 99999),
        )
        for i in range(groups)
    ]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
async def run(args: argparse.Namespace) -> None:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()

    api_key     = os.environ["FIREWORKS_API_KEY"]
    output_dir  = Path(args.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    plot_path    = output_dir / "reward_loss.png"
    if metrics_path.exists() and not args.resume_metrics:
        metrics_path.unlink()

    tasks = make_tasks(
        groups=args.groups_per_step, n_blocks=args.n_blocks, seed=args.seed)

    # ── INFERENCE CALIBRATION (no training API needed) ────────────────────────
    if args.calibrate_only and args.calibration_backend == "inference":
        print(f"[calibrate/inference] model={args.inference_model} "
              f"groups={args.groups_per_step} rollouts={args.rollouts_per_prompt}")
        t0 = time.perf_counter()
        records = await sample_rollouts_inference(
            tasks,
            api_key=api_key,
            model=args.inference_model,
            base_url=args.inference_base_url,
            rollouts_per_prompt=args.rollouts_per_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            parallelism=args.parallelism,
        )
        row = {
            "phase": "calibrate", "backend": "inference", "step": 0,
            "n_blocks": args.n_blocks,
            "num_rollouts": len(records),
            "rollout_seconds": round(time.perf_counter() - t0, 2),
            "within_group_reward_std": within_group_reward_std(records),
            **reward_stats(records),
        }
        append_jsonl(metrics_path, row)
        maybe_plot(metrics_path, plot_path)
        print(json.dumps(row, sort_keys=True))

        if args.debug_samples > 0:
            print("\n── Sample rollouts ──")
            for rec in records[: args.debug_samples]:
                print(f"  reward={rec.reward:.2f}  steps={rec.n_steps}"
                      f"  task=block_line({rec.task.n_blocks} blocks, seed={rec.task.seed})")
                print(f"  output: {rec.text[:200].strip()!r}")
        return

    # ── MANAGED SAMPLER (calibration or full training) ────────────────────────
    try:
        from fireworks.training.sdk import (
            AdaptiveConcurrencyController,
            FiretitanServiceClient,
            GradAccNormalization,
        )
        from transformers import AutoTokenizer
        import tinker
    except ImportError:
        print("ERROR: Training API dependencies not installed.")
        print("  uv add --optional fireworks fireworks-ai tinker transformers torch")
        print("  Or run with --calibrate-only --calibration-backend inference")
        sys.exit(1)

    account_id = os.environ.get("FIREWORKS_ACCOUNT_ID", "")
    if not account_id:
        print("ERROR: FIREWORKS_ACCOUNT_ID not set in .env")
        sys.exit(1)

    tokenizer   = AutoTokenizer.from_pretrained(
        args.tokenizer_model, trust_remote_code=True)
    controller  = AdaptiveConcurrencyController(initial_window=args.parallelism)
    service     = FiretitanServiceClient.from_firetitan_config(
        api_key=api_key,
        base_url=args.base_url,
        base_model=args.base_model,
        tokenizer_model=args.tokenizer_model,
        lora_rank=args.lora_rank,
        training_shape_id=args.training_shape,
        deployment_id=args.deployment_id,
        learning_rate=args.learning_rate,
        replica_count=args.replicas,
        cleanup_trainer_on_close=not args.keep_trainer,
        cleanup_deployment_on_close=None if args.keep_deployment else "scale_to_zero",
    )

    try:
        training_client = None
        if not args.calibrate_only:
            training_client = service.create_training_client(
                base_model=args.base_model, lora_rank=args.lora_rank)

        sampler = service.create_deployment_sampler(
            tokenizer=tokenizer, concurrency_controller=controller)

        n_iters = 1 if args.calibrate_only else args.steps
        for step in range(n_iters):
            phase = "calibrate" if args.calibrate_only else "train"
            print(f"[{phase}] step={step}  "
                  f"groups={args.groups_per_step}  rollouts={args.rollouts_per_prompt}")
            t0 = time.perf_counter()
            records = await sample_rollouts_managed(
                sampler, tokenizer, tasks,
                rollouts_per_prompt=args.rollouts_per_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            rollout_secs = time.perf_counter() - t0
            stats = reward_stats(records)

            for rec in records[: args.debug_samples]:
                print(f"  [sample] reward={rec.reward:.2f} steps={rec.n_steps} "
                      f"text={rec.text[:120]!r}")

            row: dict[str, Any] = {
                "phase": phase, "step": step,
                "n_blocks": args.n_blocks,
                "num_rollouts": len(records),
                "rollout_seconds": round(rollout_secs, 2),
                "within_group_reward_std": within_group_reward_std(records),
                "trainer_job_id": getattr(service, "trainer_job_id", None),
                "deployment_id":  getattr(service, "deployment_id", None),
                **stats,
            }

            if args.calibrate_only:
                append_jsonl(metrics_path, row)
                maybe_plot(metrics_path, plot_path)
                print(json.dumps(row, sort_keys=True))
                continue

            # ── GRPO update ──────────────────────────────────────────────────
            assert training_client is not None
            datums     = make_datums(records)
            advantages = advantages_by_record(records)
            loss_fn    = make_grpo_loss(records, advantages)

            fb_future = await training_client.forward_backward_custom_async(
                datums, loss_fn)
            fb = await fb_future.result_async()

            opt_future = await training_client.optim_step_async(
                tinker.AdamParams(
                    learning_rate=args.learning_rate,
                    beta1=0.9, beta2=0.999, eps=1e-8,
                    weight_decay=args.weight_decay,
                ),
                grad_accumulation_normalization=GradAccNormalization.NUM_LOSS_TOKENS,
            )
            await opt_future.result_async()
            row.update(fb.metrics)

            saved_f = await training_client.save_weights_for_sampler_async(
                f"step-{step:05d}")
            saved = await saved_f.result_async()
            row["checkpoint"] = saved.path
            sampler = service.create_deployment_sampler(
                model_path=saved.path,
                tokenizer=tokenizer,
                concurrency_controller=controller,
            )
            append_jsonl(metrics_path, row)
            maybe_plot(metrics_path, plot_path)
            print(json.dumps(row, sort_keys=True))

    finally:
        service.close()

    print(f"\nMetrics → {metrics_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Task config
    p.add_argument("--n-blocks", type=int, default=2,
                   help="Blocks to place per task (1–4). 2 = good spread for early training.")
    p.add_argument("--seed", type=int, default=42)

    # Model / API
    p.add_argument("--base-url", default=os.environ.get(
        "FIREWORKS_BASE_URL", "https://api.fireworks.ai"))
    p.add_argument("--base-model",       default=DEFAULT_BASE_MODEL)
    p.add_argument("--inference-model",  default=DEFAULT_INFERENCE_MODEL)
    p.add_argument("--tokenizer-model",  default=DEFAULT_TOKENIZER_MODEL)
    p.add_argument("--training-shape",   default=DEFAULT_TRAINING_SHAPE)
    p.add_argument("--deployment-id",    default="worldsim-construction-rl")
    p.add_argument("--output-dir",       default="runs/construction-rl")
    p.add_argument("--inference-base-url", default=DEFAULT_INFERENCE_BASE_URL)

    # Training loop
    p.add_argument("--steps",              type=int,   default=5)
    p.add_argument("--groups-per-step",    type=int,   default=6)
    p.add_argument("--rollouts-per-prompt",type=int,   default=6)
    p.add_argument("--parallelism",        type=int,   default=16)
    p.add_argument("--replicas",           type=int,   default=1)
    p.add_argument("--lora-rank",          type=int,   default=0)
    p.add_argument("--learning-rate",      type=float, default=1e-5)
    p.add_argument("--weight-decay",       type=float, default=0.01)
    p.add_argument("--temperature",        type=float, default=0.9)
    p.add_argument("--top-p",             type=float, default=0.95)
    p.add_argument("--max-tokens",         type=int,   default=1024)

    # Debug / flow
    p.add_argument("--debug-samples",  type=int, default=2,
                   help="Print this many sample rollouts per step.")
    p.add_argument("--calibrate-only", action="store_true",
                   help="Sample + grade only, no training step.")
    p.add_argument("--calibration-backend", choices=("inference", "managed"),
                   default="inference",
                   help="'inference' = cheap OpenAI-compat API, no training keys needed. "
                        "'managed' = Fireworks deployment sampler, requires training access.")
    p.add_argument("--keep-trainer",    action="store_true")
    p.add_argument("--keep-deployment", action="store_true")
    p.add_argument("--resume-metrics",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
