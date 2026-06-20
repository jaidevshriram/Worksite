"""Run a VLA policy through the Franka pick task and report a success rate.

Runs a Taskset of pick rollouts: the SDK owns the loop, the `robot` wire protocol,
and (with HUD_API_KEY set) per-step trace streaming. Each rollout is one task
(distinct seed -> varied object pose); the success rate is averaged over them, and
mean reward captures partial lift progress.

    # plumbing check (no GPU, no model):
    python run_vla.py --noop --group 1

    # the pi0.5 baseline, policy local (needs a GPU + `pip install -e '.[vla]'`):
    python run_vla.py --group 10 --max-steps 200

    # policy on a remote GPU box (this machine needs no GPU) - serve it from serve/
    # (`modal run serve/pi05_modal.py`) and pass the printed ws host:port:
    python run_vla.py --remote HOST:PORT --group 10

    # record a dataset while evaluating (env-side):
    python run_vla.py --group 5 --record ./datasets/pi05-pick
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from hud import LocalRuntime, Taskset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from environment.vla_env import vla_pick  # noqa: E402


class _UploadWatch(logging.Handler):
    """Capture the exporter's swallowed 'telemetry upload failed' warnings.

    A dropped span batch leaves exactly one WARNING and nothing else (see
    hud.telemetry.exporter._do_upload), so tapping that logger is the only
    client-side evidence of a drop. We also print each one live.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        self.records.append(msg)
        print(f"[telemetry-watch] {msg}", flush=True)


def _scan_dump(path: Path) -> dict[str, dict[str, list[int]]]:
    """Per-trace {source: sorted ticks} from a local span dump (the emit-side truth)."""
    from hud.telemetry.span import PAYLOAD_ATTRIBUTE

    out: dict[str, dict[str, list[int]]] = {}
    for jsonl in path.glob("*.jsonl"):
        per_source: dict[str, list[int]] = {}
        for line in jsonl.read_text().splitlines():
            payload = json.loads(line).get("attributes", {}).get(PAYLOAD_ATTRIBUTE, {})
            tick, source = payload.get("tick"), payload.get("source")
            if tick is not None and source in ("observation", "inference"):
                per_source.setdefault(source, []).append(tick)
        out[jsonl.stem] = {s: sorted(t) for s, t in per_source.items()}
    return out


def _gaps(ticks: list[int]) -> list[tuple[int, int]]:
    """Contiguous missing ranges inside [min..max] of a tick list."""
    present = set(ticks)
    full = range(min(ticks), max(ticks) + 1) if ticks else range(0)
    missing = [t for t in full if t not in present]
    ranges: list[tuple[int, int]] = []
    for t in missing:
        if ranges and t == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], t)
        else:
            ranges.append((t, t))
    return ranges


async def _readback(trace_id: str) -> int | None:
    """Spans the HUD platform actually stored for a trace, or None if unreadable."""
    from hud.settings import settings
    from hud.utils import make_request

    for route in (f"{settings.hud_api_url}/trace/{trace_id}/events",
                  f"{settings.hud_api_url}/telemetry/trace/{trace_id}"):
        try:
            data = await make_request(method="GET", url=route, api_key=settings.api_key,
                                      max_retries=1)
            items = data.get("events") or data.get("trajectory")
            if items is not None:
                return len(items)
        except Exception:
            continue
    return None


async def _diagnose(job: object, dump_dir: Path, watch: _UploadWatch) -> None:
    """Flush telemetry, then reconcile what was emitted vs. what beta stored."""
    from hud.telemetry import flush

    print(f"\n{'=' * 60}\n[diag] flushing telemetry exporter (drain + wait uploads)…", flush=True)
    drained = flush(timeout=180)
    print(f"[diag] flush complete={drained}  upload-failure warnings={len(watch.records)}")

    emitted = _scan_dump(dump_dir)
    runs = list(getattr(job, "runs", []))
    for i, run in enumerate(runs):
        # Print the emit-side truth FIRST - it's the ground record and must survive
        # the best-effort readback and the known sim/torch teardown segfault.
        tid = getattr(run, "trace_id", None)
        marks = emitted.get(tid or "", {})
        obs, inf = marks.get("observation", []), marks.get("inference", [])
        n_emitted = sum(len(v) for v in marks.values())
        print(f"\n[diag] rollout {i}  trace={tid}")
        print(f"       emitted spans : {n_emitted}  (observation={len(obs)} inference={len(inf)})")
        if obs:
            print(f"       obs ticks     : {obs[0]}..{obs[-1]}  gaps_in_emitted={_gaps(obs)}")
        if inf:
            step = inf[1] - inf[0] if len(inf) > 1 else None
            print(f"       inference ticks: {inf}  (replan every {step} ticks)")
        landed = await _readback(tid) if tid else None
        if landed is None:
            print("       beta readback : unavailable (endpoint/auth) - rely on emit+warnings")
        else:
            lost = n_emitted - landed
            verdict = "OK (all spans landed)" if lost <= 0 else f"DROPPED {lost} spans upstream"
            print(f"       beta stored   : {landed}  ->  {verdict}")
    if watch.records:
        print(f"\n[diag] SMOKING GUN - {len(watch.records)} upload(s) failed and were discarded:")
        for r in watch.records:
            print(f"       ! {r}")
    else:
        print("\n[diag] no upload-failure warnings this run (drop is intermittent; "
              "emit-side dump above is the ground truth)")
    print(f"[diag] span dump kept at: {dump_dir}\n{'=' * 60}")


async def run_eval(args: argparse.Namespace) -> None:
    if args.record:
        os.environ["WORLDSIM_RECORD_DIR"] = args.record  # the bridge records when this is set

    # Telemetry diagnostics: dump every emitted span to disk (the emit-side truth,
    # immune to upload drops) and tap the exporter's failure warnings.
    from hud.settings import settings
    dump_dir = Path(args.diagnose_dir or f"./telemetry-dump/{int(time.time())}").resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)
    settings.telemetry_local_dir = str(dump_dir)
    watch = _UploadWatch()
    logging.getLogger("hud.telemetry.exporter").addHandler(watch)
    print(f"[diag] span dump -> {dump_dir}  (telemetry_enabled={settings.telemetry_enabled}, "
          f"api_key={'set' if settings.api_key else 'MISSING'})")

    if args.noop:
        from agents.vla_agent import NoopAgent
        agent = NoopAgent()
        policy = "noop"
    elif args.remote:
        from agents.vla_agent import RemoteAgent
        host, _, port = args.remote.rpartition(":")
        agent = RemoteAgent(host=host or "localhost", port=int(port))
        policy = f"remote://{args.remote}"
    elif args.agent:
        # Bring your own: `--agent module.path:ClassName` (e.g. agents.vla_agent:CustomAgent).
        import importlib
        mod_name, _, cls_name = args.agent.partition(":")
        agent = getattr(importlib.import_module(mod_name), cls_name)()
        policy = args.agent
    else:
        from agents.vla_agent import PI05Agent
        agent = PI05Agent(checkpoint=args.checkpoint)
        policy = args.checkpoint

    tasks = [
        vla_pick(instruction=args.instruction, target_object=args.target_object,
                 lift_height=args.lift_height, seed=i, max_steps=args.max_steps)
        for i in range(args.group)
    ]
    print(f"VLA eval: {args.group} rollout(s) of {args.instruction!r} (policy: {policy})\n")

    job = await Taskset("worldsim-vla", tasks).run(
        agent, runtime=LocalRuntime(str(ROOT / "environment" / "vla_env.py")),
        max_concurrent=args.max_concurrent,
    )

    rewards = [run.reward or 0.0 for run in job.runs]
    successes = [r >= args.threshold for r in rewards]
    for i, (r, ok) in enumerate(zip(rewards, successes, strict=False)):
        print(f"  rollout {i:>2}: reward={r:.4f}  {'SUCCESS' if ok else 'fail'}")
    n = len(rewards) or 1
    print(f"\n{'=' * 50}\nSUCCESS RATE: {sum(successes) / n * 100:.1f}%   "
          f"mean reward: {sum(rewards) / n:.4f}\n{'=' * 50}")
    if args.record:
        print(f"dataset saved -> {args.record}/")

    await _diagnose(job, dump_dir, watch)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a VLA policy on the Franka pick task; report success rate.")
    ap.add_argument("--group", type=int, default=10, metavar="N", help="rollouts (seeds 0..N-1)")
    ap.add_argument("--noop", action="store_true", help="use the no-op agent (no GPU/model) to test the wire")
    ap.add_argument("--remote", default=None, metavar="HOST:PORT", help="run inference on a remote openpi/0 policy server (GPU box); see serve/")
    ap.add_argument("--agent", default=None, metavar="MODULE:CLASS", help="bring your own RobotAgent, e.g. agents.vla_agent:CustomAgent")
    ap.add_argument("--checkpoint", default="lerobot/pi05_libero_finetuned_v044", help="LeRobot checkpoint for the pi0.5 agent")
    ap.add_argument("--threshold", type=float, default=0.999, metavar="R", help="reward >= R counts as success (1.0 = fully lifted)")
    ap.add_argument("--max-steps", type=int, default=200, metavar="N", help="max control steps per rollout")
    ap.add_argument("--max-concurrent", type=int, default=1, metavar="N", help="parallel rollouts (one sim per env process)")
    ap.add_argument("--record", default=None, metavar="DIR", help="record (obs, action) episodes to DIR (env-side dataset)")
    ap.add_argument("--instruction", default="pick up the red block", help="language instruction handed to the policy")
    ap.add_argument("--target-object", default="block", help="scene body the task scores on lifting")
    ap.add_argument("--lift-height", type=float, default=0.55, metavar="Z", help="height (m) the object must reach for success")
    ap.add_argument("--diagnose-dir", default=None, metavar="DIR", help="where to dump emitted telemetry spans for the post-run reconciliation report")
    asyncio.run(run_eval(ap.parse_args()))


if __name__ == "__main__":
    main()
